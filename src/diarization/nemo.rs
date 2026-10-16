use super::{
    Audio, Diarization, DiarizationBundle, DiarizationError, DiarizeOptions, ExecutionDevice,
    FrameProbabilities, LoadOptions, MODEL_REVISION, Result, SpeakerTurn, ffi,
};
use std::{
    ffi::{CString, c_void},
    ptr::NonNull,
    sync::{Arc, atomic::Ordering, mpsc},
    time::Duration,
    time::Instant,
};

pub(super) struct Native {
    model: NonNull<c_void>,
    api: ffi::Api,
    options: LoadOptions,
    device: Option<String>,
}
impl Native {
    pub fn load(bundle: &DiarizationBundle, options: LoadOptions) -> Result<Self> {
        let api = ffi::Api::load(&options.native_library)?;
        let (gpu, device) = match options.device {
            ExecutionDevice::Cpu => (-1, None),
            ExecutionDevice::Metal => {
                let (index, description) = api.metal_device()?;
                (index, Some(description))
            }
        };
        let path = CString::new(bundle.model_path().to_str().ok_or_else(|| {
            DiarizationError::InvalidBundle("native model path must be UTF-8".into())
        })?)
        .map_err(|e| DiarizationError::InvalidBundle(e.to_string()))?;
        let g = options.preset.geometry();
        let config = ffi::ModelConfig {
            size: std::mem::size_of::<ffi::ModelConfig>(),
            model_path: path.as_ptr(),
            gpu,
            preset: c"v3-streaming".as_ptr(),
            chunk_frames: g.chunk,
            right_context_frames: g.right,
            left_context_frames: g.left,
            fifo_frames: g.fifo,
            spkcache_frames: g.cache,
            update_period_frames: g.update,
        };
        let mut pointer = std::ptr::null_mut();
        // SAFETY: config has the pinned C layout; SDK copies strings/config at create.
        unsafe {
            api.check((api.create)(&config, &mut pointer))?;
        }
        let model = NonNull::new(pointer)
            .ok_or_else(|| DiarizationError::InvalidOutput("null model after create".into()))?;
        let native = Self {
            model,
            api,
            options,
            device,
        };
        // SAFETY: a created model remains owned until Drop on this thread.
        unsafe {
            let cadence = (native.api.cadence)(pointer);
            if (native.api.speakers)(pointer) != 8
                || !cadence.is_finite()
                || (cadence - 0.01).abs() > 1e-9
            {
                return Err(DiarizationError::BackendUnavailable(
                    "expected V3 eight-speaker/10 ms model".into(),
                ));
            }
        }
        Ok(native)
    }
    pub fn run(&mut self, audio: Audio, options: &DiarizeOptions) -> Result<Diarization> {
        options.check()?;
        let mut stream = self.open()?;
        stream.push(&audio.samples, options)?;
        stream.finish(options)
    }
    pub fn run_live(
        &mut self,
        input: mpsc::Receiver<Vec<f32>>,
        shared: &super::stream::Shared,
        ready: mpsc::Sender<Result<()>>,
    ) -> Result<Diarization> {
        let mut stream = match self.open() {
            Ok(stream) => stream,
            Err(error) => {
                let _ = ready.send(Err(error.clone()));
                return Err(error);
            }
        };
        let _ = ready.send(Ok(()));
        let mut last_snapshot = 0;
        loop {
            shared.options.check()?;
            // An empty queue plus finish means every accepted block was consumed.
            let block = match input.try_recv() {
                Ok(block) => Some(block),
                Err(mpsc::TryRecvError::Empty)
                    if shared.finish_requested.load(Ordering::Acquire) =>
                {
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    match input.recv_timeout(Duration::from_millis(10)) {
                        Ok(block) => Some(block),
                        Err(mpsc::RecvTimeoutError::Timeout) => None,
                        Err(_) => return Err(DiarizationError::Cancelled),
                    }
                }
                Err(_) => return Err(DiarizationError::Cancelled),
            };
            if let Some(block) = block {
                stream.push(&block, &shared.options)?;
                if stream.samples - last_snapshot >= 5 * 16000 {
                    shared.publish(Arc::new(stream.results(false)?), false);
                    last_snapshot = stream.samples;
                }
            }
        }
        stream.finish(&shared.options)
    }
    fn open(&mut self) -> Result<Session<'_>> {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: only this worker accesses the native model; the stream borrows it.
        unsafe {
            self.api
                .check((self.api.open)(self.model.as_ptr(), &mut pointer))?;
        }
        Ok(Session {
            pointer: NonNull::new(pointer)
                .ok_or_else(|| DiarizationError::InvalidOutput("null stream after open".into()))?,
            native: self,
            samples: 0,
            finished: false,
            started: Instant::now(),
            compute: Duration::ZERO,
        })
    }
}
impl Drop for Native {
    fn drop(&mut self) {
        // SAFETY: borrowed sessions have already closed; API/library are live.
        unsafe {
            (self.api.destroy)(self.model.as_ptr());
        }
    }
}
struct Session<'a> {
    pointer: NonNull<c_void>,
    native: &'a mut Native,
    samples: usize,
    finished: bool,
    started: Instant,
    compute: Duration,
}
impl Session<'_> {
    fn push(&mut self, samples: &[f32], options: &DiarizeOptions) -> Result<()> {
        if self.samples + samples.len() > super::MAX_SESSION_SAMPLES {
            return Err(DiarizationError::ResourceLimit(
                "one-hour session limit".into(),
            ));
        }
        for block in samples.chunks(2560) {
            options.check()?;
            let started = Instant::now();
            // SAFETY: the PCM slice lives through this synchronous call.
            unsafe {
                self.native.api.check((self.native.api.push)(
                    self.pointer.as_ptr(),
                    block.as_ptr(),
                    block.len(),
                    16000,
                ))?;
            }
            self.compute += started.elapsed();
            self.samples += block.len();
        }
        Ok(())
    }
    fn finish(&mut self, options: &DiarizeOptions) -> Result<Diarization> {
        if self.samples < 320 {
            return Err(DiarizationError::InvalidAudio(
                "recording needs at least 20 ms of PCM".into(),
            ));
        }
        options.check()?;
        let started = Instant::now();
        // SAFETY: finish runs once; the stream is still owned here.
        unsafe {
            self.native
                .api
                .check((self.native.api.finish)(self.pointer.as_ptr()))?;
        }
        self.compute += started.elapsed();
        self.finished = true;
        options.check()?;
        self.results(options.frame_probabilities)
    }
    fn results(&self, include_frames: bool) -> Result<Diarization> {
        let duration = self.samples as f64 / 16000.;
        let expected_frames = self.samples / 160 + 1;
        let api = &self.native.api;
        let pointer = self.pointer.as_ptr();
        let invalid = |s: &str| DiarizationError::InvalidOutput(s.into());
        // SAFETY: this thread owns the finished stream and bounds all copied output.
        unsafe {
            let count = (api.frame_count)(pointer);
            if count < 0
                || count as usize > expected_frames
                || (self.finished && count as usize != expected_frames)
            {
                return Err(invalid("invalid/incomplete native frame count"));
            }
            let mut length = 0usize;
            api.check((api.segments)(
                pointer,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                &mut length,
            ))?;
            if length > 100_000 {
                return Err(invalid("excessive speaker turns"));
            }
            let mut raw = vec![ffi::Segment::default(); length];
            if length > 0 {
                api.check((api.segments)(
                    pointer,
                    std::ptr::null(),
                    raw.as_mut_ptr(),
                    raw.len(),
                    &mut length,
                ))?;
            }
            if length > raw.len() {
                return Err(invalid("native segment count changed"));
            }
            let mut turns = Vec::with_capacity(length);
            let mut previous = 0.;
            for s in raw.into_iter().take(length) {
                if !s.start.is_finite()
                    || !s.end.is_finite()
                    || s.start < 0.
                    || s.end < s.start
                    || s.start < previous
                    || s.end > count as f64 * 0.01 + 1e-4
                    || !(1..=8).contains(&s.speaker)
                {
                    return Err(invalid("invalid speaker interval"));
                }
                previous = s.start;
                // Only the proven padded terminal span is clipped. Other bad output fails above.
                if s.start < duration {
                    turns.push(SpeakerTurn {
                        start_seconds: s.start,
                        end_seconds: s.end.min(duration),
                        speaker: s.speaker as u8,
                    });
                }
            }
            let frame_probabilities = if include_frames {
                let first = (api.probs_start)(pointer);
                if first < 0 || first > count {
                    return Err(invalid("invalid retained frame range"));
                }
                let mut values = vec![[0f32; 8]; (count - first) as usize];
                api.check((api.probs)(
                    pointer,
                    values.as_mut_ptr().cast(),
                    values.len() * 8,
                ))?;
                if values
                    .iter()
                    .flatten()
                    .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
                {
                    return Err(invalid("invalid speaker probability"));
                }
                Some(FrameProbabilities {
                    first_frame: first as u64,
                    seconds_per_frame: 0.01,
                    values,
                })
            } else {
                None
            };
            Ok(Diarization {
                turns,
                audio_duration_seconds: duration,
                native_frame_count: count as u64,
                frame_probabilities,
                model_revision: MODEL_REVISION,
                runtime_version: api.version.clone(),
                geometry: self.native.options.preset.geometry(),
                execution_micros: self.started.elapsed().as_micros().min(u64::MAX as u128) as u64,
                native_compute_micros: self.compute.as_micros().min(u64::MAX as u128) as u64,
                backend: match self.native.options.device {
                    ExecutionDevice::Cpu => "cpu",
                    ExecutionDevice::Metal => "metal",
                },
                device: self.native.device.clone(),
            })
        }
    }
}
impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: no synchronous call is active; closing precedes model/library destruction.
        unsafe {
            (self.native.api.close)(self.pointer.as_ptr());
        }
    }
}
