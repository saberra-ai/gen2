//! Microphone lifecycle adapted from pio-nola's desktop flow:
//! pio-app/src-tauri/plugins/tauri-plugin-voice-capture/src/desktop.rs:1-390
//! and pio-nola-tauri/src/api/capture.rs:483-520. A recorder thread owns the
//! !Send CPAL stream, callbacks downmix, stop flushes and joins before return.
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Serialize;
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Serialize)]
pub struct CaptureState {
    pub phase: String,
    pub seconds: f64,
    pub level: f32,
    pub error: Option<String>,
    pub sections: u64,
}
struct Handle {
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<CaptureState>>,
    thread: thread::JoinHandle<Result<PathBuf, String>>,
}
#[derive(Default)]
pub struct Capture {
    handle: Mutex<Option<Handle>>,
}
impl Capture {
    pub fn start(&self, path: PathBuf) -> Result<(), String> {
        self.start_output(path, false, None)
    }
    pub fn start_live(
        &self,
        directory: PathBuf,
        speakers: Arc<crate::live_speakers::LiveSpeakers>,
    ) -> Result<(), String> {
        self.start_output(directory, true, Some(speakers))
    }
    fn start_output(
        &self,
        path: PathBuf,
        live: bool,
        speakers: Option<Arc<crate::live_speakers::LiveSpeakers>>,
    ) -> Result<(), String> {
        let mut guard = self.handle.lock().unwrap();
        if guard.is_some() {
            return Err("A recording is already active.".into());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(CaptureState {
            phase: "starting".into(),
            seconds: 0.,
            level: 0.,
            error: None,
            sections: 0,
        }));
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let thread_stop = stop.clone();
        let thread_state = state.clone();
        let handle = thread::Builder::new()
            .name("pnola-microphone".into())
            .spawn(move || {
                let result = record(
                    path,
                    live,
                    speakers,
                    thread_stop,
                    &thread_state,
                    ready_tx.clone(),
                );
                if let Err(error) = &result {
                    let mut state = thread_state.lock().unwrap();
                    state.phase = "error".into();
                    state.error = Some(error.clone());
                    let _ = ready_tx.send(Err(error.clone()));
                }
                result
            })
            .map_err(|e| e.to_string())?;
        *guard = Some(Handle {
            stop,
            state,
            thread: handle,
        });
        match ready_rx.recv_timeout(Duration::from_secs(20)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                if let Some(handle) = guard.take() {
                    handle.stop.store(true, Ordering::Release);
                    let _ = handle.thread.join();
                }
                Err(error)
            }
            Err(_) => {
                if let Some(handle) = guard.take() {
                    handle.stop.store(true, Ordering::Release);
                }
                Err("Microphone did not start within 20 seconds. Check microphone permission in System Settings.".into())
            }
        }
    }
    pub fn state(&self) -> CaptureState {
        self.handle
            .lock()
            .unwrap()
            .as_ref()
            .map(|h| h.state.lock().unwrap().clone())
            .unwrap_or(CaptureState {
                phase: "idle".into(),
                seconds: 0.,
                level: 0.,
                error: None,
                sections: 0,
            })
    }
    pub fn stop(&self) -> Result<PathBuf, String> {
        let handle = self
            .handle
            .lock()
            .unwrap()
            .take()
            .ok_or("No recording is active.")?;
        handle.stop.store(true, Ordering::Release);
        handle
            .thread
            .join()
            .map_err(|_| "Microphone recorder stopped unexpectedly.".to_string())?
    }
    pub fn stop_live(&self) -> Result<CaptureState, String> {
        let handle = self
            .handle
            .lock()
            .unwrap()
            .take()
            .ok_or("No recording is active.")?;
        handle.stop.store(true, Ordering::Release);
        let result = handle.thread.join();
        let mut state = handle.state.lock().unwrap().clone();
        if let Err(error) =
            result.unwrap_or_else(|_| Err("Microphone recorder stopped unexpectedly.".into()))
        {
            state.error = Some(error);
            state.phase = "error".into();
        }
        Ok(state)
    }
    pub fn request_stop(&self) {
        if let Some(handle) = self.handle.lock().unwrap().as_ref() {
            handle.stop.store(true, Ordering::Release);
        }
    }
    pub fn shutdown(&self) {
        if self.handle.lock().unwrap().is_some() {
            let _ = self.stop();
        }
    }
}
fn record(
    path: PathBuf,
    live: bool,
    speakers: Option<Arc<crate::live_speakers::LiveSpeakers>>,
    stop: Arc<AtomicBool>,
    state: &Arc<Mutex<CaptureState>>,
    ready: mpsc::SyncSender<Result<(), String>>,
) -> Result<PathBuf, String> {
    let device = cpal::default_host()
        .default_input_device()
        .ok_or("No microphone input device was found.")?;
    let supported = device
        .default_input_config()
        .map_err(|e| format!("Microphone unavailable: {e}"))?;
    let rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(32);
    let error_state = state.clone();
    let error_stop = stop.clone();
    let on_error = move |error| {
        let mut state = error_state.lock().unwrap();
        state.error = Some(format!("Microphone stream failed: {error}"));
        state.phase = "error".into();
        error_stop.store(true, Ordering::Release);
    };
    let callback_state = state.clone();
    let callback_stop = stop.clone();
    let emit = move |mono: Vec<f32>| {
        if mono.is_empty() || callback_stop.load(Ordering::Acquire) {
            return;
        }
        let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt();
        callback_state.lock().unwrap().level = rms.sqrt().clamp(0., 1.);
        if tx.try_send(mono).is_err() {
            let mut state = callback_state.lock().unwrap();
            state.phase = "error".into();
            state.error = Some(
                "The microphone writer could not keep up. Recording stopped to avoid losing audio."
                    .into(),
            );
            callback_stop.store(true, Ordering::Release);
        }
    };
    let stream=match format {
        cpal::SampleFormat::F32=>device.build_input_stream(config,move |data:&[f32],_|emit(downmix(data,channels,|s|s)),on_error,None),
        cpal::SampleFormat::I16=>device.build_input_stream(config,move |data:&[i16],_|emit(downmix(data,channels,|s|s as f32/32768.)),on_error,None),
        cpal::SampleFormat::U16=>device.build_input_stream(config,move |data:&[u16],_|emit(downmix(data,channels,|s|(s as f32-32768.)/32768.)),on_error,None),
        _=>return Err("This microphone's sample format is unsupported.".into())
    }.map_err(|e|format!("Could not start the microphone. Allow Pnola in System Settings → Privacy & Security → Microphone. {e}"))?;
    let mut output = if live {
        Output::Live {
            writer: crate::live_audio::Sections::new(&path)?,
            speakers,
        }
    } else {
        Output::File(
            hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .map_err(|e| e.to_string())?,
        )
    };
    stream.play().map_err(|e| e.to_string())?;
    let started = Instant::now();
    let mut resampler = Resampler::new(rate);
    let mut count = 0u64;
    state.lock().unwrap().phase = "recording".into();
    let _ = ready.send(Ok(()));
    while !stop.load(Ordering::Acquire) {
        if let Ok(samples) = rx.recv_timeout(Duration::from_millis(100)) {
            let output_samples = resampler.process(&samples);
            let length = if live {
                output_samples.len()
            } else {
                output_samples
                    .len()
                    .min((16000 * 7200u64).saturating_sub(count) as usize)
            };
            output.push(&output_samples[..length])?;
            count += length as u64;
            if !live && count >= 16000 * 7200 {
                break;
            }
        }
        {
            let mut value = state.lock().unwrap();
            value.seconds = started.elapsed().as_secs_f64();
            value.sections = count / crate::live_audio::SECTION_FRAMES as u64;
        }
    }
    drop(stream);
    while let Ok(samples) = rx.try_recv() {
        let output_samples = resampler.process(&samples);
        let length = if live {
            output_samples.len()
        } else {
            output_samples
                .len()
                .min((16000 * 7200u64).saturating_sub(count) as usize)
        };
        output.push(&output_samples[..length])?;
        count += length as u64;
    }
    output.finish()?;
    let mut state = state.lock().unwrap();
    state.seconds = count as f64 / 16000.;
    state.sections = count.div_ceil(crate::live_audio::SECTION_FRAMES as u64);
    state.level = 0.;
    if let Some(error) = &state.error {
        return Err(error.clone());
    }
    state.phase = "stopped".into();
    if !live && count < 320 {
        return Err("Recording was too short. Speak for at least a moment.".into());
    }
    Ok(path)
}
enum Output {
    File(hound::WavWriter<std::io::BufWriter<std::fs::File>>),
    Live {
        writer: crate::live_audio::Sections,
        speakers: Option<Arc<crate::live_speakers::LiveSpeakers>>,
    },
}
impl Output {
    fn push(&mut self, samples: &[i16]) -> Result<(), String> {
        match self {
            Self::Live { writer, speakers } => {
                writer.push(samples)?;
                if let Some(speakers) = speakers {
                    speakers.push(samples);
                }
                Ok(())
            }
            Self::File(writer) => {
                for sample in samples {
                    writer.write_sample(*sample).map_err(|e| e.to_string())?;
                }
                Ok(())
            }
        }
    }
    fn finish(self) -> Result<(), String> {
        match self {
            Self::Live {
                mut writer,
                speakers,
            } => {
                writer.flush()?;
                if let Some(speakers) = speakers {
                    speakers.close_input();
                }
                Ok(())
            }
            Self::File(writer) => writer.finalize().map_err(|e| e.to_string()),
        }
    }
}
fn downmix<T: Copy>(data: &[T], channels: usize, convert: impl Fn(T) -> f32) -> Vec<f32> {
    data.chunks_exact(channels)
        .map(|frame| frame.iter().map(|&s| convert(s)).sum::<f32>() / channels as f32)
        .collect()
}
// Same fractional carry across callback batches as the pio-nola reference.
struct Resampler {
    step: f64,
    pos: f64,
    last: f32,
}
impl Resampler {
    fn new(rate: u32) -> Self {
        Self {
            step: rate as f64 / 16000.,
            pos: 0.,
            last: 0.,
        }
    }
    fn process(&mut self, input: &[f32]) -> Vec<i16> {
        let mut output = Vec::new();
        if input.is_empty() {
            return output;
        }
        while self.pos < input.len() as f64 {
            let i = self.pos.floor() as usize;
            let fraction = (self.pos - i as f64) as f32;
            let a = if i == 0 { self.last } else { input[i - 1] };
            let b = input[i.min(input.len() - 1)];
            output.push(((a + (b - a) * fraction).clamp(-1., 1.) * 32767.) as i16);
            self.pos += self.step;
        }
        self.pos -= input.len() as f64;
        self.last = *input.last().unwrap();
        output
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires real Nemotron/Bark assets and PNOLA_VERIFY_AUDIO; streams the fixture at microphone cadence"]
    fn real_live_pcm_tee_preserves_audio_and_speaker_session() {
        let engine = crate::engine::Engine::new();
        engine.load(std::path::Path::new("/missing-packaged-resources"));
        let job = engine.claim("live-qualification".into()).unwrap();
        let input = std::env::var("PNOLA_VERIFY_AUDIO").expect("PNOLA_VERIFY_AUDIO required");
        let mut reader = hound::WavReader::open(&input).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 16000);
        assert_eq!(reader.spec().bits_per_sample, 16);
        let samples = reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(samples.len() >= 10 * 16000 && samples.len() <= 120 * 16000);
        let directory = tempfile::tempdir().unwrap();
        let speakers = crate::live_speakers::LiveSpeakers::start(&job);
        assert_eq!(
            speakers.snapshot().status,
            crate::live_speakers::Status::Active
        );
        let session_id = speakers.snapshot().session_id;
        let mut output = Output::Live {
            writer: crate::live_audio::Sections::new(directory.path()).unwrap(),
            speakers: Some(speakers.clone()),
        };
        let started = Instant::now();
        let mut pushed = 0;
        let mut revision = 0;
        let mut revisions = Vec::new();
        for block in samples.chunks(320) {
            output.push(block).unwrap();
            pushed += block.len();
            let snapshot = speakers.snapshot();
            assert_eq!(
                snapshot.status,
                crate::live_speakers::Status::Active,
                "{:?}",
                snapshot.reason
            );
            assert_eq!(snapshot.session_id, session_id);
            assert!(snapshot.revision >= revision);
            if snapshot.revision > revision {
                revisions.push(serde_json::json!({"revision":snapshot.revision,"elapsed_seconds":started.elapsed().as_secs_f64(),"accepted_audio_seconds":snapshot.accepted_audio_seconds,"processed_audio_seconds":snapshot.processed_audio_seconds}));
                revision = snapshot.revision;
            }
            let target = Duration::from_secs_f64(pushed as f64 / 16000.);
            if let Some(wait) = target.checked_sub(started.elapsed()) {
                thread::sleep(wait);
            }
        }
        assert!(!revisions.is_empty());
        output.finish().unwrap();
        speakers.finish();
        let final_snapshot = speakers.snapshot();
        assert_eq!(
            final_snapshot.status,
            crate::live_speakers::Status::Complete,
            "{:?}",
            final_snapshot.reason
        );
        assert_eq!(final_snapshot.session_id, session_id);
        assert!(final_snapshot.revision > revision);
        assert!(
            (final_snapshot.accepted_audio_seconds - samples.len() as f64 / 16000.).abs() < 1e-6
        );
        let mut paths = std::fs::read_dir(directory.path())
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect::<Vec<_>>();
        paths.sort();
        let mut saved = Vec::new();
        for path in &paths {
            saved.extend(
                hound::WavReader::open(path)
                    .unwrap()
                    .samples::<i16>()
                    .map(Result::unwrap),
            );
        }
        assert_eq!(saved, samples);
        let reference = job
            .diarizer
            .as_ref()
            .unwrap()
            .diarize(
                gen2::diarization::Audio::mono_16khz(
                    samples.iter().map(|s| *s as f32 / 32768.).collect(),
                )
                .unwrap(),
                Default::default(),
            )
            .unwrap();
        assert_eq!(
            final_snapshot.timeline.as_ref().unwrap().turns,
            reference.turns
        );
        assert!(
            (final_snapshot.processed_audio_seconds - reference.audio_duration_seconds).abs()
                < 1e-6
        );
        let receipt = serde_json::json!({"passed":true,"microphone_used":false,"cadence":"320 normalized samples every 20 ms through capture Output::Live","audio":input,"audio_seconds":samples.len() as f64/16000.,"section_count":paths.len(),"final":final_snapshot,"provisional_revisions":revisions,"final_matches_recording":true,"saved_pcm_matches_input":true});
        if let Ok(path) = std::env::var("PNOLA_VERIFY_OUTPUT") {
            std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        }
        drop(job);
        engine.shutdown();
    }

    #[test]
    fn downmix_averages_stereo() {
        assert_eq!(downmix(&[1., -1., 0.5, 0.5], 2, |s| s), vec![0., 0.5]);
    }
    #[test]
    fn resampling_preserves_callback_continuity() {
        let input: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.01).sin()).collect();
        let whole = Resampler::new(48000).process(&input);
        let mut streaming = Resampler::new(48000);
        let joined: Vec<_> = input
            .chunks(511)
            .flat_map(|part| streaming.process(part))
            .collect();
        assert_eq!(whole, joined);
        assert_eq!(joined.len(), 16000);
    }
}
