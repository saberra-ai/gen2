use super::*;
use std::{
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub native_library: Option<PathBuf>,
    pub queue_capacity: usize,
    pub intra_threads: usize,
    /// Conservative peak workspace plus native weights; never below 2 GiB.
    pub reservation_mb: u64,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            native_library: None,
            queue_capacity: 2,
            intra_threads: 4,
            reservation_mb: 4096,
        }
    }
}
impl LoadOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.queue_capacity == 0
            || self.queue_capacity > 64
            || self.intra_threads == 0
            || self.reservation_mb < 2048
        {
            return Err(SpeechError::ResourceLimit(
                "queue capacity must be 1–64, threads nonzero, reservation at least 2048 MB".into(),
            ));
        }
        #[cfg(feature = "onnx-dynamic")]
        if !self.native_library.as_ref().is_some_and(|p| p.is_file()) {
            return Err(SpeechError::BackendUnavailable(
                "supply an ONNX Runtime 1.24 native_library".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    pub cancellation: Cancellation,
    pub deadline: Option<Instant>,
}
impl TranscribeOptions {
    pub(crate) fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            Err(SpeechError::Cancelled)
        } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
            Err(SpeechError::DeadlineExceeded)
        } else {
            Ok(())
        }
    }
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct SpeechCapabilities {
    pub language: &'static str,
    pub sample_rate: u32,
    pub input_sample_rates: &'static [u32],
    pub maximum_channels: usize,
    pub wav_encodings: &'static [&'static str],
    pub minimum_audio_seconds: f64,
    pub maximum_audio_seconds: f64,
    pub streaming: bool,
    pub cleanup: bool,
    pub provider: &'static str,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct Transcription {
    pub text: String,
    pub duration_seconds: f64,
    pub model_revision: &'static str,
    pub manifest_sha256: &'static str,
    pub provider: &'static str,
    pub queue_micros: u64,
    pub frontend_micros: u64,
    pub encoder_micros: u64,
    pub decoder_micros: u64,
    pub execution_micros: u64,
    /// Greedy decoder decisions, useful for independent compatibility checks.
    pub token_ids: Vec<u32>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechStatus {
    Ready,
    Running,
    Stopping,
    Unloaded,
}
struct State {
    status: Mutex<SpeechStatus>,
    done: Condvar,
    stop: AtomicBool,
    active: Mutex<Option<Cancellation>>,
}
struct Inner {
    tx: mpsc::SyncSender<Job>,
    state: Arc<State>,
    capabilities: SpeechCapabilities,
    bundle: SpeechBundle,
}
impl Drop for Inner {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Release);
    }
}
#[cfg_attr(not(feature = "backend-asr-onnx"), allow(dead_code))]
struct Job {
    audio: Audio,
    options: TranscribeOptions,
    queued: Instant,
    reply: mpsc::Sender<Result<Transcription>>,
}
#[derive(Clone)]
pub struct Transcriber {
    inner: Arc<Inner>,
}
pub(crate) struct SpeechMonitor(Weak<Inner>);
impl SpeechMonitor {
    pub(crate) fn upgrade(&self) -> Option<Transcriber> {
        self.0.upgrade().map(|inner| Transcriber { inner })
    }
}
impl std::fmt::Debug for Transcriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transcriber")
            .field("status", &self.status())
            .finish()
    }
}
impl Transcriber {
    pub fn capabilities(&self) -> &SpeechCapabilities {
        &self.inner.capabilities
    }
    pub fn bundle(&self) -> &SpeechBundle {
        &self.inner.bundle
    }
    pub(crate) fn monitor(&self) -> SpeechMonitor {
        SpeechMonitor(Arc::downgrade(&self.inner))
    }
    pub fn status(&self) -> SpeechStatus {
        *self
            .inner
            .state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }
    /// Stop admission without blocking on a native kernel.
    pub fn unload(&self) {
        self.inner.state.stop.store(true, Ordering::Release);
        if let Some(active) = self
            .inner
            .state
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            active.cancel();
        }
        let mut status = self
            .inner
            .state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *status != SpeechStatus::Unloaded {
            *status = SpeechStatus::Stopping;
        }
    }
    /// Wait until native sessions and their memory reservation are released.
    pub fn shutdown(&self) {
        self.unload();
        let mut status = self
            .inner
            .state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while *status != SpeechStatus::Unloaded {
            status = self
                .inner
                .state
                .done
                .wait(status)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
    pub fn transcribe(&self, audio: Audio, options: TranscribeOptions) -> Result<Transcription> {
        options.check()?;
        if self.inner.state.stop.load(Ordering::Acquire) {
            return Err(SpeechError::Unloaded);
        }
        let (reply, rx) = mpsc::channel();
        self.inner
            .tx
            .try_send(Job {
                audio,
                options: options.clone(),
                queued: Instant::now(),
                reply,
            })
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => SpeechError::Busy,
                mpsc::TrySendError::Disconnected(_) => SpeechError::Unloaded,
            })?;
        loop {
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(result) => {
                    options.check()?;
                    return result;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(SpeechError::Unloaded),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if self.inner.state.stop.load(Ordering::Acquire) {
                        return Err(SpeechError::Unloaded);
                    }
                    options.check()?;
                }
            }
        }
    }
}
#[cfg(feature = "backend-asr-onnx")]
pub(crate) fn micros(t: Instant) -> u64 {
    t.elapsed().as_micros().min(u64::MAX as u128) as u64
}

pub(crate) fn spawn(
    bundle: SpeechBundle,
    options: LoadOptions,
    reservation: Box<dyn Send>,
) -> Result<Transcriber> {
    #[cfg(not(feature = "backend-asr-onnx"))]
    {
        let _ = (bundle, options, reservation);
        Err(SpeechError::BackendUnavailable(
            "enable asr-dynamic or backend-asr-onnx".into(),
        ))
    }
    #[cfg(feature = "backend-asr-onnx")]
    {
        options.validate()?;
        let state = Arc::new(State {
            status: Mutex::new(SpeechStatus::Ready),
            done: Condvar::new(),
            stop: AtomicBool::new(false),
            active: Mutex::new(None),
        });
        let worker_state = state.clone();
        let (tx, rx) = mpsc::sync_channel::<Job>(options.queue_capacity);
        let (ready, ready_rx) = mpsc::channel();
        let worker_bundle = bundle.clone();
        std::thread::Builder::new()
            .name("gen2-bark".into())
            .spawn(move || {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut backend = match super::onnx::OnnxBackend::load(&worker_bundle, &options)
                    {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = ready.send(Err(e));
                            return;
                        }
                    };
                    if ready.send(Ok(())).is_err() {
                        return;
                    }
                    while !worker_state.stop.load(Ordering::Acquire) {
                        let job = match rx.recv_timeout(Duration::from_millis(20)) {
                            Ok(j) => j,
                            Err(mpsc::RecvTimeoutError::Timeout) => continue,
                            Err(_) => break,
                        };
                        *worker_state
                            .active
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) =
                            Some(job.options.cancellation.clone());
                        if worker_state.stop.load(Ordering::Acquire) {
                            job.options.cancellation.cancel();
                        }
                        {
                            let mut status = worker_state
                                .status
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            if *status != SpeechStatus::Stopping {
                                *status = SpeechStatus::Running;
                            }
                        }
                        let started = Instant::now();
                        let result = job
                            .options
                            .check()
                            .and_then(|_| backend.run(job.audio, &job.options))
                            .and_then(|mut response| {
                                job.options.check()?;
                                if worker_state.stop.load(Ordering::Acquire) {
                                    return Err(SpeechError::Unloaded);
                                }
                                response.queue_micros = started
                                    .duration_since(job.queued)
                                    .as_micros()
                                    .min(u64::MAX as u128)
                                    as u64;
                                response.execution_micros = micros(started);
                                Ok(response)
                            });
                        *worker_state
                            .active
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = None;
                        let _ = job.reply.send(result);
                        let mut status = worker_state
                            .status
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        if *status != SpeechStatus::Stopping {
                            *status = SpeechStatus::Ready;
                        }
                    }
                    // Backend sessions drop before the reservation and completion signal.
                }));
                drop(rx);
                drop(reservation);
                *worker_state
                    .status
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = SpeechStatus::Unloaded;
                worker_state.done.notify_all();
            })
            .map_err(|e| SpeechError::Execution(e.to_string()))?;
        ready_rx
            .recv()
            .map_err(|_| SpeechError::Execution("speech worker failed during load".into()))??;
        Ok(Transcriber {
            inner: Arc::new(Inner {
                tx,
                state,
                bundle,
                capabilities: SpeechCapabilities {
                    language: "en",
                    sample_rate: 16000,
                    input_sample_rates: &[8000, 16000, 44100, 48000],
                    maximum_channels: 8,
                    wav_encodings: &["pcm16", "float32"],
                    minimum_audio_seconds: 0.02,
                    maximum_audio_seconds: 60.,
                    streaming: false,
                    cleanup: true,
                    provider: "CPUExecutionProvider",
                },
            }),
        })
    }
}
