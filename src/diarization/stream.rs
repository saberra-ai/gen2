use super::{Diarization, DiarizationError, DiarizeOptions, Diarizer, Result};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

/// At most 1.04 seconds of normalized PCM per transport block.
pub const MAX_PUSH_SAMPLES: usize = 16_640;
/// Hard safety cap: one hour. Release qualification is tracked separately.
pub const MAX_SESSION_SAMPLES: usize = 3600 * 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamStatus {
    Active,
    Finishing,
    Finished,
    Cancelled,
    Failed,
}

/// Replace the previous snapshot for this session; no prefix is immutable.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub session_id: uuid::Uuid,
    pub revision: u64,
    pub complete: bool,
    /// Native output position, not a stable-prefix watermark.
    pub processed_audio_seconds: f64,
    pub result: Arc<Diarization>,
}

/// Rejected PCM remains owned by the caller, including on backpressure (`Busy`).
#[derive(Debug)]
pub struct PushError {
    pub samples: Vec<f32>,
    pub error: DiarizationError,
}

struct Output {
    status: StreamStatus,
    #[cfg(feature = "backend-diar-nemo")]
    revision: u64,
    snapshot: Option<Snapshot>,
    terminal: Option<Result<Arc<Diarization>>>,
}
pub(super) struct Shared {
    pub options: DiarizeOptions,
    pub finish_requested: AtomicBool,
    id: uuid::Uuid,
    output: Mutex<Output>,
    done: Condvar,
}
impl Shared {
    pub fn new(options: DiarizeOptions) -> Arc<Self> {
        Arc::new(Self {
            options,
            finish_requested: AtomicBool::new(false),
            id: uuid::Uuid::new_v4(),
            output: Mutex::new(Output {
                status: StreamStatus::Active,
                #[cfg(feature = "backend-diar-nemo")]
                revision: 0,
                snapshot: None,
                terminal: None,
            }),
            done: Condvar::new(),
        })
    }
    #[cfg(feature = "backend-diar-nemo")]
    pub fn publish(&self, result: Arc<Diarization>, complete: bool) {
        let mut output = self.output.lock().unwrap_or_else(|e| e.into_inner());
        output.revision += 1;
        output.snapshot = Some(Snapshot {
            session_id: self.id,
            revision: output.revision,
            complete,
            processed_audio_seconds: (result.native_frame_count as f64 * 0.01)
                .min(result.audio_duration_seconds),
            result,
        });
    }
    #[cfg(feature = "backend-diar-nemo")]
    pub fn complete(&self, result: Result<Diarization>) {
        let result = result.map(Arc::new);
        if let Ok(result) = &result {
            self.publish(Arc::clone(result), true);
        }
        let mut output = self.output.lock().unwrap_or_else(|e| e.into_inner());
        output.status = match &result {
            Ok(_) => StreamStatus::Finished,
            Err(DiarizationError::Cancelled) => StreamStatus::Cancelled,
            Err(_) => StreamStatus::Failed,
        };
        output.terminal = Some(result);
        self.done.notify_all();
    }
}

/// One recording's native speaker cache. The two-block PCM queue is bounded;
/// finish and cancellation use a separate control path and cannot fill it.
pub struct DiarizationStream {
    _owner: Diarizer,
    tx: mpsc::SyncSender<Vec<f32>>,
    shared: Arc<Shared>,
    accepted: usize,
}
impl DiarizationStream {
    pub(super) fn new(
        owner: Diarizer,
        tx: mpsc::SyncSender<Vec<f32>>,
        shared: Arc<Shared>,
    ) -> Self {
        Self {
            _owner: owner,
            tx,
            shared,
            accepted: 0,
        }
    }
    pub fn session_id(&self) -> uuid::Uuid {
        self.shared.id
    }
    pub fn status(&self) -> StreamStatus {
        self.shared
            .output
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .status
    }
    pub fn accepted_samples(&self) -> usize {
        self.accepted
    }
    pub fn try_push(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> std::result::Result<(), PushError> {
        let reject = |samples, error| Err(PushError { samples, error });
        if self.status() != StreamStatus::Active {
            return reject(
                samples,
                DiarizationError::InvalidState("stream no longer accepts PCM".into()),
            );
        }
        if let Err(error) = self.shared.options.check() {
            return reject(samples, error);
        }
        if sample_rate != 16000
            || samples.is_empty()
            || samples.len() > MAX_PUSH_SAMPLES
            || samples.iter().any(|x| !x.is_finite())
        {
            return reject(
                samples,
                DiarizationError::InvalidAudio(
                    "use 1–16640 finite mono samples at 16 kHz per push".into(),
                ),
            );
        }
        if self.accepted + samples.len() > MAX_SESSION_SAMPLES {
            return reject(
                samples,
                DiarizationError::ResourceLimit("one-hour session limit".into()),
            );
        }
        let length = samples.len();
        match self.tx.try_send(samples) {
            Ok(()) => {
                self.accepted += length;
                Ok(())
            }
            Err(mpsc::TrySendError::Full(samples)) => reject(samples, DiarizationError::Busy),
            Err(mpsc::TrySendError::Disconnected(samples)) => reject(
                samples,
                DiarizationError::InvalidState("stream worker stopped".into()),
            ),
        }
    }
    /// Coalesced snapshots are produced at most once per five seconds of audio.
    /// Holding a snapshot never blocks native execution; consumers bound their own history.
    pub fn snapshot(&self) -> Option<Snapshot> {
        self.shared
            .output
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot
            .clone()
    }
    pub fn cancel(&self) {
        self.shared.options.cancellation.cancel();
    }
    /// Drain accepted input, flush the tail once, and cache success or failure.
    /// This waits for native teardown of the stream; a kernel cannot be interrupted.
    pub fn finish(&mut self) -> Result<Arc<Diarization>> {
        let mut output = self.shared.output.lock().unwrap_or_else(|e| e.into_inner());
        if output.terminal.is_none() {
            output.status = StreamStatus::Finishing;
            self.shared.finish_requested.store(true, Ordering::Release);
        }
        while output.terminal.is_none() {
            output = self
                .shared
                .done
                .wait(output)
                .unwrap_or_else(|e| e.into_inner());
        }
        output.terminal.as_ref().expect("terminal checked").clone()
    }
}
impl Drop for DiarizationStream {
    fn drop(&mut self) {
        if self
            .shared
            .output
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .terminal
            .is_none()
        {
            self.cancel();
        }
    }
}
