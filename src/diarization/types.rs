use super::{Cancellation, DiarizationError, Result};
use std::{path::PathBuf, time::Instant};

/// Effective geometry in coarse 80 ms encoder frames, not output frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Geometry {
    pub chunk: i32,
    pub right: i32,
    pub left: i32,
    pub fifo: i32,
    pub cache: i32,
    pub update: i32,
}
/// Qualified stateful streaming configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub enum Preset {
    /// Official Python low-latency configuration qualified in N0.
    #[default]
    LowLatency,
    /// SDK V3 streaming defaults, a separately qualified configuration.
    SdkStreaming,
}
impl Preset {
    pub fn geometry(self) -> Geometry {
        match self {
            Self::LowLatency => Geometry {
                chunk: 9,
                right: 4,
                left: 0,
                fifo: 264,
                cache: 264,
                update: 222,
            },
            Self::SdkStreaming => Geometry {
                chunk: 13,
                right: 1,
                left: 0,
                fifo: 80,
                cache: 264,
                update: 40,
            },
        }
    }
}
/// Explicit native placement. Metal requires a Metal device in the supplied SDK;
/// unavailable placement fails loading instead of silently running on CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionDevice {
    Cpu,
    Metal,
}
/// Native loading and admitted peak residency. No runtime/model auto-download.
/// The supplied SDK must be qualified for the `gen2-valid-mel-v1` or
/// `gen2-valid-mel-pool-v1` runtime profile on the selected host/backend.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub native_library: PathBuf,
    pub device: ExecutionDevice,
    pub preset: Preset,
    /// MiB covering native weights/workspace, bounded PCM and outputs.
    pub reservation_mb: u64,
}
impl LoadOptions {
    /// Explicit CPU execution with a conservative initial 1024 MiB reservation.
    pub fn cpu(native_library: impl Into<PathBuf>) -> Self {
        Self {
            native_library: native_library.into(),
            device: ExecutionDevice::Cpu,
            preset: Preset::default(),
            reservation_mb: 1024,
        }
    }
    /// Explicit Metal execution through a separately built native SDK on macOS.
    /// Gen2's chat `metal` feature does not select this dynamic SDK's placement.
    pub fn metal(native_library: impl Into<PathBuf>) -> Self {
        Self {
            device: ExecutionDevice::Metal,
            ..Self::cpu(native_library)
        }
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if self.device == ExecutionDevice::Metal && !cfg!(target_os = "macos") {
            return Err(DiarizationError::BackendUnavailable(
                "Metal diarization requires macOS".into(),
            ));
        }
        if !(1024..=65536).contains(&self.reservation_mb) {
            return Err(DiarizationError::ResourceLimit(
                "reservation must be 1024–65536 MiB".into(),
            ));
        }
        if !self.native_library.is_file() {
            return Err(DiarizationError::BackendUnavailable(
                "supply the qualified native C ABI library".into(),
            ));
        }
        Ok(())
    }
}
/// Per-recording controls. Native calls cooperate between bounded pushes.
#[derive(Debug, Clone, Default)]
pub struct DiarizeOptions {
    pub cancellation: Cancellation,
    pub deadline: Option<Instant>,
    /// Return bounded raw frame probabilities for diagnostics at the native cadence.
    pub frame_probabilities: bool,
}
impl DiarizeOptions {
    pub(crate) fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            Err(DiarizationError::Cancelled)
        } else if self.deadline.is_some_and(|time| Instant::now() >= time) {
            Err(DiarizationError::DeadlineExceeded)
        } else {
            Ok(())
        }
    }
}
/// Anonymous speaker turn. IDs are one-based and scoped to one recording.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeakerTurn {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub speaker: u8,
}
/// Retained probabilities at the native 10 ms cadence. Rows preserve overlap.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct FrameProbabilities {
    pub first_frame: u64,
    pub seconds_per_frame: f64,
    pub values: Vec<[f32; 8]>,
}
/// A complete recording's timeline; speaker turns are clipped to actual audio.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Diarization {
    pub turns: Vec<SpeakerTurn>,
    pub audio_duration_seconds: f64,
    pub native_frame_count: u64,
    pub frame_probabilities: Option<FrameProbabilities>,
    pub model_revision: &'static str,
    pub runtime_version: String,
    pub geometry: Geometry,
    pub execution_micros: u64,
    /// Time inside synchronous native push/finish calls, excluding idle input waits.
    pub native_compute_micros: u64,
    /// Explicit placement, admitted before model creation; never an automatic fallback.
    pub backend: &'static str,
    /// Selected Metal device description; CPU execution has no GPU device.
    pub device: Option<String>,
}
