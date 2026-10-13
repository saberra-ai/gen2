//! Reusable local speech recognition. ONNX CPU is enabled with `asr-dynamic`.
//!
//! Audio capture belongs to the host. Submit a completed English recording;
//! Bark produces a cleaned transcript without a second text-generation model.

mod audio;
pub use audio::Audio;
mod bundle;
pub use bundle::{MANIFEST_SHA256, MODEL_REPOSITORY, MODEL_REVISION, SpeechBundle};
mod provision;
pub use provision::download_bark;
mod worker;
pub use worker::{
    Cancellation, LoadOptions, SpeechCapabilities, SpeechStatus, TranscribeOptions, Transcriber,
    Transcription,
};
pub(crate) use worker::{SpeechMonitor, spawn};
#[cfg(feature = "backend-asr-onnx")]
mod frontend;
#[cfg(feature = "backend-asr-onnx")]
mod onnx;

/// Speech failures have stable categories; callers need not parse display text.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SpeechError {
    #[error("invalid audio: {0}")]
    InvalidAudio(String),
    #[error("invalid speech bundle: {0}")]
    InvalidBundle(String),
    #[error("speech runtime unavailable: {0}")]
    BackendUnavailable(String),
    #[error("speech execution failed: {0}")]
    Execution(String),
    #[error("invalid speech output: {0}")]
    InvalidOutput(String),
    #[error("speech resource limit: {0}")]
    ResourceLimit(String),
    #[error("speech queue is full")]
    Busy,
    #[error("speech request cancelled")]
    Cancelled,
    #[error("speech deadline expired")]
    DeadlineExceeded,
    #[error("speech model is unloaded")]
    Unloaded,
}
impl SpeechError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidAudio(_) => "speech_invalid_audio",
            Self::InvalidBundle(_) => "speech_invalid_bundle",
            Self::BackendUnavailable(_) => "speech_backend_unavailable",
            Self::Execution(_) => "speech_execution",
            Self::InvalidOutput(_) => "speech_invalid_output",
            Self::ResourceLimit(_) => "speech_resource_limit",
            Self::Busy => "speech_busy",
            Self::Cancelled => "speech_cancelled",
            Self::DeadlineExceeded => "speech_deadline",
            Self::Unloaded => "speech_unloaded",
        }
    }
}
pub type Result<T> = std::result::Result<T, SpeechError>;
