/// Stable failure categories for loading and running speaker diarization.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum DiarizationError {
    #[error("invalid diarization audio: {0}")]
    InvalidAudio(String),
    #[error("invalid diarization bundle: {0}")]
    InvalidBundle(String),
    #[error("diarization backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("diarization execution failed: {0}")]
    Execution(String),
    #[error("invalid diarization output: {0}")]
    InvalidOutput(String),
    #[error("diarization resource limit: {0}")]
    ResourceLimit(String),
    #[error("diarizer is busy")]
    Busy,
    #[error("diarization cancelled")]
    Cancelled,
    #[error("diarization deadline expired")]
    DeadlineExceeded,
    #[error("diarizer is unloaded")]
    Unloaded,
    #[error("invalid diarization stream state: {0}")]
    InvalidState(String),
}
impl DiarizationError {
    /// Machine-readable category; callers need not parse the message.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidAudio(_) => "diarization_invalid_audio",
            Self::InvalidBundle(_) => "diarization_invalid_bundle",
            Self::BackendUnavailable(_) => "diarization_backend_unavailable",
            Self::Execution(_) => "diarization_execution",
            Self::InvalidOutput(_) => "diarization_invalid_output",
            Self::ResourceLimit(_) => "diarization_resource_limit",
            Self::Busy => "diarization_busy",
            Self::Cancelled => "diarization_cancelled",
            Self::DeadlineExceeded => "diarization_deadline",
            Self::Unloaded => "diarization_unloaded",
            Self::InvalidState(_) => "diarization_invalid_state",
        }
    }
}
/// Diarization result with typed errors.
pub type Result<T> = std::result::Result<T, DiarizationError>;
