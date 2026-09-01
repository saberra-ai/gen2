use thiserror::Error;

impl ExecError {
    /// Wrap anything printable as an I/O failure.
    ///
    /// Lets fallible readers keep the `map_err(ExecError::io)` shape they had
    /// when this crate borrowed the host's error type.
    pub fn io(msg: impl std::fmt::Display) -> Self {
        Self::Io(msg.to_string())
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ExecError {
    #[error("model not loaded")]
    ModelNotLoaded,
    #[error("embedder not loaded")]
    EmbedderNotLoaded,
    #[error("feature not supported: {0}")]
    FeatureUnsupported(&'static str),
    #[error("invalid argument: {0}")]
    InvalidArg(&'static str),
    #[error("mmproj incompatible: {0}")]
    MmprojIncompatible(&'static str),
    #[error("settings error: {0}")]
    SettingsError(String),
    #[error("kv incompatible: {0}")]
    KvIncompatible(String),
    #[error("kv corrupt: {0}")]
    KvCorrupt(String),
    #[error("invalid model file: {0}")]
    InvalidModelFile(String),
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("io error: {0}")]
    Io(String),
    /// Context window exceeded — conversation too long.
    #[error("context overflow: {0}")]
    ContextOverflow(String),
    /// Chat template failed to parse.
    #[error("template error: {0}")]
    TemplateError(String),
    /// FFI panic — inference session state is lost.
    #[error("session poisoned: {0}")]
    SessionPoisoned(String),
    /// Metal / CUDA OOM — caught via catch_unwind around forward pass.
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    #[error("unimplemented in milestone 0")]
    Unimplemented,
    /// A generation could not be dispatched or joined (channel closed, worker
    /// gone, blocking task panicked).
    #[error("generation failed: {0}")]
    Generation(String),
    /// An error that arrived already tagged with the host's own error code —
    /// a `ControllerEvent::Error` carrying a snake_case code string.
    ///
    /// gen2 does not own that taxonomy (the host routes each code to a user
    /// action), so it carries the code through verbatim rather than lossily
    /// collapsing it into one of the variants above. The host maps it back
    /// when converting this error into its own type.
    #[error("{message}")]
    Coded { code: String, message: String },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_render() {
        assert_eq!(format!("{}", ExecError::ModelNotLoaded), "model not loaded");
        assert_eq!(
            format!("{}", ExecError::EmbedderNotLoaded),
            "embedder not loaded"
        );
        assert!(format!("{}", ExecError::FeatureUnsupported("images")).contains("images"));
        assert!(
            format!("{}", ExecError::SettingsError("bad temp".into())).contains("settings error")
        );
    }
}
