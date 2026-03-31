use thiserror::Error;

#[derive(Debug, Error)]
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
    #[error("unimplemented in milestone 0")]
    Unimplemented,
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
