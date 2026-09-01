//! The one error type the public API returns.

use thiserror::Error;

/// Anything that can go wrong driving the engine.
///
/// `#[non_exhaustive]` so new failure modes don't break callers' `match`es.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The controller loop is gone — it panicked, or shutdown already ran.
    /// Nothing further can be sent through this engine.
    #[error("the inference controller is no longer running")]
    ControllerGone,

    /// A model must be loaded before generating.
    #[error("no model is loaded")]
    ModelNotLoaded,

    /// The model file could not be loaded.
    #[error("failed to load model: {0}")]
    Load(String),

    /// Generation failed. `code` is the engine's stable identifier for the
    /// failure, suitable for routing a caller to a specific recovery.
    #[error("generation failed [{code}]: {message}")]
    Generation { code: String, message: String },

    /// The model will not run on this machine at the requested context.
    ///
    /// Carries the whole verdict — what was needed, what was available, and
    /// the largest context that would have worked — so a caller can retry
    /// smaller or tell the user precisely what is wrong.
    #[error("{0}")]
    WontFit(Box<crate::api::fit::Fit>),

    /// An engine-internal error surfaced verbatim.
    #[error(transparent)]
    Exec(#[from] crate::engine::ExecError),
}

impl Error {
    /// The engine's stable code for this failure, when it has one.
    ///
    /// Callers route on this rather than matching error text.
    pub fn code(&self) -> Option<&str> {
        match self {
            Self::Generation { code, .. } => Some(code),
            Self::WontFit(_) => Some("wont_fit"),
            _ => None,
        }
    }

    /// The fit verdict, when this failure was a sizing one.
    ///
    /// `Some` means the model didn't fit — read [`Fit::max_context`] for a
    /// context that would have.
    ///
    /// [`Fit::max_context`]: crate::Fit::max_context
    pub fn fit(&self) -> Option<&crate::api::fit::Fit> {
        match self {
            Self::WontFit(fit) => Some(fit),
            _ => None,
        }
    }

    /// Whether retrying the same request could plausibly succeed.
    ///
    /// A missing model or a dead controller will not fix itself; a generation
    /// failure may be transient.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Generation { .. })
    }
}

/// Shorthand for the crate's public results.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_errors_expose_their_code_for_routing() {
        let e = Error::Generation {
            code: "context_overflow".into(),
            message: "too long".into(),
        };
        assert_eq!(e.code(), Some("context_overflow"));
        assert!(e.is_retryable());
    }

    #[test]
    fn structural_failures_are_not_retryable() {
        assert_eq!(Error::ModelNotLoaded.code(), None);
        assert!(!Error::ModelNotLoaded.is_retryable());
        assert!(!Error::ControllerGone.is_retryable());
    }

    #[test]
    fn exec_errors_convert_without_losing_their_message() {
        let e: Error = crate::engine::ExecError::ContextOverflow("4096 exceeded".into()).into();
        assert!(e.to_string().contains("4096 exceeded"));
    }
}
