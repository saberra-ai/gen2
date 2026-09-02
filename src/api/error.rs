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

    /// The loaded model cannot do what was asked — images sent to a text-only
    /// model, say.
    ///
    /// Recoverable by construction: it is raised before anything is generated,
    /// so the conversation is untouched and a caller can drop the attachment,
    /// swap the model, or tell the user.
    #[error("the loaded model does not support {0}")]
    Unsupported(String),

    /// The agent's tools are misconfigured — a duplicate name, a missing
    /// description, or tools deferred with no way to find them.
    ///
    /// Distinct from [`Error::Load`], which is about the model: reporting a
    /// tool misconfiguration as "failed to load model" sends the reader to the
    /// wrong place entirely.
    #[error("tool configuration: {0}")]
    Tools(#[from] crate::api::tools::ToolConfigError),

    /// The request itself was malformed, and nothing was generated.
    ///
    /// Raised before inference — two labels that are the same, a schema that
    /// cannot be built — so the model was never asked and no tokens were
    /// spent. Distinct from [`Error::Generation`] for exactly that reason: the
    /// fix is in the caller's code, not in a retry.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The model generated text that did not decode into the requested type.
    ///
    /// Deliberately not a [`Error::Generation`]: generation succeeded. What
    /// failed was reading the result as `T`, which is a different problem with
    /// a different fix — and the raw text is carried so a caller can see what
    /// the model actually said instead of guessing.
    #[error("could not read the reply as {type_name}: {message}")]
    Extraction {
        /// The Rust type the caller asked for.
        type_name: &'static str,
        /// What went wrong decoding it.
        message: String,
        /// Exactly what the model produced.
        raw: String,
    },

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
            Self::Tools(_) => Some("tool_config"),
            Self::Unsupported(_) => Some("unsupported"),
            Self::InvalidRequest(_) => Some("invalid_request"),
            Self::Extraction { .. } => Some("extraction_failed"),
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
    /// A missing model or a dead controller will not fix itself. Most
    /// generation failures may be transient — but not all of them, and the
    /// exception matters: the controller distinguishes a poisoned session,
    /// whose backend state is gone, from an ordinary failure, precisely so a
    /// caller does not retry into it. Answering `true` for everything shaped
    /// like a `Generation` threw that distinction away at the API boundary.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Generation { code, .. } => !NOT_WORTH_RETRYING.contains(&code.as_str()),
            _ => false,
        }
    }
}

/// Generation failure codes a retry cannot fix.
///
/// `session_poisoned` means the backend lost the session's state, so the same
/// request against the same conversation fails the same way — the caller has
/// to start the conversation over. The controller goes out of its way to
/// distinguish it from an ordinary failure, and this is where that
/// distinction has to survive.
const NOT_WORTH_RETRYING: &[&str] = &["session_poisoned"];

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

    #[test]
    fn a_poisoned_session_is_not_retryable() {
        // The controller separates this from `generation_error` so a caller
        // does not retry into a session whose state is gone. Answering `true`
        // for anything shaped like a `Generation` threw that away at the API
        // boundary.
        let poisoned = Error::Generation {
            code: "session_poisoned".into(),
            message: "session state lost".into(),
        };
        assert!(!poisoned.is_retryable());

        let transient = Error::Generation {
            code: "generation_error".into(),
            message: "the GPU hiccuped".into(),
        };
        assert!(
            transient.is_retryable(),
            "an ordinary generation failure may well succeed on a second try"
        );
    }

    #[test]
    fn nothing_outside_a_generation_failure_is_retryable() {
        assert!(!Error::ControllerGone.is_retryable());
        assert!(!Error::Load("no such file".into()).is_retryable());
    }
}
