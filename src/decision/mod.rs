//! Typed decisions without generation, sessions, or tool execution.
//!
//! The request contract preserves state and option order. Execution is supplied
//! by a decision runtime; these types do not require a native backend feature.

mod bundle;
mod decode;
mod encode;
mod graph;
mod json;
mod long;
mod question;
mod router;
mod worker;
pub(crate) use worker::DecisionMonitor;
pub(crate) use worker::spawn as spawn_worker;
#[cfg(feature = "backend-laya-onnx")]
mod onnx;

pub use encode::{EncodeOptions, EncodedQuestion, LayaTokenizer, OverflowPolicy, SpecialTokens};

pub use bundle::{BundleManifest, Checkpoint, FileDigest, LayaBundle, SequenceEnvelope};
pub use decode::{
    Answer, AnswerValue, Calibration, CalibrationDiagnostic, Histogram, TemperatureMap,
};
pub use json::OrderedJson;
pub use long::{LongDecisionResult, LongStateOptions, WindowAnswer};
pub use question::{ChoiceOption, DecisionRequest, DecisionState, Question};
pub use router::DecisionRouter;
pub use worker::{
    Cancellation, DecisionCapabilities, DecisionModel, DecisionOptions, DecisionResult,
    DecisionStatus, ExecutionOptions, ExecutionProvider, LoadOptions,
};

/// A decision-specific failure with a stable machine-readable category.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DecisionError {
    /// A state, question, or execution policy is malformed.
    #[error("invalid decision request: {0}")]
    InvalidRequest(String),
    /// A local artifact is missing, incompatible, or fails integrity validation.
    #[error("invalid decision bundle: {0}")]
    InvalidBundle(String),
    /// Input does not fit without losing evidence or option identity.
    #[error("decision input overflow: {0}")]
    InputOverflow(String),
    /// A backend returned malformed or non-finite output.
    #[error("invalid decision output: {0}")]
    InvalidOutput(String),
    #[error("decision backend is unavailable: {0}")]
    BackendUnavailable(String),
    #[error("decision execution failed: {0}")]
    Execution(String),
    #[error("decision queue is full")]
    Busy,
    #[error("decision was cancelled")]
    Cancelled,
    #[error("decision deadline expired")]
    DeadlineExceeded,
    #[error("decision model is unloaded")]
    Unloaded,
    #[error("decision memory reservation refused: {0}")]
    ResourceLimit(String),
}

impl DecisionError {
    /// Stable identifier; callers should not parse the display text.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::InvalidBundle(_) => "decision_bundle_invalid",
            Self::InputOverflow(_) => "decision_input_overflow",
            Self::InvalidOutput(_) => "decision_invalid_output",
            Self::BackendUnavailable(_) => "decision_backend_unavailable",
            Self::Execution(_) => "decision_execution",
            Self::Busy => "decision_busy",
            Self::Cancelled => "decision_cancelled",
            Self::DeadlineExceeded => "decision_deadline",
            Self::Unloaded => "decision_unloaded",
            Self::ResourceLimit(_) => "decision_resource_limit",
        }
    }
}

/// Result returned by pure decision-contract operations.
pub type Result<T> = std::result::Result<T, DecisionError>;
