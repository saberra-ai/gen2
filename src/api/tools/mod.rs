//! Tools: the executable things an agent can call.
//!
//! The split matters. A [`Tool`] is executable; a [`ToolSpec`] is the context
//! that tells the model it exists. Backends render specs differently, tools run
//! the same everywhere.

mod function;
mod registry;
mod search;
mod set;
mod spec;

use std::sync::Arc;

use async_trait::async_trait;

pub use function::FunctionTool;
pub use registry::{ToolConfigError, ToolRegistry};
pub use search::ToolSearch;
pub use set::{IntoTool, ToolSet};
pub use spec::{ToolLoading, ToolSpec};

/// Anything the model can call.
///
/// Implement it directly for stateful tools, or use [`FunctionTool`] to build
/// one from a typed async closure — which derives the schema from the argument
/// type, so the schema and the code that reads it cannot drift.
#[async_trait]
pub trait Tool: Send + Sync {
    /// What the model sees.
    fn spec(&self) -> &ToolSpec;

    /// Run the tool.
    ///
    /// `args` has already been validated against [`ToolSpec::input_schema`], so
    /// an implementation can deserialize without re-checking shape.
    async fn call(
        &self,
        ctx: &ToolContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError>;

    /// How this tool may be scheduled. Defaults to fully parallel.
    fn execution_policy(&self) -> ExecutionPolicy {
        ExecutionPolicy::default()
    }
}

/// What a tool produces.
///
/// `#[non_exhaustive]`: images and files will join these.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ToolOutput {
    /// Plain text, fed back to the model verbatim.
    Text(String),
    /// Structured data, serialized before it reaches the model.
    Json(serde_json::Value),
}

impl ToolOutput {
    /// Render for the model.
    pub fn to_model_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Json(v) => v.to_string(),
        }
    }
}

impl From<String> for ToolOutput {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for ToolOutput {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<serde_json::Value> for ToolOutput {
    fn from(v: serde_json::Value) -> Self {
        Self::Json(v)
    }
}

/// Why a tool call failed.
///
/// The distinction that matters is whether the *model* can do anything about
/// it. A bad argument it can correct on the next turn; a dead network it
/// cannot, and telling it so just burns a round.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolError {
    /// The arguments didn't fit the schema. Reported to the model so it can
    /// retry with corrected arguments.
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    /// The tool ran and failed in a way the model can react to — a file that
    /// doesn't exist, a query with no results.
    #[error("{0}")]
    Failed(String),
    /// The tool could not run at all: network down, service unreachable.
    /// Retrying the same call may succeed; explaining it to the model won't.
    #[error("unavailable: {0}")]
    Unavailable(String),
    /// The call exceeded its time budget.
    #[error("timed out after {0:?}")]
    TimedOut(std::time::Duration),
    /// The caller declined to run it.
    #[error("denied: {0}")]
    Denied(String),
}

impl ToolError {
    /// Whether the model should be told, so it can adjust and try again.
    ///
    /// False for infrastructure failures — the model cannot fix a dead socket,
    /// and handing it one wastes a round of context.
    pub fn is_model_actionable(&self) -> bool {
        matches!(self, Self::InvalidArguments(_) | Self::Failed(_))
    }
}

/// How a tool may be scheduled.
///
/// Two independent questions, because the answers genuinely differ: `grep` is
/// safe beside anything, `git push` must not race another write but doesn't
/// touch the GPU, and local image generation contends with the model itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPolicy {
    /// May run concurrently with other tools.
    pub parallel_safe: bool,
    /// Contends with the model for the accelerator, so inference must not
    /// overlap it. The reason a local runtime needs this at all.
    pub blocks_inference: bool,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            parallel_safe: true,
            blocks_inference: false,
        }
    }
}

impl ExecutionPolicy {
    /// Must not overlap another tool — a shared write, a lock, a mutation.
    pub fn exclusive() -> Self {
        Self {
            parallel_safe: false,
            blocks_inference: false,
        }
    }

    /// Needs the accelerator: local image generation, an embedding pass.
    pub fn gpu_bound() -> Self {
        Self {
            parallel_safe: true,
            blocks_inference: true,
        }
    }
}

/// What a tool is given besides its arguments.
///
/// Somewhere for cancellation, identity, and tracing to live, so a handler
/// doesn't capture half the runtime in a closure.
#[derive(Debug, Clone)]
pub struct ToolContext {
    session_id: String,
    call_id: Option<String>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

// Constructed by the agent loop, which lands next.
#[allow(dead_code)]
impl ToolContext {
    pub(crate) fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            call_id: None,
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub(crate) fn with_call_id(mut self, id: Option<String>) -> Self {
        self.call_id = id;
        self
    }

    /// The conversation this call belongs to.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The model's id for this call, when it supplied one.
    pub fn call_id(&self) -> Option<&str> {
        self.call_id.as_deref()
    }

    /// Whether the turn has been cancelled. Long-running tools should check
    /// this and return early.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Signal cancellation to every tool sharing this context.
    pub(crate) fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_model_fixable_failures_are_worth_reporting_back() {
        assert!(ToolError::InvalidArguments("missing city".into()).is_model_actionable());
        assert!(ToolError::Failed("no such file".into()).is_model_actionable());
        // The model cannot fix these, so a round spent telling it is wasted.
        assert!(!ToolError::Unavailable("connection refused".into()).is_model_actionable());
        assert!(!ToolError::TimedOut(std::time::Duration::from_secs(30)).is_model_actionable());
        assert!(!ToolError::Denied("user declined".into()).is_model_actionable());
    }

    #[test]
    fn execution_policies_separate_the_two_questions() {
        let default = ExecutionPolicy::default();
        assert!(default.parallel_safe && !default.blocks_inference);

        // A shared write blocks other tools but leaves the GPU alone.
        let write = ExecutionPolicy::exclusive();
        assert!(!write.parallel_safe && !write.blocks_inference);

        // Image generation is the mirror image: fine beside other tools,
        // fatal beside inference.
        let gpu = ExecutionPolicy::gpu_bound();
        assert!(gpu.parallel_safe && gpu.blocks_inference);
    }

    #[test]
    fn json_output_serializes_for_the_model() {
        let out = ToolOutput::Json(serde_json::json!({"temp_c": 18}));
        assert_eq!(out.to_model_text(), r#"{"temp_c":18}"#);
        assert_eq!(ToolOutput::from("hi").to_model_text(), "hi");
    }

    #[test]
    fn cancellation_reaches_every_tool_sharing_a_context() {
        let ctx = ToolContext::new("s1");
        let clone = ctx.clone();
        assert!(!clone.is_cancelled());
        ctx.cancel();
        assert!(clone.is_cancelled(), "cancellation is shared, not copied");
    }
}
