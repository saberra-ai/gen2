//! The public API.
//!
//! Load a model, hold a conversation, read tokens back:
//!
//! ```no_run
//! use gen2::{Engine, Session};
//!
//! let engine = Engine::load("/models/model.gguf")?;
//!
//! // A conversation you own. The reply is appended to it.
//! let mut session = Session::new();
//! engine.chat(&mut session).user("Explain entropy.").send()?;
//! println!("{}", session.latest_text().unwrap_or_default());
//!
//! // A follow-up: the history is already here, so nothing is resent.
//! engine.chat(&mut session).user("Simpler?").send()?;
//!
//! // One-off, nothing kept.
//! let title = engine.infer("Title this in three words.").max_tokens(16).text()?;
//! # Ok::<(), gen2::Error>(())
//! ```
//!
//! Everything underneath — backend dispatch, session runtime, KV cache, the
//! model zoo, placement routing, residency policy — is internal, so it can
//! change without breaking callers. [`Engine::controller`] is the escape hatch
//! for what this doesn't cover.

mod agent;
mod agent_config;
mod agent_spawned;
#[cfg(feature = "tokio")]
mod asynchronous;
mod chat;
mod engine;
mod error;
pub(crate) mod fit;
mod inference;
mod session;
mod spawned;
mod stream;
pub mod tools;

pub use agent::{
    Agent, AgentStep, ApprovalMode, DEFAULT_MAX_STEPS, Decision, Risk, SEARCH_TOOL, Steering,
};
pub use agent_config::AgentConfig;
pub use agent_spawned::{AgentRun, OwnedAgent};
#[cfg(feature = "tokio")]
pub use asynchronous::{AsyncAgentRun, AsyncTurn};
pub use chat::{Chat, DEFAULT_TOOL_DEPTH};
pub use engine::{Engine, EngineBuilder};
pub use error::{Error, Result};
pub use fit::{Fit, FitVerdict, ModelInfo};
pub use inference::Inference;
pub use session::Session;
pub use spawned::{Canceller, OwnedChat, Turn, Update};
pub use stream::{Budget, Completion, Event, Finish, Struggle, TokenStream, Tokens};
pub use tools::{
    AgentTool, ExecutionPolicy, FunctionTool, IntoTool, Skill, SkillLibrary, Tool, ToolConfigError,
    ToolContext, ToolError, ToolLoading, ToolOutput, ToolRegistry, ToolSearch, ToolSet, ToolSpec,
};
