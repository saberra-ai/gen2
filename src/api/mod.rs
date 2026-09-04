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
mod classify;
mod engine;
mod error;
mod extract;
pub(crate) mod fit;
mod generation;
mod inference;
pub mod input;
pub mod model;
pub mod output;
mod response;
mod runtime;
mod session;
mod spawned;
mod stream;
pub mod tools;

// ── The inference-first facade (api_spec.md §4–§6) ──────────────────────────
// `Runtime` loads `Model`s; a model generates a `Response`. Built over the
// engine below rather than beside it: a `Model` is an `Engine` with a
// registry entry, and one-shot `Generation` is a `Chat` on a session it
// throws away.
pub use generation::Generation;
pub use input::Input;
pub use model::{Model, ModelId};
pub use response::Response;
pub use runtime::{RemoteModelBuilder, Runtime, RuntimeBuilder};

pub use agent::{
    Agent, AgentStep, ApprovalMode, DEFAULT_MAX_STEPS, Decision, Risk, SEARCH_TOOL, Steering,
};
pub use agent_config::AgentConfig;
pub use agent_spawned::{AgentRun, OwnedAgent};
#[cfg(feature = "tokio")]
pub use asynchronous::{AsyncAgentRun, AsyncTurn};
pub use chat::{Chat, DEFAULT_TOOL_DEPTH};
pub use classify::Classify;
pub use engine::{Engine, EngineBuilder};
pub use error::{Error, Result};
pub use extract::Extract;
pub use fit::{Fit, FitVerdict, ModelInfo};
pub use inference::Inference;
pub use session::Session;
pub use spawned::{Canceller, OwnedChat, Turn, Update};
pub use stream::{Budget, Completion, Event, Finish, Struggle, TokenStream, Tokens};
pub use tools::{
    AgentTool, ExecutionPolicy, FunctionTool, IntoTool, Skill, SkillLibrary, Tool, ToolConfigError,
    ToolContext, ToolError, ToolLoading, ToolOutput, ToolRegistry, ToolSearch, ToolSet, ToolSpec,
};

/// Session invariants under generated operation sequences.
#[cfg(test)]
mod session_props;

/// Agent-loop contracts, on scripted model behaviour.
#[cfg(test)]
mod agent_contract_tests;

/// The off-thread API, on scripted model behaviour.
#[cfg(test)]
mod spawned_tests;

/// Tool bundles, reusable agent configurations, and sub-agents.
#[cfg(test)]
mod composition_tests;

/// `infer` and `chat`, the two entry points most callers reach for first.
#[cfg(test)]
mod entrypoint_tests;

/// What the model was actually shown, across the whole stack.
#[cfg(test)]
mod lifecycle_tests;
