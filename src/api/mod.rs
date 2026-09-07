//! The public API, assembled.
//!
//! Nothing is reached through this module: the crate root re-exports the
//! inference-first surface (api_spec.md §25), [`crate::advanced`] the local
//! controls under it, [`crate::legacy`] the previous facade, and
//! [`crate::agent`] the loop above it. This module only declares the parts
//! and gates the agent-shaped ones on the `agent` feature.

#[cfg(feature = "agent")]
mod agent;
#[cfg(feature = "agent")]
mod agent_config;
#[cfg(feature = "agent")]
mod agent_spawned;
#[cfg(feature = "tokio")]
mod async_turn;
#[cfg(feature = "tokio")]
mod asynchronous;
mod chat;
mod classify;
mod embed;
mod engine;
mod error;
pub mod event;
mod extract;
// ── S3.1: `hf:` model references (src/api/hf.rs) ───────────────────────────
pub mod hf;
// ── end S3.1 ────────────────────────────────────────────────────────────────
pub(crate) mod fit;
mod generation;
mod inference;
pub mod input;
pub mod model;
pub mod output;
mod response;
mod runtime;
pub mod session;
mod spawned;
mod stream;
pub mod tool_defs;
#[cfg(feature = "agent")]
pub mod tools;
pub mod turn;

// ── The inference-first facade (api_spec.md §4–§6) ──────────────────────────
// `Runtime` loads `Model`s; a model generates a `Response`. Built over the
// engine below rather than beside it: a `Model` is an `Engine` with a
// registry entry, and one-shot `Generation` is a `Chat` on a session it
// throws away.
pub use embed::{Embedder, Reranker};
pub use error::{Error, Result};
pub use event::{Event, EventStream};
pub use generation::Generation;
pub use input::Input;
pub use model::Model;
pub use response::Response;
pub use runtime::{
    ModelResidency, RemoteModelBuilder, ResidencySnapshot, Runtime, RuntimeBuilder, RuntimeStats,
};
pub use session::Session;
pub use tool_defs::{ToolDefinition, ToolSet};
pub use turn::{GenerationOptions, ToolChoice, Turn};

// ── The previous facade (`crate::legacy`) ───────────────────────────────────
#[cfg(feature = "tokio")]
pub use asynchronous::AsyncTurn;
pub use chat::{Chat, DEFAULT_TOOL_DEPTH};
pub use classify::Classify;
pub use engine::{Engine, EngineBuilder};
pub use extract::Extract;
pub use fit::{Fit, FitVerdict, ModelInfo as FileModelInfo};
pub use inference::Inference;
pub use spawned::{Canceller as ChatCanceller, OwnedChat, Turn as ChatTurn, Update};
pub use stream::{Budget, Completion, Event as TokenEvent, Finish, Struggle, TokenStream, Tokens};

// ── The agent layer (`crate::agent`, feature `agent`) ───────────────────────
#[cfg(feature = "agent")]
pub use agent::{Agent, AgentStep, ApprovalMode, DEFAULT_MAX_STEPS, SEARCH_TOOL, Steering};
#[cfg(feature = "agent")]
pub use agent_config::AgentConfig;
#[cfg(feature = "agent")]
pub use agent_spawned::{AgentRun, OwnedAgent};
#[cfg(all(feature = "agent", feature = "tokio"))]
pub use asynchronous::AsyncAgentRun;
#[cfg(feature = "agent")]
pub use tools::{
    AgentTool, Decision, ExecutionPolicy, FunctionTool, IntoTool, Risk, Skill, SkillLibrary, Tool,
    ToolConfigError, ToolContext, ToolError, ToolLoading, ToolOutput, ToolRegistry, ToolSearch,
    ToolSet as ToolBundle, ToolSpec,
};

/// The spec's §28 walkthrough, compiled as written.
#[cfg(doctest)]
mod spec_walkthrough_tests;

/// Session invariants under generated operation sequences.
#[cfg(test)]
mod session_props;

/// Agent-loop contracts, on scripted model behaviour.
#[cfg(all(test, feature = "agent"))]
mod agent_contract_tests;

/// The off-thread API, on scripted model behaviour.
#[cfg(all(test, feature = "agent"))]
mod spawned_tests;

/// Tool bundles, reusable agent configurations, and sub-agents.
#[cfg(all(test, feature = "agent"))]
mod composition_tests;

/// `infer` and `chat`, the two entry points most callers reach for first.
#[cfg(test)]
mod entrypoint_tests;

/// What the model was actually shown, across the whole stack.
#[cfg(all(test, feature = "agent"))]
mod lifecycle_tests;

/// Model switching, cache identity, residency, and concurrency (api_spec.md
/// §4.2, §4.5, §17, §23), on scripted models.
#[cfg(test)]
mod switching_tests;
