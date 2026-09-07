//! The loop above the inference core, behind the `agent` feature.
//!
//! Everything here decides *what invocation happens next*: an [`Agent`] runs
//! generate → dispatch → feed results back until the model stops asking,
//! executing [`Tool`]s it owns, gating risky ones through an
//! [`ApprovalMode`], keeping to a budget, and taking [`Steering`] mid-run.
//! api_spec.md §2 and §26 put all of that above Gen2, and §29 invariant 8
//! says a harness never needs it: every part of a tool round is on
//! [`Session`](crate::Session) and [`Turn`](crate::Turn) at the root.
//!
//! It stays in the crate as an optional layer because pio-app consumes it.
//! With `default-features = false` and no `agent`, this module does not
//! exist and the crate is inference only.
//!
//! An agent runs over the previous facade's [`Engine`](crate::legacy::Engine)
//! — [`Agent::on`] borrows a [`Model`](crate::Model)'s engine so a new-API
//! consumer never names the deprecated type; a spawned [`OwnedAgent`] still
//! needs an `Arc<Engine>` from [`legacy`](crate::legacy).
//!
//! ```no_run
//! use gen2::agent::{Agent, FunctionTool, ToolOutput};
//! use gen2::Session;
//!
//! #[derive(serde::Deserialize, schemars::JsonSchema)]
//! struct City { city: String }
//!
//! let model = gen2::load("/models/model.gguf")?;
//! let mut session = Session::new();
//! let weather = FunctionTool::new("get_weather", "Current weather for a city",
//!     |_ctx, a: City| async move { Ok(ToolOutput::from(format!("{}: 18C", a.city))) });
//!
//! let done = Agent::on(&model, &mut session)
//!     .add_tool(weather)
//!     .max_steps(6)
//!     .goal("What is the weather in Paris?")?;
//! println!("{}", done.text);
//! # Ok::<(), gen2::Error>(())
//! ```

/// A spawned agent run as a `Stream`.
#[cfg(feature = "tokio")]
pub use crate::api::AsyncAgentRun;
pub use crate::api::{
    Agent, AgentConfig, AgentRun, AgentStep, ApprovalMode, DEFAULT_MAX_STEPS, Decision, OwnedAgent,
    Risk, SEARCH_TOOL, Steering,
};
/// Executable tools: the trait, the typed [`FunctionTool`], sub-agents as
/// tools, skills, the registry and its search.
pub use crate::api::{
    AgentTool, ExecutionPolicy, FunctionTool, IntoTool, Skill, SkillLibrary, Tool,
    ToolBundle as ToolSet, ToolConfigError, ToolContext, ToolError, ToolLoading, ToolOutput,
    ToolRegistry, ToolSearch, ToolSpec,
};
/// What a run reports: the [`Completion`] it ends with, why ([`Finish`],
/// [`Budget`], [`Struggle`]), and the [`Update`]s a spawned run streams.
pub use crate::api::{Budget, Completion, Finish, Struggle, Update};

/// MCP client — an external server's tools as this crate's [`Tool`]s.
pub mod mcp {
    pub use crate::mcp::{
        CallToolResult, ContentBlock, DEFAULT_TIMEOUT, InitializeResult, ListToolsResult,
        McpClient, McpError, McpTool, McpToolSet, PROTOCOL_VERSION, ServerInfo, ToolDescriptor,
    };
}

/// An append-only history and the projections that turn it into a
/// transcript: the substrate a durable, resumable agent is built on.
pub mod journal {
    pub use crate::journal::{
        Declined, EntryId, Everything, Heartbeat, InputSource, Journal, JournalEntry, JournalError,
        JsonlJournal, MIN_INTERVAL, MemoryJournal, Projection, RecentTurns, Record, Resident,
        Scratch, Turn, Wake, WakeReason, WakeScheduler, WithPreamble, round_len,
    };
}
