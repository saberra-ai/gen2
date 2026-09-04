//! The README, compiled.
//!
//! Every Rust block in it is a doc test, so an example cannot drift from the
//! API without the build saying so. That is the whole point of including it
//! here rather than letting the two live separate lives.
#![doc = include_str!("../README.md")]
//! gen2: next-generation inference engine with pluggable backends.
//!
//! Extracted from `pio-core` into a standalone crate. Everything the engine
//! needs to run a generation lives here — the wire types it speaks, the
//! hardware and memory facts it sizes itself against, and the backends it
//! dispatches to. It depends on no host application.
//!
//! # The API
//!
//! [`Engine`] loads a model; [`Session`] holds a conversation you own. Three
//! ways to call:
//!
//! - [`Engine::infer`] for one prompt with nothing kept.
//! - [`Engine::chat`] for a turn in a conversation.
//! - [`Engine::agent`] for a task carried out with your tools.
//!
//! ```no_run
//! use gen2::{Engine, Session};
//!
//! let engine = Engine::load("/models/model.gguf")?;
//!
//! let title = engine.infer("Title this in three words.").max_tokens(16).text()?;
//!
//! let mut session = Session::new();
//! engine.chat(&mut session).user("Explain entropy.").send()?;
//! engine.chat(&mut session).user("Simpler?").send()?;
//! # Ok::<(), gen2::Error>(())
//! ```
//!
//! [`controller`] is the layer underneath, reachable through
//! [`Engine::controller`] for anything the facade does not cover. Everything
//! else — backend dispatch, session runtime, KV cache, the model zoo, placement
//! routing, residency policy — is internal and free to change.
//!
//! See `docs/EXTRACTION.md` for what moved out of `pio-core`, what was
//! inverted, and the one seam (remote/flock dispatch) a host still supplies.

// A narrow API only holds if everything reachable through it can be named.
// This lint is allow-by-default, so without it a type can sit in a public
// field, be handed to callers, and still be impossible to declare or
// construct — a hole that compiles silently. It caught 19 on the day the API
// narrowed to the controller.
#![warn(unnameable_types)]

// ── The public API ──────────────────────────────────────────────────────────
pub mod api;

// Below the happy path: bring your own backend. Documented in the module,
// not here, so its links resolve in its own scope.
pub mod advanced;

/// The controller: commands, events, and handles. [`api`] is the ergonomic
/// layer over this; reach for the controller directly when you need something
/// the facade doesn't cover.
// `unused_mut` is allowed here (as it was before the split): some bindings need
// `mut` only under certain backend features.
#[allow(unused_mut)]
pub mod controller;

// ── Internals ───────────────────────────────────────────────────────────────
// Reachable within the crate only. Anything here that leaks into a public
// controller signature must be re-exported below, or rustc's
// `private_interfaces` lint will say so.
//
// These carry `dead_code`/`unused_imports` allowances because much of this
// surface exists for callers rather than for the engine's own use: it was
// `pub` before the API narrowed to the controller, and each module's `pub use`
// block still documents what that module offers. Two reasons to keep it rather
// than delete to satisfy the lint — a backend's helpers look unused whenever
// that backend's feature is off (150 of these warnings survive with
// `backend-llamacpp` on, 207 with it off), and pio-app calls into a good deal
// of it today, so the switchover decides what genuinely goes. Until then,
// deleting a working engine's internals to quiet a lint trades real capability
// for a clean build. See docs/EXTRACTION.md.
#[allow(dead_code, unused_imports)]
pub(crate) mod backend;
#[allow(dead_code, unused_imports)]
pub(crate) mod bundle;
#[allow(dead_code, unused_imports)]
pub(crate) mod engine;
#[allow(dead_code)]
pub(crate) mod executor;
#[allow(dead_code, unused_variables, unused_imports)]
pub(crate) mod generation;
/// Machine facts the engine sizes itself against: GPU backend, hardware
/// profile, and the per-platform default `Settings` they imply.
#[allow(dead_code, unused_imports)]
pub(crate) mod hardware;
#[allow(dead_code, unused_imports)]
pub(crate) mod kv;
/// MCP client — register an external server's tools as this crate's tools.
pub mod mcp;
#[allow(dead_code, unused_imports)]
pub(crate) mod media;

/// Memory governance — machine tier, pressure level, and the budgets that
/// decide whether another runtime may go resident.
#[allow(dead_code, unused_imports)]
pub(crate) mod memory;
#[allow(dead_code, unused_imports)]
pub(crate) mod residency;
#[allow(dead_code, unused_imports)]
pub(crate) mod residency_policy;
#[allow(dead_code, unused_imports)]
pub(crate) mod residency_stats;
/// Inference router — local-first placement. Picks where a generation
/// actually runs given the flock's declared capabilities. Pure function, no
/// I/O; the controller consults it during dispatch.
#[allow(dead_code, unused_imports)]
pub(crate) mod router;

/// An append-only history and the projections that turn it into a transcript.
///
/// The seam a durable agent is built on: the journal is what happened, the
/// context is a view of it, and the two are allowed to differ.
pub mod journal;

#[allow(
    dead_code,
    unused_variables,
    unused_unsafe,
    unused_imports,
    unused_assignments
)]
pub(crate) mod session_rt;
/// Panic-safe background task spawning used by the executor loop.
#[allow(dead_code, unused_imports)]
pub(crate) mod task_util;
/// Scripted backends and other test-only machinery. Never compiled into a
/// release build.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod test_support;

/// Wire types shared by every backend: messages in, execution stats out, and
/// the model/persona records a session is started from.
#[allow(dead_code, unused_imports)]
pub(crate) mod types;

/// Auxiliary model runtimes — embedding today, more to come — owned off the
/// controller thread so a slow helper cannot stall chat token scheduling.
pub(crate) mod utilities;
/// Canonical model zoo + per-platform bundle selector. Ships Gemma-4 as the
/// reference family; new models plug in by editing `resources/models/zoo.json`.
#[allow(dead_code, unused_imports)]
pub(crate) mod zoo;

// ── The controller's vocabulary ─────────────────────────────────────────────
// Public because the controller's commands, events, and return types are
// written in these terms. This is the whole nameable surface besides
// `controller` itself.

/// The primary API — see [`api`].
#[cfg(feature = "tokio")]
pub use api::{AsyncAgentRun, AsyncTurn};
/// Deriving a tool's argument schema needs the same `schemars` this crate
/// compiled against — a different version produces a `JsonSchema` impl that
/// won't satisfy [`FunctionTool`]'s bound. Use `gen2::schemars` rather than
/// adding your own dependency.
pub use schemars;

pub use api::{
    Agent, AgentConfig, AgentRun, AgentStep, ApprovalMode, Budget, Canceller, Chat, Classify,
    Completion, DEFAULT_MAX_STEPS, DEFAULT_TOOL_DEPTH, Decision, Engine, EngineBuilder, Error,
    Event, ExecutionPolicy, Extract, Finish, Fit, FitVerdict, FunctionTool, Inference, IntoTool,
    ModelInfo, OwnedChat, Result, Session, TokenStream, Tokens, Tool, ToolConfigError, ToolContext,
    ToolError, ToolLoading, ToolOutput, ToolRegistry, ToolSearch, ToolSet, Turn, Update,
};
pub use api::{AgentTool, Risk, SEARCH_TOOL, Skill, SkillLibrary, Steering, Struggle};
/// Tools served by an MCP server, alongside the tool types they sit with.
pub use mcp::{McpClient, McpError, McpTool, McpToolSet};
/// Which auxiliary runtimes are loaded, separate from what the chat model can do.
pub use utilities::{LoadedUtility, RerankResult, UtilityStatus};

/// Commands, events, and handles — see [`controller`].
pub use controller::{
    ControllerCmd, ControllerConfig, ControllerEvent, ControllerHandle, ControllerMetricsSnapshot,
    ControllerObservabilitySnapshot, ControllerPolicySnapshot, ControllerRuntimeSnapshot,
    ControllerState, InferenceHandle, Placement, RemoteDispatch, SystemTask,
};

/// What a generation is asked to do, and how it may think.
pub use generation::{GenSpec, ThinkingMode};
/// Structured payloads carried by [`ControllerEvent`].
pub use generation::{MediaBoundary, ToolCall};

/// The conversation the engine is given, and what it reports back.
pub use types::ExecutionStats;
pub use types::message::{Message, MessageBody, MessageChunk, MessageContent, ToolSpec};

/// Engine configuration accepted by [`ControllerCmd::LoadModel`] and
/// [`ControllerCmd::ApplySettings`], plus the error every fallible call
/// returns.
pub use engine::{
    Capabilities, Degraded, ExecError, LoadOutcome, MmSettings, PromptSettings, SamplingSettings,
    Settings, StoppingSettings, SystemSettings,
};

/// What machine this is — memory, cores, GPU — as read by
/// [`HardwareProfile::detect`]. The input to a fit check.
pub use hardware::{GpuBackend, HardwareProfile};
/// A model's own header metadata, reachable from [`ModelInfo`].
pub use types::model::{Model, ModelConfig, ModelMetadata};

// ── Transitively reachable types ────────────────────────────────────────────
// Not part of the controller's own signatures, but reachable *through* them —
// a field of a public struct, a member of a public snapshot. A caller can
// receive one of these, so it must be able to name one; otherwise the value is
// there but nothing can be declared, constructed, or matched against it.
// `unnameable_types` (below) is what proves this list stays complete.

/// What the active backend can do, and how fast it reaches first token.
pub use backend::caps::{BackendCaps, LatencyTier};
/// Output shaping on [`GenSpec::grammar`]: JSON schema, regex, Lark, or GBNF.
pub use backend::common::grammar::GrammarSpec;
/// What a predictor is given to draft from.
///
/// Exported because [`SpeculativePredictor::draft_with_context`] takes one: it
/// was reachable but unnameable, so the trait could not actually be
/// implemented outside this crate. Only exists on the MLX backend, which is
/// the only one that surfaces the target's hidden states.
#[cfg(feature = "backend-mlx")]
pub use backend::common::speculative::DraftContext;
/// Speculative-decoding policy reachable from the sampling settings.
pub use backend::common::speculative::{SpeculativeMode, SpeculativePredictor};

/// Tool definitions carried on a [`Message`], distinct from the
/// [`ToolCall`] event the model emits mid-stream.
pub use types::message::FunctionDefinition;
pub use types::message::ToolCall as MessageToolCall;
/// Media reference on a message chunk.
pub use types::message::Url;

/// Memory governance reachable through the controller's observability
/// snapshots.
pub use memory::{
    MachineMemoryTier, MemoryBudgets, MemoryGovernor, MemoryPolicyInput, MemoryPressureLevel,
    MemorySnapshot,
};

/// What is currently resident, and the policy deciding what may join it —
/// reachable through [`ControllerObservabilitySnapshot`].
pub use residency::{ResidencyInventory, ResidentRuntime, RuntimeKind};
pub use residency_policy::ResidencyPolicy;
pub use residency_stats::ResidencyStats;
