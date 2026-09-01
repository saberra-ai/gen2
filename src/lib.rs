//! gen2: next-generation inference engine with pluggable backends.
//!
//! Extracted from `pio-core` into a standalone crate. Everything the engine
//! needs to run a generation lives here — the wire types it speaks, the
//! hardware and memory facts it sizes itself against, and the backends it
//! dispatches to. It depends on no host application.
//!
//! # The public API is the controller
//!
//! [`controller`] is the only module another crate can reach. You start a
//! controller, send it [`ControllerCmd`]s, and read [`ControllerEvent`]s back:
//!
//! ```no_run
//! use std::sync::mpsc::sync_channel;
//! use pio_gen2::{ControllerCmd, ControllerEvent, GenSpec, Message, Settings};
//! use pio_gen2::controller::start_controller;
//!
//! let handle = start_controller();
//!
//! let (resp_tx, resp_rx) = std::sync::mpsc::channel();
//! handle.send(ControllerCmd::LoadModel {
//!     model_path: "/path/model.gguf".into(),
//!     mmproj_path: None,
//!     settings: Settings::default(),
//!     api_key: None,
//!     api_format: None,
//!     resp: resp_tx,
//! })?;
//! resp_rx.recv()??;
//!
//! let (tx, rx) = sync_channel(64);
//! handle.send(ControllerCmd::StartChat {
//!     chat_id: "chat-1".into(),
//!     messages: vec![Message::user("Hello")],
//!     gen_spec: GenSpec { max_tokens: Some(32), ..Default::default() },
//!     thinking: Default::default(),
//!     model_id: None,
//!     model_size_bytes: None,
//!     tools: None,
//!     tx,
//! })?;
//!
//! for event in rx {
//!     match event {
//!         ControllerEvent::Token(t) => print!("{t}"),
//!         ControllerEvent::Eos | ControllerEvent::Stopped => break,
//!         _ => {}
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! For prompt-in/text-out without driving the command channel yourself, use
//! the `system_infer` family on [`InferenceHandle`].
//!
//! Everything the engine does *underneath* that — backend dispatch, session
//! runtime, KV cache, the model zoo, placement routing, residency policy — is
//! internal. Those modules are `pub(crate)`: they are implementation, and
//! keeping them so is what lets them change without breaking consumers.
//!
//! The types re-exported below are public only because they appear in the
//! controller's own signatures; you cannot call the API without naming them.
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
#[allow(dead_code, unused_imports)]
pub(crate) mod media;
/// Memory governance — machine tier, pressure level, and the budgets that
/// decide whether another runtime may go resident.
#[allow(dead_code, unused_imports)]
pub(crate) mod memory;
/// Where a generation ran — the compute-provenance receipt the engine emits.
#[allow(dead_code, unused_imports)]
pub(crate) mod provenance;
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
/// Wire types shared by every backend: messages in, execution stats out, and
/// the model/persona records a session is started from.
#[allow(dead_code, unused_imports)]
pub(crate) mod types;
/// Canonical model zoo + per-platform bundle selector. Ships Gemma-4 as the
/// reference family; new models plug in by editing `resources/models/zoo.json`.
#[allow(dead_code, unused_imports)]
pub(crate) mod zoo;

// ── The controller's vocabulary ─────────────────────────────────────────────
// Public because the controller's commands, events, and return types are
// written in these terms. This is the whole nameable surface besides
// `controller` itself.

/// The primary API — see [`api`].
pub use api::{
    Chat, Completion, Engine, EngineBuilder, Error, Event, Finish, Result, TokenStream, Tokens,
};

/// Commands, events, and handles — see [`controller`].
pub use controller::{
    ControllerCmd, ControllerConfig, ControllerEvent, ControllerHandle, ControllerMetricsSnapshot,
    ControllerObservabilitySnapshot, ControllerPolicySnapshot, ControllerRuntimeSnapshot,
    ControllerState, InferenceHandle, SystemTask,
};

/// What a generation is asked to do, and how it may think.
pub use generation::{GenSpec, ThinkingMode};
/// Structured payloads carried by [`ControllerEvent`].
pub use generation::{MediaBoundary, ToolCall};

/// The conversation the engine is given, and what it reports back.
pub use types::ExecutionStats;
pub use types::message::{Message, MessageBody, MessageChunk, MessageContent, Tool};

/// Engine configuration accepted by [`ControllerCmd::LoadModel`] and
/// [`ControllerCmd::ApplySettings`], plus the error every fallible call
/// returns.
pub use engine::{
    ExecError, MmSettings, PromptSettings, SamplingSettings, Settings, StoppingSettings,
    SystemSettings,
};

/// The receipt describing where a generation ran.
pub use provenance::ComputeProvenance;

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
