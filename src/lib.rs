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
//! [`load`] a model and ask it to generate:
//!
//! ```no_run
//! let model = gen2::load("/models/model.gguf")?;
//! let text = model.generate("Explain entropy in one sentence.").max_tokens(64).text()?;
//! # Ok::<(), gen2::Error>(())
//! ```
//!
//! A [`Runtime`] holds several [`Model`]s, local or served by an
//! OpenAI-compatible endpoint; a [`Session`] is conversation state you own,
//! and [`Model::turn`] runs one invocation against it:
//!
//! ```no_run
//! use gen2::{Runtime, Session};
//!
//! let runtime = Runtime::new()?;
//! let model = runtime.load("/models/model.gguf")?;
//!
//! let mut session = Session::new().with_system("Be concise.");
//! model.turn(&mut session).user("Explain entropy.").run()?;
//! model.turn(&mut session).user("Simpler?").run()?;
//! # Ok::<(), gen2::Error>(())
//! ```
//!
//! The root is the whole happy path (api_spec.md §25). Below it:
//!
//! - [`advanced`] — local tuning, residency and hardware, the wire types,
//!   the controller, and the backend plugin seam.
//! - [`legacy`] — the previous `Engine`/`Chat`/`Inference` facade, deprecated
//!   and mapped to the new surface in api_spec.md §27.
//! - [`agent`] — the loop above the inference core, behind the `agent`
//!   feature (on by default).
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
// Assembled in `api`, named from here: the root, `advanced`, `legacy`, and
// `agent` are the only paths a consumer sees.
pub(crate) mod api;

// Below the happy path: local tuning, residency, hardware, the wire types,
// the controller, and the backend seam. Documented in the module, not here,
// so its links resolve in its own scope.
pub mod advanced;

// ── S3.1: `hf:` model references ────────────────────────────────────────────
/// Models from the Hugging Face Hub by `hf:owner/repo[:QUANT]` reference —
/// [`HfModel`](hf::HfModel), the cache, and the errors.
pub use api::hf;
// ── end S3.1 ────────────────────────────────────────────────────────────────

// The previous facade, deprecated: `Engine`, `Chat`, `Inference` and their
// types, each mapped to the new surface (api_spec.md §27).
pub mod legacy;

// The loop above the inference core — `Agent`, executable tools, approvals,
// MCP — behind the `agent` feature.
#[cfg(feature = "agent")]
pub mod agent;

// ── S5.1: the old module tree for pio-app's switchover ──────────────────────
// Hidden: not the supported surface, the list of what the host still reaches
// for. Shrinks as pio-app moves to the root API.
#[doc(hidden)]
pub mod compat;
// ── end S5.1 ────────────────────────────────────────────────────────────────

// The controller: commands, events, and handles. Reached as
// `advanced::controller`; the module itself stays crate-private so its
// vocabulary does not sit beside `Model` and `Session` (api_spec.md §25).
// `unused_mut` is allowed here (as it was before the split): some bindings need
// `mut` only under certain backend features.
#[allow(unused_mut)]
pub(crate) mod controller;

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
// MCP client — register an external server's tools as this crate's tools.
// Reached as `agent::mcp`.
#[cfg(feature = "agent")]
pub(crate) mod mcp;
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

// An append-only history and the projections that turn it into a transcript.
// The seam a durable agent is built on. Reached as `agent::journal`; the
// session runtime's truncation shares its round rule.
#[allow(dead_code, unused_imports)]
pub(crate) mod journal;

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

// ── The root namespace (api_spec.md §25) ────────────────────────────────────
// Exactly the list a normal consumer sees, plus `load`. Supporting types live
// in the modules below it; everything else is under `advanced`, `legacy`, or
// `agent`. Keep this block short — it is the crate's public identity.

#[cfg(feature = "tokio")]
pub use api::event::AsyncEventStream;
/// The inference-first surface: a [`Runtime`] loads [`Model`]s; a model
/// answers an [`Input`] with a [`Response`]; a [`Session`] holds the
/// conversation, and [`Model::turn`] runs one [`Turn`] against it with
/// [`GenerationOptions`] and a [`ToolChoice`], blocking or as an
/// [`EventStream`] of [`Event`]s.
pub use api::{
    Error, Event, EventStream, GenerationOptions, Input, Model, Response, Result, Runtime, Session,
    ToolChoice, ToolDefinition, ToolSet, Turn,
};
/// The conversation the model is given: the role-based wire message the
/// backends render (api_spec.md §9's `enum Message` is not yet this type).
pub use types::message::Message;

/// Load a model with a private runtime the returned [`Model`] keeps alive.
///
/// Shorthand for `Runtime::new()?.load(path)`; the one-liner for the common
/// case of one model.
///
/// ```no_run
/// let model = gen2::load("qwen3-8b.gguf")?;
/// println!("{}", model.generate("Why is the sky blue?").text()?);
/// # Ok::<(), gen2::Error>(())
/// ```
pub fn load(path: impl AsRef<std::path::Path>) -> Result<Model> {
    Runtime::new()?.load(path)
}

/// Deriving a tool's argument schema needs the same `schemars` this crate
/// compiled against — a different version produces a `JsonSchema` impl that
/// won't satisfy [`ToolDefinition::input_schema`]'s bound. Use
/// `gen2::schemars` rather than adding your own dependency.
pub use schemars;

// ── Supporting modules (api_spec.md §25) ────────────────────────────────────
// One block per module so a line added by another slice merges cleanly.

/// What a model is and does: [`ModelId`](model::ModelId),
/// [`ModelInfo`](model::ModelInfo), [`ModelCapabilities`](model::ModelCapabilities),
/// the [`Generation`](model::Generation) builder behind [`Model::generate`],
/// the remote-model builder, and the auxiliary [`Embedder`](model::Embedder)
/// and [`Reranker`](model::Reranker) (api_spec.md §5, §6, §21).
pub mod model {
    pub use crate::api::model::{ModelCapabilities, ModelId, ModelInfo, ModelSourceKind};
    pub use crate::api::{Embedder, Generation, RemoteModelBuilder, Reranker};
    pub use crate::generation::ThinkingMode;
    pub use crate::utilities::RerankResult;
}

/// Conversation state: ids, revisions, the append-only event log, message
/// records, tool results, and the errors a refused edit returns
/// (api_spec.md §7–§8).
pub mod session {
    pub use crate::api::session::{
        ContextFingerprint, MessageId, MessageRecord, SessionError, SessionEvent, SessionId,
        SessionRevision, ToolResult,
    };
}

/// What a model is given: [`Input`], its [`InputPart`](input::InputPart)s,
/// and an [`Image`](input::Image) (api_spec.md §9.2).
pub mod input {
    pub use crate::api::input::{Image, Input, InputPart};
}

/// What comes back: the [`AssistantMessage`](output::AssistantMessage) and its
/// parts, tool calls, usage, stats, and the finish reason (api_spec.md §14).
pub mod output {
    pub use crate::api::output::{
        AssistantMessage, FinishReason, GenerationStats, OutputPart, ToolCall, ToolCallId, Usage,
    };
}

/// Semantic streaming: [`Event`], the [`EventStream`] that yields them, and
/// the [`Canceller`](event::Canceller) that ends one (api_spec.md §15–§16).
pub mod event {
    #[cfg(feature = "tokio")]
    pub use crate::api::event::AsyncEventStream;
    pub use crate::api::event::{Canceller, Event, EventStream};
}

/// Tool definitions the model is told about, with nothing to run
/// (api_spec.md §10).
pub mod tool_defs {
    pub use crate::api::tool_defs::{ToolDefinition, ToolSet};
}
