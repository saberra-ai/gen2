//! mlxcel as a `gen2` backend — fast MLX inference embedded via the
//! [`mlxcel`] crate (Rust + MLX C++ bindings, decode ≈ mlx-lm).
//!
//! Register it with the engine builder and every safetensors model directory
//! routes here:
//!
//! ```no_run
//! let engine = gen2::Engine::builder()
//!     .model("/models/qwen3-0.6b-4bit")
//!     .backend(gen2_mlxcel::plugin())
//!     .build()?;
//! let text = engine.infer("Reply with one word: hello").max_tokens(8).text()?;
//! # Ok::<(), gen2::Error>(())
//! ```
//!
//! # Why a separate crate
//!
//! Neither `mlxcel` nor `mlxcel-core` has a crates.io release, and the
//! registry refuses a git dependency even when it is optional. So the
//! published `gen2` carries no reference to them, and this crate — a
//! workspace member, `publish = false` — supplies the backend through
//! [`gen2::advanced::plugin`], the seam any out-of-tree backend uses.
//!
//! # Do not link `backend-mlx` in the same binary
//!
//! `gen2`'s `backend-mlx` feature (mlx-rs) and this crate (mlxcel-core's cxx
//! bindings) BOTH link MLX C++. Enabling both in one binary duplicates the MLX
//! symbol surface. Pick one: this crate replaces `backend-mlx` on the Mac
//! path. Nothing enforces it at compile time any more (the two no longer share
//! a manifest), so it is a rule to know.
//!
//! ## Structure
//! - [`MlxcelEngine`] — the [`Backend`](gen2::advanced::plugin::Backend) impl.
//!   Owns the worker handle (which holds the `!Send` MLX model + tokenizer) and
//!   the engine-level [`Settings`](gen2::advanced::plugin::Settings).
//! - `session::MlxcelSession` — the
//!   [`BackendSession`](gen2::advanced::plugin::BackendSession).
//! - `puller::MlxcelTokenPuller` — the
//!   [`TokenPullerDyn`](gen2::advanced::plugin::TokenPullerDyn): PULLs decoded
//!   `(id, text)` tuples off the worker's mpsc channel.
//! - `worker::ModelWorker` — a **dedicated worker thread** owning
//!   `(LoadedModel, MlxcelTokenizer)`. MLX state is thread-confined (`!Send`),
//!   so all model touches (load, generate) run on that one thread; commands +
//!   token replies cross via `std::sync::mpsc`. This mirrors mlxcel's own
//!   `src/server/audio_worker.rs` thread-confinement pattern.
//!
//! ## FAST PATH (the whole point)
//! Generation wraps [`mlxcel::MlxInferenceSession::generate_streaming`], which
//! delegates verbatim to `CxxGenerator::generate_streaming` — the pipelined
//! decode loop that hits mlx-lm-class throughput. We do NOT drive
//! `forward` + `sample_token_optimized` per token (that loses the lookahead
//! pipeline and is ~8× slower — the exact trap that made the old backend slow).
//! `generate_streaming` is PUSH; gen2's `next_event()` is PULL, so the on-token
//! callback (which runs on the worker, where the tokenizer lives) decodes
//! id→text and pushes it down a channel the puller drains.
//!
//! ## Grammar-constrained decode
//! When `GenSpec.grammar` is set, generation CANNOT ride `generate_streaming`:
//! that fast path exposes no per-step logit hook and pipelines the next sample
//! ahead of the token callback, so a grammar mask can't be applied between
//! steps. Grammar generations take a **manual masked decode loop**
//! (`grammar::run_grammar_generation`) that mirrors mlxcel's own on-device
//! structured mask (`server/structured.rs::apply_structured_mask_to_logits`)
//! while driving gen2's canonical
//! [`GrammarMatcher`](gen2::advanced::plugin::GrammarMatcher).
//! Greedy/text generations keep the fast path unchanged.
//!
//! ## Still deferred
//! Stubbed: embeddings / multimodal / KV-snapshot → `None`; speculative decode
//! and logprobs deferred; `stats()` → `ExecutionStats::default()`.

use std::path::Path;

use gen2::advanced::BackendPlugin;

mod engine;
mod grammar;
mod puller;
mod session;
mod worker;

pub use engine::MlxcelEngine;

/// The name the backend reports, and the plugin registers under.
pub const NAME: &str = "mlxcel";

/// Whether `path` is a model directory this backend loads: a directory
/// holding at least one `*.safetensors` file. The same rule `gen2` itself
/// applies for its MLX backend, so a zoo bundle marked `mlx` lands here.
pub fn claims(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "safetensors")
    })
}

/// The backend as a plugin, ready for
/// [`EngineBuilder::backend`](gen2::EngineBuilder::backend).
///
/// Claims safetensors model directories (see [`claims`]). The engine is built
/// on the controller's thread and spawns its own MLX worker thread there.
pub fn plugin() -> BackendPlugin {
    BackendPlugin {
        name: NAME,
        claims,
        make: Box::new(|| Box::new(MlxcelEngine::new())),
    }
}
