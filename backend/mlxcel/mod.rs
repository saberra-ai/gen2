//! `backend-mlxcel` — fast MLX inference embedded via the [`mlxcel`] crate
//! (Rust + MLX C++ bindings, decode ≈ mlx-lm). Replaces `backend-mlx` (mlx-rs)
//! on the macOS/daemon path, which was measured ~8–11× slower than mlx-lm.
//! Roadmap: `docs/plans/mlxcel-embedding-roadmap.md` (slice S2).
//!
//! ## Structure (mirrors `super::mlx`)
//! - [`engine::MlxcelEngine`] — the [`Backend`](super::traits::Backend) impl.
//!   Owns the worker handle (which holds the `!Send` MLX model + tokenizer) and
//!   the engine-level [`Settings`](crate::gen2::engine::Settings).
//! - [`session::MlxcelSession`] — the [`BackendSession`](super::traits::BackendSession).
//! - [`puller::MlxcelTokenPuller`] — the [`TokenPullerDyn`](super::traits::TokenPullerDyn):
//!   PULLs decoded `(id, text)` tuples off the worker's mpsc channel.
//! - [`worker::ModelWorker`] — a **dedicated worker thread** owning
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
//! ## Grammar-constrained decode (the S4 divergence)
//! When `GenSpec.grammar` is set, generation CANNOT ride `generate_streaming`:
//! that fast path exposes no per-step logit hook and pipelines the next sample
//! ahead of the token callback, so a grammar mask can't be applied between
//! steps. Grammar generations take a **manual masked decode loop**
//! ([`grammar::run_grammar_generation`]) that mirrors mlxcel's own on-device
//! structured mask (`server/structured.rs::apply_structured_mask_to_logits`)
//! while driving pio-core's canonical [`GrammarMatcher`](crate::gen2::backend::common::grammar::GrammarMatcher).
//! Greedy/text generations keep the fast path unchanged.
//!
//! ## Tracer-bullet scope (S2, still deferred)
//! Stubbed: embeddings / multimodal / KV-snapshot → `None`; speculative decode
//! and logprobs deferred; the prompt is built simply from `SessionSpec.messages`
//! (full chat-template is a later slice); `stats()` → `ExecutionStats::default()`.

mod engine;
mod grammar;
mod puller;
mod session;
mod worker;

// Consumed by the `#[cfg(test)]` captests (drive `MlxcelEngine` directly) and
// by facade routing in a later slice (S6); unused in a plain non-test lib build.
#[allow(unused_imports)]
pub(crate) use engine::MlxcelEngine;
