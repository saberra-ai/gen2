//! Auxiliary model runtimes, off the controller thread.
//!
//! Generation is one concern. Embedding, reranking, transcription, OCR and
//! speech are different models with different lifetimes, dependencies and
//! latency profiles, and the only thing they share with the chat model is that
//! an application wants both at once.
//!
//! That is why they do not live on [`Backend`](crate::backend::Backend). Two
//! problems come from putting them there, and both are real today:
//!
//! - **Ownership.** `Backend::as_embeddings()` ties the embedder to whichever
//!   backend happens to own the chat model. An MLX chat model plus a llama.cpp
//!   GGUF embedder is a perfectly ordinary thing to want and is impossible to
//!   express, because MLX does not implement the capability.
//! - **Scheduling.** The controller thread pulls chat tokens. A helper call
//!   running there stops token scheduling for as long as it takes, which is
//!   tolerable for a small embedding and not for a five-second transcription.
//!
//! So helpers get their own thread, behind the controller, reached through
//! [`UtilityWorker`]. The controller forwards a helper request and returns
//! immediately; the worker answers the original caller directly.
//!
//! # Why one worker and not one per helper
//!
//! Because it is enough. Native model state is kept thread-confined either
//! way, which is the reason the primary backend is confined to the controller
//! thread in the first place. If helper-to-helper contention ever shows up in
//! a real workload, the worker can be sharded behind the same handle without
//! the public API noticing.

mod embedding;
mod rerank;
mod types;
mod worker;

pub use rerank::RerankResult;
pub use types::{LoadedUtility, UtilityStatus};
#[cfg(test)]
pub(crate) use worker::Factories;
pub(crate) use worker::UtilityWorker;

#[cfg(test)]
pub(crate) use embedding::{EmbeddingRuntime, ScriptedEmbedder};
#[cfg(test)]
pub(crate) use rerank::{RerankerRuntime, ScriptedReranker};

/// What the worker bought, proved through the controller rather than in
/// isolation.
#[cfg(test)]
mod acceptance_tests;
