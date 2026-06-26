//! MLX inference backend for Apple Silicon (macOS/iOS).

mod bundle;
pub(crate) mod eagle3_loader;
pub(crate) mod eagle_predictor;
mod engine;
#[cfg(test)]
mod golden;
#[cfg(test)]
mod kv_donation_probe;
mod loader;
pub(crate) mod model;
// ngram module replaced by cross-backend `common::speculative::*` —
// see that module for the trigram impl plus PLD / Lookahead / Eagle3
// alternatives. Kept as an alias for backward-compat in external code.
#[cfg(test)]
mod profile_decode;
mod puller;
mod sampler;
mod session;
mod tokenizer;
#[cfg(test)]
mod vision_parity;

pub use bundle::ModelBundle;
pub use engine::Engine;
// Re-exported for the in-crate test modules (`golden.rs`, `vision_parity.rs`),
// which drive sessions/pullers directly. Gated so the non-test build (where no
// caller references them) stays warning-free.
#[cfg(test)]
pub use puller::TokenPuller;
#[cfg(test)]
pub use session::Session;
