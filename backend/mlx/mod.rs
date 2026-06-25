//! MLX inference backend for Apple Silicon (macOS/iOS).

mod bundle;
pub(crate) mod eagle3_loader;
pub(crate) mod eagle_predictor;
mod engine;
#[cfg(test)]
mod golden;
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

pub use bundle::ModelBundle;
pub use engine::Engine;
pub use puller::TokenPuller;
pub use session::{Session, SessionId};
