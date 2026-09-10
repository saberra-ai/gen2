//! External API inference backend.
//!
//! Connects to an OpenAI-compatible server (Ollama, llama.cpp server,
//! LM Studio, vLLM, etc.) via HTTP streaming SSE. Uses `reqwest::blocking`
//! because the gen2 controller run-loop is synchronous.

pub mod anthropic_puller;
mod engine;
// `pub(crate)` so `compat::backend::external_api::puller` can name the puller
// the facade's `TokenPuller` wraps (S5.1).
pub(crate) mod puller;
mod session;
#[cfg(test)]
mod tests;

pub use engine::Engine;
// Re-exported for other backend feature configs; default surface uses neither.
#[allow(unused_imports)]
pub use session::{RemotePuller, Session};
