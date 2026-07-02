//! External API inference backend.
//!
//! Connects to an OpenAI-compatible server (Ollama, llama.cpp server,
//! LM Studio, vLLM, etc.) via HTTP streaming SSE. Uses `reqwest::blocking`
//! because the gen2 controller run-loop is synchronous.

pub mod anthropic_puller;
mod engine;
mod puller;
mod session;

pub use engine::Engine;
// Re-exported for other backend feature configs; default surface uses neither.
#[allow(unused_imports)]
pub use session::{RemotePuller, Session};
