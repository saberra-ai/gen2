//! External API inference backend.
//!
//! Connects to an OpenAI-compatible server (Ollama, llama.cpp server,
//! LM Studio, vLLM, etc.) via HTTP streaming SSE. Uses `reqwest::blocking`
//! because the gen2 controller run-loop is synchronous.

mod engine;
mod puller;
mod session;

pub use engine::Engine;
pub use puller::TokenPuller;
pub use session::{Session, SessionId};
