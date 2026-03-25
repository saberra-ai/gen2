//! ONNX Runtime inference backend.
//!
//! Supports CPU, DirectML (Windows AMD/Intel GPU), CUDA, and CoreML
//! execution providers via ort feature flags.

mod bundle;
mod engine;
mod loader;
mod puller;
mod session;

pub use bundle::ModelBundle;
pub use engine::Engine;
pub use puller::TokenPuller;
pub use session::{Session, SessionId};
