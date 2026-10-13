//! Text generation from ONNX Runtime GenAI bundles (`genai_config.json`).
//! The host supplies GenAI 0.15.2 via `GEN2_ONNX_LIBRARY`. Graph execution,
//! sampling and KV/recurrent caches belong to GenAI; templates, conversations,
//! stop sequences and tools use gen2's existing contracts. A fresh generator
//! per turn ensures edits and cancellations cannot leave hidden conversation
//! state. Laya's decision-only ONNX path stays separate.

mod config;
mod engine;
mod ffi;
mod puller;
mod session;

pub use engine::OnnxEngine;
