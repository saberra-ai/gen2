//! gen2: next-generation inference engine with pluggable backends.

pub mod backend;
pub mod bundle;
#[allow(unused_mut)]
pub mod controller;
#[allow(dead_code)]
pub mod engine;
#[allow(dead_code, unused_variables)]
pub mod generation;
pub mod kv;
pub mod media;
#[allow(dead_code, unused_variables, unused_unsafe, unused_imports, unused_assignments)]
pub mod session_rt;

// Public re-exports for the primary surface
pub use engine::{EmbedLoadRequest, Engine, ExecError, ExecutionStats, LoadRequest, Settings, validate_model_file, read_gguf_architecture, validate_model_architecture};
// Re-export message types used by SessionSpec for convenience in integration tests
pub use crate::generation::model_runner::types::{
    Message, MessageBody, MessageChunk, MessageContent,
};
