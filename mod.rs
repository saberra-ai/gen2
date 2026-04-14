//! gen2: next-generation inference engine with pluggable backends.

pub mod backend;
pub mod bundle;
#[allow(unused_mut)]
pub mod controller;
// consumed by workspace dependents (src-tauri, pio-daemon)
#[allow(dead_code)]
pub mod engine;
// consumed by workspace dependents (src-tauri, pio-daemon)
#[allow(dead_code, unused_variables)]
pub mod generation;
pub mod kv;
pub mod media;
pub mod residency;
pub mod residency_policy;
pub mod residency_stats;
#[allow(
    dead_code,
    unused_variables,
    unused_unsafe,
    unused_imports,
    unused_assignments
)]
pub mod session_rt;

// Public re-exports for the primary surface
pub use engine::{
    EmbedLoadRequest, Engine, ExecError, ExecutionStats, LoadRequest, Settings,
    read_gguf_architecture, validate_model_architecture, validate_model_file,
};
pub use residency::{ResidencyInventory, ResidentRuntime, RuntimeKind};
pub use residency_policy::{
    ContextBudget, ResidencyPolicy, default_context_budget_for_tier, estimate_resident_mb_for_path,
};
pub use residency_stats::ResidencyStats;
// Re-export message types used by SessionSpec for convenience in integration tests
pub use crate::types::message::{Message, MessageBody, MessageChunk, MessageContent};
