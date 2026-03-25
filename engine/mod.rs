//! Engine: long-lived orchestrator.

mod error;
mod stats;
pub(crate) mod telemetry;
mod types;

// Backend-specific Engine is re-exported from gen2::backend
pub use crate::gen2::backend::Engine;
pub use error::ExecError;
pub use stats::ExecutionStats;
pub use telemetry::{HookBus, HookEvent, HookListener};
pub use types::{Capabilities, ChatTemplateSpec, CtxParamsInput, EmbedLoadRequest, LoadRequest, ModelParamsInput, Settings};
