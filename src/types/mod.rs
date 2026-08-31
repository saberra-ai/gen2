//! Wire types shared by every inference backend.
//!
//! These moved out of `pio-core::types` when gen2 became its own crate: they
//! are the vocabulary the engine speaks (messages in, stats out, the model
//! record it was asked to load), with no dependency on any host application.

pub mod execution_stats;
pub mod message;
pub mod model;
pub mod persona;

pub use execution_stats::ExecutionStats;
pub use model::{Model, ModelConfig, ModelMetadata};
pub use persona::Persona;
