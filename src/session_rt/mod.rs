pub mod compaction;
pub(crate) mod media_util;
pub(crate) mod prompt;
mod spec;
pub mod truncate;

// Backend-specific Session is re-exported from gen2::backend
pub use crate::backend::{Session, SessionId};
// Llama-specific internal types (not needed by MLX backend)
#[cfg(feature = "backend-llamacpp")]
pub(crate) use crate::backend::{DecodeState, SessionCtxCell};
pub use compaction::{CompactResult, CompactionStrategy, compact_algorithmic};
pub use prompt::{PromptContext, build_prompt_context, generation_reserve, merge_prompts};
pub use spec::SessionSpec;
pub use truncate::{ColdStart, TruncationOutcome, WarmStart};
