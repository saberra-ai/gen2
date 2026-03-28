pub(crate) mod media_util;
pub(crate) mod prompt;
mod spec;

// Backend-specific Session is re-exported from gen2::backend
pub use crate::gen2::backend::{Session, SessionId};
// Llama-specific internal types (not needed by MLX backend)
#[cfg(feature = "backend-llamacpp")]
pub(crate) use crate::gen2::backend::{DecodeState, SessionCtxCell};
pub use prompt::{PromptContext, build_prompt_context, generation_reserve, merge_prompts};
pub use spec::SessionSpec;
