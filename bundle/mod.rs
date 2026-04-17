pub mod gguf;
mod meta;

// Backend-specific ModelBundle is re-exported from gen2::backend
pub use crate::gen2::backend::ModelBundle;
pub use meta::ModelMeta;
