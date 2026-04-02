//! ONNX model bundle — holds the ort Session, tokenizer, and metadata.

use crate::gen2::backend::common::tokenizer::HfTokenizer;
use crate::gen2::bundle::ModelMeta;
use crate::gen2::engine::Capabilities;
use parking_lot::Mutex;

pub struct ModelBundle {
    pub session: Mutex<ort::session::Session>,
    pub tokenizer: HfTokenizer,
    pub capabilities: Capabilities,
    pub meta: ModelMeta,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Whether the model expects `position_ids` as an input.
    pub has_position_ids: bool,
    pub model_dir: std::path::PathBuf,
    /// Jinja2 chat template, loaded once from `tokenizer_config.json` at model load time.
    pub chat_template_str: String,
    /// Decoded BOS token string, cached to avoid repeated tokenizer lookups.
    pub bos_str: String,
    /// Decoded EOS token string, cached to avoid repeated tokenizer lookups.
    pub eos_str: String,
}

impl std::fmt::Debug for ModelBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelBundle(ONNX)")
            .field("capabilities", &self.capabilities)
            .field("meta", &self.meta)
            .field("num_layers", &self.num_layers)
            .finish()
    }
}
