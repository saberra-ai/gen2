//! MLX model bundle — holds the loaded model, tokenizer, and metadata.

use super::model::{LlamaModel, ModelConfig, RotaryEmbedding};
use super::tokenizer::HfTokenizer;
use crate::gen2::bundle::ModelMeta;
use crate::gen2::engine::Capabilities;

pub struct ModelBundle {
    pub model: LlamaModel,
    pub rope: RotaryEmbedding,
    pub tokenizer: HfTokenizer,
    pub config: ModelConfig,
    pub capabilities: Capabilities,
    pub meta: ModelMeta,
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
        f.debug_struct("ModelBundle(MLX)")
            .field("capabilities", &self.capabilities)
            .field("meta", &self.meta)
            .field("num_layers", &self.config.num_hidden_layers)
            .finish()
    }
}
