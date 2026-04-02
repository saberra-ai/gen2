//! Llama3-compatible transformer model for the MLX backend.
//!
//! Weight arrays are initialized as zeros — actual values are loaded from
//! safetensors by the loader module.

mod attention;
mod ffn;
mod llama;
mod norm;
pub mod quantized;
mod rope;

pub use llama::LlamaModel;
pub use quantized::Weight;
pub use rope::RotaryEmbedding;

use serde::Deserialize;

/// One KV-cache entry per transformer layer.
/// `None` on the first forward pass; grows along the sequence dimension thereafter.
pub type KvCache = Vec<Option<(mlx_rs::Array, mlx_rs::Array)>>;

/// Model hyperparameters deserialized from HuggingFace `config.json`.
/// Compatible with Llama, Qwen 3, Mistral, and other GQA architectures.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Explicit head dimension override. Falls back to `hidden_size / num_attention_heads`.
    pub head_dim: Option<usize>,
    /// Sliding window size (Qwen 3, Mistral). None = full attention.
    pub sliding_window: Option<usize>,
    /// Whether embeddings and lm_head share weights.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_max_position_embeddings() -> usize {
    32768
}

impl ModelConfig {
    /// Per-head dimension, derived from `hidden_size / num_attention_heads`
    /// unless an explicit override is present.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}
