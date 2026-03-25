//! Llama3-compatible transformer model for the MLX backend.
//!
//! Weight arrays are initialized as zeros — actual values are loaded from
//! safetensors by the loader module.

mod attention;
mod ffn;
mod llama;
mod norm;
mod rope;

pub use llama::LlamaModel;
pub use rope::RotaryEmbedding;

use serde::Deserialize;

/// One KV-cache entry per transformer layer.
/// `None` on the first forward pass; grows along the sequence dimension thereafter.
pub type KvCache = Vec<Option<(mlx_rs::Array, mlx_rs::Array)>>;

/// Model hyperparameters deserialized from HuggingFace `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    /// Explicit head dimension override. Falls back to `hidden_size / num_attention_heads`.
    pub head_dim: Option<usize>,
}

impl ModelConfig {
    /// Per-head dimension, derived from `hidden_size / num_attention_heads`
    /// unless an explicit override is present.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}
