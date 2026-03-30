//! Llama3-compatible transformer model.

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use super::attention::Attention;
use super::ffn::FeedForward;
use super::norm::RmsNorm;
use super::rope::RotaryEmbedding;
use super::{KvCache, ModelConfig};

/// One transformer layer: pre-norm attention + post-attention-norm FFN with residuals.
pub struct TransformerBlock {
    pub attention: Attention,
    pub ffn: FeedForward,
    pub input_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
}

impl TransformerBlock {
    pub fn new(config: &ModelConfig) -> Self {
        let head_dim = config.head_dim();
        Self {
            attention: Attention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                head_dim,
            ),
            ffn: FeedForward::new(config.hidden_size, config.intermediate_size),
            input_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attn_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps),
        }
    }

    /// Forward through one transformer layer with residual connections.
    ///
    /// `x`: (batch, seq_len, hidden_size)
    fn forward(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        cache: &mut Option<(Array, Array)>,
        offset: usize,
    ) -> Array {
        // Pre-norm → attention → residual
        let normed = self.input_norm.forward(x);
        let attn_out = self.attention.forward(&normed, rope, cache, offset);
        let x = x.add(&attn_out).unwrap();

        // Post-attention norm → FFN → residual
        let normed = self.post_attn_norm.forward(&x);
        let ffn_out = self.ffn.forward(&normed);
        x.add(&ffn_out).unwrap()
    }
}

/// Complete Llama3 model: embedding → N transformer blocks → final norm → LM head.
pub struct LlamaModel {
    /// Token embedding table: (vocab_size, hidden_size)
    pub embed_tokens: Array,
    /// Transformer layers
    pub layers: Vec<TransformerBlock>,
    /// Final RMS normalization
    pub norm: RmsNorm,
    /// Output projection (lm_head): (vocab_size, hidden_size)
    pub lm_head: Array,
    /// Model configuration
    pub config: ModelConfig,
}

impl LlamaModel {
    /// Create a model with placeholder (zero) weights.
    /// The safetensors loader must overwrite every weight before inference.
    pub fn new(config: &ModelConfig) -> Self {
        let embed_tokens =
            Array::zeros::<f32>(&[config.vocab_size as i32, config.hidden_size as i32]).unwrap();

        let layers: Vec<TransformerBlock> = (0..config.num_hidden_layers)
            .map(|_| TransformerBlock::new(config))
            .collect();

        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps);

        let lm_head =
            Array::zeros::<f32>(&[config.vocab_size as i32, config.hidden_size as i32]).unwrap();

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: config.clone(),
        }
    }

    /// Run a forward pass and return logits for the last position.
    ///
    /// `tokens`: flat list of token IDs for this step (prefill or single-token generation).
    /// `cache`: per-layer KV cache, grown each call.
    /// `rope`: precomputed rotary embeddings.
    ///
    /// Returns: (batch=1, vocab_size) logits for the final sequence position.
    pub fn forward(&self, tokens: &[u32], cache: &mut KvCache, rope: &RotaryEmbedding) -> Array {
        let seq_len = tokens.len();

        // Compute position offset from the KV cache.
        // If layer 0 has a cached K, the offset is its seq dimension length.
        let offset = cache
            .first()
            .and_then(|entry| entry.as_ref())
            .map(|(k, _)| k.shape()[2] as usize)
            .unwrap_or(0);

        // --- Token embedding ---
        // Gather rows from embed_tokens for each token ID.
        let indices = Array::from_slice(
            &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[seq_len as i32],
        );
        // Use take_axis to gather along axis 0
        let x = self.embed_tokens.take_axis(&indices, 0).unwrap();

        // Reshape to (1, seq_len, hidden_size) — batch dim of 1
        let hidden = self.config.hidden_size as i32;
        let mut x = x.reshape(&[1, seq_len as i32, hidden]).unwrap();

        // --- Transformer layers ---
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, rope, &mut cache[i], offset);
        }

        // --- Final norm ---
        x = self.norm.forward(&x);

        // --- LM head: project to vocab ---
        let lm_head_t = self.lm_head.transpose_axes(&[1, 0]).unwrap();
        let logits = x.matmul(&lm_head_t).unwrap();

        // Return only the last position's logits: (1, vocab_size)
        // Use range-based indexing: logits[0:1, (seq_len-1):seq_len, 0:vocab]
        let seq_i32 = seq_len as i32;
        logits.index((0..1, (seq_i32 - 1)..seq_i32, ..))
    }
}
