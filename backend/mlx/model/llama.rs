//! Llama3-compatible transformer model.

use std::sync::OnceLock;

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use super::attention::Attention;
use super::ffn::FeedForward;
use super::norm::RmsNorm;
use super::quantized::Weight;
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
        // Qwen3 applies RMSNorm to Q and K post-projection, pre-RoPE.
        // Detected via `model_type == "qwen3"`; without it the
        // generic attention kernel produces garbage scores and the
        // model samples EOS on its first step. The norm weights are
        // still zero-initialised — the safetensors loader must
        // populate them from `self_attn.q_norm.weight` /
        // `self_attn.k_norm.weight` before inference.
        let qk_norm = config
            .model_type
            .as_deref()
            .map_or(false, |t| t == "qwen3");
        Self {
            attention: Attention::new_with_qk_norm(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                head_dim,
                qk_norm,
                config.rms_norm_eps,
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
        let x = x.add(&attn_out).expect("mlx op");

        // Post-attention norm → FFN → residual
        let normed = self.post_attn_norm.forward(&x);
        let ffn_out = self.ffn.forward(&normed);
        x.add(&ffn_out).expect("mlx op")
    }
}

/// Complete Llama3 model: embedding → N transformer blocks → final norm → LM head.
pub struct LlamaModel {
    /// Token embedding (may be quantized).
    pub embed_tokens: Weight,
    /// Transformer layers
    pub layers: Vec<TransformerBlock>,
    /// Final RMS normalization
    pub norm: RmsNorm,
    /// Output projection (lm_head, may be quantized).
    pub lm_head: Weight,
    /// Model configuration
    pub config: ModelConfig,
    /// Cached dequantized embedding table (avoids re-dequantizing on every token).
    embed_cache: OnceLock<Array>,
}

impl LlamaModel {
    /// Create a model with placeholder (zero) weights.
    /// The safetensors loader must overwrite every weight before inference.
    pub fn new(config: &ModelConfig) -> Self {
        let embed_tokens = Weight::plain(
            Array::zeros::<f32>(&[config.vocab_size as i32, config.hidden_size as i32])
                .expect("mlx op"),
        );

        let layers: Vec<TransformerBlock> = (0..config.num_hidden_layers)
            .map(|_| TransformerBlock::new(config))
            .collect();

        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps);

        let lm_head = Weight::plain(
            Array::zeros::<f32>(&[config.vocab_size as i32, config.hidden_size as i32])
                .expect("mlx op"),
        );

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: config.clone(),
            embed_cache: OnceLock::new(),
        }
    }

    /// Run a forward pass and return logits for all positions: (1, seq_len, vocab_size).
    /// Used by the speculative decoder to verify a draft batch in one shot.
    pub fn forward_all(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
        rope: &RotaryEmbedding,
    ) -> Array {
        let seq_len = tokens.len();
        let embed_full = self.embed_cache.get_or_init(|| self.embed_tokens.to_full());
        let indices = Array::from_slice(
            &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[seq_len as i32],
        );
        let x = embed_full.take_axis(&indices, 0).expect("mlx op");
        let hidden = self.config.hidden_size as i32;
        let mut x = x.reshape(&[1, seq_len as i32, hidden]).expect("mlx op");
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, rope, &mut cache[i], offset);
        }
        x = self.norm.forward(&x);
        // All positions: (1, seq_len, vocab_size)
        self.lm_head.matmul_transpose(&x)
    }

    /// Run a forward pass and return logits for the last position.
    ///
    /// `tokens`: flat list of token IDs for this step (prefill or single-token generation).
    /// `cache`: per-layer KV cache, grown each call.
    /// `rope`: precomputed rotary embeddings.
    ///
    /// Returns: (batch=1, vocab_size) logits for the final sequence position.
    pub fn forward(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
        rope: &RotaryEmbedding,
    ) -> Array {
        let seq_len = tokens.len();
        // offset is the absolute sequence position (cur_pos), passed explicitly so
        // RoPE remains correct after KV cache eviction shrinks k.shape()[2].

        // --- Token embedding ---
        // Dequantize embedding table once and cache (avoids 151K×4096 dequantize per token).
        let embed_full = self.embed_cache.get_or_init(|| self.embed_tokens.to_full());
        let indices = Array::from_slice(
            &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[seq_len as i32],
        );
        let x = embed_full.take_axis(&indices, 0).expect("mlx op");

        // Reshape to (1, seq_len, hidden_size) — batch dim of 1
        let hidden = self.config.hidden_size as i32;
        let mut x = x.reshape(&[1, seq_len as i32, hidden]).expect("mlx op");

        // --- Transformer layers ---
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, rope, &mut cache[i], offset);
        }

        // --- Final norm ---
        x = self.norm.forward(&x);

        // --- LM head: project to vocab ---
        let logits = self.lm_head.matmul_transpose(&x);

        // Return only the last position's logits: (1, vocab_size)
        let seq_i32 = seq_len as i32;
        logits.index((0..1, (seq_i32 - 1)..seq_i32, ..))
    }
}
