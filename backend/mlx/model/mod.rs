//! MLX transformer model definitions.
//!
//! Weight arrays are initialized as zeros — actual values are loaded from
//! safetensors by the loader module.

mod attention;
pub mod diffusion_gemma;
pub mod eagle3;
mod ffn;
pub mod gemma4;
mod gemma4_fast;
mod llama;
pub mod moe;
mod norm;
pub mod profile;
pub mod quantized;
pub(crate) mod rope;
pub mod vision;
pub mod vision_preprocess;

pub use diffusion_gemma::{DiffusionGemmaConfig, DiffusionGemmaModel, DiffusionGenParams};
pub use gemma4::Gemma4Model;
pub use llama::LlamaModel;
pub use quantized::Weight;
pub use rope::RotaryEmbedding;

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use serde::Deserialize;

/// One KV-cache entry per transformer layer.
/// `None` on the first forward pass; grows along the sequence dimension thereafter.
pub type KvCache = Vec<Option<(Array, Array)>>;

/// Unified model enum — dispatch at the top level keeps session/puller code unchanged.
pub enum Model {
    Llama(LlamaModel),
    Gemma4(Gemma4Model),
    /// DiffusionGemma (block-diffusion Gemma 4). Encoder/decoder, not
    /// autoregressive — the standard token-by-token `forward` paths do not
    /// apply (slice 1: model is driven via its own `encode`/`decode` API).
    /// The denoising generation loop lands in slice 2.
    DiffusionGemma(DiffusionGemmaModel),
}

impl Model {
    /// Forward pass. GemmaModel ignores `rope` and uses its own internal per-layer ropes.
    ///
    /// `offset` is the true number of positions already processed (pre-update).
    /// For sliding-window caches this is NOT equal to `cache[i].shape[2]` —
    /// the buffer gets truncated to `window_size` but RoPE must still use the
    /// absolute position.
    pub fn forward(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
        rope: &RotaryEmbedding,
    ) -> Array {
        match self {
            Model::Llama(m) => m.forward(tokens, offset, cache, rope),
            Model::Gemma4(m) => m.forward(tokens, offset, cache),
            Model::DiffusionGemma(_) => {
                // DiffusionGemma is encoder/decoder block-diffusion; it is not
                // driven by the autoregressive `forward` path. Slice 2 wires
                // the denoising loop through `encode`/`decode` directly.
                panic!("DiffusionGemma does not support autoregressive forward; use encode/decode")
            }
        }
    }

    /// Vision prefill forward: scatter projected `image_features` into the
    /// image-token rows, then run the decoder. Returns last-token logits
    /// `[1, 1, vocab]`. Only Gemma 4 supports this; other models panic (the
    /// caller gates on `bundle.vision.is_some()`, which is Gemma-4-only).
    pub fn forward_with_image(
        &self,
        tokens: &[u32],
        image_features: &Array,
        image_token_id: u32,
        offset: usize,
        cache: &mut KvCache,
    ) -> Array {
        match self {
            Model::Gemma4(m) => {
                m.forward_with_image(tokens, image_features, image_token_id, offset, cache)
            }
            _ => panic!("forward_with_image is only implemented for Gemma 4 (vision bundles)"),
        }
    }

    /// Returns logits for every input position: `(1, seq_len, vocab_size)`.
    /// Returns `None` for models that do not support batched speculative decoding.
    pub fn forward_all(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
        rope: &RotaryEmbedding,
    ) -> Option<Array> {
        match self {
            Model::Llama(m) => Some(m.forward_all(tokens, offset, cache, rope)),
            Model::Gemma4(m) => Some(m.forward_all(tokens, offset, cache)),
            Model::DiffusionGemma(_) => None,
        }
    }

    /// True when this model is running the MLX **fast path** (`PIO_MLX_FAST`).
    /// The pipelined GPU-argmax decode loop (Stage B) only engages here; every
    /// other model / the flag-off path returns `false` and keeps the existing
    /// serial CPU-sampling decode untouched. Only Gemma 4 has a fast path.
    pub fn is_fast(&self) -> bool {
        match self {
            Model::Gemma4(m) => m.fast,
            _ => false,
        }
    }

    /// Fast-path forward whose input is a **lazy** `[seq]` int32 token-id
    /// `Array` (not a host `&[u32]`), returning the LAST-position logits
    /// `[1, 1, vocab]`. This is the on-GPU decode step for Stage-B pipelining:
    /// the lazy argmax token from step N feeds straight in as step N+1's
    /// embedding-gather index, so the chain never round-trips to host —
    /// mirroring mlx-lm's `model(y[None])` with lazy `y` (`generate.py:459`).
    ///
    /// Returns `None` for any model without a fast path (caller falls back to
    /// the host-token serial path). Only valid when [`Self::is_fast`] is true.
    pub fn forward_fast_last_logits_from_ids(
        &self,
        token_ids: &Array,
        seq_len: usize,
        offset: usize,
        cache: &mut KvCache,
    ) -> Option<Array> {
        match self {
            Model::Gemma4(m) if m.fast => {
                let all = m.forward_all_fast_from_ids(token_ids, seq_len, offset, cache);
                let s = seq_len as i32;
                Some(all.index((0..1, (s - 1)..s, ..)))
            }
            _ => None,
        }
    }

    /// True for the DiffusionGemma block-diffusion model, which is driven by
    /// its own `encode`/`decode`/`diffusion_generate` path rather than the
    /// autoregressive `forward` loop. Session/puller branch on this.
    pub fn is_diffusion(&self) -> bool {
        matches!(self, Model::DiffusionGemma(_))
    }

    pub fn num_non_shared_layers(&self) -> usize {
        match self {
            Model::Llama(m) => m.config.num_hidden_layers,
            Model::Gemma4(m) => m.num_non_shared,
            Model::DiffusionGemma(m) => m.config.num_hidden_layers,
        }
    }

    /// Variant of `forward_all` that also stashes the post-block hidden
    /// state for each layer id in `aux_layer_ids`. Returns logits plus
    /// a `Vec<Array>` with one entry per requested layer (empty for
    /// backends that haven't wired this up). EAGLE-3 speculative decode
    /// uses this: the draft model consumes these aux states as its
    /// feature input.
    ///
    /// Only Gemma 4 wires this currently — Llama returns `(logits, vec![])`.
    pub fn forward_all_with_aux(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
        rope: &RotaryEmbedding,
        aux_layer_ids: &[usize],
    ) -> Option<(Array, Vec<Array>)> {
        match self {
            Model::Llama(m) => Some((m.forward_all(tokens, offset, cache, rope), Vec::new())),
            Model::Gemma4(m) => Some(m.forward_all_with_aux(tokens, offset, cache, aux_layer_ids)),
            Model::DiffusionGemma(_) => None,
        }
    }
}

/// Model hyperparameters deserialized from HuggingFace `config.json`.
/// Compatible with Llama, Qwen 3, Mistral, Gemma 4, and other GQA architectures.
#[derive(Debug, Clone, Deserialize)]
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
    /// Sliding window size (Qwen 3, Mistral, Gemma 4). None = full attention.
    pub sliding_window: Option<usize>,
    /// Whether embeddings and lm_head share weights.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // ── Gemma 4 specific fields ────────────────────────────────────────────
    /// HuggingFace `model_type` string. Used to dispatch between Llama/Gemma loaders.
    pub model_type: Option<String>,
    /// head_dim for full-attention layers (Gemma 4 has different dims per layer type).
    pub global_head_dim: Option<usize>,
    /// Fraction of head_dim that receives rotary embeddings in full-attention layers.
    pub global_partial_rotary_factor: Option<f32>,
    /// RoPE base for local/sliding-window layers.
    pub rope_local_base_freq: Option<f32>,
    /// Per-layer attention type list. "sliding_attention" or "full_attention".
    pub layer_types: Option<Vec<String>>,
    /// Number of trailing layers that share KV with paired non-shared layers.
    pub num_kv_shared_layers: Option<usize>,
    /// Hidden dim of per-layer input embeddings (0 / absent = disabled).
    pub hidden_size_per_layer_input: Option<usize>,
    /// Vocab size used for per-layer input embeddings.
    pub vocab_size_per_layer_input: Option<usize>,
    /// Whether KV-shared layers use 2× wider MLP.
    pub use_double_wide_mlp: Option<bool>,
    /// Final logit softcapping: tanh(x/cap)*cap applied before sampling.
    pub final_logit_softcapping: Option<f32>,
    /// rope_parameters nested object (Gemma 4 format). Raw JSON, parsed by loader.
    pub rope_parameters: Option<serde_json::Value>,
    /// Pattern length for alternating sliding/full attention (fallback when layer_types absent).
    pub sliding_window_pattern: Option<usize>,
    /// Whether K == V (only used in large Gemma 4 variants, false for E2B).
    pub attention_k_eq_v: Option<bool>,
    /// Number of KV heads for global (full) attention layers.
    pub num_global_key_value_heads: Option<usize>,

    // ── Gemma 4 26B MoE fields ─────────────────────────────────────────────
    /// Whether this model uses MoE blocks alongside the dense MLP.
    pub enable_moe_block: Option<bool>,
    /// Number of expert MLPs (26B = 32).
    pub num_experts: Option<usize>,
    /// Experts selected per token (typically 2-4).
    pub top_k_experts: Option<usize>,
    /// Per-expert intermediate size (distinct from dense `intermediate_size`).
    pub moe_intermediate_size: Option<usize>,
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
