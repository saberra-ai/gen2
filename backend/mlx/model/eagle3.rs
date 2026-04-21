//! EAGLE-3 draft-model architecture for MLX.
//!
//! EAGLE-3 (Li et al., 2025 — arXiv:2503.01840) is a speculative-decode
//! draft model that uses auxiliary hidden states from specific target
//! layers. Given the target model's hidden states at layers L_aux
//! (typically `[early, middle, late]`) and the last emitted token, the
//! draft model proposes the next K tokens much cheaper than running the
//! full target.
//!
//! ## Architecture (shapes verified against
//! `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` / `config.json`)
//!
//! ```text
//! Inputs:
//!   tok_ids  : [B, T]           target-vocab token ids (262144)
//!   aux_hs   : [B, T, 3*H]      concat of hidden states from target layers [2, 15, 27]
//!                               where H = target hidden size (2816 for Gemma 4 26B MoE).
//!
//! Path:
//!   e   = embed_tokens(tok_ids)            [B, T, H]       re-uses target vocab (262144, 2816)
//!   h   = fc(aux_hs)                       [B, T, H]       projects 3*H -> H
//!   (optional: h = hidden_norm(h) if norm_before_residual)
//!   x   = concat(e, h, dim=-1)             [B, T, 2H]      layer-0 input
//!
//!   // Single Llama-style decoder layer (qkv project from 2H):
//!   y   = self_attn(input_layernorm(x))    [B, T, H]
//!   x   = x_resid + y                                      (x_resid = projection of x to H)
//!   y   = mlp(post_attention_layernorm(x)) [B, T, H]       SwiGLU: down(silu(gate) * up)
//!   x   = x + y
//!
//!   x   = norm(x)                          [B, T, H]
//!   lg  = lm_head(x)                       [B, T, 32000]   draft-vocab logits
//!
//! Output:
//!   draft_token_ids = d2t[argmax(lg, dim=-1)]              target-vocab ids via d2t map
//! ```
//!
//! ## Integration status
//!
//! This module defines the architecture + a weight-loader shape
//! verification. **The target-model hidden-state extraction is NOT yet
//! plumbed.** To complete integration:
//!
//!  1. Modify `mlx/model/gemma4.rs` forward pass to stash hidden states
//!     from layers `eagle_aux_hidden_state_layer_ids` after their
//!     respective decoder blocks (currently layers [2, 15, 27] for
//!     Gemma 4 26B).
//!  2. Expose the stashed states via a new method on the target Model
//!     (e.g. `fn aux_hidden_states(&self) -> Option<&[Array]>`).
//!  3. Thread an `Option<EagleDraftModel>` through the MLX puller.
//!  4. Build an `EagleDraftPredictor: SpeculativePredictor` that, on
//!     `draft()`, reads the latest aux states from the target and runs
//!     this model's forward pass K times autoregressively (K=3 per
//!     EAGLE-3's `speculative_tokens` default) to produce the draft.
//!  5. Use `d2t[]` to map draft ids back to target ids before handing
//!     to the existing speculative-verify path.
//!
//! Expected speedup on Gemma 4 26B: 2-3× greedy decode throughput per
//! the upstream paper and vLLM benchmarks on A4B models.

use anyhow::{Context, Result};
use mlx_rs::Array;

/// EAGLE-3 draft-model configuration. Mirrors the HuggingFace
/// `Eagle3SpeculatorConfig` shape so we can deserialize their
/// `config.json` directly.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Eagle3Config {
    /// Size of the draft lm-head output (e.g. 32000 for the Gemma 4 26B
    /// speculator). Target vocab size is recovered from
    /// `transformer_layer_config.vocab_size`.
    pub draft_vocab_size: usize,
    /// Apply `hidden_norm` to aux features BEFORE adding to residual.
    /// Default in the RedHatAI Gemma 4 checkpoint: true.
    #[serde(default)]
    pub norm_before_residual: bool,
    /// Apply a pre-fc RMSNorm (present in gpt-oss drafts, absent in
    /// Gemma 4 / Llama 3 drafts). Default false.
    #[serde(default)]
    pub norm_before_fc: bool,
    /// Target-layer ids whose hidden states feed the fc projection.
    /// `None` means "use all target layers concatenated" (fallback only;
    /// every recent EAGLE-3 checkpoint specifies the triplet).
    pub eagle_aux_hidden_state_layer_ids: Option<Vec<usize>>,
    /// Inner transformer config (Llama-family).
    pub transformer_layer_config: Eagle3LayerConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Eagle3LayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
}

fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_max_pos() -> usize {
    262_144
}

impl Eagle3Config {
    /// Load from a HuggingFace-style `config.json`.
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("read eagle3 config: {}", path.display()))?;
        let cfg: Self =
            serde_json::from_str(&s).context("parse eagle3 config.json")?;
        Ok(cfg)
    }

    pub fn num_aux_layers(&self) -> usize {
        self.eagle_aux_hidden_state_layer_ids
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(1)
    }

    /// Expected shape of the aux-state concatenation dim, per the
    /// checkpoint's layer count × hidden size.
    pub fn aux_concat_dim(&self) -> usize {
        self.num_aux_layers() * self.transformer_layer_config.hidden_size
    }
}

/// EAGLE-3 draft-model parameter bundle. Owns the MLX arrays loaded
/// from `model.safetensors`. Forward pass is implemented on top of
/// this struct in a follow-up (tracked task #17); this scaffolding
/// provides a verifiable architecture container.
#[allow(dead_code)] // forward pass lives in the integration follow-up
pub struct EagleDraftModel {
    pub cfg: Eagle3Config,

    // Shared with target (vocab × hidden_size).
    pub embed_tokens: Array,

    // Aux-state projection: (hidden_size, num_aux_layers * target_hidden).
    pub fc: Array,

    // Per-layer tensors (single layer in current EAGLE-3 checkpoints).
    pub input_layernorm: Array,
    pub hidden_norm: Array,
    pub post_attention_layernorm: Array,
    pub q_proj: Array,
    pub k_proj: Array,
    pub v_proj: Array,
    pub o_proj: Array,
    pub mlp_gate_proj: Array,
    pub mlp_up_proj: Array,
    pub mlp_down_proj: Array,

    // Final LM head to DRAFT vocab (32k).
    pub norm: Array,
    pub lm_head: Array,

    /// Draft-to-target vocab mapping: `d2t[draft_id]` → target token id.
    /// int64 on disk; stored as i64 MLX array.
    pub d2t: Array,
    /// Target-to-draft availability mask: bool vector sized to target
    /// vocab. `true` at index i means target token i appears in the
    /// draft vocab. Useful for pre-masking target logits during
    /// draft-guided verification.
    pub t2d: Array,
}

impl EagleDraftModel {
    /// Validate that a loaded parameter set has the shapes expected
    /// from `cfg`. Catches checkpoint / config mismatches early rather
    /// than at the first confusing forward-pass panic.
    pub fn verify_shapes(&self) -> Result<()> {
        let h = self.cfg.transformer_layer_config.hidden_size;
        let v_tgt = self.cfg.transformer_layer_config.vocab_size;
        let v_drf = self.cfg.draft_vocab_size;
        let ff = self.cfg.transformer_layer_config.intermediate_size;
        let q_out = self.cfg.transformer_layer_config.num_attention_heads
            * self.cfg.transformer_layer_config.head_dim;
        let kv_out = self.cfg.transformer_layer_config.num_key_value_heads
            * self.cfg.transformer_layer_config.head_dim;
        let qkv_in = 2 * h;
        let aux_in = self.cfg.aux_concat_dim();

        expect_shape("embed_tokens", &self.embed_tokens, &[v_tgt as i32, h as i32])?;
        expect_shape("fc", &self.fc, &[h as i32, aux_in as i32])?;
        expect_shape("input_layernorm", &self.input_layernorm, &[h as i32])?;
        expect_shape("hidden_norm", &self.hidden_norm, &[h as i32])?;
        expect_shape(
            "post_attention_layernorm",
            &self.post_attention_layernorm,
            &[h as i32],
        )?;
        expect_shape("q_proj", &self.q_proj, &[q_out as i32, qkv_in as i32])?;
        expect_shape("k_proj", &self.k_proj, &[kv_out as i32, qkv_in as i32])?;
        expect_shape("v_proj", &self.v_proj, &[kv_out as i32, qkv_in as i32])?;
        expect_shape("o_proj", &self.o_proj, &[h as i32, q_out as i32])?;
        expect_shape(
            "mlp_gate_proj",
            &self.mlp_gate_proj,
            &[ff as i32, h as i32],
        )?;
        expect_shape("mlp_up_proj", &self.mlp_up_proj, &[ff as i32, h as i32])?;
        expect_shape(
            "mlp_down_proj",
            &self.mlp_down_proj,
            &[h as i32, ff as i32],
        )?;
        expect_shape("norm", &self.norm, &[h as i32])?;
        expect_shape("lm_head", &self.lm_head, &[v_drf as i32, h as i32])?;
        expect_shape("d2t", &self.d2t, &[v_drf as i32])?;
        expect_shape("t2d", &self.t2d, &[v_tgt as i32])?;
        Ok(())
    }
}

fn expect_shape(name: &str, arr: &Array, expected: &[i32]) -> Result<()> {
    let got = arr.shape();
    if got != expected {
        anyhow::bail!(
            "EAGLE-3 weight {name} shape mismatch: expected {expected:?}, got {got:?}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parses_gemma4_26b_eagle3_shape() {
        // Synthetic config matching the shapes in the RedHatAI repo.
        let json = r#"{
            "draft_vocab_size": 32000,
            "norm_before_residual": true,
            "norm_before_fc": false,
            "eagle_aux_hidden_state_layer_ids": [2, 15, 27],
            "transformer_layer_config": {
                "hidden_size": 2816,
                "intermediate_size": 2112,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "vocab_size": 262144,
                "rms_norm_eps": 0.000001
            }
        }"#;
        let cfg: Eagle3Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.draft_vocab_size, 32000);
        assert_eq!(cfg.num_aux_layers(), 3);
        assert_eq!(cfg.aux_concat_dim(), 3 * 2816);
        assert_eq!(cfg.transformer_layer_config.head_dim, 256);
        assert!(cfg.norm_before_residual);
    }
}
