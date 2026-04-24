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
use mlx_rs::ops::indexing::IndexOp;

use super::rope::RotaryEmbedding;

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
        let cfg: Self = serde_json::from_str(&s).context("parse eagle3 config.json")?;
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

        expect_shape(
            "embed_tokens",
            &self.embed_tokens,
            &[v_tgt as i32, h as i32],
        )?;
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
        expect_shape("mlp_gate_proj", &self.mlp_gate_proj, &[ff as i32, h as i32])?;
        expect_shape("mlp_up_proj", &self.mlp_up_proj, &[ff as i32, h as i32])?;
        expect_shape("mlp_down_proj", &self.mlp_down_proj, &[h as i32, ff as i32])?;
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
        anyhow::bail!("EAGLE-3 weight {name} shape mismatch: expected {expected:?}, got {got:?}");
    }
    Ok(())
}

impl EagleDraftModel {
    /// Single-step greedy draft. Produces the next target-vocab token id
    /// given the last emitted token and the target model's auxiliary
    /// hidden states at that position.
    ///
    /// Ports the vLLM reference forward pass
    /// (`vllm/model_executor/models/llama_eagle3.py`) for layer_idx=0.
    /// All operations run on position `pos` (offset from sequence start)
    /// for RoPE. Batch size 1, sequence length 1.
    ///
    /// * `last_token_id`  — target-vocab id of the last accepted token.
    /// * `aux_hidden_states` — `[1, 1, num_aux_layers * hidden_size]`:
    ///   concatenated hidden states from the target at the
    ///   configured aux layers (e.g. [2, 15, 27] for Gemma 4 26B).
    /// * `pos` — absolute position index for RoPE (target's cur_pos).
    /// * `rope` — precomputed RoPE table (theta=10000, head_dim=256 for
    ///   Gemma 4 26B).
    ///
    /// Returns `(target_token_id, draft_hidden_prenorm)`. The prenorm
    /// state can be used to chain additional draft steps (EAGLE-3's
    /// autoregressive K>1 path — handled by a separate iterative
    /// wrapper; this method is the single-step primitive).
    pub fn forward_step_argmax(
        &self,
        last_token_id: u32,
        aux_hidden_states: &Array,
        pos: usize,
        rope: &RotaryEmbedding,
    ) -> Result<(u32, Array)> {
        let cfg = &self.cfg.transformer_layer_config;
        let hidden_size = cfg.hidden_size as i32;
        let head_dim = cfg.head_dim as i32;
        let n_heads = cfg.num_attention_heads as i32;
        let n_kv = cfg.num_key_value_heads as i32;
        let eps = cfg.rms_norm_eps;

        // ── 1. Embed last token → [1, 1, H] ──────────────────────────
        // Look up the embedding row. embed_tokens is [V_target, H].
        let ids = Array::from_slice(&[last_token_id as i32], &[1, 1]);
        let embeds = self
            .embed_tokens
            .index(&ids) // [1, 1, H]
            .reshape(&[1, 1, hidden_size])
            .context("eagle3 embed reshape")?;

        // ── 2. Combine aux hidden states via fc: [1, 1, 3H] → [1, 1, H] ──
        let hidden = matmul_transpose(aux_hidden_states, &self.fc);

        // ── 3. Layer-0 forward (layer_idx == 0 path) ──────────────────
        //   a. embeds = input_layernorm(embeds)
        //   b. if norm_before_residual:
        //        hidden = hidden_norm(hidden); residual = hidden
        //      else:
        //        residual = hidden; hidden = hidden_norm(hidden)
        //   c. x = concat(embeds, hidden, dim=-1)  [1, 1, 2H]
        //   d. attn_out = self_attn(x)  [1, 1, H]
        //   e. hidden = residual + attn_out; residual = hidden
        //      normed = post_attention_layernorm(hidden)
        //   f. mlp_out = swiglu(normed)
        //   g. hidden = residual + mlp_out
        let embeds = rms_norm(&embeds, &self.input_layernorm, eps);
        let (hidden, residual) = if self.cfg.norm_before_residual {
            let h = rms_norm(&hidden, &self.hidden_norm, eps);
            (h.clone(), h)
        } else {
            let res = hidden.clone();
            let h = rms_norm(&hidden, &self.hidden_norm, eps);
            (h, res)
        };
        let x = mlx_rs::ops::concatenate_axis(&[&embeds, &hidden], -1)
            .context("concat embeds+hidden")?;

        let attn_out = self.self_attn_forward(&x, pos, n_heads, n_kv, head_dim, rope)?;
        let hidden = residual.add(&attn_out).context("residual add attn")?;
        let residual = hidden.clone();
        let normed = rms_norm(&hidden, &self.post_attention_layernorm, eps);
        let mlp_out = self.mlp_forward(&normed);
        let hidden = residual.add(&mlp_out).context("residual add mlp")?;

        // ── 4. Final norm + lm_head → draft-vocab logits ─────────────
        let hidden_prenorm = hidden.clone();
        let normed = rms_norm(&hidden, &self.norm, eps);
        let draft_logits = matmul_transpose(&normed, &self.lm_head); // [1, 1, 32000]

        // ── 5. Greedy argmax over draft vocab, map to target vocab ───
        let draft_id = mlx_rs::ops::indexing::argmax_axis(&draft_logits, -1, None)
            .context("argmax draft logits")?; // [1, 1] i32
        let draft_idx_i64: i64 = draft_id.item::<i32>() as i64;
        // d2t is i64[32000]; lookup position draft_idx.
        let target_id_i64: i64 = self.d2t.index(draft_idx_i64 as i32).item::<i64>();
        let target_id = target_id_i64 as u32;
        Ok((target_id, hidden_prenorm))
    }

    /// Self-attention forward for EAGLE-3 layer 0.
    ///
    /// Unlike a standard decoder layer, the input has dim `2H` (concat
    /// of embeds + hidden), so q/k/v project from `2H → q_out` / `kv_out`.
    /// The output is `[1, 1, H]` after o_proj.
    ///
    /// Single-token path (T=1): no KV cache needed since EAGLE runs
    /// fresh per draft step. Full attention over the single query.
    fn self_attn_forward(
        &self,
        x: &Array,
        pos: usize,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        rope: &RotaryEmbedding,
    ) -> Result<Array> {
        // Projections: [1, 1, 2H] → q: [1, 1, n_heads*head_dim], k/v: [1, 1, n_kv*head_dim]
        let q = matmul_transpose(x, &self.q_proj);
        let k = matmul_transpose(x, &self.k_proj);
        let v = matmul_transpose(x, &self.v_proj);

        // Reshape to [1, 1, heads, head_dim] → [1, heads, 1, head_dim]
        let q = q
            .reshape(&[1, 1, n_heads, head_dim])
            .context("q reshape")?
            .transpose_axes(&[0, 2, 1, 3])
            .context("q transpose")?;
        let k = k
            .reshape(&[1, 1, n_kv, head_dim])
            .context("k reshape")?
            .transpose_axes(&[0, 2, 1, 3])
            .context("k transpose")?;
        let v = v
            .reshape(&[1, 1, n_kv, head_dim])
            .context("v reshape")?
            .transpose_axes(&[0, 2, 1, 3])
            .context("v transpose")?;

        // RoPE at pos (offset). Same RoPE table as the target.
        let q = rope.forward(&q, pos);
        let k = rope.forward(&k, pos);

        // GQA: repeat KV heads to match Q heads.
        let (k, v) = if n_kv < n_heads {
            let repeats = (n_heads / n_kv) as usize;
            (repeat_kv(&k, repeats)?, repeat_kv(&v, repeats)?)
        } else {
            (k, v)
        };

        // Scaled dot-product attention (T_kv=1, T_q=1; causal is trivial).
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let scores = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2]).context("k^T")?)
            .context("q @ k^T")?;
        let scale_arr = Array::from_f32(scale);
        let scores = scores.multiply(&scale_arr).context("scale scores")?;
        let probs = mlx_rs::ops::softmax_axes(&scores, &[-1], None).context("attn softmax")?;
        let attn = probs.matmul(&v).context("attn @ v")?;

        // Transpose back: [1, heads, 1, head_dim] → [1, 1, heads*head_dim]
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])
            .context("attn transpose back")?
            .reshape(&[1, 1, n_heads * head_dim])
            .context("attn reshape")?;

        // Output projection.
        Ok(matmul_transpose(&attn, &self.o_proj))
    }

    /// SwiGLU MLP: down(silu(gate(x)) * up(x)).
    fn mlp_forward(&self, x: &Array) -> Array {
        let gate = matmul_transpose(x, &self.mlp_gate_proj);
        let sig = mlx_rs::ops::sigmoid(&gate).expect("sigmoid");
        let gate = gate.multiply(&sig).expect("silu");
        let up = matmul_transpose(x, &self.mlp_up_proj);
        let hidden = gate.multiply(&up).expect("gate*up");
        matmul_transpose(&hidden, &self.mlp_down_proj)
    }
}

/// Plain-weight matmul with transpose, mirroring
/// `quantized::Weight::matmul_transpose` for un-quantized tensors.
/// `w` is stored as `(out_features, in_features)`; `x @ w.T`.
fn matmul_transpose(x: &Array, w: &Array) -> Array {
    // mlx-rs auto-broadcasts the leading dims; matmul handles it.
    let dims = w.shape().len();
    let w_t_axes: Vec<i32> = (0..dims as i32 - 2)
        .chain([dims as i32 - 1, dims as i32 - 2])
        .collect();
    let w_t = w.transpose_axes(&w_t_axes).expect("w transpose");
    x.matmul(&w_t).expect("matmul")
}

/// Standard RMSNorm (no Gemma +1 offset) — EAGLE-3 uses plain LlamaRMSNorm.
fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Array {
    let x_sq = x.multiply(x).expect("x*x");
    let variance = x_sq.mean_axis(-1, true).expect("mean");
    let eps_a = Array::from_f32(eps);
    let var_eps = variance.add(&eps_a).expect("var+eps");
    let norm_factor = var_eps.rsqrt().expect("rsqrt");
    let normalized = x.multiply(&norm_factor).expect("x * norm");
    normalized.multiply(weight).expect("norm * weight")
}

/// Repeat KV heads along axis 1 by the given factor (GQA → MHA).
/// `kv`: `[B, n_kv, T, head_dim]` → `[B, n_kv * repeats, T, head_dim]`.
fn repeat_kv(kv: &Array, repeats: usize) -> Result<Array> {
    if repeats == 1 {
        return Ok(kv.clone());
    }
    let shape = kv.shape();
    let b = shape[0];
    let n_kv = shape[1];
    let t = shape[2];
    let hd = shape[3];
    // Expand: [B, n_kv, 1, T, hd] → [B, n_kv, repeats, T, hd] → [B, n_kv*repeats, T, hd]
    let expanded = kv
        .reshape(&[b, n_kv, 1, t, hd])
        .context("kv reshape for repeat")?;
    let broadcast_shape: Vec<i32> = vec![b, n_kv, repeats as i32, t, hd];
    let broadcasted =
        mlx_rs::ops::broadcast_to(&expanded, &broadcast_shape).context("broadcast kv")?;
    broadcasted
        .reshape(&[b, n_kv * repeats as i32, t, hd])
        .context("kv reshape after repeat")
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
