//! Gemma 4 **vision tower** (NaFlex SigLIP2) + multimodal projector.
//!
//! A faithful port of mlx-vlm `models/gemma4/vision.py` and the
//! `MultimodalEmbedder` from `models/gemma4/gemma4.py` (clone at
//! `~/workspace/mlx-vlm`, the same reference gen2's decode mirrored from
//! mlx-lm). Every component cites the exact reference line it translates.
//!
//! Arch is **gemma4**, not gemma3: Linear patchify (no Conv2d), 2D
//! multidimensional RoPE, RMSNorm (not LayerNorm), one-hot position table, and
//! a position-aware avg-pooler — confirmed by the weight-key layout
//! (`vision_tower.patch_embedder.input_proj.*`).
//!
//! ## Bundle reality (gemma-4-e2b-it-4bit) vs the plan
//! - `use_clipped_linears = TRUE` for this bundle (the plan guessed False).
//!   The vision q/k/v/o + gate/up/down projections are **ClippableLinear**:
//!   the weight key is `...<proj>.linear.weight` (the `nn.Linear` submodule),
//!   and four **finite** scalar clip bounds are present
//!   (`input_min/input_max/output_min/output_max`). We apply
//!   `clip(x, in_min, in_max)` before and `clip(out, out_min, out_max)` after
//!   the matmul (vision.py:35-41). `patch_embedder.input_proj` is a plain
//!   `nn.Linear` (no clip).
//! - Vision-tower linears are **plain BF16** (unquantized).
//! - `embed_vision.embedding_projection` is **4-bit quantized**.

// Staged port: the tower/projector forward methods are exercised by the
// vision-parity tests (Stages 3-5) and the engine merge path (Stage 6); allow
// dead_code until every consumer lands so each stage commits clean.
#![allow(dead_code)]

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use super::norm::{RmsNorm, rms_norm_no_scale_fast};
use super::quantized::Weight;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Vision config (mirrors `gemma4/config.py:VisionConfig` :27-57). Values are
/// read from `config.json`'s `vision_config`; defaults match the dataclass.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,             // 768
    pub intermediate_size: usize,       // 3072
    pub num_hidden_layers: usize,       // 16
    pub num_attention_heads: usize,     // 12
    pub num_key_value_heads: usize,     // 12
    pub head_dim: usize,                // 64
    pub rms_norm_eps: f32,              // 1e-6
    pub patch_size: usize,              // 16
    pub position_embedding_size: usize, // 10240
    pub default_output_length: usize,   // 280
    pub pooling_kernel_size: usize,     // 3
    pub rope_theta: f32,                // 100.0
    pub use_clipped_linears: bool,      // true for e2b/e4b 4bit
    pub standardize: bool,              // false
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 16,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            patch_size: 16,
            position_embedding_size: 10240,
            default_output_length: 280,
            pooling_kernel_size: 3,
            rope_theta: 100.0,
            use_clipped_linears: false,
            standardize: false,
        }
    }
}

impl VisionConfig {
    /// Parse from the root `config.json` value's `vision_config` sub-object.
    /// Missing keys fall back to the dataclass defaults.
    pub fn from_root_json(root: &serde_json::Value) -> Self {
        let d = Self::default();
        let vc = root.get("vision_config");
        let g_usize = |k: &str, def: usize| -> usize {
            vc.and_then(|v| v.get(k))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(def)
        };
        let g_f32 = |k: &str, def: f32| -> f32 {
            vc.and_then(|v| v.get(k))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(def)
        };
        let g_bool = |k: &str, def: bool| -> bool {
            vc.and_then(|v| v.get(k))
                .and_then(|v| v.as_bool())
                .unwrap_or(def)
        };
        // rope_theta nests under rope_parameters.rope_theta.
        let rope_theta = vc
            .and_then(|v| v.get("rope_parameters"))
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(d.rope_theta);
        Self {
            hidden_size: g_usize("hidden_size", d.hidden_size),
            intermediate_size: g_usize("intermediate_size", d.intermediate_size),
            num_hidden_layers: g_usize("num_hidden_layers", d.num_hidden_layers),
            num_attention_heads: g_usize("num_attention_heads", d.num_attention_heads),
            num_key_value_heads: g_usize("num_key_value_heads", d.num_attention_heads),
            head_dim: g_usize("head_dim", d.head_dim),
            rms_norm_eps: g_f32("rms_norm_eps", d.rms_norm_eps),
            patch_size: g_usize("patch_size", d.patch_size),
            position_embedding_size: g_usize("position_embedding_size", d.position_embedding_size),
            default_output_length: g_usize("default_output_length", d.default_output_length),
            pooling_kernel_size: g_usize("pooling_kernel_size", d.pooling_kernel_size),
            rope_theta,
            use_clipped_linears: g_bool("use_clipped_linears", d.use_clipped_linears),
            standardize: g_bool("standardize", d.standardize),
        }
    }

    pub fn max_patches(&self) -> usize {
        self.default_output_length * self.pooling_kernel_size * self.pooling_kernel_size
    }
}

// ─── ClippableLinear ─────────────────────────────────────────────────────────

/// Linear with optional input/output clamping — mirrors `ClippableLinear`
/// (vision.py:10-41). When `use_clipping` is true the checkpoint ships four
/// scalar bounds; `__call__` does `clip(x, in_min, in_max)` → `linear(x)` →
/// `clip(out, out_min, out_max)` (vision.py:35-41). The vision linears here are
/// plain (unquantized) BF16 weights keyed `...<name>.linear.weight`.
pub struct ClippableLinear {
    pub weight: Weight,
    /// `Some((in_min, in_max, out_min, out_max))` when clipping is active.
    pub clip: Option<(Array, Array, Array, Array)>,
}

impl ClippableLinear {
    pub fn placeholder(out_features: usize, in_features: usize, use_clipping: bool) -> Self {
        let weight = Weight::plain(
            Array::zeros::<f32>(&[out_features as i32, in_features as i32]).expect("mlx op"),
        );
        // Bounds initialized to ±inf (no-op) until loaded — matches
        // ClippableLinear.__init__ (vision.py:30-33). When use_clipping is
        // false we keep `clip = None` and skip the clamp ops entirely.
        let clip = if use_clipping {
            Some((
                Array::from_f32(f32::NEG_INFINITY),
                Array::from_f32(f32::INFINITY),
                Array::from_f32(f32::NEG_INFINITY),
                Array::from_f32(f32::INFINITY),
            ))
        } else {
            None
        };
        Self { weight, clip }
    }

    /// `clip(x, in_min, in_max)` → `x @ Wᵀ` → `clip(out, out_min, out_max)`.
    pub fn forward(&self, x: &Array) -> Array {
        let x = match &self.clip {
            Some((in_min, in_max, _, _)) => {
                mlx_rs::ops::clip(x, (in_min, in_max)).expect("mlx op: clip in")
            }
            None => x.clone(),
        };
        let out = self.weight.matmul_transpose(&x);
        match &self.clip {
            Some((_, _, out_min, out_max)) => {
                mlx_rs::ops::clip(&out, (out_min, out_max)).expect("mlx op: clip out")
            }
            None => out,
        }
    }
}

// ─── 2D multidimensional RoPE (vision.py:103-158) ────────────────────────────

fn rotate_half(x: &Array) -> Array {
    // `[-x2, x1]` (vision.py:96-100).
    let last = *x.shape().last().expect("rank>=1");
    let half = last / 2;
    let x1 = x.index((.., .., .., ..half));
    let x2 = x.index((.., .., .., half..));
    let neg_x2 = x2.negative().expect("mlx op");
    mlx_rs::ops::concatenate_axis(&[&neg_x2, &x1], -1).expect("mlx op")
}

/// Apply multidimensional (2D) RoPE — mirrors `apply_multidimensional_rope`
/// (vision.py:103-158) for the `positions.ndim == 3` (2D) case.
///
/// Splits the head dim into `ndim=2` equal partitions and rotates **within
/// each partition independently** (one per spatial axis). This is the trap the
/// plan flags: a naive 1D RoPE reuse mixes the spatial axes and is wrong.
///
/// `inputs`: `[B, L, N, H]` (per-head, NOT yet transposed to [B,N,L,H]).
/// `positions`: `[B, L, 2]` int (the (x, y) grid coordinate per patch).
fn apply_multidimensional_rope(inputs: &Array, positions: &Array, base_frequency: f32) -> Array {
    let head_dim = *inputs.shape().last().expect("rank>=1");
    let ndim: i32 = 2;
    // channels_per_dim = 2 * (head_dim // (2*ndim)); half_per_dim = .. / 2.
    let channels_per_dim = 2 * (head_dim / (2 * ndim)); // 32 for head_dim=64
    let half_per_dim = channels_per_dim / 2; // 16

    let positions_f = positions.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op");

    let mut parts: Vec<Array> = Vec::with_capacity(ndim as usize);
    for d in 0..ndim {
        // x_part = inputs[..., d*cpd : (d+1)*cpd]
        let lo = d * channels_per_dim;
        let hi = (d + 1) * channels_per_dim;
        let x_part = inputs.index((.., .., .., lo..hi));

        // freq_exponents = (2/cpd) * arange(0, half_per_dim)
        // timescale = base ** freq_exponents
        let ar =
            mlx_rs::ops::arange::<_, f32>(0.0f32, half_per_dim as f32, 1.0f32).expect("mlx op");
        let scale = Array::from_f32(2.0 / channels_per_dim as f32);
        let freq_exponents = ar.multiply(&scale).expect("mlx op");
        let base = Array::from_f32(base_frequency);
        let timescale = mlx_rs::ops::power(&base, &freq_exponents).expect("mlx op");

        // sinusoid_inp = positions[..., d:d+1] / timescale  -> [B, L, half_per_dim]
        let pos_d = positions_f.index((.., .., d..(d + 1))); // [B, L, 1]
        let sinusoid = pos_d.divide(&timescale).expect("mlx op"); // broadcast [B,L,half]
        let cos_d = sinusoid.cos().expect("mlx op");
        let sin_d = sinusoid.sin().expect("mlx op");
        // duplicate: [B,L,half] -> [B,L,cpd]
        let cos_d = mlx_rs::ops::concatenate_axis(&[&cos_d, &cos_d], -1).expect("mlx op");
        let sin_d = mlx_rs::ops::concatenate_axis(&[&sin_d, &sin_d], -1).expect("mlx op");
        // expand axis=2 -> [B, L, 1, cpd]
        let cos_d = mlx_rs::ops::expand_dims(&cos_d, 2).expect("mlx op");
        let sin_d = mlx_rs::ops::expand_dims(&sin_d, 2).expect("mlx op");
        let cos_d = cos_d.as_dtype(inputs.dtype()).expect("mlx op");
        let sin_d = sin_d.as_dtype(inputs.dtype()).expect("mlx op");

        // y_part = x_part*cos + rotate_half(x_part)*sin  (within this partition)
        let term_a = x_part.multiply(&cos_d).expect("mlx op");
        let term_b = rotate_half(&x_part).multiply(&sin_d).expect("mlx op");
        parts.push(term_a.add(&term_b).expect("mlx op"));
    }
    let refs: Vec<&Array> = parts.iter().collect();
    mlx_rs::ops::concatenate_axis(&refs, -1).expect("mlx op")
}

/// `VisionRMSNorm` (vision.py:49-65): **float32** RMS norm with a learnable
/// scale, casting back to the input dtype. mlx-vlm explicitly upcasts the
/// variance reduction to f32 — the shared `RmsNorm::forward` keeps the input
/// dtype (bf16), which is a small but real precision divergence for the
/// vision tower's large-magnitude activations. Reproducing the f32 upcast here
/// tightens parity. `weight`: `[head_dim]`.
fn vision_rms_norm_f32(x: &Array, weight: &Array, eps: f32) -> Array {
    let in_dtype = x.dtype();
    let xf = x.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op");
    let var = xf
        .multiply(&xf)
        .expect("mlx op")
        .mean_axis(-1, true)
        .expect("mlx op");
    let eps = Array::from_f32(eps);
    let inv = var.add(&eps).expect("mlx op").rsqrt().expect("mlx op");
    let normed = xf.multiply(&inv).expect("mlx op");
    let wf = weight.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op");
    normed
        .multiply(&wf)
        .expect("mlx op")
        .as_dtype(in_dtype)
        .expect("mlx op")
}

/// `VisionRMSNormNoScale` (vision.py:68-81): f32 RMS norm without a learnable
/// scale (the v_norm).
fn vision_rms_norm_no_scale_f32(x: &Array, eps: f32) -> Array {
    let in_dtype = x.dtype();
    let xf = x.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op");
    let var = xf
        .multiply(&xf)
        .expect("mlx op")
        .mean_axis(-1, true)
        .expect("mlx op");
    let eps = Array::from_f32(eps);
    let inv = var.add(&eps).expect("mlx op").rsqrt().expect("mlx op");
    xf.multiply(&inv)
        .expect("mlx op")
        .as_dtype(in_dtype)
        .expect("mlx op")
}

// ─── Vision RMSNorm helpers ──────────────────────────────────────────────────
//
// q_norm/k_norm are VisionRMSNorm (learnable scale, vision.py:49-65) — reuse the
// shared `RmsNorm` (no offset). v_norm is VisionRMSNormNoScale (vision.py:68-81)
// — reuse `rms_norm_no_scale_fast`. The block's 4 norms are plain `RMSNorm`
// (vision.py:84-93) — `RmsNorm` (no offset), same as text.

// ─── Attention (vision.py:161-231) ───────────────────────────────────────────

pub struct VisionAttention {
    pub q_proj: ClippableLinear,
    pub k_proj: ClippableLinear,
    pub v_proj: ClippableLinear,
    pub o_proj: ClippableLinear,
    pub q_norm: RmsNorm, // VisionRMSNorm(head_dim)
    pub k_norm: RmsNorm,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl VisionAttention {
    fn new(cfg: &VisionConfig) -> Self {
        let h = cfg.hidden_size;
        let clip = cfg.use_clipped_linears;
        Self {
            q_proj: ClippableLinear::placeholder(cfg.num_attention_heads * cfg.head_dim, h, clip),
            k_proj: ClippableLinear::placeholder(cfg.num_key_value_heads * cfg.head_dim, h, clip),
            v_proj: ClippableLinear::placeholder(cfg.num_key_value_heads * cfg.head_dim, h, clip),
            o_proj: ClippableLinear::placeholder(h, cfg.num_attention_heads * cfg.head_dim, clip),
            q_norm: RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps),
            k_norm: RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
        }
    }

    /// `x`: [B, L, hidden]; `positions`: [B, L, 2]; `mask`: [B, 1, L, L]
    /// additive bias (0 / -1e4). Mirrors `VisionAttention.__call__`
    /// (vision.py:200-231): q/k/v proj → reshape → q/k/v norm → 2D RoPE →
    /// transpose to [B,H,L,D] → SDPA(scale=1.0, mask) → o_proj.
    fn forward(&self, x: &Array, positions: &Array, mask: &Array) -> Array {
        let sh = x.shape();
        let (b, l) = (sh[0], sh[1]);
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;

        let q = self
            .q_proj
            .forward(x)
            .reshape(&[b, l, nh, hd])
            .expect("mlx op");
        let k = self
            .k_proj
            .forward(x)
            .reshape(&[b, l, nkv, hd])
            .expect("mlx op");
        let v = self
            .v_proj
            .forward(x)
            .reshape(&[b, l, nkv, hd])
            .expect("mlx op");

        let q = vision_rms_norm_f32(&q, &self.q_norm.weight, self.eps);
        let k = vision_rms_norm_f32(&k, &self.k_norm.weight, self.eps);
        let v = vision_rms_norm_no_scale_f32(&v, self.eps);

        // 2D RoPE applied on [B, L, N, D] BEFORE the transpose (vision.py:214).
        let q = apply_multidimensional_rope(&q, positions, self.rope_theta);
        let k = apply_multidimensional_rope(&k, positions, self.rope_theta);

        // transpose to [B, H, L, D] for SDPA.
        let q = q.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let k = k.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let v = v.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");

        // Fused SDPA, scale=1.0 (vision.py:226 ensure_fused_sdpa). num_kv ==
        // num_heads here (12==12), no GQA expansion needed. The additive bias
        // `mask` zeroes valid pairs and applies -1e4 to padded ones.
        let sdpa_mask = mlx_rs::fast::ScaledDotProductAttentionMask::Array(mask);
        let out =
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, 1.0, sdpa_mask, None::<&Array>)
                .expect("mlx op: vision sdpa");

        // [B, H, L, D] -> [B, L, H*D]
        let out = out.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let out = out.reshape(&[b, l, nh * hd]).expect("mlx op");
        self.o_proj.forward(&out)
    }
}

// ─── MLP (vision.py:234-249) ─────────────────────────────────────────────────

pub struct VisionMlp {
    pub gate_proj: ClippableLinear,
    pub up_proj: ClippableLinear,
    pub down_proj: ClippableLinear,
}

impl VisionMlp {
    fn new(cfg: &VisionConfig) -> Self {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let clip = cfg.use_clipped_linears;
        Self {
            gate_proj: ClippableLinear::placeholder(i, h, clip),
            up_proj: ClippableLinear::placeholder(i, h, clip),
            down_proj: ClippableLinear::placeholder(h, i, clip),
        }
    }

    /// `down(gelu_approx(gate(x)) * up(x))` (vision.py:248).
    fn forward(&self, x: &Array) -> Array {
        let gate = self.gate_proj.forward(x);
        let gate = mlx_rs::nn::gelu_approximate(&gate).expect("mlx op");
        let up = self.up_proj.forward(x);
        let gated = gate.multiply(&up).expect("mlx op");
        self.down_proj.forward(&gated)
    }
}

// ─── Transformer block (vision.py:252-279) ───────────────────────────────────

pub struct VisionBlock {
    pub self_attn: VisionAttention,
    pub mlp: VisionMlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub pre_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm: RmsNorm,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig) -> Self {
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        Self {
            self_attn: VisionAttention::new(cfg),
            mlp: VisionMlp::new(cfg),
            input_layernorm: RmsNorm::new(h, eps),
            post_attention_layernorm: RmsNorm::new(h, eps),
            pre_feedforward_layernorm: RmsNorm::new(h, eps),
            post_feedforward_layernorm: RmsNorm::new(h, eps),
        }
    }

    /// Gemma sandwich-norm block (vision.py:268-279):
    /// `h = x + post_attn_norm(attn(input_norm(x)))`;
    /// `return h + post_ffw_norm(mlp(pre_ffw_norm(h)))`.
    fn forward(&self, x: &Array, positions: &Array, mask: &Array) -> Array {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, positions, mask);
        let attn_out = self.post_attention_layernorm.forward(&attn_out);
        let h = x.add(&attn_out).expect("mlx op");

        let normed_h = self.pre_feedforward_layernorm.forward(&h);
        let ffw = self.mlp.forward(&normed_h);
        let ffw = self.post_feedforward_layernorm.forward(&ffw);
        h.add(&ffw).expect("mlx op")
    }
}

// ─── Patch embedder (vision.py:282-332) ──────────────────────────────────────

pub struct VisionPatchEmbedder {
    /// `nn.Linear(3*patch², hidden)` — PLAIN (not clippable). Key
    /// `patch_embedder.input_proj.weight`.
    pub input_proj: Weight,
    /// `[2, position_embedding_size, hidden]` one-hot position table.
    pub position_embedding_table: Array,
    pub hidden_size: usize,
    pub patch_size: usize,
    pub position_embedding_size: usize,
}

impl VisionPatchEmbedder {
    fn new(cfg: &VisionConfig) -> Self {
        let h = cfg.hidden_size;
        Self {
            input_proj: Weight::plain(
                Array::zeros::<f32>(&[h as i32, (3 * cfg.patch_size * cfg.patch_size) as i32])
                    .expect("mlx op"),
            ),
            position_embedding_table: Array::zeros::<f32>(&[
                2,
                cfg.position_embedding_size as i32,
                h as i32,
            ])
            .expect("mlx op"),
            hidden_size: h,
            patch_size: cfg.patch_size,
            position_embedding_size: cfg.position_embedding_size,
        }
    }

    /// `_patchify` (vision.py:308-320): reshape `[B,C,H,W]` → patches, center to
    /// `2*(x-0.5)`, then `input_proj`.
    fn patchify(&self, pixel_values: &Array) -> Array {
        let sh = pixel_values.shape();
        let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let p = self.patch_size as i32;
        let ph = h / p;
        let pw = w / p;
        // [B,C,pH,p,pW,p] -> transpose [B,pH,pW,p,p,C] -> [B, pH*pW, C*p*p]
        let patches = pixel_values.reshape(&[b, c, ph, p, pw, p]).expect("mlx op");
        let patches = patches.transpose_axes(&[0, 2, 4, 3, 5, 1]).expect("mlx op");
        let patches = patches.reshape(&[b, ph * pw, c * p * p]).expect("mlx op");
        // 2*(x - 0.5)
        let half = Array::from_f32(0.5);
        let two = Array::from_f32(2.0);
        let patches = patches
            .subtract(&half)
            .expect("mlx op")
            .multiply(&two)
            .expect("mlx op");
        // cast to the input_proj weight dtype before matmul (vision.py:320).
        let patches = patches
            .as_dtype(self.input_proj.to_full().dtype())
            .expect("mlx op");
        self.input_proj.matmul_transpose(&patches)
    }

    /// `_position_embeddings` (vision.py:295-306): one-hot(positions) @ table,
    /// sum over the 2 axes, zero out padded patches.
    ///
    /// `patch_positions`: [B, L, 2] int; `padding_positions`: [B, L] bool.
    fn position_embeddings(&self, patch_positions: &Array, padding_positions: &Array) -> Array {
        let table = &self.position_embedding_table;
        let table_dtype = table.dtype();
        // one_hot(indices, pos_size): (expand_dims(idx,-1) == arange(pos_size))
        // -> [B, L, 2, pos_size]
        let oh = one_hot(patch_positions, self.position_embedding_size as i32);
        // transpose to [B, 2, L, pos_size], cast to table dtype.
        let oh = oh.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let oh = oh.as_dtype(table_dtype).expect("mlx op");
        // oh @ table: [B,2,L,pos_size] @ [2,pos_size,hidden] -> [B,2,L,hidden]
        // mlx matmul broadcasts the leading batch dims; table is [2,pos,hidden].
        let pe = oh.matmul(table).expect("mlx op");
        // sum over axis 1 (the 2 spatial axes) -> [B, L, hidden]
        let pe = pe.sum_axes(&[1], false).expect("mlx op");
        // zero out padded patches: where(expand_dims(padding,-1), 0, pe)
        let pad = mlx_rs::ops::expand_dims(padding_positions, -1).expect("mlx op");
        let zero = Array::from_f32(0.0).as_dtype(pe.dtype()).expect("mlx op");
        let zero = mlx_rs::ops::broadcast_to(&zero, pe.shape()).expect("mlx op");
        mlx_rs::ops::r#where(&pad, &zero, &pe).expect("mlx op")
    }

    /// `__call__` (vision.py:322-332): patchify + position embeddings.
    fn forward(
        &self,
        pixel_values: &Array,
        patch_positions: &Array,
        padding_positions: &Array,
    ) -> Array {
        let hidden = self.patchify(pixel_values);
        let pe = self.position_embeddings(patch_positions, padding_positions);
        let pe = pe.as_dtype(hidden.dtype()).expect("mlx op");
        hidden.add(&pe).expect("mlx op")
    }
}

/// `one_hot(indices, num_classes)` (vision.py:44-46):
/// `(expand_dims(indices,-1) == arange(num_classes)).astype(f32)`.
fn one_hot(indices: &Array, num_classes: i32) -> Array {
    let idx = indices.as_dtype(mlx_rs::Dtype::Int32).expect("mlx op");
    let idx = mlx_rs::ops::expand_dims(&idx, -1).expect("mlx op");
    let ar = mlx_rs::ops::arange::<_, i32>(0i32, num_classes, 1i32).expect("mlx op");
    let eq = idx.eq(&ar).expect("mlx op");
    eq.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op")
}

// ─── Pooler (vision.py:335-372) ──────────────────────────────────────────────

pub struct VisionPooler {
    pub hidden_size: usize,
    pub default_output_length: usize,
    pub root_hidden_size: f32,
}

impl VisionPooler {
    fn new(cfg: &VisionConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            default_output_length: cfg.default_output_length,
            root_hidden_size: (cfg.hidden_size as f32).sqrt(),
        }
    }

    /// `_avg_pool_by_positions` (vision.py:342-354) → (output, valid_mask).
    fn avg_pool_by_positions(
        &self,
        x: &Array,
        patch_positions: &Array,
        length: i32,
    ) -> (Array, Array) {
        let input_seq_len = x.shape()[1];
        // k = int((input_seq_len // length) ** 0.5); k_squared = k*k
        let k = (((input_seq_len / length) as f32).sqrt()) as i32;
        let k_squared = (k * k) as f32;

        // clamped = clip(patch_positions, 0, None)
        let zero = Array::from_int(0);
        let clamped = mlx_rs::ops::clip(patch_positions, (&zero, ())).expect("mlx op: clip pos");
        let clamped_f = clamped.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op");

        // max_x = max(clamped[...,0], -1, keepdims) + 1
        let col0 = clamped.index((.., .., 0)); // [B, L]
        let max_x = col0.max_axis(-1, true).expect("mlx op"); // [B, 1] int
        let one = Array::from_int(1);
        let max_x = max_x.add(&one).expect("mlx op");

        // kernel_idxs = floor(clamped/k); then [...,0] + (max_x//k)*[...,1]
        let kf = Array::from_f32(k as f32);
        let ki = clamped_f
            .divide(&kf)
            .expect("mlx op")
            .floor()
            .expect("mlx op");
        let ki = ki.as_dtype(mlx_rs::Dtype::Int32).expect("mlx op"); // [B,L,2]
        let ki0 = ki.index((.., .., 0)); // [B,L]
        let ki1 = ki.index((.., .., 1));
        let max_x_div_k = max_x
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("mlx op")
            .divide(&kf)
            .expect("mlx op")
            .floor()
            .expect("mlx op")
            .as_dtype(mlx_rs::Dtype::Int32)
            .expect("mlx op"); // [B,1]
        let kernel_idxs = ki0
            .add(max_x_div_k.multiply(&ki1).expect("mlx op"))
            .expect("mlx op"); // [B,L]

        // weights = one_hot(kernel_idxs, length) / k_squared   [B, L, length]
        let weights = one_hot(&kernel_idxs, length);
        let inv = Array::from_f32(1.0 / k_squared);
        let weights = weights.multiply(&inv).expect("mlx op");

        // output = einsum("bLl,bLd->bld", weights, x)
        let x_f = x.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op");
        let output = mlx_rs::ops::einsum("bLl,bLd->bld", [&weights, &x_f]).expect("mlx op");
        let output = output.as_dtype(x.dtype()).expect("mlx op");

        // mask = logical_not(all(weights == 0, axis=1))   [B, length]
        let wzero = Array::from_f32(0.0);
        let is_zero = weights.eq(&wzero).expect("mlx op");
        let all_zero = is_zero.all_axes(&[1], false).expect("mlx op"); // [B, length]
        let valid = all_zero.logical_not().expect("mlx op");
        (output, valid)
    }

    /// `__call__` (vision.py:356-372): zero padded tokens, pool to
    /// `default_output_length`, `* root_hidden_size`. Returns (pooled, mask).
    fn forward(
        &self,
        hidden_states: &Array,
        patch_positions: &Array,
        padding_positions: &Array,
    ) -> (Array, Array) {
        // Zero out padding tokens before pooling (vision.py:360-362).
        let pad = mlx_rs::ops::expand_dims(padding_positions, -1).expect("mlx op");
        let zero = Array::from_f32(0.0)
            .as_dtype(hidden_states.dtype())
            .expect("mlx op");
        let zero = mlx_rs::ops::broadcast_to(&zero, hidden_states.shape()).expect("mlx op");
        let hidden_states = mlx_rs::ops::r#where(&pad, &zero, hidden_states).expect("mlx op");

        let length = self.default_output_length as i32;
        let (pooled, mask) = if hidden_states.shape()[1] == length {
            (hidden_states.clone(), padding_positions.clone())
        } else {
            self.avg_pool_by_positions(&hidden_states, patch_positions, length)
        };
        let root = Array::from_f32(self.root_hidden_size)
            .as_dtype(pooled.dtype())
            .expect("mlx op");
        let pooled = pooled.multiply(&root).expect("mlx op");
        (pooled, mask)
    }
}

// ─── Top-level VisionTower (vision.py:392-538) ───────────────────────────────

pub struct VisionTower {
    pub patch_embedder: VisionPatchEmbedder,
    pub layers: Vec<VisionBlock>,
    pub pooler: VisionPooler,
    pub config: VisionConfig,
}

impl VisionTower {
    pub fn new(cfg: &VisionConfig) -> Self {
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| VisionBlock::new(cfg))
            .collect();
        Self {
            patch_embedder: VisionPatchEmbedder::new(cfg),
            layers,
            pooler: VisionPooler::new(cfg),
            config: cfg.clone(),
        }
    }

    /// Build patch positions + padding mask for a single fixed-size image —
    /// `_patch_positions_single` (vision.py:417-439). Returns
    /// `(positions [1, max_patches, 2] int32, padding [1, max_patches] bool,
    ///   num_real)`.
    fn patch_positions_single(&self, h: i32, w: i32) -> (Array, Array, usize) {
        let p = self.config.patch_size as i32;
        let ph = (h / p) as usize;
        let pw = (w / p) as usize;
        let num_patches = ph * pw;
        let max_patches = self.config.max_patches();

        // meshgrid(grid_x=arange(pW), grid_y=arange(pH), indexing="xy") then
        // stack([gx.flatten(), gy.flatten()], -1). For "xy" the row index is y,
        // column is x: position[r] = (x = r % pW, y = r // pW).
        let mut pos: Vec<i32> = Vec::with_capacity(num_patches * 2);
        for y in 0..ph {
            for x in 0..pw {
                pos.push(x as i32);
                pos.push(y as i32);
            }
        }
        // pad to max_patches with (-1, -1).
        let num_padding = max_patches.saturating_sub(num_patches);
        for _ in 0..num_padding {
            pos.push(-1);
            pos.push(-1);
        }
        let positions = Array::from_slice(&pos, &[1, max_patches as i32, 2]);

        // padding_mask: false for real, true for padded.
        let mut pad: Vec<bool> = vec![false; max_patches];
        for v in pad.iter_mut().skip(num_patches) {
            *v = true;
        }
        let pad_i: Vec<i32> = pad.iter().map(|&b| b as i32).collect();
        let padding = Array::from_slice(&pad_i, &[1, max_patches as i32])
            .as_dtype(mlx_rs::Dtype::Bool)
            .expect("mlx op");
        (positions, padding, num_patches.min(max_patches))
    }

    /// `__call__` (vision.py:441-538) for the single fixed-size image / batch=1
    /// `all_same_size` branch. Returns trimmed pooled hidden `[1, n_valid, 768]`.
    pub fn forward(&self, pixel_values: &Array) -> Array {
        let (_, _, trimmed) = self.forward_parts(pixel_values);
        trimmed
    }

    /// Same as [`Self::forward`] but also returns the intermediate tensors so
    /// the Stage-3 parity test can check the **pre-pool** encoder hidden state
    /// (`vision.py:517`) and the **post-pool** pooled state (`vision.py:519`)
    /// separately. Returns `(encoder_hidden_prepool, pooled_full, trimmed)`.
    pub fn forward_parts(&self, pixel_values: &Array) -> (Array, Array, Array) {
        let sh = pixel_values.shape();
        let (b, _c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        assert_eq!(b, 1, "VisionTower v1 supports batch=1");
        let max_patches = self.config.max_patches() as i32;

        let (positions, padding, num_real_usize) = self.patch_positions_single(h, w);
        let num_real = num_real_usize as i32;

        // patch_embedder over the REAL patches only (vision.py:468-472).
        let pos_real = positions.index((.., 0..num_real, ..));
        let pad_real = padding.index((.., 0..num_real));
        let mut inputs_embeds = self
            .patch_embedder
            .forward(pixel_values, &pos_real, &pad_real);

        // pad embeds up to max_patches with zeros (vision.py:474-479).
        let num_padding = max_patches - num_real;
        if num_padding > 0 {
            let hidden = self.config.hidden_size as i32;
            let pad_embeds = Array::zeros::<f32>(&[1, num_padding, hidden])
                .expect("mlx op")
                .as_dtype(inputs_embeds.dtype())
                .expect("mlx op");
            inputs_embeds =
                mlx_rs::ops::concatenate_axis(&[&inputs_embeds, &pad_embeds], 1).expect("mlx op");
        }

        // Bidirectional attn mask [B,1,L,L], -1e4 on padded pairs
        // (vision.py:506-515). valid = ~padding; attn = valid[:,None]*valid[:,:,None].
        let valid = padding.logical_not().expect("mlx op"); // [B, L] bool
        let valid_a = mlx_rs::ops::expand_dims(&valid, 1).expect("mlx op"); // [B,1,L]
        let valid_b = mlx_rs::ops::expand_dims(&valid, 2).expect("mlx op"); // [B,L,1]
        let valid_a = valid_a.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op");
        let valid_b = valid_b.as_dtype(mlx_rs::Dtype::Float32).expect("mlx op");
        let both = valid_a.multiply(&valid_b).expect("mlx op"); // [B,L,L] 1/0
        let both_bool = both.gt(Array::from_f32(0.5)).expect("mlx op");
        let zeros = Array::zeros::<f32>(both.shape())
            .expect("mlx op")
            .as_dtype(inputs_embeds.dtype())
            .expect("mlx op");
        let fill = Array::from_f32(-1e4)
            .as_dtype(inputs_embeds.dtype())
            .expect("mlx op");
        let fill = mlx_rs::ops::broadcast_to(&fill, both.shape()).expect("mlx op");
        let attn_mask = mlx_rs::ops::r#where(&both_bool, &zeros, &fill).expect("mlx op");
        let attn_mask = mlx_rs::ops::expand_dims(&attn_mask, 1).expect("mlx op"); // [B,1,L,L]

        // Run encoder over the (padded) full grid with tiled positions.
        let mut hidden_states = inputs_embeds;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &positions, &attn_mask);
        }

        // Pool (vision.py:519-521).
        let (pooled, pool_mask) = self.pooler.forward(&hidden_states, &positions, &padding);

        // Trim to the valid soft tokens (vision.py:523-533). pool_mask is
        // `valid` (true == keep) from avg_pool_by_positions; count per batch.
        // For batch=1: take the first n_valid rows of pooled. The pooler emits
        // valid tokens contiguously at the front for a single contiguous image.
        let n_valid = pool_mask
            .as_dtype(mlx_rs::Dtype::Int32)
            .expect("mlx op")
            .sum(None)
            .expect("mlx op")
            .item::<i32>();
        let trimmed = pooled.index((0..1, 0..n_valid, ..));

        // standardize branch (vision.py:535-536) — false for these bundles.
        (hidden_states, pooled, trimmed)
    }
}

// ─── Multimodal projector — MultimodalEmbedder (gemma4.py:22-34) ─────────────

pub struct EmbedVision {
    /// `nn.Linear(vision_hidden, text_hidden, bias=False)` — QUANTIZED 4-bit in
    /// this bundle. Key `embed_vision.embedding_projection`.
    pub embedding_projection: Weight,
    /// `RMSNormNoScale(vision_hidden)` — no learnable scale (gemma4.py:30).
    pub eps: f32,
}

impl EmbedVision {
    pub fn new(cfg: &VisionConfig, text_hidden: usize) -> Self {
        Self {
            embedding_projection: Weight::plain(
                Array::zeros::<f32>(&[text_hidden as i32, cfg.hidden_size as i32]).expect("mlx op"),
            ),
            eps: cfg.rms_norm_eps,
        }
    }

    /// `projection(pre_projection_norm(x))` (gemma4.py:32-34). NO
    /// `hidden**0.5` divide here — that scale lives in the pooler
    /// (`* root_hidden_size`); do not copy the gemma3 merge-time divide.
    pub fn forward(&self, x: &Array) -> Array {
        let normed = rms_norm_no_scale_fast(x, self.eps);
        self.embedding_projection.matmul_transpose(&normed)
    }
}

// ─── Combined vision model (tower + projector), held by the bundle ──────────

pub struct VisionModel {
    pub tower: VisionTower,
    pub projector: EmbedVision,
    /// `image_token_id` (config.py:130) — the decoder row id replaced by image
    /// features at merge time.
    pub image_token_id: u32,
    /// `<boi>` begin-of-image marker string (tokenizer_config `boi_token`,
    /// `<|image>` for the gemma-4 bundles). Opens the soft-token run.
    pub boi_token: String,
    /// `image_token` placeholder string (tokenizer_config `image_token`,
    /// `<|image|>` == `image_token_id`). Repeated `n_soft` times per image.
    pub image_token: String,
    /// `<eoi>` end-of-image marker string (tokenizer_config `eoi_token`,
    /// `<image|>`). Closes the soft-token run.
    pub eoi_token: String,
}

impl VisionModel {
    /// `embed_vision(vision_tower(pixels))` (gemma4.py:126-127) — the projected
    /// image features `[1, n_soft, text_hidden]`.
    pub fn encode_image(&self, pixel_values: &Array) -> Array {
        let feats = self.tower.forward(pixel_values);
        self.projector.forward(&feats)
    }

    /// The Gemma-4 image-placeholder expansion for one image:
    /// `<boi>` + `image_token` × `n_soft` + `<eoi>`. This is the exact string
    /// the HF/mlx-vlm processor substitutes for each image placeholder
    /// (`processing_gemma4.py:504-507`), where `n_soft` is the per-image soft
    /// token count (`Gemma4ImageProcessor::num_soft_tokens`). Tokenizing it
    /// yields exactly `n_soft` `image_token_id` rows — the scatter targets that
    /// `forward_with_image` fills with the pooled vision features.
    pub fn image_placeholder_expansion(&self, n_soft: usize) -> String {
        let mut s = String::with_capacity(
            self.boi_token.len() + self.image_token.len() * n_soft + self.eoi_token.len(),
        );
        s.push_str(&self.boi_token);
        for _ in 0..n_soft {
            s.push_str(&self.image_token);
        }
        s.push_str(&self.eoi_token);
        s
    }
}
