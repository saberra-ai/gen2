//! Gemma 4 E2B transformer model.
//!
//! Key architectural differences from Llama:
//! - Standard RMSNorm (no +1 weight offset)
//! - Per-head q/k/v norms in every attention layer
//! - GEGLU feed-forward network
//! - KV sharing: layers num_non_shared..N reuse KV from layers 0..num_non_shared
//! - Partial RoPE: full-attention layers apply rotary only to the first rope_dim head dims
//! - 4 norms per transformer block (input, post-attn, pre-ffn, post-ffn)
//! - Per-layer input embedding contribution
//! - Learnable per-layer scalar
//! - Final logit softcapping: tanh(x/cap)*cap

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use super::norm::RmsNorm;
use super::quantized::Weight;
use super::rope::RotaryEmbedding;
use super::{KvCache, ModelConfig};

// ─── RMSNorm without learnable weight (used for v_norm) ──────────────────────

struct RmsNormNoScale {
    eps: f32,
}

impl RmsNormNoScale {
    fn new(eps: f32) -> Self {
        Self { eps }
    }

    fn forward(&self, x: &Array) -> Array {
        let x_sq = x.multiply(x).expect("mlx op");
        let variance = x_sq.mean_axis(-1, true).expect("mlx op");
        let eps = Array::from_f32(self.eps);
        let var_eps = variance.add(&eps).expect("mlx op");
        let norm_factor = var_eps.rsqrt().expect("mlx op");
        x.multiply(&norm_factor).expect("mlx op")
    }
}

// ─── Partial RoPE note ────────────────────────────────────────────────────────
//
// Partial rotary embedding (rotating only the first `rotated_dim` elements of
// each head while passing the rest through) is now handled inside
// `RotaryEmbedding::with_freq_divisor`: it builds cos/sin tables sized for the
// full head_dim and zeroes out angles past `rotated_dim/2`, so `rope.forward`
// can be called directly on the full-width tensor. The previous split-then-
// concat helper rotated the wrong pairs (see rope.rs docstring).

// ─── Causal + sliding-window attention mask ───────────────────────────────────

/// Build an additive attention bias of shape [1, 1, query_len, kv_len].
///
/// 0.0 where attention is allowed; -inf where masked out.
/// Applies both causal constraint and optional sliding window.
fn build_causal_mask(query_len: usize, kv_len: usize, window: Option<usize>) -> Array {
    let q = query_len as i32;
    let k = kv_len as i32;
    let neg_inf = f32::NEG_INFINITY;
    let mut data = vec![0.0f32; (q * k) as usize];

    for qi in 0..q {
        let abs_q = k - q + qi; // absolute kv position of this query
        for ki in 0..k {
            let causal_ok = ki <= abs_q;
            let window_ok = window.map_or(true, |w| ki > abs_q - w as i32);
            if !causal_ok || !window_ok {
                data[(qi * k + ki) as usize] = neg_inf;
            }
        }
    }

    let raw = Array::from_slice(&data, &[q, k]);
    raw.reshape(&[1, 1, q, k]).expect("mlx op")
}

// ─── GQA head repetition ─────────────────────────────────────────────────────

fn repeat_kv_heads(x: &Array, repeats: usize) -> Array {
    if repeats == 1 {
        return x.clone();
    }
    let shape = x.shape();
    let (batch, n_kv, seq, hd) = (shape[0], shape[1], shape[2], shape[3]);
    let expanded = x.reshape(&[batch, n_kv, 1, seq, hd]).expect("mlx op");
    let refs: Vec<&Array> = (0..repeats).map(|_| &expanded).collect();
    mlx_rs::ops::concatenate_axis(&refs, 2)
        .expect("mlx op")
        .reshape(&[batch, n_kv * repeats as i32, seq, hd])
        .expect("mlx op")
}

// ─── Attention ────────────────────────────────────────────────────────────────

pub struct Gemma4Attention {
    pub q_proj: Weight,
    pub k_proj: Weight,
    /// `None` when `attention_k_eq_v` is true and this is a full-attention layer
    /// (large Gemma 4 variants — 31B). V is then a pre-norm clone of K.
    pub v_proj: Option<Weight>,
    pub o_proj: Weight,
    /// Per-head Q norm (learnable scale, size = head_dim).
    pub q_norm: RmsNorm,
    /// Per-head K norm (learnable scale, size = head_dim).
    pub k_norm: RmsNorm,
    /// Per-head V norm (no learnable scale).
    v_norm: RmsNormNoScale,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Number of head dimensions that receive rotary embeddings.
    /// Equals head_dim for sliding layers; head_dim/4 for full-attention layers.
    pub rope_dim: usize,
    pub is_sliding: bool,
    pub sliding_window: Option<usize>,
    /// When true, V is reused from K (before norm). Disables `v_proj`.
    pub use_k_eq_v: bool,
}

impl Gemma4Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        eps: f32,
        is_sliding: bool,
        sliding_window: Option<usize>,
        use_k_eq_v: bool,
    ) -> Self {
        let q_proj = Weight::plain(
            Array::zeros::<f32>(&[(num_heads * head_dim) as i32, hidden as i32]).expect("mlx op"),
        );
        let k_proj = Weight::plain(
            Array::zeros::<f32>(&[(num_kv_heads * head_dim) as i32, hidden as i32])
                .expect("mlx op"),
        );
        let v_proj = if use_k_eq_v {
            None
        } else {
            Some(Weight::plain(
                Array::zeros::<f32>(&[(num_kv_heads * head_dim) as i32, hidden as i32])
                    .expect("mlx op"),
            ))
        };
        let o_proj = Weight::plain(
            Array::zeros::<f32>(&[hidden as i32, (num_heads * head_dim) as i32]).expect("mlx op"),
        );
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RmsNorm::new(head_dim, eps),
            k_norm: RmsNorm::new(head_dim, eps),
            v_norm: RmsNormNoScale::new(eps),
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            is_sliding,
            sliding_window,
            use_k_eq_v,
        }
    }

    /// Forward pass.
    ///
    /// `shared_kv`: `Some((K, V))` for KV-shared layers, which skip k/v projection
    ///              and read directly from their non-shared partner's cache slot.
    ///              Cache is NOT updated when shared_kv is Some.
    pub fn forward(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        cache: &mut Option<(Array, Array)>,
        offset: usize,
        shared_kv: Option<&(Array, Array)>,
    ) -> Array {
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;

        let prof = super::profile::enabled();

        // Q: project → reshape → norm → partial RoPE → transpose to [B, H, S, D]
        let q = self.q_proj.matmul_transpose(x);
        let q = q.reshape(&[batch, seq, nh, hd]).expect("mlx op");
        if prof {
            super::profile::prof_eval("attn.qkv_proj", &[&q]);
        }
        let q = self.q_norm.forward(&q);
        let q = q.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let q = rope.forward(&q, offset);
        if prof {
            super::profile::prof_eval("attn.qknorm_rope", &[&q]);
        }

        // K, V: either compute fresh (non-shared) or borrow from shared partner's cache
        let (k, v) = if let Some((ck, cv)) = shared_kv {
            (ck.clone(), cv.clone())
        } else {
            let k = self.k_proj.matmul_transpose(x);
            // 31B `attention_k_eq_v`: V is K (pre-norm) instead of a separate projection.
            let v = match &self.v_proj {
                Some(v_w) => v_w.matmul_transpose(x),
                None => k.clone(),
            };

            let k = k.reshape(&[batch, seq, nkv, hd]).expect("mlx op");
            let v = v.reshape(&[batch, seq, nkv, hd]).expect("mlx op");
            let k = self.k_norm.forward(&k);
            let v = self.v_norm.forward(&v);

            let k = k.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
            let v = v.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
            let k = rope.forward(&k, offset);
            if prof {
                super::profile::prof_eval("attn.qkv_proj", &[&k, &v]);
            }

            // Append to KV cache along the sequence axis
            let (k, v) = if let Some((prev_k, prev_v)) = cache.take() {
                let k = mlx_rs::ops::concatenate_axis(&[&prev_k, &k], 2).expect("mlx op");
                let v = mlx_rs::ops::concatenate_axis(&[&prev_v, &v], 2).expect("mlx op");
                (k, v)
            } else {
                (k, v)
            };
            if prof {
                super::profile::prof_eval("attn.kv_concat", &[&k, &v]);
            }

            // Sliding-window truncation: drop oldest frames beyond the window
            let (k, v) = if let Some(w) = self.sliding_window {
                let kv_len = k.shape()[2] as usize;
                if kv_len > w {
                    let start = (kv_len - w) as i32;
                    let end = kv_len as i32;
                    let k = k.index((.., .., start..end, ..));
                    let v = v.index((.., .., start..end, ..));
                    (k, v)
                } else {
                    (k, v)
                }
            } else {
                (k, v)
            };

            *cache = Some((k.clone(), v.clone()));
            (k, v)
        };

        // Decode fast path: seq=1 → use fused SDPA Metal kernel (no mask needed
        // since a single query token can attend to all cached K/V positions).
        // This path handles GQA internally so we don't pre-tile k/v.
        // Prefill (seq>1) still uses manual Q@Kᵀ → softmax → @V with our
        // float causal+sliding mask, since the fused kernel's mask dtype
        // convention differs from ours and mixing them broke content when we
        // tried previously.
        let out = if seq == 1 {
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, 1.0, None, None::<&Array>)
                .expect("mlx op")
        } else {
            // GQA: expand KV heads to match the number of Q heads.
            let (k, v) = if self.num_kv_heads < self.num_heads {
                let reps = self.num_heads / self.num_kv_heads;
                (repeat_kv_heads(&k, reps), repeat_kv_heads(&v, reps))
            } else {
                (k, v)
            };

            let k_t = k.transpose_axes(&[0, 1, 3, 2]).expect("mlx op");
            let mut scores = q.matmul(&k_t).expect("mlx op");

            let kv_len = scores.shape()[3] as usize;
            let mask = build_causal_mask(seq as usize, kv_len, self.sliding_window);
            scores = scores.add(&mask).expect("mlx op");

            let attn_w = mlx_rs::ops::softmax_axes(&scores, &[-1], None).expect("mlx op");
            attn_w.matmul(&v).expect("mlx op")
        };
        if prof {
            super::profile::prof_eval("attn.sdpa", &[&out]);
        }

        // Merge heads: [B, H, S, D] → [B, S, H·D]
        let out = out.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let out = out.reshape(&[batch, seq, nh * hd]).expect("mlx op");

        let out = self.o_proj.matmul_transpose(&out);
        if prof {
            super::profile::prof_eval("attn.o_proj", &[&out]);
        }
        out
    }

    /// Fast forward (`PIO_MLX_FAST`) — bf16 activations + fused SDPA (native
    /// GQA, NO repeat_kv) + step-buffer KV cache. Mirrors `gemma4_text.py`
    /// `Attention.__call__` (~line 226): q_norm/k_norm/v_norm via fast RMSNorm,
    /// partial RoPE, `scaled_dot_product_attention(..., scale=1.0)` (:202, :262).
    ///
    /// `x` is expected to already be bf16. Returns bf16.
    /// Returns `(attn_output, (k_view, v_view))` — the second element is the
    /// filled KV view this layer produced, which the model threads to any
    /// KV-shared partner layer (mirrors mlx-lm `intermediates`).
    pub fn forward_fast(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        cache: &mut Option<(Array, Array)>,
        offset: usize,
        shared_kv: Option<&(Array, Array)>,
    ) -> (Array, (Array, Array)) {
        use super::gemma4_fast::{fast_sdpa, step_buffer_update};
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;

        // Q: project → reshape → q_norm (fast) → transpose → partial RoPE.
        let q = self.q_proj.matmul_transpose(x);
        let q = q.reshape(&[batch, seq, nh, hd]).expect("mlx op");
        let q = self.q_norm.forward_fast(&q);
        let q = q.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let q = rope.forward_fast(&q, offset);

        // K, V: fresh (non-shared) or borrowed from the shared partner's cache.
        let (k, v) = if let Some((ck, cv)) = shared_kv {
            (ck.clone(), cv.clone())
        } else {
            let k = self.k_proj.matmul_transpose(x);
            let v = match &self.v_proj {
                Some(v_w) => v_w.matmul_transpose(x),
                None => k.clone(),
            };
            let k = k.reshape(&[batch, seq, nkv, hd]).expect("mlx op");
            let v = v.reshape(&[batch, seq, nkv, hd]).expect("mlx op");
            let k = self.k_norm.forward_fast(&k);
            let v = super::norm::rms_norm_no_scale_fast(&v, self.v_norm.eps);
            let k = k.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
            let v = v.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
            let k = rope.forward_fast(&k, offset);

            // Step-buffer cache update + sliding-window truncation
            // (cache.py KVCache.update_and_fetch). `offset` is the true fill
            // before this chunk (session tracks it as cur_pos).
            step_buffer_update(cache, &k, &v, offset, self.sliding_window)
        };

        // Fused SDPA — native GQA (no repeat_kv), scale=1.0 (Gemma 4).
        let out = fast_sdpa(&q, &k, &v, seq as usize, self.sliding_window);

        let out = out.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let out = out.reshape(&[batch, seq, nh * hd]).expect("mlx op");
        let out = self.o_proj.matmul_transpose(&out);
        (out, (k, v))
    }
}

// ─── GEGLU feed-forward network ───────────────────────────────────────────────

pub struct Gemma4Ffn {
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
}

impl Gemma4Ffn {
    pub fn new(hidden: usize, intermediate: usize) -> Self {
        let zero = |rows: usize, cols: usize| {
            Weight::plain(Array::zeros::<f32>(&[rows as i32, cols as i32]).expect("mlx op"))
        };
        Self {
            gate_proj: zero(intermediate, hidden),
            up_proj: zero(intermediate, hidden),
            down_proj: zero(hidden, intermediate),
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        let gate = self.gate_proj.matmul_transpose(x);
        let gate = mlx_rs::nn::gelu_approximate(&gate).expect("mlx op");
        let up = self.up_proj.matmul_transpose(x);
        let gated = gate.multiply(&up).expect("mlx op");
        self.down_proj.matmul_transpose(&gated)
    }
}

// ─── Per-layer input embedding system ────────────────────────────────────────

pub struct Gemma4PerLayerInput {
    /// Linear(hidden → hidden_per_layer): gates the embedding contribution.
    pub per_layer_input_gate: Weight,
    /// Linear(hidden_per_layer → hidden): projects gated embedding back.
    pub per_layer_projection: Weight,
    pub post_per_layer_input_norm: RmsNorm,
}

impl Gemma4PerLayerInput {
    pub fn new(hidden: usize, hidden_per_layer: usize, eps: f32) -> Self {
        let zero = |rows: usize, cols: usize| {
            Weight::plain(Array::zeros::<f32>(&[rows as i32, cols as i32]).expect("mlx op"))
        };
        Self {
            per_layer_input_gate: zero(hidden_per_layer, hidden),
            per_layer_projection: zero(hidden, hidden_per_layer),
            post_per_layer_input_norm: RmsNorm::new(hidden, eps),
        }
    }

    /// Compute the per-layer embedding contribution to add to the residual stream.
    ///
    /// Matches the golden `DecoderLayer.__call__` per-layer-input block:
    /// `gate = gelu_approx(W_gate·h) * per_layer_input; out = norm(W_proj·gate)`.
    pub fn forward(&self, x: &Array, per_layer_input: &Array) -> Array {
        let gate = self.per_layer_input_gate.matmul_transpose(x);
        let gate = mlx_rs::nn::gelu_approximate(&gate).expect("mlx op");
        let gated = gate.multiply(per_layer_input).expect("mlx op");
        let projected = self.per_layer_projection.matmul_transpose(&gated);
        self.post_per_layer_input_norm.forward(&projected)
    }
}

// ─── Transformer block ────────────────────────────────────────────────────────

pub struct Gemma4TransformerBlock {
    pub attention: Gemma4Attention,
    pub ffn: Gemma4Ffn,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub pre_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm: RmsNorm,
    pub per_layer_input: Gemma4PerLayerInput,
    /// Learnable scalar multiplied at the end of each block (shape [1]).
    pub layer_scalar: Array,

    // ── MoE (Gemma 4 26B only) ─────────────────────────────────────────────
    /// `Some` when `enable_moe_block=true`. Block runs dense MLP + sparse
    /// expert branch in parallel and sums the outputs.
    pub moe: Option<Gemma4MoeBlock>,
}

pub struct Gemma4MoeBlock {
    pub router: super::moe::Router,
    pub experts: super::moe::Experts,
    pub pre_feedforward_layernorm_2: RmsNorm,
    pub post_feedforward_layernorm_1: RmsNorm,
    pub post_feedforward_layernorm_2: RmsNorm,
}

impl Gemma4TransformerBlock {
    pub fn new(config: &ModelConfig, is_sliding: bool) -> Self {
        let head_dim = if is_sliding {
            config.head_dim.unwrap_or(256)
        } else {
            config.global_head_dim.unwrap_or(512)
        };
        let rope_dim = if is_sliding {
            head_dim
        } else {
            ((head_dim as f32) * config.global_partial_rotary_factor.unwrap_or(0.25)).round()
                as usize
        };
        let hidden = config.hidden_size;
        // PLE is opt-in. Treat None == Some(0) == "no PLE" — both pre-allocate
        // zero-sized placeholders that the loader leaves alone for non-PLE
        // variants (12B / 26B / 31B). E2B / E4B keep the real dim from config.
        let hpl = config.hidden_size_per_layer_input.unwrap_or(0);
        let eps = config.rms_norm_eps;
        let sw = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        // 31B `attention_k_eq_v`: only full-attention layers; sliding always has v_proj.
        let use_k_eq_v = !is_sliding && config.attention_k_eq_v.unwrap_or(false);
        let num_kv_heads = if use_k_eq_v {
            config
                .num_global_key_value_heads
                .unwrap_or(config.num_key_value_heads)
        } else {
            config.num_key_value_heads
        };

        Self {
            attention: Gemma4Attention::new(
                hidden,
                config.num_attention_heads,
                num_kv_heads,
                head_dim,
                rope_dim,
                eps,
                is_sliding,
                sw,
                use_k_eq_v,
            ),
            ffn: Gemma4Ffn::new(hidden, config.intermediate_size),
            input_layernorm: RmsNorm::new(hidden, eps),
            post_attention_layernorm: RmsNorm::new(hidden, eps),
            pre_feedforward_layernorm: RmsNorm::new(hidden, eps),
            post_feedforward_layernorm: RmsNorm::new(hidden, eps),
            per_layer_input: Gemma4PerLayerInput::new(hidden, hpl, eps),
            layer_scalar: Array::ones::<f32>(&[1]).expect("mlx op"),
            moe: if config.enable_moe_block.unwrap_or(false) {
                let n_exp = config.num_experts.unwrap_or(0);
                let top_k = config.top_k_experts.unwrap_or(2);
                let moe_inter = config
                    .moe_intermediate_size
                    .unwrap_or(config.intermediate_size);
                Some(Gemma4MoeBlock {
                    router: super::moe::Router::new(hidden, n_exp, top_k, eps),
                    experts: super::moe::Experts::new(hidden, moe_inter, n_exp),
                    pre_feedforward_layernorm_2: RmsNorm::new(hidden, eps),
                    post_feedforward_layernorm_1: RmsNorm::new(hidden, eps),
                    post_feedforward_layernorm_2: RmsNorm::new(hidden, eps),
                })
            } else {
                None
            },
        }
    }

    pub fn forward(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        cache: &mut Option<(Array, Array)>,
        offset: usize,
        shared_kv: Option<&(Array, Array)>,
        per_layer_input: Option<&Array>,
    ) -> Array {
        // Pre-norm → attention → post-attn-norm → residual
        let h = self.input_layernorm.forward(x);
        let h = self.attention.forward(&h, rope, cache, offset, shared_kv);
        let h = self.post_attention_layernorm.forward(&h);
        let x = x.add(&h).expect("mlx op");

        // Feed-forward: dense path, plus MoE branch (summed) when enabled.
        let prof = super::profile::enabled();
        let h = if let Some(moe) = &self.moe {
            let h1 = self.pre_feedforward_layernorm.forward(&x);
            let h1 = self.ffn.forward(&h1);
            let h1 = moe.post_feedforward_layernorm_1.forward(&h1);
            if prof {
                super::profile::prof_eval("ffn.dense", &[&h1]);
            }
            let (idx, w) = moe.router.forward(&x);
            if prof {
                super::profile::prof_eval("moe.router", &[&idx, &w]);
            }
            let h2 = moe.pre_feedforward_layernorm_2.forward(&x);
            let h2 = moe.experts.forward(&h2, &idx, &w);
            let h2 = moe.post_feedforward_layernorm_2.forward(&h2);
            if prof {
                super::profile::prof_eval("moe.experts", &[&h2]);
            }
            h1.add(&h2).expect("mlx op")
        } else {
            let h = self.pre_feedforward_layernorm.forward(&x);
            self.ffn.forward(&h)
        };
        let h = self.post_feedforward_layernorm.forward(&h);
        let x = x.add(&h).expect("mlx op");

        // Per-layer input contribution → residual (skipped on non-PLE variants).
        let x = match per_layer_input {
            Some(pl) => {
                let pli = self.per_layer_input.forward(&x, pl);
                x.add(&pli).expect("mlx op")
            }
            None => x,
        };

        // Layer scalar (broadcast over [batch, seq, hidden])
        x.multiply(&self.layer_scalar).expect("mlx op")
    }

    /// Fast (`PIO_MLX_FAST`) block forward — same structure as `forward` but
    /// bf16 throughout with fast RMSNorms + fused-SDPA attention. Mirrors
    /// `DecoderLayer.__call__` (`gemma4_text.py:324`). `x` is bf16; returns bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fast(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        cache: &mut Option<(Array, Array)>,
        offset: usize,
        shared_kv: Option<&(Array, Array)>,
        per_layer_input: Option<&Array>,
    ) -> (Array, (Array, Array)) {
        // Pre-norm → attention → post-attn-norm → residual
        let h = self.input_layernorm.forward_fast(x);
        let (h, kv_view) = self
            .attention
            .forward_fast(&h, rope, cache, offset, shared_kv);
        let h = self.post_attention_layernorm.forward_fast(&h);
        let x = x.add(&h).expect("mlx op");

        // Feed-forward: dense path, plus MoE branch (summed) when enabled.
        let h = if let Some(moe) = &self.moe {
            let h1 = self.pre_feedforward_layernorm.forward_fast(&x);
            let h1 = self.ffn.forward(&h1);
            let h1 = moe.post_feedforward_layernorm_1.forward_fast(&h1);
            let (idx, w) = moe.router.forward(&x);
            let h2 = moe.pre_feedforward_layernorm_2.forward_fast(&x);
            let h2 = moe.experts.forward(&h2, &idx, &w);
            let h2 = moe.post_feedforward_layernorm_2.forward_fast(&h2);
            h1.add(&h2).expect("mlx op")
        } else {
            let h = self.pre_feedforward_layernorm.forward_fast(&x);
            self.ffn.forward(&h)
        };
        let h = self.post_feedforward_layernorm.forward_fast(&h);
        // Keep the residual stream in bf16: the dense FFN / MoE experts may
        // dequantize to f16 and promote bf16×f16 matmuls to f32 (MLX promotion).
        // Re-cast so the next block sees bf16 (mlx-lm runs the whole trunk bf16).
        let h = super::gemma4_fast::to_bf16(&h);
        let x = x.add(&h).expect("mlx op");

        // Per-layer input contribution (skipped on non-PLE variants — 26B).
        let x = match per_layer_input {
            Some(pl) => {
                let pli = self.per_layer_input.forward(&x, pl);
                x.add(&pli).expect("mlx op")
            }
            None => x,
        };

        let x = x.multiply(&self.layer_scalar).expect("mlx op");
        (x, kv_view)
    }
}

// ─── Full model ───────────────────────────────────────────────────────────────

pub struct Gemma4Model {
    pub embed_tokens: Weight,
    /// Full per-layer embedding table. Stored as a single (possibly quantized)
    /// weight; rows are gathered per-token and dequantized at forward time
    /// via `Weight::embedding_lookup` — never materialized in full.
    pub embed_tokens_per_layer: Weight,
    /// Linear(hidden → num_layers × hpl). Feeds the residual stream back into
    /// the per-layer input before it's mixed with `embed_tokens_per_layer`.
    pub per_layer_model_projection: Weight,
    /// RMSNorm applied to the projected per-layer input (over hpl).
    pub per_layer_projection_norm: RmsNorm,
    pub layers: Vec<Gemma4TransformerBlock>,
    pub norm: RmsNorm,
    /// RoPE for sliding-attention layers: full head_dim (256), theta=10000.
    pub local_rope: RotaryEmbedding,
    /// RoPE for full-attention layers: rope_dim only (128), theta=1 000 000.
    pub global_rope: RotaryEmbedding,
    embed_scale: f32,
    /// sqrt(hpl) — scales the per-layer embedding lookup before mix.
    embed_tokens_per_layer_scale: f32,
    /// 1/sqrt(hidden) — scales the per-layer model projection before mix.
    per_layer_projection_scale: f32,
    /// 1/sqrt(2) — applied to the sum of projection + embedding contributions.
    per_layer_input_scale: f32,
    /// Number of layers that own their own KV cache slot (= num_hidden_layers − num_kv_shared_layers).
    pub num_non_shared: usize,
    /// For each layer i, the cache slot to read/write from. Non-shared layers
    /// map to their own slot; shared layers map to the *last* non-shared layer
    /// of the same type (sliding/full) — matches `previous_kvs` in mlx-lm.
    pub cache_slot: Vec<usize>,
    final_logit_softcapping: Option<f32>,
    hidden_per_layer_input: usize,
    /// `PIO_MLX_FAST` flag, read **once** at construction (mirrors the
    /// `PIO_MLX_DENOISING_STEPS` env convention in session.rs). When true,
    /// `forward` / `forward_all` route through the bf16 + fused-SDPA +
    /// step-buffer fast path; when false the default f32 path is byte-identical
    /// to before. A single binary supports both, selected at runtime.
    pub fast: bool,
}

/// Read the `PIO_MLX_FAST` runtime flag. `"1"` / `"on"` (case-insensitive)
/// enable the fast path; unset / `"0"` / anything else keep the default path.
pub fn fast_flag_enabled() -> bool {
    std::env::var("PIO_MLX_FAST")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "on" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

impl Gemma4Model {
    pub fn new(config: &ModelConfig) -> Self {
        let vocab = config.vocab_size;
        let hidden = config.hidden_size;
        let n = config.num_hidden_layers;
        let n_shared = config.num_kv_shared_layers.unwrap_or(0);
        let num_non_shared = n.saturating_sub(n_shared);
        // PLE is opt-in. Treat None == Some(0) == "no PLE" — both pre-allocate
        // zero-sized placeholders that the loader leaves alone for non-PLE
        // variants (12B / 26B / 31B). E2B / E4B keep the real dim from config.
        let hpl = config.hidden_size_per_layer_input.unwrap_or(0);
        let eps = config.rms_norm_eps;
        let embed_scale = (hidden as f32).sqrt();

        let layer_types = build_layer_types(config);
        let layers = (0..n)
            .map(|i| {
                let is_sliding = layer_types[i] != "full_attention";
                Gemma4TransformerBlock::new(config, is_sliding)
            })
            .collect();

        let sliding_head_dim = config.head_dim.unwrap_or(256);
        let global_head_dim = config.global_head_dim.unwrap_or(512);
        let global_rope_dim = ((global_head_dim as f32)
            * config.global_partial_rotary_factor.unwrap_or(0.25))
        .round() as usize;
        let local_theta = config.rope_local_base_freq.unwrap_or(10_000.0);
        let global_theta = config.rope_theta;
        let max_seq = config.max_position_embeddings;

        // KV-sharing slot map: shared layers map to the *last* non-shared
        // layer of the same type. Mirrors `previous_kvs` construction in
        // `Gemma4TextModel.__init__`.
        let mut cache_slot = vec![0usize; n];
        let mut last_by_type: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for i in 0..num_non_shared {
            cache_slot[i] = i;
            let t = if layer_types[i] == "full_attention" {
                "full"
            } else {
                "sliding"
            };
            last_by_type.insert(t, i);
        }
        for i in num_non_shared..n {
            let t = if layer_types[i] == "full_attention" {
                "full"
            } else {
                "sliding"
            };
            cache_slot[i] = *last_by_type.get(t).unwrap_or(&0);
        }

        Self {
            embed_tokens: Weight::plain(
                Array::zeros::<f32>(&[vocab as i32, hidden as i32]).expect("mlx op"),
            ),
            embed_tokens_per_layer: Weight::plain(
                Array::zeros::<f32>(&[vocab as i32, (hpl * n) as i32]).expect("mlx op"),
            ),
            per_layer_model_projection: Weight::plain(
                Array::zeros::<f32>(&[(hpl * n) as i32, hidden as i32]).expect("mlx op"),
            ),
            per_layer_projection_norm: RmsNorm::new(hpl, eps),
            layers,
            norm: RmsNorm::new(hidden, eps),
            // Sliding-attn: standard RoPE (rotated_dim == head_dim).
            local_rope: RotaryEmbedding::new(sliding_head_dim, max_seq, local_theta),
            // Full-attn: ProportionalRoPE — rotates only the first
            // global_rope_dim elements but uses the FULL global_head_dim as
            // the frequency divisor. This is what Gemma 4's
            // `rope_type: "proportional"` actually does in mlx-lm.
            global_rope: RotaryEmbedding::with_freq_divisor(
                global_rope_dim,
                global_head_dim,
                max_seq,
                global_theta,
            ),
            embed_scale,
            embed_tokens_per_layer_scale: (hpl as f32).sqrt(),
            per_layer_projection_scale: 1.0 / (hidden as f32).sqrt(),
            per_layer_input_scale: 1.0 / 2.0_f32.sqrt(),
            num_non_shared,
            cache_slot,
            final_logit_softcapping: config.final_logit_softcapping,
            hidden_per_layer_input: hpl,
            fast: fast_flag_enabled(),
        }
    }

    /// Forward pass. Returns logits for the last token: shape [1, 1, vocab_size].
    ///
    /// `offset` is the **true** absolute token position of the first element
    /// in `tokens` — caller tracks it in session state. Do NOT infer it from
    /// `cache[0].shape[2]`: layer 0 is a sliding-attention layer whose cache
    /// length is capped at `sliding_window`, so once the conversation passes
    /// the window boundary the inferred offset stalls while full-attention
    /// layers still need the true position for RoPE. That bug caused the
    /// post-turn-4 collapse on 26B (sliding_window=512) and post-turn-8 on
    /// 31B (sliding_window=1024).
    pub fn forward(&self, tokens: &[u32], offset: usize, cache: &mut KvCache) -> Array {
        let all_logits = self.forward_all(tokens, offset, cache);
        let s = tokens.len() as i32;
        all_logits.index((0..1, (s - 1)..s, ..))
    }

    /// Forward pass returning logits for EVERY position: shape `[1, seq_len, vocab_size]`.
    ///
    /// Used by speculative decoding (n-gram drafts in the puller): samples one
    /// token per position in a single batched forward pass, so accepted drafts
    /// amortize the decode cost.
    pub fn forward_all(&self, tokens: &[u32], offset: usize, cache: &mut KvCache) -> Array {
        if self.fast {
            return self.forward_all_fast(tokens, offset, cache);
        }
        let seq_len = tokens.len();

        let idx: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_indices = Array::from_slice(&idx, &[seq_len as i32]);

        // Main embedding: gather + dequantize only the rows we need.
        let x0 = self.embed_tokens.embedding_lookup(&token_indices);
        let scale = Array::from_f32(self.embed_scale);
        let x0 = x0.multiply(&scale).expect("mlx op");
        let hidden = self.layers[0].input_layernorm.weight.shape()[0];
        let mut x = x0.reshape(&[1, seq_len as i32, hidden]).expect("mlx op");
        super::profile::prof_eval("embedding", &[&x]);

        // ── Per-layer inputs (PLE — Gemma 3n / Gemma 4 E-series only) ──────
        // Mirrors `if self.hidden_size_per_layer_input:` in the Python golden
        // impl. 12B / 26B / 31B ship `hidden_size_per_layer_input = 0` and
        // skip this entire pipeline (no `embed_tokens_per_layer`,
        // `per_layer_model_projection`, etc. in their checkpoints).
        let n_layers = self.layers.len() as i32;
        let hpl = self.hidden_per_layer_input as i32;
        let per_layer: Option<Array> = if hpl > 0 {
            // (a) embed_tokens_per_layer(tokens) * sqrt(hpl) → [1, seq, n_layers, hpl]
            let per_layer_embed = self.embed_tokens_per_layer.embedding_lookup(&token_indices);
            let ptl_scale = Array::from_f32(self.embed_tokens_per_layer_scale);
            let per_layer_embed = per_layer_embed.multiply(&ptl_scale).expect("mlx op");
            let per_layer_embed = per_layer_embed
                .reshape(&[1, seq_len as i32, n_layers, hpl])
                .expect("mlx op");

            // (b) per_layer_model_projection(h) * (1/sqrt(hidden)) → reshape, then RMSNorm
            let per_layer_proj = self.per_layer_model_projection.matmul_transpose(&x);
            let proj_scale = Array::from_f32(self.per_layer_projection_scale);
            let per_layer_proj = per_layer_proj.multiply(&proj_scale).expect("mlx op");
            let per_layer_proj = per_layer_proj
                .reshape(&[1, seq_len as i32, n_layers, hpl])
                .expect("mlx op");
            let per_layer_proj = self.per_layer_projection_norm.forward(&per_layer_proj);

            // (c) (proj + embed) * (1/sqrt(2))
            let mixed = per_layer_proj.add(&per_layer_embed).expect("mlx op");
            let input_scale = Array::from_f32(self.per_layer_input_scale);
            let pl = mixed.multiply(&input_scale).expect("mlx op");
            super::profile::prof_eval("ple", &[&pl]);
            Some(pl)
        } else {
            None
        };

        let n = self.num_non_shared;

        for (i, layer) in self.layers.iter().enumerate() {
            let rope = if layer.attention.is_sliding {
                &self.local_rope
            } else {
                &self.global_rope
            };

            let cache_idx = self.cache_slot[i];
            let is_shared = i >= n;

            let shared_kv: Option<(Array, Array)> = if is_shared {
                cache[cache_idx]
                    .as_ref()
                    .map(|(k, v)| (k.clone(), v.clone()))
            } else {
                None
            };

            // Per-layer input slice for layer i: [1, seq, hpl]. None when no PLE.
            let per_layer_i = per_layer.as_ref().map(|pl| {
                let li = i as i32;
                pl.index((.., .., li..li + 1, ..))
                    .reshape(&[1, seq_len as i32, hpl])
                    .expect("mlx op")
            });

            x = layer.forward(
                &x,
                rope,
                &mut cache[cache_idx],
                offset,
                shared_kv.as_ref(),
                per_layer_i.as_ref(),
            );
        }

        // Final norm
        x = self.norm.forward(&x);
        super::profile::prof_eval("final_norm", &[&x]);

        // LM head (weights tied to embed_tokens)
        let logits = self.embed_tokens.matmul_transpose(&x);

        // Final logit softcapping: tanh(x / cap) * cap
        let out = if let Some(cap) = self.final_logit_softcapping {
            let cap_arr = Array::from_f32(cap);
            let scaled = logits.divide(&cap_arr).expect("mlx op");
            let tanhed = mlx_rs::ops::tanh(&scaled).expect("mlx op");
            tanhed.multiply(&cap_arr).expect("mlx op")
        } else {
            logits
        };
        super::profile::prof_eval("lm_head+softcap", &[&out]);
        out
    }

    /// Fast (`PIO_MLX_FAST`) forward returning logits for every position.
    /// Mirrors `Gemma4TextModel.__call__` + `Model.__call__`
    /// (`gemma4_text.py:508` / `:580`) in bf16:
    ///   - embedding × embed_scale, cast to **bf16** (`:518-519`),
    ///   - PLE pipeline in bf16 when present (26B has none),
    ///   - each block via `forward_fast` (fused SDPA + step buffer),
    ///   - final RMSNorm (fast) + tied lm_head + logit softcapping (`:597`).
    ///
    /// The KV cache slots hold bf16 step buffers (see `gemma4_fast`). `offset`
    /// is the true fill before this chunk — the session tracks it as `cur_pos`.
    pub fn forward_all_fast(&self, tokens: &[u32], offset: usize, cache: &mut KvCache) -> Array {
        let seq_len = tokens.len();
        let idx: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_indices = Array::from_slice(&idx, &[seq_len as i32]);
        self.forward_all_fast_from_ids(&token_indices, seq_len, offset, cache)
    }

    /// Fast (`PIO_MLX_FAST`) forward whose input is a **lazy** `[seq]` int32
    /// token-id `Array` rather than a host `&[u32]`. This is the Stage-B
    /// pipelining entry: the next token sampled on-GPU (lazy) is fed straight
    /// in here as the embedding-gather index, so the decode chain never syncs
    /// to host between steps — mirroring mlx-lm's `model(y[None])` where `y` is
    /// the lazy argmax token (`generate.py:459`).
    ///
    /// `seq_len` must equal the token array's length (caller-known; for the
    /// per-token decode step it is `1`). Returns logits for every position,
    /// `[1, seq_len, vocab]`, same as [`Self::forward_all_fast`].
    pub fn forward_all_fast_from_ids(
        &self,
        token_indices: &Array,
        seq_len: usize,
        offset: usize,
        cache: &mut KvCache,
    ) -> Array {
        // Embedding × sqrt(hidden), then cast to bf16 for the whole trunk.
        let x0 = self.embed_tokens.embedding_lookup(token_indices);
        let scale = Array::from_f32(self.embed_scale);
        let x0 = x0.multiply(&scale).expect("mlx op");
        let hidden = self.layers[0].input_layernorm.weight.shape()[0];
        let x0 = x0.reshape(&[1, seq_len as i32, hidden]).expect("mlx op");
        let mut x = super::gemma4_fast::to_bf16(&x0);

        // ── Per-layer inputs (PLE — E-series only; 26B skips this) ──────
        let n_layers = self.layers.len() as i32;
        let hpl = self.hidden_per_layer_input as i32;
        let per_layer: Option<Array> = if hpl > 0 {
            let per_layer_embed = self.embed_tokens_per_layer.embedding_lookup(token_indices);
            let ptl_scale = Array::from_f32(self.embed_tokens_per_layer_scale);
            let per_layer_embed = per_layer_embed.multiply(&ptl_scale).expect("mlx op");
            let per_layer_embed = super::gemma4_fast::to_bf16(&per_layer_embed);
            let per_layer_embed = per_layer_embed
                .reshape(&[1, seq_len as i32, n_layers, hpl])
                .expect("mlx op");
            let per_layer_proj = self.per_layer_model_projection.matmul_transpose(&x);
            let proj_scale = Array::from_f32(self.per_layer_projection_scale);
            let per_layer_proj = per_layer_proj.multiply(&proj_scale).expect("mlx op");
            let per_layer_proj = per_layer_proj
                .reshape(&[1, seq_len as i32, n_layers, hpl])
                .expect("mlx op");
            let per_layer_proj = self.per_layer_projection_norm.forward_fast(&per_layer_proj);
            let mixed = per_layer_proj.add(&per_layer_embed).expect("mlx op");
            let input_scale = Array::from_f32(self.per_layer_input_scale);
            Some(super::gemma4_fast::to_bf16(
                &mixed.multiply(&input_scale).expect("mlx op"),
            ))
        } else {
            None
        };

        let n = self.num_non_shared;
        // `intermediates[slot]` holds the (keys, values) VIEW returned by the
        // owning non-shared layer this pass — exactly mlx-lm's `intermediates`
        // (`gemma4_text.py:543`). KV-shared layers read THIS view, NOT the raw
        // cache slot: the fast step-buffer slot is over-allocated with zero
        // padding, so reading it raw would make shared layers attend over zeros
        // (the post-turn-2 collapse). The default path stores the exact filled
        // prefix so it can read the slot directly; the fast path must route the
        // view through here instead.
        let mut intermediates: Vec<Option<(Array, Array)>> = vec![None; self.layers.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            let rope = if layer.attention.is_sliding {
                &self.local_rope
            } else {
                &self.global_rope
            };
            let cache_idx = self.cache_slot[i];
            let is_shared = i >= n;
            let shared_kv: Option<(Array, Array)> = if is_shared {
                intermediates[cache_idx].clone()
            } else {
                None
            };
            let per_layer_i = per_layer.as_ref().map(|pl| {
                let li = i as i32;
                pl.index((.., .., li..li + 1, ..))
                    .reshape(&[1, seq_len as i32, hpl])
                    .expect("mlx op")
            });
            let (x_next, kv_view) = layer.forward_fast(
                &x,
                rope,
                &mut cache[cache_idx],
                offset,
                shared_kv.as_ref(),
                per_layer_i.as_ref(),
            );
            x = x_next;
            intermediates[i] = Some(kv_view);
        }

        // Final norm (fast) → tied lm_head → logit softcapping.
        x = self.norm.forward_fast(&x);
        let logits = self.embed_tokens.matmul_transpose(&x);
        if let Some(cap) = self.final_logit_softcapping {
            let cap_arr = Array::from_f32(cap);
            let scaled = logits.divide(&cap_arr).expect("mlx op");
            let tanhed = mlx_rs::ops::tanh(&scaled).expect("mlx op");
            tanhed.multiply(&cap_arr).expect("mlx op")
        } else {
            logits
        }
    }

    /// Forward pass that ALSO stashes the post-block hidden state for
    /// each layer id in `aux_layer_ids`. Returns `(logits, aux_states)`
    /// where `aux_states[k]` is `[1, seq_len, hidden_size]` — the hidden
    /// state immediately after layer `aux_layer_ids[k]` (pre-final-norm).
    ///
    /// Used by EAGLE-3 speculative decoding: the draft model takes the
    /// concatenation of these aux states as its feature input.
    pub fn forward_all_with_aux(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: &mut KvCache,
        aux_layer_ids: &[usize],
    ) -> (Array, Vec<Array>) {
        let seq_len = tokens.len();
        let idx: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_indices = Array::from_slice(&idx, &[seq_len as i32]);

        let x0 = self.embed_tokens.embedding_lookup(&token_indices);
        let scale = Array::from_f32(self.embed_scale);
        let x0 = x0.multiply(&scale).expect("mlx op");
        let hidden = self.layers[0].input_layernorm.weight.shape()[0];
        let mut x = x0.reshape(&[1, seq_len as i32, hidden]).expect("mlx op");

        let n_layers = self.layers.len() as i32;
        let hpl = self.hidden_per_layer_input as i32;
        let per_layer: Option<Array> = if hpl > 0 {
            let per_layer_embed = self.embed_tokens_per_layer.embedding_lookup(&token_indices);
            let ptl_scale = Array::from_f32(self.embed_tokens_per_layer_scale);
            let per_layer_embed = per_layer_embed.multiply(&ptl_scale).expect("mlx op");
            let per_layer_embed = per_layer_embed
                .reshape(&[1, seq_len as i32, n_layers, hpl])
                .expect("mlx op");
            let per_layer_proj = self.per_layer_model_projection.matmul_transpose(&x);
            let proj_scale = Array::from_f32(self.per_layer_projection_scale);
            let per_layer_proj = per_layer_proj.multiply(&proj_scale).expect("mlx op");
            let per_layer_proj = per_layer_proj
                .reshape(&[1, seq_len as i32, n_layers, hpl])
                .expect("mlx op");
            let per_layer_proj = self.per_layer_projection_norm.forward(&per_layer_proj);
            let mixed = per_layer_proj.add(&per_layer_embed).expect("mlx op");
            let input_scale = Array::from_f32(self.per_layer_input_scale);
            Some(mixed.multiply(&input_scale).expect("mlx op"))
        } else {
            None
        };

        let n = self.num_non_shared;
        let mut aux_states: Vec<Array> = Vec::with_capacity(aux_layer_ids.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let rope = if layer.attention.is_sliding {
                &self.local_rope
            } else {
                &self.global_rope
            };
            let cache_idx = self.cache_slot[i];
            let is_shared = i >= n;
            let shared_kv: Option<(Array, Array)> = if is_shared {
                cache[cache_idx]
                    .as_ref()
                    .map(|(k, v)| (k.clone(), v.clone()))
            } else {
                None
            };
            let per_layer_i = per_layer.as_ref().map(|pl| {
                let li = i as i32;
                pl.index((.., .., li..li + 1, ..))
                    .reshape(&[1, seq_len as i32, hpl])
                    .expect("mlx op")
            });
            x = layer.forward(
                &x,
                rope,
                &mut cache[cache_idx],
                offset,
                shared_kv.as_ref(),
                per_layer_i.as_ref(),
            );
            // Stash a clone of the post-block hidden state if this is one
            // of the configured aux layers. Clone is cheap (mlx refcount).
            if aux_layer_ids.contains(&i) {
                aux_states.push(x.clone());
            }
        }

        x = self.norm.forward(&x);
        let logits = self.embed_tokens.matmul_transpose(&x);
        let logits = if let Some(cap) = self.final_logit_softcapping {
            let cap_arr = Array::from_f32(cap);
            let scaled = logits.divide(&cap_arr).expect("mlx op");
            let tanhed = mlx_rs::ops::tanh(&scaled).expect("mlx op");
            tanhed.multiply(&cap_arr).expect("mlx op")
        } else {
            logits
        };
        (logits, aux_states)
    }
}

// ─── Layer type helpers ───────────────────────────────────────────────────────

fn build_layer_types(config: &ModelConfig) -> Vec<String> {
    if let Some(types) = &config.layer_types {
        return types.clone();
    }
    // Fallback: sliding_window_pattern-based alternation (every Nth layer is full attention)
    let pattern = config.sliding_window_pattern.unwrap_or(5);
    (0..config.num_hidden_layers)
        .map(|i| {
            if (i + 1) % pattern == 0 {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 1536,
            intermediate_size: 6144,
            num_attention_heads: 8,
            num_hidden_layers: 35,
            num_key_value_heads: 1,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 131072,
            head_dim: Some(256),
            sliding_window: Some(512),
            tie_word_embeddings: true,
            model_type: Some("gemma4_text".into()),
            global_head_dim: Some(512),
            global_partial_rotary_factor: Some(0.25),
            rope_local_base_freq: Some(10_000.0),
            layer_types: None,
            num_kv_shared_layers: Some(20),
            hidden_size_per_layer_input: Some(256),
            vocab_size_per_layer_input: Some(262144),
            use_double_wide_mlp: Some(true),
            final_logit_softcapping: Some(30.0),
            rope_parameters: None,
            sliding_window_pattern: Some(5),
            attention_k_eq_v: None,
            num_global_key_value_heads: None,
            enable_moe_block: None,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
        }
    }

    /// E2B-style config: no k_eq_v, no MoE. v_proj must be Some, moe must be None.
    #[test]
    fn e2b_config_has_v_proj_and_no_moe() {
        let cfg = base_config();
        let block_sliding = Gemma4TransformerBlock::new(&cfg, true);
        let block_full = Gemma4TransformerBlock::new(&cfg, false);
        assert!(block_sliding.attention.v_proj.is_some());
        assert!(block_full.attention.v_proj.is_some());
        assert!(!block_sliding.attention.use_k_eq_v);
        assert!(!block_full.attention.use_k_eq_v);
        assert!(block_sliding.moe.is_none());
        assert!(block_full.moe.is_none());
    }

    /// 31B-style config (`attention_k_eq_v=true`): full-attn layers drop v_proj
    /// and use the global kv-head count; sliding-attn layers stay normal.
    #[test]
    fn k_eq_v_only_affects_full_attention_layers() {
        let mut cfg = base_config();
        cfg.attention_k_eq_v = Some(true);
        cfg.num_global_key_value_heads = Some(2);

        let sliding = Gemma4TransformerBlock::new(&cfg, true);
        assert!(
            !sliding.attention.use_k_eq_v,
            "sliding layer must NOT use k_eq_v"
        );
        assert!(sliding.attention.v_proj.is_some());
        assert_eq!(sliding.attention.num_kv_heads, cfg.num_key_value_heads);

        let full = Gemma4TransformerBlock::new(&cfg, false);
        assert!(
            full.attention.use_k_eq_v,
            "full-attn layer must use k_eq_v when config says so"
        );
        assert!(
            full.attention.v_proj.is_none(),
            "v_proj must be None when k_eq_v is active"
        );
        assert_eq!(
            full.attention.num_kv_heads,
            cfg.num_global_key_value_heads.unwrap(),
            "full-attn layer must adopt num_global_key_value_heads"
        );
    }

    /// 26B-style config (`enable_moe_block=true`): every block carries an MoE
    /// sub-block; experts are sized from `moe_intermediate_size`; routers
    /// from `num_experts` and `top_k_experts`.
    #[test]
    /// Regression for the PLE-zero panic: 12B / 26B / 31B variants ship
    /// `hidden_size_per_layer_input = 0` literally, which used to drive
    /// `unwrap_or(256) → 0 → reshape into [.., 0]` and crash. The model
    /// must now treat hpl=0 as "no PLE pipeline" — verified by ensuring
    /// `Gemma4Model::new` doesn't panic and `num_non_shared` still computes.
    #[test]
    fn ple_zero_does_not_panic_in_constructor() {
        let mut cfg = base_config();
        cfg.hidden_size_per_layer_input = Some(0); // 31B / 26B / 12B variant
        cfg.num_kv_shared_layers = Some(0); // simpler — no KV sharing
        cfg.num_hidden_layers = 4; // small for test speed
        // Provide explicit layer_types so build_layer_types doesn't divide-by-zero.
        cfg.layer_types = Some(vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ]);
        // Should not panic.
        let _model = Gemma4Model::new(&cfg);
    }

    /// `hpl = None` and `hpl = Some(0)` must behave identically — both mean
    /// "no PLE". Avoids the historical bug where unwrap_or(256) treated the
    /// two cases differently.
    #[test]
    fn ple_none_and_zero_are_equivalent() {
        let mut a = base_config();
        a.hidden_size_per_layer_input = None;
        a.num_kv_shared_layers = Some(0);
        a.num_hidden_layers = 2;
        a.layer_types = Some(vec!["full_attention".into(), "full_attention".into()]);

        let mut b = base_config();
        b.hidden_size_per_layer_input = Some(0);
        b.num_kv_shared_layers = Some(0);
        b.num_hidden_layers = 2;
        b.layer_types = Some(vec!["full_attention".into(), "full_attention".into()]);

        let m_a = Gemma4Model::new(&a);
        let m_b = Gemma4Model::new(&b);
        assert_eq!(m_a.num_non_shared, m_b.num_non_shared);
        // Both should have the per-layer-projection-norm tensor allocated to
        // the SAME placeholder shape — actual loading is gated separately.
        assert_eq!(
            m_a.per_layer_projection_norm.weight.shape(),
            m_b.per_layer_projection_norm.weight.shape()
        );
    }

    #[test]
    fn moe_block_present_and_sized_from_config() {
        let mut cfg = base_config();
        cfg.enable_moe_block = Some(true);
        cfg.num_experts = Some(32);
        cfg.top_k_experts = Some(2);
        cfg.moe_intermediate_size = Some(2048);

        let block = Gemma4TransformerBlock::new(&cfg, true);
        let moe = block.moe.as_ref().expect("moe should be present");
        assert_eq!(moe.router.num_experts, 32);
        assert_eq!(moe.router.top_k, 2);
        assert_eq!(moe.experts.num_experts, 32);
        // gate_proj shape: [num_experts, moe_intermediate, hidden].
        let gp = moe.experts.gate_proj.to_full();
        let shape = gp.shape();
        assert_eq!(shape, &[32, 2048, 1536]);
        let dp = moe.experts.down_proj.to_full();
        assert_eq!(dp.shape(), &[32, 1536, 2048]);
    }

    /// Disabling MoE means no MoE-only norms and no router/expert tiles
    /// allocated — important so E2B / 31B don't pay the memory cost.
    #[test]
    fn moe_disabled_by_default_when_flag_unset() {
        let cfg = base_config();
        let block = Gemma4TransformerBlock::new(&cfg, true);
        assert!(block.moe.is_none());
    }
}
