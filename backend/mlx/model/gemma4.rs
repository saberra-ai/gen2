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

// ─── Partial RoPE ─────────────────────────────────────────────────────────────

/// Apply RoPE to only the first `rope_dim` elements of the last axis.
///
/// Input shape: [batch, heads, seq, full_head_dim]
fn apply_partial_rope(x: &Array, rope: &RotaryEmbedding, offset: usize, rope_dim: usize) -> Array {
    let full_dim = x.shape()[3] as usize;
    if rope_dim >= full_dim {
        return rope.forward(x, offset);
    }
    let rd = rope_dim as i32;
    let fd = full_dim as i32;
    let x_rope = x.index((.., .., .., 0..rd));
    let x_pass = x.index((.., .., .., rd..fd));
    let x_rotated = rope.forward(&x_rope, offset);
    mlx_rs::ops::concatenate_axis(&[&x_rotated, &x_pass], 3).expect("mlx op")
}

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
    pub v_proj: Weight,
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
}

impl Gemma4Attention {
    pub fn new(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        eps: f32,
        is_sliding: bool,
        sliding_window: Option<usize>,
    ) -> Self {
        let q_proj = Weight::plain(
            Array::zeros::<f32>(&[(num_heads * head_dim) as i32, hidden as i32]).expect("mlx op"),
        );
        let k_proj = Weight::plain(
            Array::zeros::<f32>(&[(num_kv_heads * head_dim) as i32, hidden as i32])
                .expect("mlx op"),
        );
        let v_proj = Weight::plain(
            Array::zeros::<f32>(&[(num_kv_heads * head_dim) as i32, hidden as i32])
                .expect("mlx op"),
        );
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

        // Q: project → reshape → norm → partial RoPE → transpose to [B, H, S, D]
        let q = self.q_proj.matmul_transpose(x);
        let q = q.reshape(&[batch, seq, nh, hd]).expect("mlx op");
        let q = self.q_norm.forward(&q);
        let q = q.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let q = apply_partial_rope(&q, rope, offset, self.rope_dim);

        // K, V: either compute fresh (non-shared) or borrow from shared partner's cache
        let (k, v) = if let Some((ck, cv)) = shared_kv {
            (ck.clone(), cv.clone())
        } else {
            let k = self.k_proj.matmul_transpose(x);
            let v = self.v_proj.matmul_transpose(x);

            let k = k.reshape(&[batch, seq, nkv, hd]).expect("mlx op");
            let v = v.reshape(&[batch, seq, nkv, hd]).expect("mlx op");
            let k = self.k_norm.forward(&k);
            let v = self.v_norm.forward(&v);

            let k = k.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
            let v = v.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
            let k = apply_partial_rope(&k, rope, offset, self.rope_dim);

            // Append to KV cache along the sequence axis
            let (k, v) = if let Some((prev_k, prev_v)) = cache.take() {
                let k = mlx_rs::ops::concatenate_axis(&[&prev_k, &k], 2).expect("mlx op");
                let v = mlx_rs::ops::concatenate_axis(&[&prev_v, &v], 2).expect("mlx op");
                (k, v)
            } else {
                (k, v)
            };

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

        // GQA: expand KV heads to match the number of Q heads
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let reps = self.num_heads / self.num_kv_heads;
            (repeat_kv_heads(&k, reps), repeat_kv_heads(&v, reps))
        } else {
            (k, v)
        };

        // Attention score = Q · Kᵀ (scale = 1.0 per Gemma 4 spec)
        let k_t = k.transpose_axes(&[0, 1, 3, 2]).expect("mlx op");
        let mut scores = q.matmul(&k_t).expect("mlx op");

        // Causal + sliding-window mask (prefill only; seq_len == 1 during decode)
        if seq > 1 {
            let kv_len = scores.shape()[3] as usize;
            let mask = build_causal_mask(seq as usize, kv_len, self.sliding_window);
            scores = scores.add(&mask).expect("mlx op");
        }

        let attn_w = mlx_rs::ops::softmax_axes(&scores, &[-1], None).expect("mlx op");
        let out = attn_w.matmul(&v).expect("mlx op");

        // Merge heads: [B, H, S, D] → [B, S, H·D]
        let out = out.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let out = out.reshape(&[batch, seq, nh * hd]).expect("mlx op");

        self.o_proj.matmul_transpose(&out)
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
            Weight::plain(
                Array::zeros::<f32>(&[rows as i32, cols as i32]).expect("mlx op"),
            )
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
            Weight::plain(
                Array::zeros::<f32>(&[rows as i32, cols as i32]).expect("mlx op"),
            )
        };
        Self {
            per_layer_input_gate: zero(hidden_per_layer, hidden),
            per_layer_projection: zero(hidden, hidden_per_layer),
            post_per_layer_input_norm: RmsNorm::new(hidden, eps),
        }
    }

    /// Compute the per-layer embedding contribution to add to the residual stream.
    ///
    /// `embed_lookup`: already-gathered per-token embedding for this layer,
    /// shape `[batch, seq, hidden_per_layer]`. The caller (`Gemma4Model`) runs
    /// the gather+dequantize once for all layers and slices the per-layer
    /// chunk out for each call.
    pub fn forward(&self, x: &Array, embed_lookup: &Array) -> Array {
        // Gate: sigmoid(Linear(x, hidden → hpl)) → [batch, seq, hpl]
        let gate = self.per_layer_input_gate.matmul_transpose(x);
        let gate = mlx_rs::ops::sigmoid(&gate).expect("mlx op");

        // Gated multiply → project back → normalize
        let gated = gate.multiply(embed_lookup).expect("mlx op");
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
        let hpl = config.hidden_size_per_layer_input.unwrap_or(256);
        let eps = config.rms_norm_eps;
        let sw = if is_sliding { config.sliding_window } else { None };

        Self {
            attention: Gemma4Attention::new(
                hidden,
                config.num_attention_heads,
                config.num_key_value_heads,
                head_dim,
                rope_dim,
                eps,
                is_sliding,
                sw,
            ),
            ffn: Gemma4Ffn::new(hidden, config.intermediate_size),
            input_layernorm: RmsNorm::new(hidden, eps),
            post_attention_layernorm: RmsNorm::new(hidden, eps),
            pre_feedforward_layernorm: RmsNorm::new(hidden, eps),
            post_feedforward_layernorm: RmsNorm::new(hidden, eps),
            per_layer_input: Gemma4PerLayerInput::new(hidden, hpl, eps),
            layer_scalar: Array::ones::<f32>(&[1]).expect("mlx op"),
        }
    }

    pub fn forward(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        cache: &mut Option<(Array, Array)>,
        offset: usize,
        shared_kv: Option<&(Array, Array)>,
        embed_lookup: &Array,
    ) -> Array {
        // Pre-norm → attention → post-attn-norm → residual
        let h = self.input_layernorm.forward(x);
        let h = self.attention.forward(&h, rope, cache, offset, shared_kv);
        let h = self.post_attention_layernorm.forward(&h);
        let x = x.add(&h).expect("mlx op");

        // Pre-ffn-norm → FFN → post-ffn-norm → residual
        let h = self.pre_feedforward_layernorm.forward(&x);
        let h = self.ffn.forward(&h);
        let h = self.post_feedforward_layernorm.forward(&h);
        let x = x.add(&h).expect("mlx op");

        // Per-layer input contribution → residual
        let pli = self.per_layer_input.forward(&x, embed_lookup);
        let x = x.add(&pli).expect("mlx op");

        // Layer scalar (broadcast over [batch, seq, hidden])
        x.multiply(&self.layer_scalar).expect("mlx op")
    }
}

// ─── Full model ───────────────────────────────────────────────────────────────

pub struct Gemma4Model {
    pub embed_tokens: Weight,
    /// Full per-layer embedding table. Stored as a single (possibly quantized)
    /// weight; rows are gathered per-token and dequantized at forward time
    /// via `Weight::embedding_lookup` — never materialized in full.
    pub embed_tokens_per_layer: Weight,
    pub layers: Vec<Gemma4TransformerBlock>,
    pub norm: RmsNorm,
    /// RoPE for sliding-attention layers: full head_dim (256), theta=10000.
    pub local_rope: RotaryEmbedding,
    /// RoPE for full-attention layers: rope_dim only (128), theta=1 000 000.
    pub global_rope: RotaryEmbedding,
    embed_scale: f32,
    /// Number of layers that own their own KV cache slot (= num_hidden_layers − num_kv_shared_layers).
    pub num_non_shared: usize,
    final_logit_softcapping: Option<f32>,
    hidden_per_layer_input: usize,
}

impl Gemma4Model {
    pub fn new(config: &ModelConfig) -> Self {
        let vocab = config.vocab_size;
        let hidden = config.hidden_size;
        let n = config.num_hidden_layers;
        let n_shared = config.num_kv_shared_layers.unwrap_or(0);
        let num_non_shared = n.saturating_sub(n_shared);
        let hpl = config.hidden_size_per_layer_input.unwrap_or(256);
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
        let global_rope_dim =
            ((global_head_dim as f32) * config.global_partial_rotary_factor.unwrap_or(0.25))
                .round() as usize;
        let local_theta = config.rope_local_base_freq.unwrap_or(10_000.0);
        let global_theta = config.rope_theta;
        let max_seq = config.max_position_embeddings;

        Self {
            embed_tokens: Weight::plain(
                Array::zeros::<f32>(&[vocab as i32, hidden as i32]).expect("mlx op"),
            ),
            embed_tokens_per_layer: Weight::plain(
                Array::zeros::<f32>(&[vocab as i32, (hpl * n) as i32]).expect("mlx op"),
            ),
            layers,
            norm: RmsNorm::new(hidden, eps),
            local_rope: RotaryEmbedding::new(sliding_head_dim, max_seq, local_theta),
            global_rope: RotaryEmbedding::new(global_rope_dim, max_seq, global_theta),
            embed_scale,
            num_non_shared,
            final_logit_softcapping: config.final_logit_softcapping,
            hidden_per_layer_input: hpl,
        }
    }

    /// Forward pass. Returns logits for the last token: shape [1, vocab_size].
    pub fn forward(&self, tokens: &[u32], cache: &mut KvCache) -> Array {
        let seq_len = tokens.len();
        let offset = cache
            .first()
            .and_then(|e| e.as_ref())
            .map(|(k, _)| k.shape()[2] as usize)
            .unwrap_or(0);

        let idx: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_indices = Array::from_slice(&idx, &[seq_len as i32]);

        // Main embedding: gather + dequantize only the rows we need.
        let x = self.embed_tokens.embedding_lookup(&token_indices);
        let scale = Array::from_f32(self.embed_scale);
        let x = x.multiply(&scale).expect("mlx op");
        let hidden = self.layers[0].input_layernorm.weight.shape()[0];
        let mut x = x.reshape(&[1, seq_len as i32, hidden]).expect("mlx op");

        // Per-layer embeddings: one gather+dequantize for all layers.
        // Output: [seq_len, num_layers × hpl] → reshape [1, seq_len, num_layers, hpl].
        let n_layers = self.layers.len() as i32;
        let hpl = self.hidden_per_layer_input as i32;
        let per_layer_flat = self.embed_tokens_per_layer.embedding_lookup(&token_indices);
        let per_layer = per_layer_flat
            .reshape(&[1, seq_len as i32, n_layers, hpl])
            .expect("mlx op");

        let n = self.num_non_shared;

        for (i, layer) in self.layers.iter().enumerate() {
            let rope = if layer.attention.is_sliding {
                &self.local_rope
            } else {
                &self.global_rope
            };

            // Shared layers re-use the cache slot of their non-shared partner.
            // The modulo handles the second cycle (layers 30–34 → same slots as 0–4).
            let cache_idx = if i < n { i } else { (i - n) % n };
            let is_shared = i >= n;

            let shared_kv: Option<(Array, Array)> = if is_shared {
                cache[cache_idx].as_ref().map(|(k, v)| (k.clone(), v.clone()))
            } else {
                None
            };

            // Per-layer embedding slice for layer i: [1, seq, hpl]
            let li = i as i32;
            let embed_lookup = per_layer
                .index((.., .., li..li + 1, ..))
                .reshape(&[1, seq_len as i32, hpl])
                .expect("mlx op");

            x = layer.forward(
                &x,
                rope,
                &mut cache[cache_idx],
                offset,
                shared_kv.as_ref(),
                &embed_lookup,
            );
        }

        // Final norm
        x = self.norm.forward(&x);

        // LM head (weights tied to embed_tokens)
        let logits = self.embed_tokens.matmul_transpose(&x);

        // Final logit softcapping: tanh(x / cap) * cap
        let logits = if let Some(cap) = self.final_logit_softcapping {
            let cap_arr = Array::from_f32(cap);
            let scaled = logits.divide(&cap_arr).expect("mlx op");
            let tanhed = mlx_rs::ops::tanh(&scaled).expect("mlx op");
            tanhed.multiply(&cap_arr).expect("mlx op")
        } else {
            logits
        };

        // Return the last position's logits: [1, vocab_size]
        let s = seq_len as i32;
        logits.index((0..1, (s - 1)..s, ..))
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
