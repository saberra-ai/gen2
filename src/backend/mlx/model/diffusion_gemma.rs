//! DiffusionGemma (block-diffusion Gemma 4) text model — slice 1.
//!
//! This is a native MLX (`mlx-rs`) translation of mlx-vlm's
//! `diffusion_gemma/language.py`. It is an **encoder/decoder block-diffusion**
//! model, *not* an autoregressive decoder:
//!
//! - **Encoder**: embeds the prompt and runs the shared transformer layers
//!   over it, producing a per-layer KV cache. Bidirectional (full) attention
//!   layers see the whole prompt; sliding layers see a 1024-token window.
//!   Each encoder layer multiplies its output by a distinct learnable
//!   `layer_scalar` (the *encoder* scalar, separate from the decoder one).
//! - **Decoder**: embeds a fixed-length canvas (256 tokens) plus a
//!   self-conditioning signal, then runs the *same* transformer layers,
//!   cross-attending to the encoder KV cache concatenated with the canvas.
//!   The decoder masks are bidirectional within the canvas (this is a
//!   diffusion model — every canvas position attends to every other).
//!
//! Each `DecoderLayer` runs a **dual feed-forward**: a dense `MLP` branch and
//! a MoE `Router`+`Experts` branch in parallel, combined as
//! `post_feedforward_layernorm(mlp_out + moe_out)`. There are 5 feed-forward
//! norms plus a `layer_scalar`.
//!
//! Attention is QK-normed (per-head RMSNorm on Q and K, RMSNormNoScale on V)
//! with **different head dims per layer type**: sliding layers use head_dim
//! 256 with 8 KV heads; full (global) layers use head_dim 512 with 2 KV heads
//! and reuse K as V (no `v_proj`).
//!
//! Slice 1 goal: load the checkpoint and run encoder + one decoder forward to
//! produce logits of shape `(1, canvas_length, vocab)` without panicking.
//! Numerical parity is a later slice; the denoising generation loop is slice 2.

use mlx_rs::Array;

use super::moe::{Experts, Router};
use super::norm::RmsNorm;
use super::quantized::Weight;
use super::rope::RotaryEmbedding;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Parsed `text_config` for DiffusionGemma. Mirrors the reference `TextConfig`
/// for the fields slice 1 needs (text-only; vision is skipped entirely).
#[derive(Debug, Clone)]
pub struct DiffusionGemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub final_logit_softcapping: Option<f32>,
    pub canvas_length: usize,
    /// Per-layer attention type, length `num_hidden_layers`.
    /// "sliding_attention" or "full_attention".
    pub layer_types: Vec<String>,
    pub rope_theta_local: f32,
    pub rope_theta_global: f32,
    pub global_partial_rotary_factor: f32,
}

impl DiffusionGemmaConfig {
    /// Parse from the raw `config.json` value (the whole top-level object).
    pub fn from_json(raw: &serde_json::Value) -> Result<Self, String> {
        let tc = raw
            .get("text_config")
            .ok_or_else(|| "missing text_config".to_string())?;

        let get_usize = |v: &serde_json::Value, k: &str, default: usize| -> usize {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(default)
        };
        let get_f32 = |v: &serde_json::Value, k: &str, default: f32| -> f32 {
            v.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(default)
        };

        let num_hidden_layers = get_usize(tc, "num_hidden_layers", 30);

        // layer_types: prefer explicit list; fall back to the 5-sliding-then-full pattern.
        let layer_types: Vec<String> = match tc.get("layer_types").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect(),
            None => (0..num_hidden_layers)
                .map(|i| {
                    if (i + 1) % 6 == 0 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect(),
        };

        // RoPE params nested per layer type.
        let rope = tc.get("rope_parameters");
        let rope_theta_local = rope
            .and_then(|r| r.get("sliding_attention"))
            .map(|s| get_f32(s, "rope_theta", 10_000.0))
            .unwrap_or(10_000.0);
        let rope_theta_global = rope
            .and_then(|r| r.get("full_attention"))
            .map(|s| get_f32(s, "rope_theta", 1_000_000.0))
            .unwrap_or(1_000_000.0);
        let global_partial_rotary_factor = rope
            .and_then(|r| r.get("full_attention"))
            .map(|s| get_f32(s, "partial_rotary_factor", 0.25))
            .unwrap_or(0.25);

        let final_softcap = tc.get("final_logit_softcapping").and_then(|x| x.as_f64());

        Ok(Self {
            vocab_size: get_usize(tc, "vocab_size", 262144),
            hidden_size: get_usize(tc, "hidden_size", 2816),
            intermediate_size: get_usize(tc, "intermediate_size", 2112),
            moe_intermediate_size: get_usize(tc, "moe_intermediate_size", 704),
            num_hidden_layers,
            num_attention_heads: get_usize(tc, "num_attention_heads", 16),
            num_key_value_heads: get_usize(tc, "num_key_value_heads", 8),
            num_global_key_value_heads: get_usize(tc, "num_global_key_value_heads", 2),
            head_dim: get_usize(tc, "head_dim", 256),
            global_head_dim: get_usize(tc, "global_head_dim", 512),
            rms_norm_eps: get_f32(tc, "rms_norm_eps", 1e-6),
            max_position_embeddings: get_usize(tc, "max_position_embeddings", 262144),
            sliding_window: get_usize(tc, "sliding_window", 1024),
            num_experts: get_usize(tc, "num_experts", 128),
            top_k_experts: get_usize(tc, "top_k_experts", 8),
            final_logit_softcapping: final_softcap.map(|x| x as f32),
            canvas_length: get_usize(raw, "canvas_length", 256),
            layer_types,
            rope_theta_local,
            rope_theta_global,
            global_partial_rotary_factor,
        })
    }

    fn is_sliding(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "sliding_attention")
            .unwrap_or(true)
    }
}

// ─── RMSNorm without learnable scale (v_norm, self-conditioning post_norm) ────

struct RmsNormNoScale {
    eps: f32,
}

impl RmsNormNoScale {
    fn new(eps: f32) -> Self {
        Self { eps }
    }

    fn forward(&self, x: &Array) -> Array {
        let x_sq = x.multiply(x).expect("mlx op");
        let var = x_sq.mean_axis(-1, true).expect("mlx op");
        let var_eps = var.add(Array::from_f32(self.eps)).expect("mlx op");
        let norm_factor = var_eps.rsqrt().expect("mlx op");
        x.multiply(&norm_factor).expect("mlx op")
    }
}

// ─── GEGLU helper ─────────────────────────────────────────────────────────────

/// `gelu_approx(gate) * x` — matches the reference `geglu(gate, x)`.
fn geglu(gate: &Array, x: &Array) -> Array {
    let g = mlx_rs::nn::gelu_approximate(gate).expect("mlx op");
    g.multiply(x).expect("mlx op")
}

// ─── Dense MLP ────────────────────────────────────────────────────────────────

pub struct Mlp {
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
}

impl Mlp {
    fn new(hidden: usize, intermediate: usize) -> Self {
        let zero = |r: usize, c: usize| {
            Weight::plain(Array::zeros::<f32>(&[r as i32, c as i32]).expect("mlx op"))
        };
        Self {
            gate_proj: zero(intermediate, hidden),
            up_proj: zero(intermediate, hidden),
            down_proj: zero(hidden, intermediate),
        }
    }

    fn forward(&self, x: &Array) -> Array {
        let gate = self.gate_proj.matmul_transpose(x);
        let up = self.up_proj.matmul_transpose(x);
        let gated = geglu(&gate, &up);
        self.down_proj.matmul_transpose(&gated)
    }
}

// ─── Attention ────────────────────────────────────────────────────────────────

/// Repeat KV heads to match Q heads (GQA expansion).
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

pub struct Attention {
    pub q_proj: Weight,
    pub k_proj: Weight,
    /// `None` for global (full-attention) layers: V is reused from K (pre-norm).
    pub v_proj: Option<Weight>,
    pub o_proj: Weight,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    v_norm: RmsNormNoScale,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub is_sliding: bool,
}

impl Attention {
    fn new(config: &DiffusionGemmaConfig, layer_idx: usize) -> Self {
        let is_sliding = config.is_sliding(layer_idx);
        let head_dim = if is_sliding {
            config.head_dim
        } else {
            config.global_head_dim
        };
        let num_heads = config.num_attention_heads;
        let num_kv_heads = if is_sliding {
            config.num_key_value_heads
        } else {
            config.num_global_key_value_heads
        };
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps;
        let zero = |r: usize, c: usize| {
            Weight::plain(Array::zeros::<f32>(&[r as i32, c as i32]).expect("mlx op"))
        };
        Self {
            q_proj: zero(num_heads * head_dim, hidden),
            k_proj: zero(num_kv_heads * head_dim, hidden),
            // Global layers reuse K as V (the checkpoint has no v_proj for them).
            v_proj: if is_sliding {
                Some(zero(num_kv_heads * head_dim, hidden))
            } else {
                None
            },
            o_proj: zero(hidden, num_heads * head_dim),
            q_norm: RmsNorm::new(head_dim, eps),
            k_norm: RmsNorm::new(head_dim, eps),
            v_norm: RmsNormNoScale::new(eps),
            num_heads,
            num_kv_heads,
            head_dim,
            is_sliding,
        }
    }

    /// Compute Q/K/V for `x` at sequence `offset`. Returns
    /// `(q, k, v)` each `[B, H_or_KV, S, head_dim]` with RoPE already applied
    /// to Q and K. V is v-normed (no RoPE).
    fn qkv(&self, x: &Array, rope: &RotaryEmbedding, offset: usize) -> (Array, Array, Array) {
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;

        let q = self.q_proj.matmul_transpose(x);
        let q = q.reshape(&[batch, seq, nh, hd]).expect("mlx op");
        let q = self.q_norm.forward(&q);
        let q = q.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let q = rope.forward(&q, offset);

        let k = self.k_proj.matmul_transpose(x);
        let v = match &self.v_proj {
            Some(w) => w.matmul_transpose(x),
            None => k.clone(),
        };
        let k = k.reshape(&[batch, seq, nkv, hd]).expect("mlx op");
        let v = v.reshape(&[batch, seq, nkv, hd]).expect("mlx op");
        let k = self.k_norm.forward(&k);
        let v = self.v_norm.forward(&v);
        let k = k.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let v = v.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let k = rope.forward(&k, offset);

        (q, k, v)
    }

    /// Run SDPA given precomputed q/k/v and an optional additive mask
    /// `[B, 1, q_len, kv_len]` (0 = attend, -inf = masked). Merges heads and
    /// applies `o_proj`.
    fn attend(&self, q: &Array, k: &Array, v: &Array, mask: Option<&Array>) -> Array {
        let shape = q.shape();
        let batch = shape[0];
        let seq = shape[2];
        let nh = self.num_heads as i32;
        let hd = self.head_dim as i32;

        // GQA expand.
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let reps = self.num_heads / self.num_kv_heads;
            (repeat_kv_heads(k, reps), repeat_kv_heads(v, reps))
        } else {
            (k.clone(), v.clone())
        };

        // scores = scale * q @ kᵀ. Reference uses scale = 1.0.
        let k_t = k.transpose_axes(&[0, 1, 3, 2]).expect("mlx op");
        let mut scores = q.matmul(&k_t).expect("mlx op");
        if let Some(m) = mask {
            scores = scores.add(m).expect("mlx op");
        }
        let attn_w = mlx_rs::ops::softmax_axes(&scores, &[-1], None).expect("mlx op");
        let out = attn_w.matmul(&v).expect("mlx op");

        let out = out.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let out = out.reshape(&[batch, seq, nh * hd]).expect("mlx op");
        self.o_proj.matmul_transpose(&out)
    }
}

// ─── MoE block (Router + Experts) ─────────────────────────────────────────────

pub struct MoeBlock {
    pub router: Router,
    pub experts: Experts,
}

// ─── Decoder layer ────────────────────────────────────────────────────────────

pub struct DecoderLayer {
    pub layer_idx: usize,
    pub is_sliding: bool,
    pub self_attn: Attention,
    pub mlp: Mlp,
    pub moe: MoeBlock,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub pre_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm: RmsNorm,
    pub pre_feedforward_layernorm_2: RmsNorm,
    pub post_feedforward_layernorm_1: RmsNorm,
    pub post_feedforward_layernorm_2: RmsNorm,
    /// Decoder per-layer scalar (shape `[1]`).
    pub layer_scalar: Array,
}

impl DecoderLayer {
    fn new(config: &DiffusionGemmaConfig, layer_idx: usize) -> Self {
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps;
        Self {
            layer_idx,
            is_sliding: config.is_sliding(layer_idx),
            self_attn: Attention::new(config, layer_idx),
            mlp: Mlp::new(hidden, config.intermediate_size),
            moe: MoeBlock {
                router: Router::new(hidden, config.num_experts, config.top_k_experts, eps),
                experts: Experts::new(hidden, config.moe_intermediate_size, config.num_experts),
            },
            input_layernorm: RmsNorm::new(hidden, eps),
            post_attention_layernorm: RmsNorm::new(hidden, eps),
            pre_feedforward_layernorm: RmsNorm::new(hidden, eps),
            post_feedforward_layernorm: RmsNorm::new(hidden, eps),
            pre_feedforward_layernorm_2: RmsNorm::new(hidden, eps),
            post_feedforward_layernorm_1: RmsNorm::new(hidden, eps),
            post_feedforward_layernorm_2: RmsNorm::new(hidden, eps),
            layer_scalar: Array::ones::<f32>(&[1]).expect("mlx op"),
        }
    }

    /// Feed-forward + residual + layer-scalar tail shared by encoder/decoder.
    ///
    /// `attn_out` is the post-attention residual stream `h` (after the
    /// `residual + post_attention_layernorm(attn)` step). `layer_scalar`
    /// overrides the decoder scalar (the encoder passes its own).
    fn feedforward_tail(&self, h: &Array, layer_scalar: &Array) -> Array {
        let residual = h;

        // Dense MLP branch.
        let h1 = self.pre_feedforward_layernorm.forward(residual);
        let h1 = self.mlp.forward(&h1);
        let h1 = self.post_feedforward_layernorm_1.forward(&h1);

        // MoE branch (operates on flattened [B*S, H], reshaped back).
        let shape = residual.shape();
        let (b, s, hdim) = (shape[0], shape[1], shape[2]);
        let flat = residual.reshape(&[1, b * s, hdim]).expect("mlx op");
        let (idx, w) = self.moe.router.forward(&flat);
        let h2 = self.pre_feedforward_layernorm_2.forward(&flat);
        let h2 = self.moe.experts.forward(&h2, &idx, &w);
        let h2 = h2.reshape(&[b, s, hdim]).expect("mlx op");
        let h2 = self.post_feedforward_layernorm_2.forward(&h2);

        let h = h1.add(&h2).expect("mlx op");
        let h = self.post_feedforward_layernorm.forward(&h);
        let h = residual.add(&h).expect("mlx op");
        h.multiply(layer_scalar).expect("mlx op")
    }

    /// Encoder forward: self-attention over `x` with a causal/sliding mask,
    /// returns `(output, k, v)` where `(k, v)` are this layer's cache entries.
    fn forward_encoder(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        mask: Option<&Array>,
        layer_scalar: &Array,
    ) -> (Array, Array, Array) {
        let residual = x;
        let h = self.input_layernorm.forward(x);
        let (q, k, v) = self.self_attn.qkv(&h, rope, 0);
        let attn = self.self_attn.attend(&q, &k, &v, mask);
        let attn = self.post_attention_layernorm.forward(&attn);
        let h = residual.add(&attn).expect("mlx op");
        let out = self.feedforward_tail(&h, layer_scalar);
        (out, k, v)
    }

    /// Decoder forward: cross-attend over `[encoder_kv ++ canvas_kv]`.
    /// `offset` is the encoder length (the position the canvas begins at).
    fn forward_decoder(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        encoder_kv: &(Array, Array),
        mask: Option<&Array>,
        offset: usize,
    ) -> Array {
        let residual = x;
        let h = self.input_layernorm.forward(x);
        let (q, k, v) = self.self_attn.qkv(&h, rope, offset);

        // Concatenate encoder KV (positions 0..offset) with canvas KV.
        let (enc_k, enc_v) = encoder_kv;
        let k = mlx_rs::ops::concatenate_axis(&[enc_k, &k], 2).expect("mlx op");
        let v = mlx_rs::ops::concatenate_axis(&[enc_v, &v], 2).expect("mlx op");

        let attn = self.self_attn.attend(&q, &k, &v, mask);
        let attn = self.post_attention_layernorm.forward(&attn);
        let h = residual.add(&attn).expect("mlx op");
        self.feedforward_tail(&h, &self.layer_scalar)
    }
}

// ─── Self-conditioning ────────────────────────────────────────────────────────

pub struct SelfConditioning {
    pub pre_norm: RmsNorm,
    post_norm: RmsNormNoScale,
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
}

impl SelfConditioning {
    fn new(config: &DiffusionGemmaConfig) -> Self {
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps;
        let zero = |r: usize, c: usize| {
            Weight::plain(Array::zeros::<f32>(&[r as i32, c as i32]).expect("mlx op"))
        };
        Self {
            pre_norm: RmsNorm::new(hidden, eps),
            post_norm: RmsNormNoScale::new(eps),
            gate_proj: zero(config.intermediate_size, hidden),
            up_proj: zero(config.intermediate_size, hidden),
            down_proj: zero(hidden, config.intermediate_size),
        }
    }

    /// `post_norm(inputs_embeds + down(geglu(gate(pre_norm(signal)), up(...))))`.
    fn forward(&self, inputs_embeds: &Array, signal: &Array) -> Array {
        let normed = self.pre_norm.forward(signal);
        let gate = self.gate_proj.matmul_transpose(&normed);
        let up = self.up_proj.matmul_transpose(&normed);
        let g = geglu(&gate, &up);
        let sig = self.down_proj.matmul_transpose(&g);
        let summed = inputs_embeds.add(&sig).expect("mlx op");
        self.post_norm.forward(&summed)
    }
}

// ─── Top-level model ──────────────────────────────────────────────────────────

pub struct DiffusionGemmaModel {
    pub config: DiffusionGemmaConfig,
    pub embed_tokens: Weight,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub self_conditioning: SelfConditioning,
    /// Encoder per-layer scalars (shape `[1]` each), one per layer.
    pub encoder_layer_scalars: Vec<Array>,
    local_rope: RotaryEmbedding,
    global_rope: RotaryEmbedding,
    embed_scale: f32,
}

/// Encoder KV cache: one `(K, V)` entry per layer, each `[1, kv_heads, enc_len, head_dim]`.
pub type EncoderCache = Vec<(Array, Array)>;

impl DiffusionGemmaModel {
    pub fn new(config: DiffusionGemmaConfig) -> Self {
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let eps = config.rms_norm_eps;
        let max_seq = config.max_position_embeddings;

        let layers: Vec<DecoderLayer> = (0..config.num_hidden_layers)
            .map(|i| DecoderLayer::new(&config, i))
            .collect();
        let encoder_layer_scalars = (0..config.num_hidden_layers)
            .map(|_| Array::ones::<f32>(&[1]).expect("mlx op"))
            .collect();

        let local_rope = RotaryEmbedding::new(config.head_dim, max_seq, config.rope_theta_local);
        let global_rope_dim = ((config.global_head_dim as f32)
            * config.global_partial_rotary_factor)
            .round() as usize;
        let global_rope = RotaryEmbedding::with_freq_divisor(
            global_rope_dim,
            config.global_head_dim,
            max_seq,
            config.rope_theta_global,
        );

        Self {
            embed_tokens: Weight::plain(
                Array::zeros::<f32>(&[vocab as i32, hidden as i32]).expect("mlx op"),
            ),
            layers,
            norm: RmsNorm::new(hidden, eps),
            self_conditioning: SelfConditioning::new(&config),
            encoder_layer_scalars,
            local_rope,
            global_rope,
            embed_scale: (hidden as f32).sqrt(),
            config,
        }
    }

    fn rope_for(&self, is_sliding: bool) -> &RotaryEmbedding {
        if is_sliding {
            &self.local_rope
        } else {
            &self.global_rope
        }
    }

    fn embed(&self, token_ids: &[u32]) -> Array {
        let idx: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        let token_indices = Array::from_slice(&idx, &[token_ids.len() as i32]);
        let emb = self.embed_tokens.embedding_lookup(&token_indices);
        let scale = Array::from_f32(self.embed_scale);
        let emb = emb.multiply(&scale).expect("mlx op");
        let hidden = self.config.hidden_size as i32;
        emb.reshape(&[1, token_ids.len() as i32, hidden])
            .expect("mlx op")
    }

    /// Run the encoder over `prompt_ids`. Returns the per-layer KV cache.
    /// Encoder attention is causal + sliding-window (matches the reference
    /// `create_attention_mask` defaults for the prefill case).
    pub fn encode(&self, prompt_ids: &[u32]) -> EncoderCache {
        let seq = prompt_ids.len();
        let mut h = self.embed(prompt_ids);
        let mut cache: EncoderCache = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let rope = self.rope_for(layer.is_sliding);
            let window = if layer.is_sliding {
                Some(self.config.sliding_window)
            } else {
                None
            };
            let mask = build_causal_mask(seq, seq, window);
            let scalar = &self.encoder_layer_scalars[i];
            let (out, k, v) = layer.forward_encoder(&h, rope, Some(&mask), scalar);
            h = out;
            cache.push((k, v));
        }
        cache
    }

    /// Run a single decoder forward over `canvas_ids` (length `canvas_length`),
    /// cross-attending to the encoder cache. `self_conditioning_logits` is the
    /// previous step's logits used as the conditioning signal (None on the
    /// first step → zero signal). Returns logits `(1, canvas_length, vocab)`.
    pub fn decode(
        &self,
        canvas_ids: &[u32],
        encoder_cache: &EncoderCache,
        self_conditioning_logits: Option<&Array>,
    ) -> Array {
        let canvas_len = canvas_ids.len();
        let encoder_len = if encoder_cache.is_empty() {
            0
        } else {
            encoder_cache[0].0.shape()[2] as usize
        };

        // Embed canvas + self-conditioning.
        let inputs_embeds = self.embed(canvas_ids);
        let soft = match self_conditioning_logits {
            None => Array::zeros::<f32>(inputs_embeds.shape()).expect("mlx op"),
            Some(logits) => {
                // probs = softmax(logits); soft = (probs @ embed_table) * embed_scale.
                let probs = mlx_rs::ops::softmax_axes(logits, &[-1], None).expect("mlx op");
                let table = self.embed_tokens.to_full(); // [vocab, hidden]
                let soft = probs.matmul(&table).expect("mlx op");
                soft.multiply(Array::from_f32(self.embed_scale))
                    .expect("mlx op")
            }
        };
        let mut h = self.self_conditioning.forward(&inputs_embeds, &soft);

        // Bidirectional decoder masks, by layer type.
        let full_mask = build_decoder_mask(canvas_len, encoder_len, encoder_len, None);
        let window_prefix = self.config.sliding_window.saturating_sub(1);
        let sliding_mask =
            build_decoder_mask(canvas_len, encoder_len, encoder_len, Some(window_prefix));

        for layer in self.layers.iter() {
            let rope = self.rope_for(layer.is_sliding);
            let mask = if layer.is_sliding {
                sliding_mask.as_ref()
            } else {
                full_mask.as_ref()
            };
            h = layer.forward_decoder(&h, rope, &encoder_cache[layer.layer_idx], mask, encoder_len);
        }

        let h = self.norm.forward(&h);

        // Tied LM head: logits = embed_tokens.as_linear(h) → quantized matmul.
        let logits = self.embed_tokens.matmul_transpose(&h);

        // Softcap: tanh(x / cap) * cap.
        match self.config.final_logit_softcapping {
            Some(cap) => {
                let cap_arr = Array::from_f32(cap);
                let scaled = logits.divide(&cap_arr).expect("mlx op");
                let tanhed = mlx_rs::ops::tanh(&scaled).expect("mlx op");
                tanhed.multiply(&cap_arr).expect("mlx op")
            }
            None => logits,
        }
    }
}

// ─── Slice 2: entropy-bound denoising generation loop ──────────────────────────

/// Parameters for the entropy-bound denoising sampler, sourced from the
/// checkpoint `generation_config`.
#[derive(Debug, Clone)]
pub struct DiffusionGenParams {
    /// Number of denoising steps (`max_denoising_steps`, default 48).
    pub max_denoising_steps: usize,
    /// Entropy acceptance bound (`sampler_config.entropy_bound`, default 0.1).
    pub entropy_bound: f32,
    /// Linear temperature schedule floor (`t_min`, default 0.4).
    pub t_min: f32,
    /// Linear temperature schedule ceiling (`t_max`, default 0.8).
    pub t_max: f32,
    /// Stop-token ids (`eos_token_id`, default `[1, 106, 50]`).
    pub eos_token_ids: Vec<u32>,
}

impl Default for DiffusionGenParams {
    fn default() -> Self {
        Self {
            max_denoising_steps: 48,
            entropy_bound: 0.1,
            t_min: 0.4,
            t_max: 0.8,
            eos_token_ids: vec![1, 106, 50],
        }
    }
}

impl DiffusionGenParams {
    /// Parse from the raw `config.json` value (the whole top-level object).
    pub fn from_json(raw: &serde_json::Value) -> Self {
        let mut p = Self::default();
        let gc = raw.get("generation_config");
        if let Some(gc) = gc {
            if let Some(v) = gc.get("max_denoising_steps").and_then(|x| x.as_u64()) {
                p.max_denoising_steps = v as usize;
            }
            if let Some(v) = gc
                .get("sampler_config")
                .and_then(|s| s.get("entropy_bound"))
                .and_then(|x| x.as_f64())
            {
                p.entropy_bound = v as f32;
            }
            if let Some(v) = gc.get("t_min").and_then(|x| x.as_f64()) {
                p.t_min = v as f32;
            }
            if let Some(v) = gc.get("t_max").and_then(|x| x.as_f64()) {
                p.t_max = v as f32;
            }
            if let Some(arr) = gc.get("eos_token_id").and_then(|x| x.as_array()) {
                let ids: Vec<u32> = arr
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect();
                if !ids.is_empty() {
                    p.eos_token_ids = ids;
                }
            }
        }
        p
    }
}

/// Per-position entropy for logits `[seq, vocab]` (batch flattened to seq).
///
/// `entropy = -sum(softmax(logits) * log_softmax(logits))`, computed via the
/// numerically-stable `log_probs = logits - logsumexp(logits)` form (matches
/// the reference `_diffusion_token_entropy`).
fn token_entropy(logits: &Array) -> Array {
    let lse = mlx_rs::ops::logsumexp_axis(logits, -1, true).expect("mlx op");
    let log_probs = logits.subtract(&lse).expect("mlx op");
    let probs = mlx_rs::ops::exp(&log_probs).expect("mlx op");
    let prod = probs.multiply(&log_probs).expect("mlx op");
    let summed = prod.sum_axis(-1, false).expect("mlx op");
    summed.negative().expect("mlx op")
}

/// Entropy transfer (acceptance) mask over a 1D entropy vector `[seq]`.
///
/// Ports `_diffusion_entropy_transfer_mask` exactly:
///   sort entropy ascending → cumsum & cummax of sorted entropy →
///   `(cumsum - cummax) <= bound` → scatter the boolean selection back to the
///   original positions via `put_along_axis`.
///
/// Returns a `bool` array `[seq]`: `true` = accept this position's sampled token.
fn entropy_transfer_mask(entropy: &Array, entropy_bound: f32) -> Array {
    // 1D argsort (ascending). Batch is always 1 so flattening is exact.
    let sorted_indices = mlx_rs::ops::argsort(entropy).expect("mlx op");
    let sorted_entropy =
        mlx_rs::ops::indexing::take_along_axis(entropy, &sorted_indices, 0).expect("mlx op");
    let cumulative = sorted_entropy.cumsum(0, false, true).expect("mlx op");
    let cumulative_max = sorted_entropy.cummax(0, false, true).expect("mlx op");
    let diff = cumulative.subtract(&cumulative_max).expect("mlx op");
    let bound = Array::from_f32(entropy_bound);
    let sorted_selection = diff.le(&bound).expect("mlx op"); // bool [seq]

    // Scatter the sorted selection back to original positions.
    let zeros = Array::from_slice(&vec![false; entropy.size()], &[entropy.size() as i32]);
    mlx_rs::ops::indexing::put_along_axis(&zeros, &sorted_indices, &sorted_selection, 0)
        .expect("mlx op")
}

impl DiffusionGemmaModel {
    /// Build a fresh random canvas of `canvas_len` token ids in `[0, vocab)`.
    /// Returned as a host `Vec<u32>` so it can feed `decode(canvas_ids)`.
    fn random_canvas(&self, canvas_len: usize, key: &Array) -> Vec<u32> {
        let lo = Array::from_int(0);
        let hi = Array::from_int(self.config.vocab_size as i32);
        let canvas = mlx_rs::random::randint::<_, i32>(lo, hi, &[canvas_len as i32][..], Some(key))
            .expect("mlx op");
        canvas.as_slice::<i32>().iter().map(|&v| v as u32).collect()
    }

    /// Entropy-bound denoising generation. Returns the generated token ids
    /// (a single 256-canvas, EOS-trimmed).
    ///
    /// Faithful port of mlx-vlm's `stream_diffusion_generate` inner loop for the
    /// `entropy-bound` sampler at `temperature <= 0` (greedy denoiser): random
    /// canvas init → for each of `max_denoising_steps` steps, decode logits,
    /// apply the linear temperature schedule, accept low-entropy positions
    /// (`_diffusion_entropy_transfer_mask`) and re-randomize the rest, carrying
    /// the previous step's logits forward as the self-conditioning signal. The
    /// final canvas is `argmax(logits)` of the last step.
    pub fn diffusion_generate(&self, prompt_ids: &[u32], params: &DiffusionGenParams) -> Vec<u32> {
        let canvas_len = self.config.canvas_length;
        let max_steps = params.max_denoising_steps;

        let encoder_cache = self.encode(prompt_ids);

        // Deterministic-ish RNG key seeded once; split per re-randomization.
        let key = mlx_rs::random::key(0x5151_5151).expect("mlx op");
        let split = |k: &Array| -> (Array, Array) { mlx_rs::random::split(k, 2).expect("mlx op") };

        let (k0, mut key_rest) = split(&key);
        let mut canvas: Vec<u32> = self.random_canvas(canvas_len, &k0);

        // Self-conditioning carries the *previous* step's raw logits (slice-1
        // `decode` does softmax + soft-embedding internally).
        let mut self_cond: Option<Array> = None;
        let mut final_logits: Option<Array> = None;

        // for cur_step in reversed(range(1, max_steps + 1))
        for cur_step in (1..=max_steps).rev() {
            let logits = self.decode(&canvas, &encoder_cache, self_cond.as_ref());
            // logits: (1, canvas_len, vocab) → flatten batch to (canvas_len, vocab).
            let logits2d = logits
                .reshape(&[canvas_len as i32, self.config.vocab_size as i32])
                .expect("mlx op");

            // Linear temperature schedule, applied to logits.
            let t =
                params.t_min + (params.t_max - params.t_min) * (cur_step as f32 / max_steps as f32);
            let processed = logits2d.divide(Array::from_f32(t)).expect("mlx op");

            // argmax over vocab → [canvas_len]. argmax returns Int64; cast to
            // Int32 so it matches the random canvas dtype (for `where`) and the
            // host read-out below.
            let argmax_canvas = mlx_rs::ops::indexing::argmax_axis(&processed, -1, false)
                .expect("mlx op")
                .as_type::<i32>()
                .expect("mlx op");

            // Final step: output is argmax of the last processed logits; no transfer.
            if cur_step == 1 {
                final_logits = Some(argmax_canvas);
                break;
            }

            // Greedy denoiser (temperature <= 0): sampled tokens == argmax.
            let denoiser_canvas = argmax_canvas.clone();

            // Entropy per position + acceptance mask.
            let entropy = token_entropy(&processed); // [canvas_len]
            let accept = entropy_transfer_mask(&entropy, params.entropy_bound); // bool [canvas_len]

            // Re-randomize rejected positions.
            let (k_rand, k_next) = split(&key_rest);
            key_rest = k_next;
            let rand_ids = mlx_rs::random::randint::<_, i32>(
                Array::from_int(0),
                Array::from_int(self.config.vocab_size as i32),
                &[canvas_len as i32][..],
                Some(&k_rand),
            )
            .expect("mlx op");

            // current_canvas = where(accept, denoiser_canvas, rand_ids)
            let next_canvas =
                mlx_rs::ops::r#where(&accept, &denoiser_canvas, &rand_ids).expect("mlx op");
            canvas = next_canvas
                .as_slice::<i32>()
                .iter()
                .map(|&v| v as u32)
                .collect();

            // Self-conditioning for the next step = this step's processed logits.
            self_cond = Some(processed);
        }

        // Materialize the output canvas (argmax ids of the final step).
        let out = match final_logits {
            Some(a) => a,
            None => {
                // max_steps == 0 edge case: argmax the only decode we have.
                let logits = self.decode(&canvas, &encoder_cache, self_cond.as_ref());
                let l2 = logits
                    .reshape(&[canvas_len as i32, self.config.vocab_size as i32])
                    .expect("mlx op");
                mlx_rs::ops::indexing::argmax_axis(&l2, -1, false)
                    .expect("mlx op")
                    .as_type::<i32>()
                    .expect("mlx op")
            }
        };
        let ids: Vec<u32> = out.as_slice::<i32>().iter().map(|&v| v as u32).collect();

        // Trim at the first EOS / stop token.
        let eos = &params.eos_token_ids;
        match ids.iter().position(|t| eos.contains(t)) {
            Some(pos) => ids[..pos].to_vec(),
            None => ids,
        }
    }
}

// ─── Masks ────────────────────────────────────────────────────────────────────

/// Causal (optionally sliding) additive bias `[1, 1, q_len, kv_len]`.
/// Used by the encoder prefill.
fn build_causal_mask(query_len: usize, kv_len: usize, window: Option<usize>) -> Array {
    let q = query_len as i32;
    let k = kv_len as i32;
    let neg_inf = f32::NEG_INFINITY;
    let mut data = vec![0.0f32; (q * k) as usize];
    for qi in 0..q {
        let abs_q = k - q + qi;
        for ki in 0..k {
            let causal_ok = ki <= abs_q;
            let window_ok = window.is_none_or(|w| ki > abs_q - w as i32);
            if !causal_ok || !window_ok {
                data[(qi * k + ki) as usize] = neg_inf;
            }
        }
    }
    let raw = Array::from_slice(&data, &[q, k]);
    raw.reshape(&[1, 1, q, k]).expect("mlx op")
}

/// Bidirectional decoder mask `[1, 1, canvas_len, encoder_len + canvas_len]`.
///
/// Ports `_make_decoder_masks` for the `decoder_attention_mask is None` path:
/// - The canvas block attends to every canvas position (fully bidirectional).
/// - Full-attention layers (`window_prefix = None`) attend to all valid
///   encoder positions.
/// - Sliding layers (`window_prefix = Some(w)`) only attend to the last `w`
///   encoder positions.
///
/// Returns `None` when no positions need masking (everything is attended),
/// matching the reference which returns `None` for that layer type.
fn build_decoder_mask(
    canvas_len: usize,
    encoder_len: usize,
    valid_encoder_len: usize,
    window_prefix: Option<usize>,
) -> Option<Array> {
    let key_len = encoder_len + canvas_len;

    // Determine the encoder start index that is attended.
    let start = match window_prefix {
        // Full attention: attend all valid encoder positions.
        None => {
            if encoder_len == valid_encoder_len {
                // Everything valid → no mask needed.
                return None;
            }
            0
        }
        Some(w) => {
            if encoder_len == valid_encoder_len && encoder_len <= w {
                return None;
            }
            valid_encoder_len.saturating_sub(w)
        }
    };

    // Row vector over key positions: 1.0 where allowed, -inf where masked.
    let mut row = vec![0.0f32; key_len];
    for (ki, slot) in row.iter_mut().enumerate().take(encoder_len) {
        let keep = ki >= start && ki < valid_encoder_len;
        if !keep {
            *slot = f32::NEG_INFINITY;
        }
    }
    // Canvas positions (encoder_len..key_len) are always attended → stay 0.0.

    let row_arr = Array::from_slice(&row, &[1, 1, 1, key_len as i32]);
    // Broadcast across the canvas query dim.
    let mask = mlx_rs::ops::broadcast_to(&row_arr, &[1, 1, canvas_len as i32, key_len as i32])
        .expect("mlx op");
    Some(mask)
}
