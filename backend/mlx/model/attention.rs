//! Multi-head attention with grouped query attention (GQA) and RoPE.

use mlx_rs::Array;

use super::norm::RmsNorm;
use super::quantized::Weight;
use super::rope::RotaryEmbedding;

/// Grouped Query Attention layer.
///
/// Supports the GQA pattern where `num_kv_heads < num_heads`.
/// K and V heads are repeated to match the number of Q heads.
///
/// **Q/K normalisation.** Qwen3 applies RMSNorm to the projected Q and
/// K vectors before RoPE — a Qwen3-specific architectural change from
/// Qwen2. When `q_norm` / `k_norm` weights are populated (by the
/// safetensors loader), they're applied post-projection, pre-RoPE.
/// Defaults to identity (weight vector of ones) so Llama/Mistral/etc.
/// behaviour is unchanged when the loader doesn't overwrite them.
pub struct Attention {
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    /// Optional pre-RoPE RMSNorm on Q. Active iff the corresponding
    /// safetensor (`self_attn.q_norm.weight`) was loaded. Default
    /// weight is all-ones → identity, so un-loaded attention modules
    /// behave as if the norm weren't there.
    pub q_norm: Option<RmsNorm>,
    /// Same treatment for K.
    pub k_norm: Option<RmsNorm>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self::new_with_qk_norm(hidden_size, num_heads, num_kv_heads, head_dim, false, 1e-6)
    }

    /// Construct with optional Q/K RMSNorm slots. Callers that know
    /// the model is a Qwen3-family (config.model_type == "qwen3") pass
    /// `qk_norm = true`; the RMSNorm weights are still zero-initialised
    /// and must be overwritten by the safetensors loader before
    /// inference, or the first forward pass will produce zeros.
    pub fn new_with_qk_norm(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        qk_norm: bool,
        rms_norm_eps: f32,
    ) -> Self {
        // Weights are (out_features, in_features) — transposed during matmul.
        // Initialized as zeros; the safetensors loader overwrites them.
        let q_proj = Weight::plain(
            Array::zeros::<f32>(&[(num_heads * head_dim) as i32, hidden_size as i32])
                .expect("q_proj alloc"),
        );
        let k_proj = Weight::plain(
            Array::zeros::<f32>(&[(num_kv_heads * head_dim) as i32, hidden_size as i32])
                .expect("mlx op"),
        );
        let v_proj = Weight::plain(
            Array::zeros::<f32>(&[(num_kv_heads * head_dim) as i32, hidden_size as i32])
                .expect("mlx op"),
        );
        let o_proj = Weight::plain(
            Array::zeros::<f32>(&[hidden_size as i32, (num_heads * head_dim) as i32])
                .expect("mlx op"),
        );
        let q_norm = qk_norm.then(|| RmsNorm::new(head_dim, rms_norm_eps));
        let k_norm = qk_norm.then(|| RmsNorm::new(head_dim, rms_norm_eps));

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Run one attention pass.
    ///
    /// `x`: (batch, seq_len, hidden_size)
    /// `cache`: layer-level KV cache, grown along the sequence dimension.
    /// `offset`: position offset for RoPE (sum of previously cached tokens).
    ///
    /// Returns: (batch, seq_len, hidden_size)
    pub fn forward(
        &self,
        x: &Array,
        rope: &RotaryEmbedding,
        cache: &mut Option<(Array, Array)>,
        offset: usize,
    ) -> Array {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;

        // --- Linear projections (quantized or plain) ---
        let q = self.q_proj.matmul_transpose(x);
        let k = self.k_proj.matmul_transpose(x);
        let v = self.v_proj.matmul_transpose(x);

        // Reshape to (batch, seq_len, num_heads, head_dim)
        let q = q.reshape(&[batch, seq_len, nh, hd]).expect("mlx op");
        let k = k.reshape(&[batch, seq_len, nkv, hd]).expect("mlx op");
        let v = v.reshape(&[batch, seq_len, nkv, hd]).expect("mlx op");

        // Transpose to (batch, num_heads, seq_len, head_dim)
        let q = q.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let k = k.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let v = v.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");

        // --- Qwen3-style Q/K RMSNorm (pre-RoPE) ---
        // No-op on models where `q_norm` / `k_norm` is `None` (Llama,
        // Mistral, Qwen2). Qwen3 requires this normalisation — without
        // it attention scores are garbage.
        let q = match &self.q_norm {
            Some(n) => n.forward(&q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(n) => n.forward(&k),
            None => k,
        };

        // --- RoPE ---
        let q = rope.forward(&q, offset);
        let k = rope.forward(&k, offset);

        // --- GQA: repeat KV heads if needed ---
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeats = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&k, repeats);
            let v = repeat_kv(&v, repeats);
            (k, v)
        } else {
            (k, v)
        };

        // --- KV cache update ---
        let (k, v) = if let Some((prev_k, prev_v)) = cache.take() {
            // Concatenate along the sequence dimension (axis=2)
            let k = mlx_rs::ops::concatenate_axis(&[&prev_k, &k], 2).expect("mlx op");
            let v = mlx_rs::ops::concatenate_axis(&[&prev_v, &v], 2).expect("mlx op");
            (k, v)
        } else {
            (k, v)
        };
        // Store updated cache
        *cache = Some((k.clone(), v.clone()));

        // --- Scaled dot-product attention ---
        let scale = Array::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let k_t = k.transpose_axes(&[0, 1, 3, 2]).expect("mlx op");
        let mut scores = q.matmul(&k_t).expect("mlx op");
        scores = scores.multiply(&scale).expect("mlx op");

        // Causal mask: prevent attending to future positions.
        // Only needed when seq_len > 1 (prefill); during generation seq_len == 1.
        let kv_len = scores.shape()[3];
        if seq_len > 1 {
            scores = apply_causal_mask(&scores, seq_len, kv_len);
        }

        let attn_weights = mlx_rs::ops::softmax_axes(&scores, &[-1], None).expect("mlx op");
        let attn_out = attn_weights.matmul(&v).expect("mlx op");

        // --- Merge heads ---
        // (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3]).expect("mlx op");
        let hidden = nh * hd;
        let attn_out = attn_out.reshape(&[batch, seq_len, hidden]).expect("mlx op");

        // Output projection
        self.o_proj.matmul_transpose(&attn_out)
    }
}

/// Repeat KV heads to match the number of query heads for GQA.
///
/// Input: (batch, num_kv_heads, seq_len, head_dim)
/// Output: (batch, num_kv_heads * repeats, seq_len, head_dim)
fn repeat_kv(x: &Array, repeats: usize) -> Array {
    if repeats == 1 {
        return x.clone();
    }
    let shape = x.shape();
    let batch = shape[0];
    let n_kv = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // (batch, n_kv, 1, seq_len, head_dim) → broadcast → (batch, n_kv, repeats, seq_len, head_dim)
    let expanded = x
        .reshape(&[batch, n_kv, 1, seq_len, head_dim])
        .expect("mlx op");
    // Tile along the repeat axis by concatenating copies
    let refs: Vec<&Array> = (0..repeats).map(|_| &expanded).collect();
    let tiled = mlx_rs::ops::concatenate_axis(&refs, 2).expect("mlx op");
    tiled
        .reshape(&[batch, n_kv * repeats as i32, seq_len, head_dim])
        .expect("mlx op")
}

/// Apply an upper-triangular causal mask to attention scores.
///
/// Positions where the query attends to a future key get a large
/// negative penalty (large enough that softmax treats them as zero).
///
/// **Important**: we build the penalty directly as `{0, -1e9}` per
/// position rather than `(1 - mask) * -inf`. The multiplicative form
/// hits the IEEE-754 gotcha where `0 * -inf = NaN` — at *allowed*
/// positions where `inv_mask = 0`, the penalty would be NaN, and
/// `scores + NaN` propagates NaN through softmax and the rest of the
/// forward pass. That bug is invisible on Gemma-4 (its own
/// `gemma4.rs` module has its own mask) and only surfaces when a
/// generic Llama-family model (Qwen3, Mistral, etc.) runs through
/// this path — which is how task #82 caught it via the matrix sweep.
fn apply_causal_mask(scores: &Array, query_len: i32, kv_len: i32) -> Array {
    // Per-position penalty: 0 at allowed cells, -1e9 at masked cells.
    // -1e9 is large enough that `softmax(score - 1e9)` ≈ 0 for any
    // realistic score magnitude (attention scores are scaled by
    // 1/sqrt(head_dim), so they typically live in ±100 range). Using
    // a finite sentinel avoids the `0 * -inf` NaN trap.
    const MASK_PENALTY: f32 = -1.0e9;
    let mut penalty_data = vec![0.0f32; (query_len * kv_len) as usize];
    for q in 0..query_len {
        let max_k = kv_len - query_len + q;
        for k in (max_k + 1)..kv_len {
            penalty_data[(q * kv_len + k) as usize] = MASK_PENALTY;
        }
    }
    let penalty = Array::from_slice(&penalty_data, &[1, 1, query_len, kv_len]);
    scores.add(&penalty).expect("mlx op")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for task #82 — the `0 * -inf = NaN` trap.
    /// Applying the mask to any finite score tensor must produce a
    /// finite result at allowed positions; masked positions must be
    /// very negative but still finite. If a future refactor switches
    /// back to the `(1 - mask) * -inf` formulation, this test catches
    /// it before another model silently degenerates.
    #[test]
    fn causal_mask_does_not_produce_nan_at_allowed_positions() {
        // 4×4 scores with mixed signs — standard attention-score shape.
        // Reshape to (1, 1, 4, 4) to match the mask's expected rank.
        let scores = Array::from_slice(
            &[
                0.5f32, 1.0, -2.0, 3.0,
                -1.5, 0.25, 0.0, -0.75,
                2.0, -0.5, 1.25, -0.1,
                0.0, 0.0, 0.0, 0.0,
            ],
            &[1, 1, 4, 4],
        );
        let out = apply_causal_mask(&scores, 4, 4);
        let mn = mlx_rs::ops::min(&out, None).expect("mlx op").item::<f32>();
        let mx = mlx_rs::ops::max(&out, None).expect("mlx op").item::<f32>();
        assert!(
            mn.is_finite() || mn < -1.0e8,
            "mask produced non-finite/non-penalty min: {mn}"
        );
        assert!(mx.is_finite(), "mask produced NaN max: {mx}");
        // Softmax must give finite, positive sums along the last axis.
        let sm = mlx_rs::ops::softmax_axes(&out, &[-1], None).expect("mlx op");
        let sm_min = mlx_rs::ops::min(&sm, None).expect("mlx op").item::<f32>();
        let sm_max = mlx_rs::ops::max(&sm, None).expect("mlx op").item::<f32>();
        assert!(sm_min.is_finite() && sm_max.is_finite(), "softmax NaN");
        assert!(sm_max <= 1.0 + 1e-5 && sm_min >= 0.0);
    }
}
