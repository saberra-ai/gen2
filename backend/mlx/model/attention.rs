//! Multi-head attention with grouped query attention (GQA) and RoPE.

use mlx_rs::Array;

use super::rope::RotaryEmbedding;

/// Grouped Query Attention layer.
///
/// Supports the GQA pattern where `num_kv_heads < num_heads`.
/// K and V heads are repeated to match the number of Q heads.
pub struct Attention {
    pub q_proj: Array,
    pub k_proj: Array,
    pub v_proj: Array,
    pub o_proj: Array,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        // Weights are (out_features, in_features) — transposed during matmul.
        // Initialized as zeros; the safetensors loader overwrites them.
        let q_proj =
            Array::zeros::<f32>(&[(num_heads * head_dim) as i32, hidden_size as i32]).unwrap();
        let k_proj =
            Array::zeros::<f32>(&[(num_kv_heads * head_dim) as i32, hidden_size as i32]).unwrap();
        let v_proj =
            Array::zeros::<f32>(&[(num_kv_heads * head_dim) as i32, hidden_size as i32]).unwrap();
        let o_proj =
            Array::zeros::<f32>(&[hidden_size as i32, (num_heads * head_dim) as i32]).unwrap();

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
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

        // --- Linear projections ---
        let q_proj_t = self.q_proj.transpose_axes(&[1, 0]).unwrap();
        let k_proj_t = self.k_proj.transpose_axes(&[1, 0]).unwrap();
        let v_proj_t = self.v_proj.transpose_axes(&[1, 0]).unwrap();

        let q = x.matmul(&q_proj_t).unwrap();
        let k = x.matmul(&k_proj_t).unwrap();
        let v = x.matmul(&v_proj_t).unwrap();

        // Reshape to (batch, seq_len, num_heads, head_dim)
        let q = q.reshape(&[batch, seq_len, nh, hd]).unwrap();
        let k = k.reshape(&[batch, seq_len, nkv, hd]).unwrap();
        let v = v.reshape(&[batch, seq_len, nkv, hd]).unwrap();

        // Transpose to (batch, num_heads, seq_len, head_dim)
        let q = q.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let k = k.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let v = v.transpose_axes(&[0, 2, 1, 3]).unwrap();

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
            let k = mlx_rs::ops::concatenate_axis(&[&prev_k, &k], 2).unwrap();
            let v = mlx_rs::ops::concatenate_axis(&[&prev_v, &v], 2).unwrap();
            (k, v)
        } else {
            (k, v)
        };
        // Store updated cache
        *cache = Some((k.clone(), v.clone()));

        // --- Scaled dot-product attention ---
        let scale = Array::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let k_t = k.transpose_axes(&[0, 1, 3, 2]).unwrap();
        let mut scores = q.matmul(&k_t).unwrap();
        scores = scores.multiply(&scale).unwrap();

        // Causal mask: prevent attending to future positions.
        // Only needed when seq_len > 1 (prefill); during generation seq_len == 1.
        let kv_len = scores.shape()[3];
        if seq_len > 1 {
            scores = apply_causal_mask(&scores, seq_len, kv_len);
        }

        let attn_weights = mlx_rs::ops::softmax_axes(&scores, &[-1], None).unwrap();
        let attn_out = attn_weights.matmul(&v).unwrap();

        // --- Merge heads ---
        // (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let hidden = nh * hd;
        let attn_out = attn_out.reshape(&[batch, seq_len, hidden]).unwrap();

        // Output projection
        let o_proj_t = self.o_proj.transpose_axes(&[1, 0]).unwrap();
        attn_out.matmul(&o_proj_t).unwrap()
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
    let expanded = x.reshape(&[batch, n_kv, 1, seq_len, head_dim]).unwrap();
    // Tile along the repeat axis by concatenating copies
    let refs: Vec<&Array> = (0..repeats).map(|_| &expanded).collect();
    let tiled = mlx_rs::ops::concatenate_axis(&refs, 2).unwrap();
    tiled
        .reshape(&[batch, n_kv * repeats as i32, seq_len, head_dim])
        .unwrap()
}

/// Apply an upper-triangular causal mask to attention scores.
///
/// Positions where the query attends to a future key get -inf.
fn apply_causal_mask(scores: &Array, query_len: i32, kv_len: i32) -> Array {
    // Build a (query_len, kv_len) mask: 1 where allowed, 0 where masked
    let mut mask_data = vec![0.0f32; (query_len * kv_len) as usize];
    for q in 0..query_len {
        // For each query position, it can attend to all kv positions
        // up to (kv_len - query_len + q) inclusive.
        let max_k = kv_len - query_len + q;
        for k in 0..=max_k {
            mask_data[(q * kv_len + k) as usize] = 1.0;
        }
    }
    let mask = Array::from_slice(&mask_data, &[query_len, kv_len]);

    // Where mask == 0 → replace score with -inf
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let ones = Array::from_f32(1.0f32);

    // inverted_mask: 1 where masked, 0 where allowed
    let inv_mask = ones.subtract(&mask).unwrap();
    let penalty = inv_mask.multiply(&neg_inf).unwrap();

    // Broadcast (query_len, kv_len) across (batch, heads, query_len, kv_len)
    let penalty = penalty.reshape(&[1, 1, query_len, kv_len]).unwrap();
    scores.add(&penalty).unwrap()
}
