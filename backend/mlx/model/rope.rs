//! Rotary positional embedding (RoPE) for Llama-style models.

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

/// Precomputed cos/sin tables for rotary embeddings.
///
/// Frequencies follow the standard formulation:
///   freq_i = 1.0 / (theta ^ (2i / dim))  for i in 0..dim/2
pub struct RotaryEmbedding {
    /// Shape: (max_seq_len, head_dim/2)
    cos: Array,
    /// Shape: (max_seq_len, head_dim/2)
    sin: Array,
}

impl RotaryEmbedding {
    /// Standard RoPE: frequency divisor equals the rotated dim.
    /// Used by Llama / Qwen / Mistral / Gemma 4 sliding-attn / etc.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self::with_freq_divisor(head_dim, head_dim, max_seq_len, theta)
    }

    /// Proportional RoPE: rotates only the first `rotated_dim` elements of the
    /// head but uses `freq_divisor` (= full_head_dim) as the exponent denominator
    /// AND as the pairing stride. Used by Gemma 4 full-attention layers
    /// (`rope_type: "proportional"`).
    ///
    /// The non-traditional RoPE pair for dim i is (i, i + freq_divisor/2). So
    /// rotating only the first `rotated_dim` elements means rotating pairs
    /// `(0, freq_divisor/2), (1, freq_divisor/2 + 1), ..., (rotated_dim/2 - 1, ...)`
    /// — NOT pairs `(0, rotated_dim/2), ..., (rotated_dim/2 - 1, rotated_dim - 1)`.
    /// (That second, naive pairing is what we had before; it scrambles the
    /// attention geometry and produces garbage output across all full-attn
    /// layers.)
    ///
    /// Mirrors Python `ProportionalRoPE` in `mlx_lm/models/rope_utils.py`:
    /// freqs has length `freq_divisor/2`; first `rotated_dim/2` entries are real
    /// rotation rates, the rest are zero (identity rotation → pass-through).
    pub fn with_freq_divisor(
        rotated_dim: usize,
        freq_divisor: usize,
        max_seq_len: usize,
        theta: f32,
    ) -> Self {
        assert!(rotated_dim <= freq_divisor);
        let half_full = freq_divisor / 2;
        let half_rot = rotated_dim / 2;

        // freqs[i] = 1.0 / theta^(2i / freq_divisor) for i < half_rot,
        // else 0.0 (identity — cos(0)=1, sin(0)=0 → pass-through).
        let freq_data: Vec<f32> = (0..half_full)
            .map(|i| {
                if i < half_rot {
                    1.0 / theta.powf(2.0 * i as f32 / freq_divisor as f32)
                } else {
                    0.0
                }
            })
            .collect();
        let freqs = Array::from_slice(&freq_data, &[half_full as i32]);

        let t_data: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Array::from_slice(&t_data, &[max_seq_len as i32]);

        let t_col = t.reshape(&[max_seq_len as i32, 1]).expect("mlx op");
        let f_row = freqs.reshape(&[1, half_full as i32]).expect("mlx op");
        let angles = t_col.multiply(&f_row).expect("mlx op");

        let cos = angles.cos().expect("mlx op");
        let sin = angles.sin().expect("mlx op");

        Self { cos, sin }
    }

    /// Apply rotary embedding to `x` at the given sequence offset.
    ///
    /// `x` shape: (batch, num_heads, seq_len, head_dim)
    /// Returns tensor of same shape with RoPE applied.
    pub fn forward(&self, x: &Array, offset: usize) -> Array {
        let shape = x.shape();
        let seq_len = shape[2] as usize;
        let head_dim = shape[3] as usize;
        let half_dim = head_dim / 2;

        // Slice cos/sin for [offset..offset+seq_len, :] using range-based indexing
        let offset_i32 = offset as i32;
        let end_i32 = (offset + seq_len) as i32;
        let cos_slice = self.cos.index((offset_i32..end_i32, ..));
        let sin_slice = self.sin.index((offset_i32..end_i32, ..));

        // Split x into two halves along the last dimension
        let half_dim_i32 = half_dim as i32;
        let x1 = x.index((.., .., .., ..half_dim_i32));
        let x2 = x.index((.., .., .., half_dim_i32..));

        // x_rotated = concat(-x2, x1) along last dim
        let neg_x2 = x2.negative().expect("mlx op");
        let x_rotated = mlx_rs::ops::concatenate_axis(&[&neg_x2, &x1], -1).expect("mlx op");

        // Broadcast cos/sin: (seq_len, half_dim) → (1, 1, seq_len, half_dim)
        let cos_broad = cos_slice
            .reshape(&[1, 1, seq_len as i32, half_dim as i32])
            .expect("mlx op");
        let sin_broad = sin_slice
            .reshape(&[1, 1, seq_len as i32, half_dim as i32])
            .expect("mlx op");

        // Full cos/sin by repeating for both halves: (1,1,seq,half) → (1,1,seq,dim)
        let cos_full =
            mlx_rs::ops::concatenate_axis(&[&cos_broad, &cos_broad], -1).expect("mlx op");
        let sin_full =
            mlx_rs::ops::concatenate_axis(&[&sin_broad, &sin_broad], -1).expect("mlx op");

        // result = x * cos + x_rotated * sin
        let term_a = x.multiply(&cos_full).expect("mlx op");
        let term_b = x_rotated.multiply(&sin_full).expect("mlx op");
        term_a.add(&term_b).expect("mlx op")
    }
}
