//! Quantized weight support for MLX models.
//!
//! Wraps packed uint32 weights + scales + biases for 1-bit/2-bit/4-bit/8-bit
//! quantized models. Provides `matmul` (via `quantized_matmul`) and
//! `dequantize` (for embedding lookup) that work transparently alongside
//! unquantized float weights.

use mlx_rs::Array;
use mlx_rs::ops::{dequantize, quantized_matmul};

/// A weight that may be quantized (packed uint32 + scales + biases)
/// or unquantized (plain float array).
pub enum Weight {
    /// Unquantized float weight (f16 or f32).
    Plain(Array),
    /// Quantized weight: packed uint32 + per-group scales and biases.
    Quantized {
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
}

impl Weight {
    /// Create an unquantized weight.
    pub fn plain(array: Array) -> Self {
        Self::Plain(array)
    }

    /// Create a quantized weight from the (weight, scales, biases) triple.
    pub fn quantized(
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self::Quantized {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Matrix multiplication: `x @ self^T`.
    ///
    /// For plain weights: `x.matmul(&self.transpose())`.
    /// For quantized weights: `quantized_matmul(x, w, scales, biases, transpose=true)`.
    pub fn matmul_transpose(&self, x: &Array) -> Array {
        match self {
            Self::Plain(w) => {
                let wt = w.transpose_axes(&[1, 0]).expect("mlx op");
                x.matmul(&wt).expect("mlx op")
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => quantized_matmul(x, weight, scales, biases, true, *group_size, *bits)
                .expect("mlx op"),
        }
    }

    /// Dequantize to a full float array.
    ///
    /// For plain weights: returns self.
    /// For quantized weights: `dequantize(w, scales, biases, group_size, bits)`.
    pub fn to_full(&self) -> Array {
        match self {
            Self::Plain(w) => w.clone(),
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => dequantize(weight, scales, biases, *group_size, *bits).expect("mlx op"),
        }
    }

    /// Embedding lookup: gather rows by `indices` and dequantize only those.
    ///
    /// Mirrors `nn.QuantizedEmbedding` in mlx-lm — avoids materializing the
    /// full `[vocab, dim]` table, which can be enormous for per-layer
    /// embeddings (35 × 1536 columns on Gemma 4 E2B).
    ///
    /// Returns shape `[indices.shape, dim]`.
    pub fn embedding_lookup(&self, indices: &Array) -> Array {
        match self {
            Self::Plain(w) => w.take_axis(indices, 0).expect("mlx op"),
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let w_rows = weight.take_axis(indices, 0).expect("mlx op");
                let s_rows = scales.take_axis(indices, 0).expect("mlx op");
                let b_rows = biases.take_axis(indices, 0).expect("mlx op");
                dequantize(&w_rows, &s_rows, &b_rows, *group_size, *bits).expect("mlx op")
            }
        }
    }
}

impl Default for Weight {
    fn default() -> Self {
        Self::Plain(Array::zeros::<f32>(&[1, 1]).expect("mlx op"))
    }
}
