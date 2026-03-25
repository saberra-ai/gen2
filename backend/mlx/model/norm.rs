//! RMS normalization layer (pre-norm variant used in Llama).

use mlx_rs::Array;

/// Root-mean-square layer normalization.
///
/// Weight is initialized as ones; the loader overwrites it from safetensors.
pub struct RmsNorm {
    pub weight: Array,
    eps: f32,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let weight = Array::ones::<f32>(&[hidden_size as i32]).unwrap();
        Self { weight, eps }
    }

    /// x_norm = x * rsqrt(mean(x^2, axis=-1, keepdims=true) + eps) * weight
    pub fn forward(&self, x: &Array) -> Array {
        // x^2
        let x_sq = x.multiply(x).unwrap();

        // mean(x^2) along last axis, keepdims
        let variance = x_sq.mean_axis(-1, true).unwrap();

        // variance + eps
        let eps = Array::from_f32(self.eps);
        let var_eps = variance.add(&eps).unwrap();

        // rsqrt(variance + eps)
        let norm_factor = var_eps.rsqrt().unwrap();

        // x * rsqrt(...)
        let normalized = x.multiply(&norm_factor).unwrap();

        // normalized * weight
        normalized.multiply(&self.weight).unwrap()
    }
}
