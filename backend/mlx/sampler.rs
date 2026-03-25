//! MLX sampler — thin wrapper around common sampler using mlx_rs::Array.

use mlx_rs::Array;
use crate::gen2::backend::common::sampler::Sampler as CommonSampler;

pub struct Sampler {
    inner: CommonSampler,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: Option<f32>, top_k: Option<i32>) -> Self {
        Self {
            inner: CommonSampler::new(temperature, top_p, top_k),
        }
    }

    /// Sample a token ID from an MLX logits array of shape (vocab_size,).
    pub fn sample(&mut self, logits: &Array) -> u32 {
        let logits_slice: &[f32] = logits.as_slice::<f32>();
        self.inner.sample_from_logits(logits_slice)
    }
}
