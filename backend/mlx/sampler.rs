//! MLX sampler — thin wrapper around common sampler using mlx_rs::Array.

use crate::gen2::backend::common::sampler::Sampler as CommonSampler;
use mlx_rs::Array;

pub struct Sampler {
    inner: CommonSampler,
}

impl Sampler {
    pub fn new(
        temperature: f32,
        top_p: Option<f32>,
        top_k: Option<i32>,
        repetition_penalty: Option<f32>,
    ) -> Self {
        Self {
            inner: CommonSampler::new(temperature, top_p, top_k, repetition_penalty),
        }
    }

    /// Sample a token ID from an MLX logits array of shape (vocab_size,).
    pub fn sample(&mut self, logits: &Array) -> u32 {
        let logits_slice: &[f32] = logits.as_slice::<f32>();
        self.inner.sample_from_logits(logits_slice)
    }

    /// Record an emitted token for the repetition-penalty window. Call once
    /// per decode step after the token has been committed.
    pub fn observe(&mut self, token: u32) {
        self.inner.observe(token);
    }
}
