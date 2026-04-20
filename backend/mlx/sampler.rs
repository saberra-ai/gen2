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

    /// Tight post-answer loop detection — see [`CommonSampler::is_in_token_loop`].
    pub fn is_in_token_loop(&self, window: usize, max_unique: usize) -> bool {
        self.inner.is_in_token_loop(window, max_unique)
    }

    /// N-gram phrase-repetition detection — see [`CommonSampler::is_in_ngram_loop`].
    pub fn is_in_ngram_loop(&self, n: usize) -> bool {
        self.inner.is_in_ngram_loop(n)
    }

    /// Combined multi-size phrase-loop detector — see
    /// [`CommonSampler::is_in_any_ngram_loop`].
    pub fn is_in_any_ngram_loop(&self) -> bool {
        self.inner.is_in_any_ngram_loop()
    }

    /// Cycle-period scanner — see [`CommonSampler::is_in_cycle`].
    pub fn is_in_cycle(&self, max_period: usize) -> bool {
        self.inner.is_in_cycle(max_period)
    }
}
