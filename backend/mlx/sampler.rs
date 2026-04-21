//! MLX sampler — thin wrapper around common sampler using mlx_rs::Array.

use crate::gen2::backend::common::sampler::{
    DryParams, Sampler as CommonSampler, XtcParams,
};
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

    /// Passthrough to [`CommonSampler::with_eot_bias`].
    pub fn with_eot_bias(mut self, ids: Vec<u32>, bias: f32) -> Self {
        self.inner = self.inner.with_eot_bias(ids, bias);
        self
    }

    /// Passthrough to [`CommonSampler::with_min_p`].
    pub fn with_min_p(mut self, min_p: Option<f32>) -> Self {
        self.inner = self.inner.with_min_p(min_p);
        self
    }

    /// Passthrough to [`CommonSampler::with_dry`].
    pub fn with_dry(mut self, params: Option<DryParams>) -> Self {
        self.inner = self.inner.with_dry(params);
        self
    }

    /// Passthrough to [`CommonSampler::with_xtc`].
    pub fn with_xtc(mut self, params: Option<XtcParams>) -> Self {
        self.inner = self.inner.with_xtc(params);
        self
    }

    /// Sample a token ID from an MLX logits array of shape (vocab_size,).
    pub fn sample(&mut self, logits: &Array) -> u32 {
        let logits_slice: &[f32] = logits.as_slice::<f32>();
        self.inner.sample_from_logits(logits_slice)
    }

    /// Sample a token with an optional grammar mask applied pre-sampling.
    /// When `grammar` is `Some`, the logits are masked so only grammar-
    /// valid tokens remain, and the matcher is advanced with the chosen
    /// token. Falls back to identical behaviour as `sample()` when
    /// `grammar` is `None`.
    pub fn sample_with_grammar(
        &mut self,
        logits: &Array,
        grammar: Option<&mut crate::gen2::backend::common::grammar::GrammarMatcher>,
    ) -> u32 {
        let Some(g) = grammar else {
            return self.sample(logits);
        };
        let mut buf: Vec<f32> = logits.as_slice::<f32>().to_vec();
        if let Err(e) = g.apply_mask(&mut buf) {
            tracing::warn!(?e, "grammar mask application failed; falling back");
            return self.sample(logits);
        }
        let token_id = self.inner.sample_from_logits(&buf);
        if let Err(e) = g.observe(token_id) {
            tracing::warn!(?e, "grammar observe failed after sample");
        }
        token_id
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
