//! MLX sampler — thin wrapper around common sampler using mlx_rs::Array.

use crate::gen2::backend::common::sampler::{DryParams, Sampler as CommonSampler, XtcParams};
use mlx_rs::Array;

pub struct Sampler {
    inner: CommonSampler,
    /// Lazily-built dense `[vocab]` f32 bias vector for the GPU-argmax fast
    /// path: zero everywhere except `+bias` at each end-of-turn id. Built once
    /// on first GPU-argmax call (vocab is known only then, from the logits
    /// shape) and reused every step. `None` once built with no eot bias active.
    gpu_eot_bias: Option<Option<Array>>,
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
            gpu_eot_bias: None,
        }
    }

    /// True when this step is a pure greedy argmax (no penalties / sampling)
    /// — the GPU-argmax fast-path eligibility gate. See
    /// [`CommonSampler::is_greedy_argmax`].
    pub fn is_greedy_argmax(&self) -> bool {
        self.inner.is_greedy_argmax()
    }

    /// GPU greedy argmax over the LAST-position logits, mirroring mlx-lm's
    /// default `sampler = lambda x: mx.argmax(x, axis=-1)`
    /// (`generate.py:386`) but with the SAME additive `eot_bias` the CPU
    /// `sample_from_logits` applies (step 4 of its pipeline), so the chosen
    /// token is byte-identical to the Stage-A CPU greedy path.
    ///
    /// `last_logits` is `[1, 1, vocab]` (single position). Returns a LAZY
    /// `[1]` int32 token-id array — NOT synced to host. The caller chains it
    /// straight into the next forward (embedding gather) so the whole decode
    /// stays on-GPU, exactly like mlx-lm feeding `y` (lazy) into `model(y[None])`
    /// (`generate.py:459`). No `as_slice` / `.item()` happens here.
    pub fn argmax_gpu(&mut self, last_logits: &Array) -> Array {
        // Build (once) the dense bias vector from eot ids, then add it. The
        // add is a 262k-wide f32 op on GPU — negligible vs the forward, and it
        // makes the GPU argmax match `argmax(logits + eot_bias)` exactly.
        let vocab = *last_logits.shape().last().expect("logits rank >= 1");
        if self.gpu_eot_bias.is_none() {
            self.gpu_eot_bias = Some(self.build_eot_bias(vocab));
        }
        let biased = match self.gpu_eot_bias.as_ref().and_then(|o| o.as_ref()) {
            Some(bias) => last_logits.add(bias).expect("mlx op: eot bias add"),
            None => last_logits.clone(),
        };
        // argmax over the vocab axis → [1, 1]; reshape to [1] as the lazy
        // next-token id fed to the next forward's embedding gather. int32 to
        // match the embedding `take_axis` index dtype used elsewhere.
        //
        // CRITICAL: run the argmax on the **CPU stream** (`Stream::cpu`). The
        // Metal/GPU argmax reduction is non-associative over the 262k vocab and
        // breaks ties differently than the default path's CPU argmax
        // (`as_slice::<f32>()` + first-max), so a GPU argmax flips the winner at
        // near-ties (~3 / 1000 tokens here). On a single 262k row that flip
        // snowballs the whole conversation (the post-Stage-A divergence: the
        // turn-1 answer drifts, the model then refuses later context recalls).
        // The CPU-stream argmax is the SAME deterministic first-max reduction
        // the serial sampler uses, so the fast path's greedy tokens stay
        // byte-identical to Stage A — and we still skip the 262k host copy
        // (MLX moves only the scalar result, the data movement is internal and
        // lazy). This is the documented "reduction order on GPU varies … near
        // argmax tiebreaks" hazard from golden.rs, resolved on-device.
        let cpu = mlx_rs::Stream::cpu();
        let idx = mlx_rs::ops::indexing::argmax_axis_device(&biased, -1, None, &cpu)
            .expect("mlx op: argmax (cpu stream)");
        idx.reshape(&[1])
            .expect("mlx op: token reshape")
            .as_dtype(mlx_rs::Dtype::Int32)
            .expect("mlx op: token int32")
    }

    /// Materialize the dense `[vocab]` f32 eot-bias vector (zeros + `bias` at
    /// each id), or `None` when no eot bias is active.
    fn build_eot_bias(&self, vocab: i32) -> Option<Array> {
        let (ids, bias) = self.inner.eot_bias()?;
        if ids.is_empty() || bias == 0.0 {
            return None;
        }
        let mut v = vec![0.0f32; vocab as usize];
        for &id in ids {
            if (id as i32) < vocab {
                v[id as usize] += bias;
            }
        }
        Some(Array::from_slice(&v, &[vocab]))
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

    /// Passthrough to [`CommonSampler::with_presence_penalty`].
    pub fn with_presence_penalty(mut self, penalty: Option<f32>) -> Self {
        self.inner = self.inner.with_presence_penalty(penalty);
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
