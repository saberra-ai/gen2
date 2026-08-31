//! MLX token generation iterator with n-gram speculative decoding.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use mlx_rs::ops::indexing::IndexOp;
use parking_lot::Mutex;

use super::bundle::ModelBundle;
use super::sampler::Sampler;
use super::session::DecodeState;
use crate::backend::common::grammar::GrammarMatcher;
use crate::backend::common::output_filter::OutputFilter;
use crate::backend::common::speculative::{
    DEFAULT_DRAFT_LEN, SpeculativeMode, SpeculativePredictor,
};
use crate::backend::common::stop_matcher::StopMatcher;
use crate::engine::{ExecError, ExecutionStats, HookBus, HookEvent};
use crate::generation::{GenSpec, Token, TokenEvent};

/// Token generation puller.
///
/// Two shapes are interface-compatible from the caller's perspective — both
/// yield `TokenEvent::Token{..}` one at a time, then a terminal `Eos`/`Stopped`:
///
/// - [`TokenPuller::Ar`] — the autoregressive path. Runs one forward pass per
///   decode step (with speculative batching), sampling token-by-token.
/// - [`TokenPuller::Precomputed`] — the DiffusionGemma path. Generation is
///   *non-streaming* (a whole 256-token canvas is denoised to completion before
///   the puller is built); the puller then emits the precomputed ids one-by-one,
///   decoding text per token, ending with `Eos`. See [`PrecomputedPuller`].
pub enum TokenPuller {
    // Boxed: `ArPuller` is far larger than `PrecomputedPuller`, so an unboxed
    // enum would size every `TokenPuller` to the AR footprint (clippy
    // `large_enum_variant`).
    Ar(Box<ArPuller>),
    Precomputed(PrecomputedPuller),
}

impl TokenPuller {
    /// Drive one event out of whichever inner puller this is. Used by the
    /// `Iterator` impl and the `TokenPullerDyn` trait.
    fn next_inner(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        match self {
            TokenPuller::Ar(p) => p.next(),
            TokenPuller::Precomputed(p) => p.next(),
        }
    }

    /// Test-only: read execution stats. For the precomputed path this reports
    /// the canvas/output token counts; for AR it forwards `stats_now`.
    #[cfg(test)]
    pub(super) fn snapshot_stats(&self) -> ExecutionStats {
        match self {
            TokenPuller::Ar(p) => p.stats_now(),
            TokenPuller::Precomputed(p) => p.stats_now(),
        }
    }
}

impl Iterator for TokenPuller {
    type Item = Result<TokenEvent, ExecError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_inner()
    }
}

impl crate::backend::traits::TokenPullerDyn for TokenPuller {
    fn next_event(
        &mut self,
    ) -> Option<Result<crate::generation::TokenEvent, crate::engine::ExecError>> {
        self.next_inner()
    }
}

// ─── PrecomputedPuller ──────────────────────────────────────────────────────────

/// Streams a pre-generated list of token ids one-by-one as `TokenEvent::Token`,
/// decoding text per token, then terminates with `Eos`.
///
/// DiffusionGemma denoises a whole canvas at once (not token-by-token), so its
/// generation is run to completion *before* this puller is built (see
/// `Session::pull`). This puller exists only to expose that result through the
/// same `TokenEvent` interface AR models use, so callers (controller, app) are
/// unchanged. Compute is non-streaming; emission is sequential.
pub struct PrecomputedPuller {
    session_id: u64,
    hooks: Arc<HookBus>,
    bundle: Arc<ModelBundle>,
    tokens: VecDeque<u32>,
    prompt_tokens: usize,
    produced: usize,
    start_us: u64,
    first_token_us: Option<u64>,
    done: bool,
}

impl PrecomputedPuller {
    pub(crate) fn new(
        session_id: u64,
        hooks: Arc<HookBus>,
        bundle: Arc<ModelBundle>,
        tokens: Vec<u32>,
        prompt_tokens: usize,
    ) -> Self {
        Self {
            session_id,
            hooks,
            bundle,
            tokens: tokens.into_iter().collect(),
            prompt_tokens,
            produced: 0,
            start_us: now_us(),
            first_token_us: None,
            done: false,
        }
    }

    fn stats_now(&self) -> ExecutionStats {
        let elapsed_us = now_us().saturating_sub(self.start_us);
        let elapsed_s = (elapsed_us as f64) / 1_000_000.0;
        let avg_tps = if elapsed_s > 0.0 {
            (self.produced as f64 / elapsed_s) as f32
        } else {
            0.0
        };
        ExecutionStats {
            prompt_tokens: self.prompt_tokens as u32,
            decode_tokens: self.produced as u32,
            first_token_us: self.first_token_us.unwrap_or(0),
            avg_tps,
            ..Default::default()
        }
    }

    fn next(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        if self.done {
            return None;
        }
        match self.tokens.pop_front() {
            Some(id) => {
                if self.first_token_us.is_none() {
                    self.first_token_us = Some(now_us().saturating_sub(self.start_us));
                }
                let text = self.bundle.tokenizer.decode(&[id]).unwrap_or_default();
                self.produced += 1;
                self.hooks.emit(HookEvent::DecodeStep {
                    session_id: self.session_id,
                    token_id: id,
                    text_len: text.len(),
                });
                Some(Ok(TokenEvent::Token(Token {
                    id,
                    text,
                    logprob: None,
                })))
            }
            None => {
                self.done = true;
                let stats = self.stats_now();
                self.hooks.emit(HookEvent::FinalStats {
                    session_id: self.session_id,
                    stats,
                });
                Some(Ok(TokenEvent::Eos))
            }
        }
    }
}

pub struct ArPuller {
    session_id: u64,
    hooks: Arc<HookBus>,
    bundle: Arc<ModelBundle>,

    state: Option<DecodeState>,
    sampler: Sampler,
    /// Swappable speculative predictor (n-gram / PLD / Hybrid / Off).
    /// Selected via `GenSpec.speculative` or the `PIO_MLX_SPEC_MODE` env
    /// override — see `common/speculative.rs` for options.
    predictor: Box<dyn SpeculativePredictor>,

    prompt_tokens: usize,
    produced: usize,
    max_tokens: Option<usize>,
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    start_us: u64,
    first_token_us: Option<u64>,
    done: bool,

    /// Cumulative draft tokens submitted across all speculative batches.
    spec_drafted: usize,
    /// Cumulative draft tokens accepted by the target model.
    spec_accepted: usize,

    /// Shared cross-backend token-emit filter: runs every sampled token
    /// through the stop-matcher with hold-queue semantics, owns the pending
    /// event queue that the Iterator drains, and is the one place a
    /// terminal-stop (max_tokens / loop detector / explicit stop) pushes
    /// Eos from. Same helper used by the llama and ONNX backends.
    filter: OutputFilter,

    /// Optional grammar-constrained sampler. When set, every step's
    /// logits are masked so only grammar-valid tokens remain sampleable;
    /// the matcher is then advanced with the chosen token. Same
    /// llguidance-backed engine the llama backend uses — grammars that
    /// work there work here.
    grammar: Option<GrammarMatcher>,

    /// Target-layer ids the active predictor wants aux hidden states
    /// from. Populated at construction from `predictor.aux_layer_ids()`;
    /// empty for token-only predictors (Ngram / PLD / Lookahead / Off).
    /// When non-empty, the puller routes forward calls through
    /// `Model::forward_all_with_aux` and stashes the aux states.
    aux_layer_ids: Vec<usize>,
    /// Aux hidden states at the last generated position (one per layer
    /// in `aux_layer_ids`, each [1, 1, H]). Populated by the forward
    /// pass, consumed by the next `draft_with_context()` call.
    pending_aux: Option<Vec<mlx_rs::Array>>,

    state_slot: Weak<Mutex<Option<DecodeState>>>,

    /// Stage-B fast-path pipeline state. `Some` once the pipelined GPU-argmax
    /// decode loop has been entered (see [`ArPuller::try_step_fast_pipeline`]);
    /// holds the LAZY token id whose host work (EOS/decode/emit) we still owe.
    /// The forward+argmax for the FOLLOWING token has already been built and
    /// `async_eval`'d before we sync this one — mirroring generate_step's
    /// double-buffer (`generate.py:457-469`). `None` ⇒ not pipelining (flag
    /// off, non-greedy, grammar, aux, or `PIO_MLX_PIPELINE=0`).
    fast_pipe: Option<FastPipe>,

    /// DIAGNOSTIC fixed-workload gate (`PIO_MLX_FIXED_STEPS=N`, read once at
    /// construction). When `Some(n)`, decode EXACTLY `n` forward passes for
    /// timing, IGNORING every early-termination path (EOS / stop-id / loop
    /// detector). Correctness is irrelevant under this gate — its only purpose
    /// is a trustworthy ms/token denominator (`decode_tokens == n`) that is
    /// immune to garbage-model EOS/loop stops confounding ablations. Unset in
    /// normal operation (`None` ⇒ all stop conditions live, default behaviour
    /// byte-identical). Honoured by BOTH the serial and the fast-pipeline path.
    fixed_steps: Option<usize>,

    /// Dedicated GPU generation stream for the Stage-B pipeline, created once
    /// per puller (lazily on first pipeline entry). Mirrors mlx-lm's module
    /// `generation_stream = mx.new_stream(mx.default_device())`
    /// (`generate.py:226`): the fast-path forward's argmax runs on THIS stream
    /// so the `forward → argmax → next forward` chain stays on a single GPU
    /// stream with no cross-stream CPU dependency, keeping the GPU fed (the
    /// inter-token-stall fix). `None` until the pipeline is first entered.
    gen_stream: Option<mlx_rs::Stream>,
}

/// Double-buffered lazy decode state for the Stage-B pipeline. Mirrors
/// generate_step's `y` / `next_y` pair (`generate.py:457-469`): we hold token
/// `y` (lazy, not yet synced) while step N+1's forward consuming `y` has
/// already been dispatched and `async_eval`'d.
struct FastPipe {
    /// The lazy `[1]` int32 id of token N — the one we still owe host work for.
    y: mlx_rs::Array,
    /// True once step N+1's `forward(y) → logits → argmax → next_y` has been
    /// built + `async_eval`'d. When set, `next` holds `(next_y, next_logits)`.
    /// Only false for the very first decode token (N=0), whose N+1 hasn't been
    /// scheduled yet at first entry.
    has_next: bool,
    /// `(next_y, _next_logits)` for token N+1, already dispatched on GPU. The
    /// logits are kept only to keep the eval alive / for parity with mlx-lm's
    /// `logprobs` (unused for greedy emit).
    next_y: Option<mlx_rs::Array>,
}

impl ArPuller {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session_id: u64,
        hooks: Arc<HookBus>,
        bundle: Arc<ModelBundle>,
        state_slot: Weak<Mutex<Option<DecodeState>>>,
        state: DecodeState,
        gen_spec: GenSpec,
        paused: Arc<AtomicBool>,
        stopped: Arc<AtomicBool>,
    ) -> Self {
        let temperature = gen_spec.temperature.unwrap_or(0.7);
        // Defaults — caller can override any of these via `GenSpec`:
        //   top_p = 0.9 (Llama/Gemma/Qwen reference default)
        //   top_k = None
        //   min_p = None (off; set via GenSpec)
        //   DRY/XTC = None (off; set via GenSpec)
        //   eot_bias = 2.0 (calibrated for Gemma 4 26B-4bit; see below)
        //
        // `<turn|>` ranks #2 at answer boundaries with only a ~0.6 logit
        // gap to the winning `\n` on Gemma 4 26B-4bit (verified against
        // mlx-lm's golden reference). A +2.0 bias tips EOT over the line
        // at boundaries without affecting mid-sentence sampling (gap is
        // several logits wide there).
        let env_eot_bias: Option<f32> = std::env::var("PIO_MLX_EOT_BIAS")
            .ok()
            .and_then(|s| s.parse().ok());
        let eot_bias = gen_spec.eot_bias.or(env_eot_bias).unwrap_or(2.0);

        let dry_params = gen_spec.dry_multiplier.map(|m| {
            use crate::backend::common::sampler::DryParams;
            DryParams {
                multiplier: m,
                base: gen_spec.dry_base.unwrap_or(1.75),
                allowed_length: gen_spec.dry_allowed_length.unwrap_or(2),
            }
        });
        let xtc_params = gen_spec.xtc_probability.map(|p| {
            use crate::backend::common::sampler::XtcParams;
            XtcParams {
                probability: p,
                threshold: gen_spec.xtc_threshold.unwrap_or(0.1),
            }
        });

        let sampler = Sampler::new(
            temperature,
            Some(gen_spec.top_p.unwrap_or(0.9)),
            gen_spec.top_k,
            gen_spec.penalty_repeat,
        )
        .with_eot_bias(bundle.tokenizer.stop_ids().to_vec(), eot_bias)
        .with_min_p(gen_spec.min_p)
        .with_presence_penalty(gen_spec.penalty_present)
        .with_dry(dry_params)
        .with_xtc(xtc_params);

        let grammar: Option<GrammarMatcher> = gen_spec.grammar.clone().and_then(|spec| {
            match GrammarMatcher::new(&bundle.tokenizer, spec) {
                Ok(g) => Some(g),
                Err(e) => {
                    tracing::warn!("grammar build failed, continuing unconstrained: {e:?}");
                    None
                }
            }
        });

        // Resolve speculative mode: explicit `GenSpec.speculative` wins,
        // else the `PIO_MLX_SPEC_MODE` env override, else default (Lookahead).
        let spec_mode = gen_spec
            .speculative
            .clone()
            .or_else(|| {
                std::env::var("PIO_MLX_SPEC_MODE")
                    .ok()
                    .and_then(|s| SpeculativeMode::from_str_opt(&s))
            })
            .unwrap_or_default();
        // EAGLE-3 gets backend-specific handling: load the draft model
        // from disk and wrap in `EagleDraftPredictor` (which consumes
        // the target's aux hidden states). Other modes use the common
        // `build()` path.
        let predictor: Box<dyn SpeculativePredictor> = match &spec_mode {
            SpeculativeMode::Eagle3 { model_path } => {
                match super::eagle3_loader::load_from_dir(std::path::Path::new(model_path)) {
                    Ok(draft) => {
                        tracing::info!(model_path, "EAGLE-3 draft model loaded");
                        Box::new(super::eagle_predictor::EagleDraftPredictor::new(draft))
                    }
                    Err(e) => {
                        tracing::warn!(
                            ?e,
                            model_path,
                            "EAGLE-3 load failed; falling back to Lookahead"
                        );
                        SpeculativeMode::Lookahead.build()
                    }
                }
            }
            _ => spec_mode.clone().build(),
        };
        let predictor_aux_layer_ids: Vec<usize> = predictor.aux_layer_ids().to_vec();

        Self {
            session_id,
            hooks,
            bundle,
            prompt_tokens: state.cur_pos,
            state: Some(state),
            sampler,
            predictor,
            produced: 0,
            max_tokens: gen_spec.max_tokens,
            paused,
            stopped,
            start_us: now_us(),
            first_token_us: None,
            done: false,
            spec_drafted: 0,
            spec_accepted: 0,
            filter: OutputFilter::new(StopMatcher::gemma4_chat_defaults()),
            grammar,
            aux_layer_ids: predictor_aux_layer_ids,
            pending_aux: None,
            state_slot,
            fast_pipe: None,
            fixed_steps: std::env::var("PIO_MLX_FIXED_STEPS")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .filter(|&n| n > 0),
            gen_stream: None,
        }
    }

    /// Lazily create + return the dedicated GPU generation stream for the
    /// pipeline argmax. Mirrors mlx-lm's `generation_stream` (a GPU stream).
    /// We use the **default GPU stream** (`Stream::gpu()`) so the pipeline's
    /// argmax runs on the SAME GPU stream the fast forward already uses — the
    /// `forward → argmax → next forward` chain stays on one GPU stream with no
    /// cross-stream CPU dependency (the inter-token-stall fix). `PIO_MLX_GEN_STREAM=new`
    /// instead allocates a fresh GPU stream (`mx.new_stream`) for an A/B.
    fn gpu_gen_stream(&mut self) -> mlx_rs::Stream {
        if self.gen_stream.is_none() {
            let use_new = std::env::var("PIO_MLX_GEN_STREAM")
                .map(|v| v.eq_ignore_ascii_case("new"))
                .unwrap_or(false);
            let s = if use_new {
                mlx_rs::Stream::new_with_device(&mlx_rs::Device::gpu())
            } else {
                mlx_rs::Stream::gpu()
            };
            self.gen_stream = Some(s);
        }
        self.gen_stream.as_ref().expect("gen_stream set").clone()
    }

    /// The pipeline's greedy argmax. Defaults to the GPU generation stream (the
    /// inter-token-stall fix — keeps the decode chain on one GPU stream). The
    /// `PIO_MLX_ARGMAX_STREAM=cpu` A/B knob reverts to the prior CPU-stream
    /// argmax so the stall-fix's tok/s delta can be measured in isolation.
    fn pipeline_argmax(&mut self, logits: &mlx_rs::Array) -> mlx_rs::Array {
        let cpu = std::env::var("PIO_MLX_ARGMAX_STREAM")
            .map(|v| v.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        if cpu {
            self.sampler.argmax_gpu(logits)
        } else {
            let stream = self.gpu_gen_stream();
            self.sampler.argmax_gpu_on_stream(logits, &stream)
        }
    }

    /// Seed the speculative predictor with the session's prompt tokens.
    /// Only PLD / Hybrid use this; n-gram / Off ignore it. Call once
    /// immediately after construction when the prompt tokens are in hand.
    /// Retained for the PLD seeding path; not invoked on the default decode
    /// route yet.
    #[allow(dead_code)]
    pub(crate) fn seed_predictor(&mut self, prompt: &[u32]) {
        self.predictor.seed_prompt(prompt);
    }

    fn stats_now(&self) -> ExecutionStats {
        let elapsed_us = now_us().saturating_sub(self.start_us);
        let elapsed_s = (elapsed_us as f64) / 1_000_000.0;
        let avg_tps = if elapsed_s > 0.0 {
            (self.produced as f64 / elapsed_s) as f32
        } else {
            0.0
        };
        let (cache_tokens, cache_budget, evictions) = self
            .state
            .as_ref()
            .map(|s| {
                (
                    s.cache_len as u32,
                    s.policy.evict_trigger as u32,
                    s.evictions,
                )
            })
            .unwrap_or_default();
        ExecutionStats {
            reasoning_ms: None,
            prompt_tokens: self.prompt_tokens as u32,
            decode_tokens: self.produced as u32,
            first_token_us: self.first_token_us.unwrap_or(0),
            avg_tps,
            cache_tokens,
            cache_budget,
            evictions,
            spec_drafted: self.spec_drafted as u32,
            spec_accepted: self.spec_accepted as u32,
        }
    }
}

impl Drop for ArPuller {
    fn drop(&mut self) {
        if let Some(slot) = self.state_slot.upgrade()
            && let Some(state) = self.state.take()
        {
            *slot.lock() = Some(state);
        }
    }
}

impl ArPuller {
    /// True while the diagnostic fixed-steps gate (`PIO_MLX_FIXED_STEPS=N`) is
    /// active AND we still owe forwards (`produced < n`). When this holds,
    /// callers suppress every early-termination branch (EOS / stop-id / loop)
    /// so decode runs the full fixed workload for clean timing. Returns false
    /// when the gate is unset (`None`) — normal behaviour, all stops live.
    #[inline]
    fn fixed_steps_active(&self) -> bool {
        self.fixed_steps.map(|n| self.produced < n).unwrap_or(false)
    }

    /// The effective hard token budget for this step: the smaller of the user's
    /// `max_tokens` and the diagnostic `fixed_steps` cap. Under the fixed-steps
    /// gate this guarantees decode stops at EXACTLY `n` forwards (the gate's
    /// whole point — a trustworthy `decode_tokens == n` denominator), even when
    /// the caller passed a larger / no `max_tokens`.
    #[inline]
    fn effective_max_tokens(&self) -> Option<usize> {
        match (self.max_tokens, self.fixed_steps) {
            (Some(m), Some(n)) => Some(m.min(n)),
            (Some(m), None) => Some(m),
            (None, Some(n)) => Some(n),
            (None, None) => None,
        }
    }

    /// Pull one event, draining held filter events and stepping the decode
    /// loop until a token / terminal event is produced. Mirrors the old
    /// `Iterator::next` body — the enum `TokenPuller` delegates here.
    fn next(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        loop {
            if let Some(ev) = self.filter.pop() {
                return Some(ev);
            }
            if self.done || self.filter.is_done() {
                return None;
            }
            self.step_once();
        }
    }

    /// True when this step may run the Stage-B pipelined GPU-argmax decode.
    /// Requires the MLX fast path AND a pure greedy argmax with nothing that
    /// depends on host-side state or per-position masking:
    ///   - `bundle.model.is_fast()` (FAST PATH ONLY),
    ///   - greedy argmax sampler (no penalties — see `is_greedy_argmax`),
    ///   - no grammar (mask-per-step is CPU work),
    ///   - no aux hidden states (EAGLE-3 needs them stashed per step),
    ///   - `PIO_MLX_PIPELINE=1` (OPT-IN — see below).
    ///
    /// Speculative drafting is bypassed here: the pipeline is single-token but
    /// samples the next token on-GPU (`mlx_rs::ops::argmax` on the CPU stream)
    /// and, with `PIO_MLX_PIPELINE_ASYNC=1`, hides the per-token host sync
    /// behind the next forward (mlx-lm's `generate_step` double-buffer).
    ///
    /// WHY OPT-IN (not default): the async double-buffer (`generate.py:457-469`)
    /// is the real throughput win (~+20% on this machine: 24.9 → 29.9 tok/s),
    /// but leaving the decode chain lazy across steps lets MLX fuse the
    /// multi-step graph in a reduction order that differs from the serial
    /// path's per-token-evaluated order. On greedy near-ties (~3 / 1000 tokens)
    /// the winner flips, and on a multi-turn chat that single flip snowballs
    /// (verified: the async chain stops recalling user context on turns
    /// 5/6/11). That's a *valid* greedy trajectory — mlx-lm's own `mx.argmax`
    /// async loop has the same property — but it is NOT byte-identical to the
    /// frozen Stage-A goldens, so it cannot be the default. Flag OFF / default
    /// fast path keep the unchanged serial CPU-sampling decode (Stage A), which
    /// the goldens lock. Opt into the pipeline with `PIO_MLX_PIPELINE=1`.
    fn fast_pipeline_eligible(&self) -> bool {
        if !self.bundle.model.is_fast() {
            return false;
        }
        if !self.sampler.is_greedy_argmax() {
            return false;
        }
        if self.grammar.is_some() || !self.aux_layer_ids.is_empty() {
            return false;
        }
        std::env::var("PIO_MLX_PIPELINE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("on"))
            .unwrap_or(false)
    }

    /// Stage-B pipelined GPU-argmax decode step (FAST PATH ONLY).
    ///
    /// Faithfully mirrors mlx-lm's `generate_step` loop (`generate.py:455-470`),
    /// adapted to the puller's one-token-per-call shape via the [`FastPipe`]
    /// double-buffer:
    ///
    /// generate_step does, per iteration:
    ///   `next_y, _ = _step(y)`            # build forward(y)+argmax for N+1
    ///   `mx.async_eval(next_y, ...)`      # dispatch it, don't sync
    ///   `yield y.item()`                  # NOW sync token N to host + emit
    ///   `y = next_y`
    ///
    /// So step N+1's forward (consuming the LAZY token y from step N) is built
    /// and dispatched BEFORE token N is synced — the host work for N overlaps
    /// N+1's GPU compute. `_step` itself samples on GPU (argmax) and returns a
    /// lazy token (`generate.py:421-422`), never `as_slice`-ing the 262k vocab.
    ///
    /// Returns `true` once it has handled the step (emitted a token or set a
    /// terminal). The first call bootstraps `y` from the prefill's pending
    /// logits.
    fn step_once_fast_pipeline(&mut self) -> bool {
        // Bootstrap: turn the prefill (or first) pending logits into the first
        // lazy token y (= token N=0), via GPU argmax. No host sync yet.
        if self.fast_pipe.is_none() {
            let pending = {
                let state = match self.state.as_mut() {
                    Some(s) => s,
                    None => {
                        self.done = true;
                        self.filter
                            .push_err(ExecError::InvalidArg("state already consumed"));
                        return true;
                    }
                };
                match state.pending_logits.take() {
                    Some(p) => p,
                    None => {
                        // No pending logits to seed from — let the serial path
                        // handle this (shouldn't happen on decode entry).
                        return false;
                    }
                }
            };
            // Greedy argmax over the last-position logits → lazy [1] int32
            // token, on the GPU generation stream (mlx-lm runs its sampler on
            // `generation_stream`, NOT the CPU stream — see `pipeline_argmax`).
            // pending is [1,1,vocab]; argmax reduces the vocab axis.
            let y = self.pipeline_argmax(&pending);
            mlx_rs::transforms::async_eval([&y]).ok();
            self.fast_pipe = Some(FastPipe {
                y,
                has_next: false,
                next_y: None,
            });
        }

        // 1) Build + async_eval step N+1 BEFORE syncing token N — but only if
        //    we have budget left for another token. This is generate_step's
        //    `if n != max_tokens: next_y = _step(y); async_eval(next_y)`
        //    (`generate.py:458-460`).
        let want_next = self
            .effective_max_tokens()
            .map(|m| self.produced + 1 < m)
            .unwrap_or(true);
        let pipe_has_next = self.fast_pipe.as_ref().map(|p| p.has_next).unwrap_or(false);
        // `built_next` ⇒ this call ran `fast_build_next`, which wrote token N to
        // the KV cache and advanced `cur_pos`/`cache_len` by one. Needed for the
        // EOS rollback below (the serial path never caches the EOS token).
        let built_next = want_next && !pipe_has_next;
        if built_next && let Err(e) = self.fast_build_next() {
            self.done = true;
            self.filter.push_err(e);
            return true;
        }

        // 2) Sync token N to host (single scalar `.item()`, NOT a 262k
        //    as_slice) and do all host work — EOS / stop / loop / decode /
        //    emit — while N+1 is already running on GPU.
        let token_id = {
            let pipe = self.fast_pipe.as_ref().expect("fast_pipe set above");
            // Single-scalar host sync — generate_step's `y.item()`
            // (`generate.py:466`). The lazy token was async_eval'd when it was
            // produced, so this just reads the already-computed scalar.
            pipe.y.item::<i32>() as u32
        };

        // Record the committed token. The KV-cache offset advance for this
        // token already happened inside `fast_build_next` (the forward that
        // consumed `y` wrote it at `cur_pos` and bumped the offsets) — see the
        // invariant there. So here we only set `last_token`.
        if let Some(state) = self.state.as_mut() {
            state.last_token = token_id;
        }

        // DIAGNOSTIC: under the fixed-steps gate, suppress EOS / stop-id / loop
        // termination so the pipeline decodes the full fixed workload (see the
        // `fixed_steps` field). `effective_max_tokens` still stops us at `n`.
        let suppress_stops = self.fixed_steps_active();

        // EOS / stop-id check.
        if !suppress_stops && self.bundle.tokenizer.stop_ids().contains(&token_id) {
            // The serial path never writes the EOS token to the KV cache (it
            // returns before its `cur_pos += 1`). Our pipeline built N+1
            // eagerly for overlap, which advanced `cur_pos`/`cache_len` for
            // this EOS token in `fast_build_next`. Roll that one position back
            // so the post-turn `cur_pos` (= prefix_len for next turn's
            // delta-prefill) is byte-identical to the serial path. The phantom
            // cache write past `cur_pos` is harmless — next turn's delta
            // forward overwrites it from `prefix_len`.
            if built_next && let Some(state) = self.state.as_mut() {
                state.cur_pos = state.cur_pos.saturating_sub(1);
                state.cache_len = state.cache_len.saturating_sub(1);
            }
            let stats = self.stats_now();
            self.hooks.emit(HookEvent::FinalStats {
                session_id: self.session_id,
                stats,
            });
            self.done = true;
            self.filter.finalize(TokenEvent::Eos);
            return true;
        }

        // Loop detectors operate on the observed-token window.
        self.sampler.observe(token_id);
        self.predictor.observe(token_id);
        if !suppress_stops && (self.sampler.is_in_cycle(48) || self.sampler.is_in_token_loop(16, 2))
        {
            // Same rollback rationale as the EOS branch: serial finalizes on a
            // loop hit before committing this token's cache position.
            if built_next && let Some(state) = self.state.as_mut() {
                state.cur_pos = state.cur_pos.saturating_sub(1);
                state.cache_len = state.cache_len.saturating_sub(1);
            }
            let stats = self.stats_now();
            self.hooks.emit(HookEvent::FinalStats {
                session_id: self.session_id,
                stats,
            });
            self.done = true;
            self.filter.finalize(TokenEvent::Eos);
            return true;
        }

        let text = self
            .bundle
            .tokenizer
            .decode(&[token_id])
            .unwrap_or_default();

        if self.first_token_us.is_none() {
            self.first_token_us = Some(now_us().saturating_sub(self.start_us));
        }
        self.produced += 1;

        self.hooks.emit(HookEvent::DecodeStep {
            session_id: self.session_id,
            token_id,
            text_len: text.len(),
        });

        // 3) Swap buffers: y = next_y (generate.py:469). If there was no next
        //    (budget exhausted), the pipeline is drained — the next call hits
        //    the max_tokens guard in `step_once` and finalizes.
        if let Some(pipe) = self.fast_pipe.as_mut()
            && let Some(next_y) = pipe.next_y.take()
        {
            pipe.y = next_y;
            pipe.has_next = false;
        }

        self.filter.push_token(token_id, text);
        true
    }

    /// Build step N+1: run the fast forward consuming the LAZY token `y`
    /// (embedding gather on the lazy id — never synced to host), GPU-argmax the
    /// last-position logits into `next_y`, and `async_eval` it. Mirrors
    /// `next_y, _ = _step(y); mx.async_eval(next_y, ...)` (`generate.py:459-460`).
    fn fast_build_next(&mut self) -> Result<(), ExecError> {
        let bundle = &self.bundle;
        let (y, pos) = {
            let pipe = self.fast_pipe.as_ref().expect("fast_pipe set");
            let state = self.state.as_ref().expect("state present");
            (pipe.y.clone(), state.cur_pos)
        };

        // Forward consuming the token id → last-position logits [1,1,vocab].
        // The cache is advanced by one position inside the fast forward (step
        // buffer write at `pos`).
        //
        // The pipeline (opt-in via PIO_MLX_PIPELINE=1) keeps the decode chain
        // LAZY by default: `y` flows straight into the next forward's embedding
        // gather, and the per-token host sync is hidden behind that forward —
        // the mlx-lm `generate_step` double-buffer that delivers the throughput
        // win. This intentionally accepts the near-tie trajectory shift vs the
        // synchronous Stage-A goldens (see `fast_pipeline_eligible`).
        //
        // `PIO_MLX_PIPELINE_ASYNC=0` forces a per-step `.item()` sync instead:
        // it materializes `y` to a host scalar (forcing the prior forward —
        // incl. its KV writes — to evaluate in the serial path's canonical
        // reduction order), yielding Stage-A-identical greedy tokens at the cost
        // of the overlap (≈ Stage A speed). Useful for a deterministic-vs-serial
        // A/B without flipping the whole fast flag.
        let async_overlap = std::env::var("PIO_MLX_PIPELINE_ASYNC")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("off")))
            .unwrap_or(true);
        let y_in = if async_overlap {
            y.clone()
        } else {
            let t = y.item::<i32>();
            mlx_rs::Array::from_slice(&[t], &[1])
        };
        let state = self.state.as_mut().expect("state present");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bundle
                .model
                .forward_fast_last_logits_from_ids(&y_in, 1, pos, &mut state.cache)
                .expect("fast path returns logits")
        }));
        let logits = match result {
            Ok(l) => l,
            Err(e) => return Err(super::classify_forward_panic(e)),
        };

        // Greedy argmax (+ eot bias) on the GPU generation stream → lazy next
        // token. mlx-lm runs its sampler on `generation_stream` (the SAME GPU
        // stream as the forward), so the `forward → argmax → next forward`
        // chain never leaves the GPU and the GPU stays fed between tokens. The
        // prior CPU-stream argmax (`Sampler::argmax_gpu`) forced a GPU→CPU→GPU
        // hop every token — the ~6.2ms inter-token GPU-idle stall the trace
        // showed. We async_eval the token + logits so the forward's cache
        // writes are materialized and the next step's embedding gather can
        // start while we sync this step's token to host (`generate.py:459-460`).
        let next_y = self.pipeline_argmax(&logits);
        mlx_rs::transforms::async_eval([&next_y, &logits]).ok();

        // INVARIANT: each `fast_build_next` ran exactly one forward that wrote
        // the token `y` at position `cur_pos`. Advance the offsets by one now —
        // byte-identical to the serial path's `cur_pos += 1; cache_len += 1;
        // maybe_evict()` for that token (puller serial path, ~line 774-776).
        if let Some(state) = self.state.as_mut() {
            state.cur_pos += 1;
            state.cache_len += 1;
            state.maybe_evict();
        }

        let pipe = self.fast_pipe.as_mut().expect("fast_pipe set");
        pipe.next_y = Some(next_y);
        pipe.has_next = true;
        Ok(())
    }

    /// Run a single decode step — drives forward pass, samples, routes
    /// emitted tokens through `self.filter`. Sets `self.done` on terminal
    /// conditions and pushes the appropriate terminal event into the
    /// filter so the outer loop sees it.
    fn step_once(&mut self) {
        if self.done {
            return;
        }
        if self.stopped.load(Ordering::SeqCst) {
            let stats = self.stats_now();
            self.hooks.emit(HookEvent::FinalStats {
                session_id: self.session_id,
                stats,
            });
            self.done = true;
            self.filter.finalize(TokenEvent::Stopped);
            return;
        }
        if self.paused.load(Ordering::SeqCst) {
            self.filter.push_event(TokenEvent::Paused);
            return;
        }
        if let Some(limit) = self.effective_max_tokens()
            && self.produced >= limit
        {
            let stats = self.stats_now();
            self.hooks.emit(HookEvent::FinalStats {
                session_id: self.session_id,
                stats,
            });
            self.done = true;
            self.filter.finalize(TokenEvent::Eos);
            return;
        }

        // ── Stage-B fast pipeline (FAST PATH ONLY) ───────────────────────────
        // When the MLX fast path is active AND this step is a pure greedy
        // argmax (no grammar / penalties / aux), run the pipelined GPU-argmax
        // + async_eval decode loop instead of the serial CPU-sampling path.
        // Everything below this point is the unchanged default/serial path.
        if self.fast_pipeline_eligible() && self.step_once_fast_pipeline() {
            return;
        }

        // Snapshot gate-derived budget values BEFORE borrowing `state` mutably
        // below (these read only copyable `self` fields; computing them while
        // `state` is borrowed would be a double-borrow).
        let eff_max_tokens = self.effective_max_tokens();
        let suppress_stops = self.fixed_steps_active();
        let fixed_steps_set = self.fixed_steps.is_some();

        let state = match self.state.as_mut() {
            Some(s) => s,
            None => {
                self.done = true;
                self.filter
                    .push_err(ExecError::InvalidArg("state already consumed"));
                return;
            }
        };

        // Get logits: consume pending prefill logits on the first step,
        // then run one forward pass per decode step feeding the last sampled token.
        let logits = if let Some(pending) = state.pending_logits.take() {
            pending
        } else {
            // ── Speculative batched decode ────────────────────────────────────────
            // Ask the swappable predictor for a draft, cap by remaining
            // token budget (leaving room for the bonus). `PIO_MLX_SPEC=0`
            // forces single-token decode; grammar constraints also
            // disable speculative (mask-per-position is non-trivial).
            let remaining = eff_max_tokens.map(|m| m.saturating_sub(self.produced));
            // Force single-token decode under the fixed-steps gate so #forwards
            // == #emitted tokens exactly (speculative commits a variable batch
            // per forward, which would make `decode_tokens` an untrustworthy
            // ms/token denominator — the original ablation-confounding bug).
            let spec_off = self.grammar.is_some()
                || fixed_steps_set
                || std::env::var("PIO_MLX_SPEC")
                    .map(|v| v == "0" || v.eq_ignore_ascii_case("off"))
                    .unwrap_or(false);
            let draft_cap = remaining.map_or(DEFAULT_DRAFT_LEN, |r| {
                r.saturating_sub(1).min(DEFAULT_DRAFT_LEN)
            });
            let drafts = if spec_off || draft_cap == 0 {
                Vec::new()
            } else if self.predictor.needs_context() {
                // Hidden-state-aware predictor (EAGLE-3): draft only if
                // we have aux states stashed from the prior step.
                if let Some(aux_refs) = self.pending_aux.as_ref() {
                    let ctx = crate::backend::common::speculative::DraftContext {
                        last_token: state.last_token,
                        aux_hidden_states: aux_refs,
                        pos: state.cur_pos,
                    };
                    self.predictor.draft_with_context(&ctx, draft_cap)
                } else {
                    // First step after prefill — aux not yet stashed.
                    Vec::new()
                }
            } else {
                self.predictor.draft(draft_cap)
            };
            let k = if !drafts.is_empty() && !spec_off {
                drafts.len().min(draft_cap)
            } else {
                0
            };

            if k > 0 {
                let mut input: Vec<u32> = Vec::with_capacity(k + 1);
                input.push(state.last_token);
                input.extend_from_slice(&drafts[..k]);

                let old_cache_len = state.cache_len;
                let old_cur_pos = state.cur_pos;
                let bundle = &self.bundle;

                let needs_aux = !self.aux_layer_ids.is_empty();
                let aux_layer_ids_clone = self.aux_layer_ids.clone();
                let spec_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if needs_aux {
                        bundle
                            .model
                            .forward_all_with_aux(
                                &input,
                                old_cur_pos,
                                &mut state.cache,
                                &bundle.rope,
                                &aux_layer_ids_clone,
                            )
                            .map(|(logits, aux)| (logits, Some(aux)))
                    } else {
                        bundle
                            .model
                            .forward_all(&input, old_cur_pos, &mut state.cache, &bundle.rope)
                            .map(|logits| (logits, None))
                    }
                }));

                match spec_result {
                    Err(e) => {
                        self.done = true;
                        self.filter.push_err(super::classify_forward_panic(e));
                        return;
                    }
                    Ok(None) => {
                        // Model doesn't support batched speculative decode — fall through.
                    }
                    Ok(Some((all_logits, aux_batch))) => {
                        // Sample from every position in the batch: (1, k+1, vocab_size)
                        let mut sampled: Vec<u32> = Vec::with_capacity(k + 1);
                        for p in 0..=(k as i32) {
                            let row = all_logits.index((0..1, p..(p + 1), ..));
                            sampled.push(self.sampler.sample(&row));
                        }

                        // Speculative accept: count consecutive draft tokens where
                        // the sampled token from the target distribution matches the
                        // draft. For a point-mass draft (n-gram), this is equivalent
                        // to rejection sampling against p(x) and preserves the target
                        // distribution exactly.
                        let mut accepted = 0usize;
                        for i in 0..k {
                            if sampled[i] == drafts[i] {
                                accepted += 1;
                            } else {
                                break;
                            }
                        }
                        self.spec_drafted += k;
                        self.spec_accepted += accepted;
                        let total = accepted + 1; // accepted drafts + bonus/corrected token

                        // Truncate KV cache when drafts were partially rejected.
                        // The forward pass appended k+1 positions; keep only `total`.
                        if accepted < k {
                            let keep = (old_cache_len + total) as i32;
                            for slot in state.cache.iter_mut() {
                                if let Some(kv) = slot
                                    && kv.0.shape()[2] as usize > old_cache_len + total
                                {
                                    kv.0 = kv.0.index((.., .., 0..keep, ..));
                                    kv.1 = kv.1.index((.., .., 0..keep, ..));
                                }
                            }
                        }

                        // Commit state for the whole speculative batch.
                        let bonus = sampled[accepted];
                        state.cur_pos = old_cur_pos + total;
                        state.cache_len = old_cache_len + total;
                        state.last_token = bonus;
                        state.maybe_evict();

                        // Stash aux hidden states at the accepted position
                        // (= `accepted`, zero-indexed within the k+1 batch)
                        // for the next EAGLE draft call. The "accepted"
                        // position is where bonus was sampled; we want its
                        // aux state so the next draft continues from
                        // there.
                        if let Some(aux_arr) = aux_batch.as_ref() {
                            let idx = accepted as i32;
                            let last_pos_aux: Vec<mlx_rs::Array> = aux_arr
                                .iter()
                                .map(|a| a.index((0..1, idx..idx + 1, ..)))
                                .collect();
                            self.pending_aux = Some(last_pos_aux);
                        }
                        // state is not accessed below this point in this branch

                        if self.first_token_us.is_none() {
                            self.first_token_us = Some(now_us().saturating_sub(self.start_us));
                        }
                        self.produced += total;

                        // Feed confirmed tokens into the n-gram predictor AND
                        // the sampler's repetition-penalty window. Without
                        // the latter, speculative-accepted tokens don't
                        // suppress their own re-emission, and the
                        // "it's a great way to X … it's a great way to X"
                        // loops the n-gram predictor eagerly accepts would
                        // keep accelerating.
                        for &t in &drafts[..accepted] {
                            self.predictor.observe(t);
                            self.sampler.observe(t);
                        }
                        self.predictor.observe(bonus);
                        self.sampler.observe(bonus);

                        // Build token events (EOS check before emitting DecodeStep).
                        let stop_ids: Vec<u32> = self.bundle.tokenizer.stop_ids().to_vec();

                        // Loop detectors: speculative can commit an entire
                        // cycle's worth of tokens in one batch (the n-gram
                        // predictor eagerly accepts the cycle), so we must
                        // check after the batch is observed into the sampler.
                        let loop_hit =
                            self.sampler.is_in_cycle(48) || self.sampler.is_in_token_loop(16, 2);

                        // `i` indexes `drafts` for accepted positions AND selects
                        // the `bonus` token at `i == accepted`, so a plain
                        // iterator over `drafts` can't express the bonus tail.
                        #[allow(clippy::needless_range_loop)]
                        for i in 0..=accepted {
                            let tok = if i < accepted { drafts[i] } else { bonus };
                            if stop_ids.contains(&tok) {
                                let stats = self.stats_now();
                                self.hooks.emit(HookEvent::FinalStats {
                                    session_id: self.session_id,
                                    stats,
                                });
                                self.done = true;
                                self.filter.finalize(TokenEvent::Eos);
                                return;
                            }
                            let text = self.bundle.tokenizer.decode(&[tok]).unwrap_or_default();
                            self.hooks.emit(HookEvent::DecodeStep {
                                session_id: self.session_id,
                                token_id: tok,
                                text_len: text.len(),
                            });
                            // Route through the shared filter — may hold,
                            // or trigger Full stop (which marks filter done).
                            self.filter.push_token(tok, text);
                            if self.filter.is_done() {
                                self.done = true;
                                return;
                            }
                        }
                        if loop_hit {
                            let stats = self.stats_now();
                            self.hooks.emit(HookEvent::FinalStats {
                                session_id: self.session_id,
                                stats,
                            });
                            self.done = true;
                            self.filter.finalize(TokenEvent::Eos);
                        }
                        return;
                    }
                }
            }

            // ── Single-token decode (no draft or unsupported model) ───────────────
            let token = state.last_token;
            let pos = state.cur_pos;
            let bundle = &self.bundle;
            let needs_aux = !self.aux_layer_ids.is_empty();
            let aux_layer_ids = self.aux_layer_ids.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if needs_aux {
                    // EAGLE-3 path: get aux states at the configured
                    // target layers along with logits.
                    let (logits, aux) = bundle
                        .model
                        .forward_all_with_aux(
                            &[token],
                            pos,
                            &mut state.cache,
                            &bundle.rope,
                            &aux_layer_ids,
                        )
                        .expect("forward_all_with_aux should succeed");
                    let s = logits.shape();
                    let seq = s[1];
                    let last_logits = logits.index((0..1, (seq - 1)..seq, ..));
                    (last_logits, Some(aux))
                } else {
                    let logits =
                        bundle
                            .model
                            .forward(&[token], pos, &mut state.cache, &bundle.rope);
                    (logits, None)
                }
            }));
            let (logits_arr, aux_opt) = match result {
                Ok((l, a)) => (l, a),
                Err(e) => {
                    self.done = true;
                    self.filter.push_err(super::classify_forward_panic(e));
                    return;
                }
            };
            // Stash last-position aux for the next draft_with_context call.
            if let Some(aux) = aux_opt {
                let last_pos_aux: Vec<mlx_rs::Array> = aux
                    .iter()
                    .map(|a| {
                        let s = a.shape();
                        let seq = s[1];
                        a.index((0..1, (seq - 1)..seq, ..))
                    })
                    .collect();
                self.pending_aux = Some(last_pos_aux);
            }
            // Unwrap to Ok so the subsequent `match result` below works
            // without restructuring the surrounding function.
            logits_arr
        };

        // ── Single-token path: sample, check EOS, emit ───────────────────────────
        let token_id = self
            .sampler
            .sample_with_grammar(&logits, self.grammar.as_mut());
        state.last_token = token_id;

        // DIAGNOSTIC: under the fixed-steps gate, IGNORE EOS / stop-id / loop
        // early termination — decode the full fixed workload so the ms/token
        // denominator is exactly `n` (see `fixed_steps`). The cache write +
        // `produced` bump below still run, so the token still counts as a
        // forward; the outer `effective_max_tokens` guard stops us at `n`.
        // (`suppress_stops` was snapshotted at the top of `step_once`.)

        // Check EOS / EOT (chat models need both — Gemma 4's `<turn|>`, Llama 3's `<|eot_id|>`).
        if !suppress_stops && self.bundle.tokenizer.stop_ids().contains(&token_id) {
            let stats = self.stats_now();
            self.hooks.emit(HookEvent::FinalStats {
                session_id: self.session_id,
                stats,
            });
            self.done = true;
            self.filter.finalize(TokenEvent::Eos);
            return;
        }

        self.sampler.observe(token_id);
        self.predictor.observe(token_id);
        if !suppress_stops && (self.sampler.is_in_cycle(48) || self.sampler.is_in_token_loop(16, 2))
        {
            let stats = self.stats_now();
            self.hooks.emit(HookEvent::FinalStats {
                session_id: self.session_id,
                stats,
            });
            self.done = true;
            self.filter.finalize(TokenEvent::Eos);
            return;
        }

        let text = self
            .bundle
            .tokenizer
            .decode(&[token_id])
            .unwrap_or_default();

        state.cur_pos += 1;
        state.cache_len += 1;
        state.maybe_evict();

        if self.first_token_us.is_none() {
            self.first_token_us = Some(now_us().saturating_sub(self.start_us));
        }
        self.produced += 1;

        self.hooks.emit(HookEvent::DecodeStep {
            session_id: self.session_id,
            token_id,
            text_len: text.len(),
        });

        // Route through the shared filter — may hold (partial suffix match) or
        // trigger Full stop (marks filter done).
        self.filter.push_token(token_id, text);
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}
