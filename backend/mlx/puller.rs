//! MLX token generation iterator with n-gram speculative decoding.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use mlx_rs::ops::indexing::IndexOp;
use parking_lot::Mutex;

use super::bundle::ModelBundle;
use super::sampler::Sampler;
use super::session::DecodeState;
use crate::gen2::backend::common::grammar::GrammarMatcher;
use crate::gen2::backend::common::output_filter::OutputFilter;
use crate::gen2::backend::common::speculative::{
    DEFAULT_DRAFT_LEN, SpeculativeMode, SpeculativePredictor,
};
use crate::gen2::backend::common::stop_matcher::StopMatcher;
use crate::gen2::engine::{ExecError, ExecutionStats, HookBus, HookEvent};
use crate::gen2::generation::{GenSpec, TokenEvent};

pub struct TokenPuller {
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
}

impl TokenPuller {
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
            use crate::gen2::backend::common::sampler::DryParams;
            DryParams {
                multiplier: m,
                base: gen_spec.dry_base.unwrap_or(1.75),
                allowed_length: gen_spec.dry_allowed_length.unwrap_or(2),
            }
        });
        let xtc_params = gen_spec.xtc_probability.map(|p| {
            use crate::gen2::backend::common::sampler::XtcParams;
            XtcParams {
                probability: p,
                threshold: gen_spec.xtc_threshold.unwrap_or(0.1),
            }
        });

        let sampler = Sampler::new(
            temperature,
            Some(gen_spec.top_p.unwrap_or(0.9)),
            gen_spec.top_k,
            None,
        )
        .with_eot_bias(bundle.tokenizer.stop_ids().to_vec(), eot_bias)
        .with_min_p(gen_spec.min_p)
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
        }
    }

    /// Seed the speculative predictor with the session's prompt tokens.
    /// Only PLD / Hybrid use this; n-gram / Off ignore it. Call once
    /// immediately after construction when the prompt tokens are in hand.
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

    /// Test-only: read `stats_now` from outside the puller. Used by the golden
    /// test suite to verify per-session counters after drain.
    #[cfg(test)]
    pub(super) fn snapshot_stats(&self) -> ExecutionStats {
        self.stats_now()
    }
}

impl Drop for TokenPuller {
    fn drop(&mut self) {
        if let Some(slot) = self.state_slot.upgrade() {
            if let Some(state) = self.state.take() {
                *slot.lock() = Some(state);
            }
        }
    }
}

impl Iterator for TokenPuller {
    type Item = Result<TokenEvent, ExecError>;

    fn next(&mut self) -> Option<Self::Item> {
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
}

impl TokenPuller {
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
        if let Some(limit) = self.max_tokens
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
            let remaining = self.max_tokens.map(|m| m.saturating_sub(self.produced));
            let spec_off = self.grammar.is_some()
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
                    let ctx = crate::gen2::backend::common::speculative::DraftContext {
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
                        self.filter.push_err(ExecError::OutOfMemory(panic_msg(e)));
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
                                if let Some(kv) = slot {
                                    if kv.0.shape()[2] as usize > old_cache_len + total {
                                        kv.0 = kv.0.index((.., .., 0..keep, ..));
                                        kv.1 = kv.1.index((.., .., 0..keep, ..));
                                    }
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
                    self.filter.push_err(ExecError::OutOfMemory(panic_msg(e)));
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

        // Check EOS / EOT (chat models need both — Gemma 4's `<turn|>`, Llama 3's `<|eot_id|>`).
        if self.bundle.tokenizer.stop_ids().contains(&token_id) {
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
        if self.sampler.is_in_cycle(48) || self.sampler.is_in_token_loop(16, 2) {
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

/// Extract a human-readable message from a caught panic payload.
fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "MLX forward pass panicked (likely OOM)".to_string()
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

impl crate::gen2::backend::traits::TokenPullerDyn for TokenPuller {
    fn next_event(
        &mut self,
    ) -> Option<Result<crate::gen2::generation::TokenEvent, crate::gen2::engine::ExecError>> {
        <Self as Iterator>::next(self)
    }
}
