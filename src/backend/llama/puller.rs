use std::sync::atomic::{AtomicBool, Ordering};

use encoding_rs::UTF_8;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

use super::bundle::ModelBundle;
use super::session::{DecodeState, SessionCtxCell};
use crate::backend::common::grammar::GrammarMatcher;
use crate::backend::common::output_filter::OutputFilter;
use crate::backend::common::stop_matcher::StopMatcher;
use crate::engine::ExecError;
use crate::engine::{ExecutionStats, HookBus, HookEvent};
use crate::generation::{GenSpec, TokenEvent};
use std::collections::VecDeque;

use parking_lot::Mutex;
use std::sync::{Arc, Weak};

#[cfg(feature = "llama-dflash2")]
#[derive(Default)]
struct DFlashDecode {
    enabled: bool,
    queued: VecDeque<LlamaToken>,
    anchor: Option<LlamaToken>,
    drafted: u32,
    accepted: u32,
    failed: bool,
}

pub struct TokenPuller {
    session_id: u64,
    hooks: Arc<HookBus>,
    bundle: Arc<ModelBundle>,

    // OWNED decode state parts (wrapped so we can move them back on Drop)
    ctx_cell: Option<SessionCtxCell>,
    sampler: Option<LlamaSampler>,
    /// Grammar-constrained decoding (JSON schema / regex / lark). When
    /// `Some`, each step masks the logits via llguidance against the GGUF
    /// vocab BEFORE the base sampler runs, then `observe`s the chosen
    /// token. Bypasses llama.cpp's built-in `LlamaSampler::llguidance`,
    /// whose Matcher rejected the opening token at this dep rev.
    grammar: Option<GrammarMatcher>,

    batch: LlamaBatch<'static>,
    prompt_tokens: usize,
    cur_pos: i32,
    logits_i: i32,
    produced: usize,
    max_tokens: Option<usize>,
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    utf8_decoder: encoding_rs::Decoder,
    start_us: u64,
    first_token_us: Option<u64>,
    pre_events: VecDeque<TokenEvent>,
    done: bool,
    #[cfg(feature = "llama-dflash2")]
    dflash: DFlashDecode,

    /// Cross-backend stop-pattern filter. Every emitted token funnels through
    /// here so the llama backend gets the same zero-garbage chat behaviour
    /// as MLX (see `gen2/backend/common/output_filter.rs`).
    filter: OutputFilter,

    // weak link to Session.state (Mutex<Option<DecodeState>>)
    state_slot: Weak<Mutex<Option<DecodeState>>>,
}

impl TokenPuller {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn _new(
        session_id: u64,
        hooks: Arc<HookBus>,
        bundle: Arc<ModelBundle>,
        ctx_cell: SessionCtxCell,
        sampler: LlamaSampler,
        grammar: Option<GrammarMatcher>,
        prompt_tokens: usize,
        cur_pos: i32,
        gen_spec: GenSpec,
        paused: Arc<AtomicBool>,
        stopped: Arc<AtomicBool>,
        pre_events: VecDeque<TokenEvent>,
        initial_logits_i: i32,
    ) -> Self {
        let batch = LlamaBatch::new(1, 1);
        Self {
            session_id,
            hooks,
            bundle,
            ctx_cell: Some(ctx_cell),
            sampler: Some(sampler),
            grammar,
            batch,
            prompt_tokens,
            cur_pos,
            logits_i: initial_logits_i,
            produced: 0,
            max_tokens: gen_spec.max_tokens,
            paused,
            stopped,
            utf8_decoder: UTF_8.new_decoder(),
            start_us: ggml_time_us() as u64,
            first_token_us: None,
            pre_events,
            done: false,
            #[cfg(feature = "llama-dflash2")]
            dflash: DFlashDecode::default(),
            filter: OutputFilter::new(StopMatcher::gemma4_chat_defaults()),
            state_slot: Default::default(),
        }
    }

    /// Arm the output parser's enabled-tool gate for this pull.
    pub(crate) fn arm_enabled_tools(&mut self, tools: std::collections::HashSet<String>) {
        self.filter.set_enabled_tools(tools);
    }

    fn emit_final_stats(&self) {
        self.hooks.emit(HookEvent::FinalStats {
            session_id: self.session_id,
            stats: self.stats_now(),
        });
        if let Some(tally) = self.filter.tool_call_tally()
            && !tally.is_empty()
        {
            self.hooks.emit(HookEvent::ToolCallOutcomes {
                session_id: self.session_id,
                tally,
            });
        }
    }

    fn stats_now(&self) -> ExecutionStats {
        let elapsed_us = (ggml_time_us() as u64).saturating_sub(self.start_us);
        let elapsed_s = (elapsed_us as f64) / 1_000_000.0;
        let avg_tps = if elapsed_s > 0.0 {
            (self.produced as f64 / elapsed_s) as f32
        } else {
            0.0
        };
        ExecutionStats {
            reasoning_ms: None,
            prompt_tokens: self.prompt_tokens as u32,
            decode_tokens: self.produced as u32,
            first_token_us: self.first_token_us.unwrap_or(0),
            avg_tps,
            // llama backend doesn't track engine-level KV cache budget
            cache_tokens: 0,
            cache_budget: 0,
            evictions: 0,
            spec_drafted: {
                #[cfg(feature = "llama-dflash2")]
                {
                    self.dflash.drafted
                }
                #[cfg(not(feature = "llama-dflash2"))]
                {
                    0
                }
            },
            spec_accepted: {
                #[cfg(feature = "llama-dflash2")]
                {
                    self.dflash.accepted
                }
                #[cfg(not(feature = "llama-dflash2"))]
                {
                    0
                }
            },
        }
    }

    /// Sample one token from the logits at `logits_i`. With no grammar
    /// this is the plain `LlamaSampler::sample`. With a [`GrammarMatcher`]
    /// it masks the raw logits via llguidance (forcing schema validity),
    /// runs the base sampler chain on the masked candidates, and advances
    /// the matcher with the chosen token. Borrows of `sampler` / `grammar`
    /// / `ctx_cell` are disjoint fields, released before the caller does
    /// its own `&mut self` bookkeeping.
    fn sample_one(&mut self) -> Result<LlamaToken, ExecError> {
        let logits_i = self.logits_i;
        let ctx_ref = self
            .ctx_cell
            .as_ref()
            .ok_or(ExecError::InvalidArg("context already consumed"))?;
        let sampler = self
            .sampler
            .as_mut()
            .ok_or(ExecError::InvalidArg("sampler already consumed"))?;
        match self.grammar.as_mut() {
            Some(grammar) => {
                // `apply_mask` uses llguidance's `compute_mask_or_eos`, so a
                // completed grammar yields an EOS-only mask here — the base
                // chain then samples EOS and the caller's `is_eog_token`
                // check finalizes the stream. No special-casing needed.
                let token = ctx_ref.with_dependent(|_, ctx| -> Result<LlamaToken, ExecError> {
                    let mut logits = ctx.get_logits_ith(logits_i).to_vec();
                    grammar.apply_mask(&mut logits).map_err(ExecError::Other)?;
                    let mut data = LlamaTokenDataArray::from_iter(
                        logits
                            .iter()
                            .enumerate()
                            .map(|(id, &l)| LlamaTokenData::new(LlamaToken(id as i32), l, 0.0)),
                        false,
                    );
                    // Base chain (penalties / top-k / min-p / temp / dist)
                    // applied to grammar-valid candidates only.
                    sampler.apply(&mut data);
                    Ok(data
                        .selected_token()
                        .unwrap_or_else(|| data.sample_token_greedy()))
                })?;
                // Advance grammar state with the accepted token (no-op once
                // the parser has stopped; ignore that benign error).
                if let Err(e) = grammar.observe(token.0 as u32) {
                    tracing::debug!(?e, "grammar observe failed (accepting/exhausted)");
                }
                Ok(token)
            }
            None => Ok(ctx_ref.with_dependent(|_, ctx| sampler.sample(ctx, logits_i))),
        }
    }

    fn decode_single(&mut self, token: LlamaToken) -> Result<(), ExecError> {
        self.batch.clear();
        self.batch
            .add(token, self.cur_pos, &[0], true)
            .map_err(|e| ExecError::Other(e.into()))?;
        let result = self
            .ctx_cell
            .as_mut()
            .ok_or(ExecError::InvalidArg("context already consumed"))?
            .with_dependent_mut(|_, ctx| ctx.decode(&mut self.batch));
        #[cfg(feature = "llama-dflash2")]
        if result.is_err() && self.bundle.draft.is_some() {
            self.dflash.failed = true;
        }
        result?;
        self.cur_pos += 1;
        self.logits_i = 0;
        Ok(())
    }

    #[cfg(feature = "llama-dflash2")]
    fn verify_dflash_block(&mut self, anchor: LlamaToken) -> Result<(), ExecError> {
        use super::session::DepCtx;
        let draft_model = self
            .bundle
            .draft
            .as_ref()
            .ok_or(ExecError::InvalidArg("no DFlash2 model"))?;
        let ctx_size = self
            .ctx_cell
            .as_ref()
            .unwrap()
            .with_dependent(|_, ctx| ctx.n_ctx()) as usize;
        // The trained noise block still needs room even when output is capped.
        if self.cur_pos as usize + draft_model.block_size > ctx_size {
            return self.decode_single(anchor);
        }
        let mut proposed = self
            .ctx_cell
            .as_mut()
            .unwrap()
            .with_dependent_mut(|_, ctx| {
                match ctx {
                    DepCtx::DFlash2(spec) => {
                        let proposed = spec
                            .draft(self.cur_pos, anchor, &[])
                            .map_err(|e| ExecError::Other(e.into()))?;
                        // Drafting writes a temporary noise block. As in llama-server,
                        // remove it before injecting the verified target features.
                        spec.draft_context_mut()
                            .kv_cache_seq_rm(0, Some(self.cur_pos as u32), None)
                            .map_err(|e| ExecError::Other(e.into()))?;
                        Ok(proposed)
                    }
                    _ => Err(ExecError::InvalidArg("no DFlash2 context")),
                }
            })?;
        let drafted = proposed.len();
        self.dflash.drafted = self.dflash.drafted.saturating_add(drafted as u32);
        if let Some(limit) = self.max_tokens {
            proposed.truncate(limit.saturating_sub(self.produced));
        }
        self.batch = LlamaBatch::new(proposed.len() + 1, 1);
        self.batch
            .add(anchor, self.cur_pos, &[0], true)
            .map_err(|e| ExecError::Other(e.into()))?;
        for (i, &token) in proposed.iter().enumerate() {
            self.batch
                .add(token, self.cur_pos + i as i32 + 1, &[0], true)
                .map_err(|e| ExecError::Other(e.into()))?;
        }
        self.ctx_cell
            .as_mut()
            .unwrap()
            .with_dependent_mut(|_, ctx| ctx.decode(&mut self.batch))?;
        self.logits_i = 0;
        let mut accepted = 0usize;
        for token in proposed.iter().copied() {
            let verified = self.sample_one()?;
            if verified != token || self.bundle.model.is_eog_token(verified) {
                break;
            }
            self.sampler.as_mut().unwrap().accept(verified);
            self.dflash.queued.push_back(verified);
            accepted += 1;
            self.logits_i += 1;
        }
        let keep = self.cur_pos + accepted as i32 + 1;
        self.ctx_cell
            .as_mut()
            .unwrap()
            .with_dependent_mut(|_, ctx| -> Result<(), ExecError> {
                if accepted < proposed.len() {
                    ctx.kv_cache_seq_rm(0, Some(keep as u32), None)
                        .map_err(|e| ExecError::Other(e.into()))?;
                }
                if drafted > 0
                    && let DepCtx::DFlash2(spec) = ctx
                {
                    spec.draft_context_mut()
                        .kv_cache_seq_rm(0, Some(keep as u32), None)
                        .map_err(|e| ExecError::Other(e.into()))?;
                    spec.accept(accepted as u16)
                        .map_err(|e| ExecError::Other(e.into()))?;
                }
                Ok(())
            })?;
        self.cur_pos = keep;
        self.dflash.accepted = self.dflash.accepted.saturating_add(accepted as u32);
        Ok(())
    }

    fn next_token(&mut self) -> Result<LlamaToken, ExecError> {
        #[cfg(feature = "llama-dflash2")]
        if self.dflash.enabled {
            if let Some(anchor) = self.dflash.anchor.take() {
                if let Err(e) = self.verify_dflash_block(anchor) {
                    self.dflash.failed = true;
                    return Err(e);
                }
            }
            if let Some(token) = self.dflash.queued.pop_front() {
                return Ok(token);
            }
            let ctx_size = self
                .ctx_cell
                .as_ref()
                .unwrap()
                .with_dependent(|_, ctx| ctx.n_ctx()) as i32;
            if self.cur_pos >= ctx_size {
                return Err(ExecError::InvalidArg("DFlash2 context exhausted"));
            }
            let token = self.sample_one()?;
            self.sampler.as_mut().unwrap().accept(token);
            if !self.bundle.model.is_eog_token(token) {
                // Emit the anchor before running the next block, matching llama-server TTFT.
                self.dflash.anchor = Some(token);
            }
            return Ok(token);
        }
        let token = self.sample_one()?;
        self.sampler.as_mut().unwrap().accept(token);
        if !self.bundle.model.is_eog_token(token) {
            self.decode_single(token)?;
        }
        Ok(token)
    }

    #[cfg(feature = "llama-dflash2")]
    fn settle_dflash(&mut self) -> Result<(), ExecError> {
        if self.dflash.failed {
            return Err(ExecError::InvalidArg(
                "DFlash2 decode failed; discard context",
            ));
        }
        if !self.dflash.enabled {
            return Ok(());
        }
        if let Some(anchor) = self.dflash.anchor.take() {
            self.decode_single(anchor)?;
        }
        if !self.dflash.queued.is_empty() {
            let count = self.dflash.queued.len() as i32;
            let keep = self.cur_pos - count;
            self.ctx_cell.as_mut().unwrap().with_dependent_mut(
                |_, ctx| -> Result<(), ExecError> {
                    ctx.kv_cache_seq_rm(0, Some(keep as u32), None)
                        .map_err(|e| ExecError::Other(e.into()))?;
                    if let super::session::DepCtx::DFlash2(spec) = ctx {
                        spec.draft_context_mut()
                            .kv_cache_seq_rm(0, Some(keep as u32), None)
                            .map_err(|e| ExecError::Other(e.into()))?;
                    }
                    Ok(())
                },
            )?;
            self.cur_pos = keep;
            self.logits_i -= count;
            self.dflash.queued.clear();
        }
        Ok(())
    }
}

impl TokenPuller {
    #[cfg(feature = "llama-dflash2")]
    pub(crate) fn enable_dflash2(&mut self, enabled: bool) {
        self.dflash.enabled = enabled;
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_from_session(
        session_id: u64,
        hooks: Arc<HookBus>,
        bundle: Arc<ModelBundle>,
        state_slot: Weak<Mutex<Option<DecodeState>>>,
        state: DecodeState, // taken from Session
        sampler: LlamaSampler,
        grammar: Option<GrammarMatcher>,
        gen_spec: GenSpec,
        paused: Arc<AtomicBool>,
        stopped: Arc<AtomicBool>,
        pre_events: VecDeque<TokenEvent>,
    ) -> Self {
        let batch = LlamaBatch::new(1, 1);
        Self {
            session_id,
            hooks,
            bundle,
            ctx_cell: Some(state.ctx_cell),
            sampler: Some(sampler),
            grammar,
            batch,
            prompt_tokens: state.cur_pos.max(0) as usize,
            cur_pos: state.cur_pos,
            logits_i: state.logits_i,
            produced: 0,
            max_tokens: gen_spec.max_tokens,
            paused,
            stopped,
            utf8_decoder: UTF_8.new_decoder(),
            // Use prefill start time for accurate TTFT (includes prefill, not just decode)
            start_us: state.prefill_start_us,
            first_token_us: None,
            pre_events,
            done: false,
            #[cfg(feature = "llama-dflash2")]
            dflash: DFlashDecode::default(),
            filter: OutputFilter::new(StopMatcher::gemma4_chat_defaults()),
            state_slot,
        }
    }
}

impl Drop for TokenPuller {
    fn drop(&mut self) {
        #[cfg(feature = "llama-dflash2")]
        if let Err(e) = self.settle_dflash() {
            tracing::error!(%e, "DFlash2 state could not be restored");
            self.stopped.store(true, Ordering::Release);
            return;
        }
        if let Some(slot) = self.state_slot.upgrade()
            && let (Some(ctx_cell), Some(_sampler)) = (self.ctx_cell.take(), self.sampler.take())
        {
            let mut g = slot.lock();
            *g = Some(DecodeState {
                ctx_cell,
                cur_pos: self.cur_pos,
                logits_i: self.logits_i,
                prefill_start_us: self.start_us,
            });
        }
    }
}

impl Iterator for TokenPuller {
    type Item = Result<TokenEvent, ExecError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Outer loop handles stop-matcher hold-back: if a sampled token
        // ends up held (partial-match suffix), we sample again rather
        // than return `None` early. Same pattern as the MLX puller.
        loop {
            if let Some(ev) = self.filter.pop() {
                return Some(ev);
            }
            if self.done || self.filter.is_done() {
                return None;
            }
            if let Some(ev) = self.pre_events.pop_front() {
                return Some(Ok(ev));
            }
            if self.stopped.load(Ordering::Acquire) {
                tracing::info!(
                    target: "pio::gen2::llama::term",
                    produced = self.produced,
                    "termination: stopped flag"
                );
                self.emit_final_stats();
                self.done = true;
                self.filter.finalize(TokenEvent::Stopped);
                continue;
            }
            if self.paused.load(Ordering::Acquire) {
                return Some(Ok(TokenEvent::Paused));
            }
            if let Some(limit) = self.max_tokens
                && self.produced >= limit
            {
                tracing::info!(
                    target: "pio::gen2::llama::term",
                    produced = self.produced,
                    limit,
                    "termination: max_tokens"
                );
                self.emit_final_stats();
                self.done = true;
                self.filter.finalize(TokenEvent::Eos);
                continue;
            }

            // ── Sample + decode one token ──────────────────────────────
            // (grammar-aware: masks logits via llguidance when a
            // GrammarMatcher is present — see `sample_one`).
            let token = match self.next_token() {
                Ok(t) => t,
                Err(e) => {
                    self.emit_final_stats();
                    self.done = true;
                    self.filter.push_err(e);
                    continue;
                }
            };

            if self.bundle.model.is_eog_token(token) {
                tracing::info!(
                    target: "pio::gen2::llama::term",
                    produced = self.produced,
                    token_id = token.0,
                    "termination: model emitted EOG token"
                );
                self.emit_final_stats();
                self.done = true;
                self.filter.finalize(TokenEvent::Eos);
                continue;
            }

            let bytes = match self
                .bundle
                .model
                .token_to_piece_bytes(token, 32, true, None)
            {
                Ok(b) => b,
                Err(e) => {
                    self.emit_final_stats();
                    self.filter.push_err(ExecError::Other(e.into()));
                    continue;
                }
            };
            let mut out = String::with_capacity(32);
            let _ = self.utf8_decoder.decode_to_string(&bytes, &mut out, false);

            if self.first_token_us.is_none() {
                self.first_token_us = Some((ggml_time_us() as u64).saturating_sub(self.start_us));
            }
            self.produced += 1;
            self.hooks.emit(HookEvent::DecodeStep {
                session_id: self.session_id,
                token_id: token.0 as u32,
                text_len: out.len(),
            });

            // Route through the shared filter. It may hold (partial
            // match) — in that case we loop back and sample again.
            // On Clean/Full it pushes events into the pending queue
            // which the loop top drains.
            self.filter.push_token(token.0 as u32, out);
        }
    }
}

// ─── Trait impls (Phase 2) ─────────────────────────────────────────────────

impl crate::backend::traits::TokenPullerDyn for TokenPuller {
    fn next_event(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        <Self as Iterator>::next(self)
    }
}
