use std::sync::atomic::{AtomicBool, Ordering};

use encoding_rs::UTF_8;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;

use super::bundle::ModelBundle;
use super::session::{DecodeState, SessionCtxCell};
use crate::gen2::engine::ExecError;
use crate::gen2::engine::{ExecutionStats, HookBus, HookEvent};
use crate::gen2::generation::{GenSpec, Token, TokenEvent};
use std::collections::VecDeque;

use parking_lot::Mutex;
use std::sync::{Arc, Weak};

pub struct TokenPuller {
    session_id: u64,
    hooks: Arc<HookBus>,
    bundle: Arc<ModelBundle>,

    // OWNED decode state parts (wrapped so we can move them back on Drop)
    ctx_cell: Option<SessionCtxCell>,
    sampler: Option<LlamaSampler>,

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

    // weak link to Session.state (Mutex<Option<DecodeState>>)
    state_slot: Weak<Mutex<Option<DecodeState>>>,
}

impl TokenPuller {
    pub(crate) fn _new(
        session_id: u64,
        hooks: Arc<HookBus>,
        bundle: Arc<ModelBundle>,
        ctx_cell: SessionCtxCell,
        sampler: LlamaSampler,
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
            state_slot: Default::default(),
        }
    }

    fn emit_final_stats(&self) {
        self.hooks.emit(HookEvent::FinalStats {
            session_id: self.session_id,
            stats: self.stats_now(),
        });
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
            prompt_tokens: self.prompt_tokens as u32,
            decode_tokens: self.produced as u32,
            first_token_us: self.first_token_us.unwrap_or(0),
            avg_tps,
        }
    }
}

impl TokenPuller {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_from_session(
        session_id: u64,
        hooks: Arc<HookBus>,
        bundle: Arc<ModelBundle>,
        state_slot: Weak<Mutex<Option<DecodeState>>>,
        state: DecodeState, // taken from Session
        sampler: LlamaSampler,
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
            state_slot,
        }
    }
}

impl Drop for TokenPuller {
    fn drop(&mut self) {
        if let Some(slot) = self.state_slot.upgrade() {
            if let (Some(ctx_cell), Some(_sampler)) = (self.ctx_cell.take(), self.sampler.take()) {
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
}

impl Iterator for TokenPuller {
    type Item = Result<TokenEvent, ExecError>;

    fn next(&mut self) -> Option<Self::Item> {
        // tracing::trace!("puller.next");
        if self.done {
            return None;
        }
        if let Some(ev) = self.pre_events.pop_front() {
            return Some(Ok(ev));
        }
        if self.stopped.load(Ordering::Acquire) {
            self.emit_final_stats();
            self.done = true;
            return Some(Ok(TokenEvent::Stopped));
        }
        if self.paused.load(Ordering::Acquire) {
            return Some(Ok(TokenEvent::Paused));
        }
        if let Some(limit) = self.max_tokens {
            if self.produced >= limit {
                self.emit_final_stats();
                self.done = true;
                return Some(Ok(TokenEvent::Eos));
            }
        }

        // Sample next token
        let Some(sampler) = self.sampler.as_mut() else {
            self.emit_final_stats();
            self.done = true;
            return Some(Err(ExecError::InvalidArg("sampler already consumed")));
        };
        let Some(ctx_ref) = self.ctx_cell.as_ref() else {
            self.emit_final_stats();
            self.done = true;
            return Some(Err(ExecError::InvalidArg("context already consumed")));
        };
        let token = ctx_ref.with_dependent(|_, ctx| sampler.sample(ctx, self.logits_i));
        // accept updates the sampler's internal state → needs &mut
        let Some(sampler) = self.sampler.as_mut() else {
            self.emit_final_stats();
            self.done = true;
            return Some(Err(ExecError::InvalidArg("sampler already consumed")));
        };
        sampler.accept(token);

        if self.bundle.model.is_eog_token(token) {
            self.emit_final_stats();
            self.done = true;
            return Some(Ok(TokenEvent::Eos));
        }

        // Convert token bytes to utf8 string
        let bytes = match self
            .bundle
            .model
            .token_to_piece_bytes(token, 32, true, None)
        {
            Ok(b) => b,
            Err(e) => {
                self.emit_final_stats();
                return Some(Err(ExecError::Other(e.into())));
            }
        };
        let mut out = String::with_capacity(32);
        let _ = self.utf8_decoder.decode_to_string(&bytes, &mut out, false);

        // Prepare batch for next decode step
        self.batch.clear();
        if let Err(e) = self.batch.add(token, self.cur_pos, &[0], true) {
            self.emit_final_stats();
            return Some(Err(ExecError::Other(e.into())));
        }
        self.cur_pos += 1;

        let Some(ctx_mut) = self.ctx_cell.as_mut() else {
            self.emit_final_stats();
            self.done = true;
            return Some(Err(ExecError::InvalidArg("context already consumed")));
        };
        if let Err(e) = ctx_mut.with_dependent_mut(|_, ctx| ctx.decode(&mut self.batch)) {
            self.emit_final_stats();
            return Some(Err(ExecError::Other(e.into())));
        }

        // after single-token decode, logits index for next sample is the last element in that batch (0)
        self.logits_i = (self.batch.n_tokens() as i32) - 1;

        if self.first_token_us.is_none() {
            self.first_token_us = Some((ggml_time_us() as u64).saturating_sub(self.start_us));
        }
        self.produced += 1;
        self.hooks.emit(HookEvent::DecodeStep {
            session_id: self.session_id,
            token_id: token.0 as u32,
            text_len: out.len(),
        });

        Some(Ok(TokenEvent::Token(Token {
            id: token.0 as u32,
            text: out,
            logprob: None,
        })))
    }
}
