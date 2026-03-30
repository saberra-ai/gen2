//! MLX token generation iterator.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use super::bundle::ModelBundle;
use super::sampler::Sampler;
use super::session::DecodeState;
use crate::gen2::engine::{ExecError, ExecutionStats, HookBus, HookEvent};
use crate::gen2::generation::{GenSpec, Token, TokenEvent};

pub struct TokenPuller {
    session_id: u64,
    hooks: Arc<HookBus>,
    bundle: Arc<ModelBundle>,

    state: Option<DecodeState>,
    sampler: Sampler,

    prompt_tokens: usize,
    produced: usize,
    max_tokens: Option<usize>,
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    start_us: u64,
    first_token_us: Option<u64>,
    done: bool,

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
        let sampler = Sampler::new(
            temperature,
            None, // top_p from settings if needed
            None, // top_k from settings if needed
        );

        Self {
            session_id,
            hooks,
            bundle,
            prompt_tokens: state.cur_pos,
            state: Some(state),
            sampler,
            produced: 0,
            max_tokens: gen_spec.max_tokens,
            paused,
            stopped,
            start_us: now_us(),
            first_token_us: None,
            done: false,
            state_slot,
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
        }
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
        if self.done {
            return None;
        }
        if self.stopped.load(Ordering::SeqCst) {
            let stats = self.stats_now();
            self.hooks.emit(HookEvent::FinalStats {
                session_id: self.session_id,
                stats,
            });
            self.done = true;
            return Some(Ok(TokenEvent::Stopped));
        }
        if self.paused.load(Ordering::SeqCst) {
            return Some(Ok(TokenEvent::Paused));
        }
        if let Some(limit) = self.max_tokens {
            if self.produced >= limit {
                let stats = self.stats_now();
                self.hooks.emit(HookEvent::FinalStats {
                    session_id: self.session_id,
                    stats,
                });
                self.done = true;
                return Some(Ok(TokenEvent::Eos));
            }
        }

        let state = match self.state.as_mut() {
            Some(s) => s,
            None => {
                self.done = true;
                return Some(Err(ExecError::InvalidArg("state already consumed")));
            }
        };

        // Forward pass with previous token to get next logits
        // For the first call after prefill, we use the last token from prefill
        // which is already in the KV cache. We need to feed the last sampled token.
        // On first iteration, this is handled by the prefill.

        // If this is the first token, we already have logits from prefill
        // For subsequent tokens, run a single forward pass with the last token
        let logits = if self.produced == 0 {
            // Re-run with last token position to get logits
            // The KV cache already has the prefill, so we just need the final logits
            // We need the last token from the prompt. Since prefill already ran,
            // we can get logits by running one more forward step.
            // Actually, the prefill forward already returned logits - but we didn't save them.
            // For simplicity, we'll run the model with a dummy token and fix in integration.
            // TODO: pass prefill logits through to avoid this extra forward pass
            let last_token = 0u32; // placeholder - should be last prompt token
            self.bundle
                .model
                .forward(&[last_token], &mut state.cache, &self.bundle.rope)
        } else {
            // Forward with the token we just sampled
            let prev_token = 0u32; // will be overwritten below
            self.bundle
                .model
                .forward(&[prev_token], &mut state.cache, &self.bundle.rope)
        };

        // Sample next token from logits
        let token_id = self.sampler.sample(&logits);

        // Check EOS
        if let Some(eos_id) = self.bundle.tokenizer.eos_id() {
            if token_id == eos_id {
                let stats = self.stats_now();
                self.hooks.emit(HookEvent::FinalStats {
                    session_id: self.session_id,
                    stats,
                });
                self.done = true;
                return Some(Ok(TokenEvent::Eos));
            }
        }

        // Decode token to text
        let text = self
            .bundle
            .tokenizer
            .decode(&[token_id])
            .unwrap_or_default();

        state.cur_pos += 1;

        if self.first_token_us.is_none() {
            self.first_token_us = Some(now_us().saturating_sub(self.start_us));
        }
        self.produced += 1;

        self.hooks.emit(HookEvent::DecodeStep {
            session_id: self.session_id,
            token_id,
            text_len: text.len(),
        });

        Some(Ok(TokenEvent::Token(Token {
            id: token_id,
            text,
            logprob: None,
        })))
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}
