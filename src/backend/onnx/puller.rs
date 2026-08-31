//! ONNX token generation iterator.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use ndarray::Array2;
use parking_lot::Mutex;

use super::bundle::ModelBundle;
use super::session::DecodeState;
use crate::backend::common::output_filter::OutputFilter;
use crate::backend::common::sampler::Sampler;
use crate::backend::common::stop_matcher::StopMatcher;
use crate::engine::{ExecError, ExecutionStats, HookBus, HookEvent};
use crate::generation::{GenSpec, TokenEvent};

pub struct TokenPuller {
    session_id: u64,
    hooks: Arc<HookBus>,
    bundle: Arc<ModelBundle>,

    state: Option<DecodeState>,
    sampler: Sampler,
    last_token_id: Option<u32>,

    prompt_tokens: usize,
    produced: usize,
    max_tokens: Option<usize>,
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    start_us: u64,
    first_token_us: Option<u64>,
    done: bool,

    /// Cross-backend stop-pattern filter (same helper as MLX and llama).
    filter: OutputFilter,

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
        // Full CommonSampler pipeline — same knobs the MLX backend uses.
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
            None,
        )
        .with_min_p(gen_spec.min_p)
        .with_dry(dry_params)
        .with_xtc(xtc_params);

        Self {
            session_id,
            hooks,
            bundle,
            prompt_tokens: state.cur_pos,
            state: Some(state),
            sampler,
            last_token_id: None,
            produced: 0,
            max_tokens: gen_spec.max_tokens,
            paused,
            stopped,
            start_us: now_us(),
            first_token_us: None,
            done: false,
            filter: OutputFilter::new(StopMatcher::gemma4_chat_defaults()),
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
            ..Default::default()
        }
    }

    /// Run a single-token forward pass through the ort session and return logits.
    fn forward_one_token(&mut self, token_id: u32) -> Result<Vec<f32>, ExecError> {
        let state = self
            .state
            .as_mut()
            .ok_or(ExecError::InvalidArg("state consumed"))?;

        let ids_array = Array2::from_shape_vec((1, 1), vec![token_id as i64])
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;
        let total_seq = state.cur_pos + 1;
        let mask_array = Array2::from_shape_vec((1, total_seq), vec![1i64; total_seq])
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;
        let pos_array = Array2::from_shape_vec((1, 1), vec![state.cur_pos as i64])
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;

        let ids_tensor = ort::value::Tensor::from_array(ids_array)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;
        let mask_tensor = ort::value::Tensor::from_array(mask_array)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;

        let (logits, new_cache) = {
            let mut ort_session = self.bundle.session.lock();

            // Build inputs: ids + mask + optional position_ids + KV cache
            let mut inputs: Vec<(String, ort::value::Value)> = vec![
                ("input_ids".to_string(), ids_tensor.into()),
                ("attention_mask".to_string(), mask_tensor.into()),
            ];

            if self.bundle.has_position_ids {
                let pos_tensor = ort::value::Tensor::from_array(pos_array)
                    .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;
                inputs.push(("position_ids".to_string(), pos_tensor.into()));
            }

            // Pass KV cache from previous step
            let num_kv_heads = self.bundle.num_kv_heads;
            let head_dim = self.bundle.head_dim;
            if num_kv_heads > 0 && head_dim > 0 {
                for (layer, (k, v)) in state.cache.iter().enumerate() {
                    let k_tensor = ort::value::Tensor::from_array(k.clone())
                        .map_err(|e| ExecError::Other(anyhow::anyhow!("ort kv: {}", e)))?;
                    let v_tensor = ort::value::Tensor::from_array(v.clone())
                        .map_err(|e| ExecError::Other(anyhow::anyhow!("ort kv: {}", e)))?;
                    inputs.push((format!("past_key_values.{}.key", layer), k_tensor.into()));
                    inputs.push((format!("past_key_values.{}.value", layer), v_tensor.into()));
                }
            }

            let input_refs: Vec<(&str, &ort::value::Value)> =
                inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

            let outputs = ort_session
                .run(input_refs)
                .map_err(|e| ExecError::Other(anyhow::anyhow!("ort run: {}", e)))?;

            // Extract logits — shape (batch=1, seq_len=1, vocab_size)
            let logits_output = outputs
                .get("logits")
                .ok_or_else(|| ExecError::Other(anyhow::anyhow!("missing logits output")))?;
            let (shape, data) = logits_output
                .try_extract_tensor::<f32>()
                .map_err(|e| ExecError::Other(anyhow::anyhow!("extract logits: {}", e)))?;

            let vocab_size = *shape.last().unwrap_or(&0) as usize;
            let logits = data[data.len().saturating_sub(vocab_size)..].to_vec();

            // Extract updated KV cache
            let new_cache =
                super::session::Session::extract_kv_cache(&outputs, self.bundle.num_layers)?;

            (logits, new_cache)
        };

        // Update state with new KV cache
        let state = self
            .state
            .as_mut()
            .expect("OnnxPuller::next called after init");
        state.cache = new_cache;
        state.cur_pos += 1;

        Ok(logits)
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
            if self.stopped.load(Ordering::SeqCst) {
                let stats = self.stats_now();
                self.hooks.emit(HookEvent::FinalStats {
                    session_id: self.session_id,
                    stats,
                });
                self.done = true;
                self.filter.finalize(TokenEvent::Stopped);
                continue;
            }
            if self.paused.load(Ordering::SeqCst) {
                return Some(Ok(TokenEvent::Paused));
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
                continue;
            }

            let feed_token = self.last_token_id.unwrap_or(0);
            let logits = match self.forward_one_token(feed_token) {
                Ok(l) => l,
                Err(e) => {
                    self.done = true;
                    self.filter.push_err(e);
                    continue;
                }
            };

            let token_id = self.sampler.sample_from_logits(&logits);
            self.last_token_id = Some(token_id);

            if let Some(eos_id) = self.bundle.tokenizer.eos_id()
                && token_id == eos_id
            {
                let stats = self.stats_now();
                self.hooks.emit(HookEvent::FinalStats {
                    session_id: self.session_id,
                    stats,
                });
                self.done = true;
                self.filter.finalize(TokenEvent::Eos);
                continue;
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

            self.filter.push_token(token_id, text);
        }
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

impl crate::backend::traits::TokenPullerDyn for TokenPuller {
    fn next_event(
        &mut self,
    ) -> Option<Result<crate::generation::TokenEvent, crate::engine::ExecError>> {
        <Self as Iterator>::next(self)
    }
}
