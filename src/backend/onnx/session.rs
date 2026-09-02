//! ONNX inference session — manages KV cache and generation state.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use ndarray::Array2;

use super::bundle::ModelBundle;
use super::puller::TokenPuller;
use crate::Message;
use crate::backend::common::chat_template::ChatTemplate;
use crate::engine::{ExecError, HookBus, HookEvent, Settings};
use crate::generation::GenSpec;
use crate::session_rt::prompt::merge_prompts;
use crate::types::message::{MessageBody, MessageContent, TokenizerConfigToken};

use parking_lot::{Mutex, RwLock};

pub type SessionId = u64;

/// KV cache state for ONNX — stored as raw f32 tensors per layer.
pub(crate) type KvCache = Vec<(ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)>;

pub(crate) struct DecodeState {
    pub cache: KvCache,
    pub cur_pos: usize,
}

impl std::fmt::Debug for DecodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeState(ONNX)")
            .field("cur_pos", &self.cur_pos)
            .field("layers", &self.cache.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct Session {
    pub id: SessionId,
    pub bundle: Arc<ModelBundle>,
    hooks: Arc<HookBus>,
    settings: Settings,
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    state: Arc<Mutex<Option<DecodeState>>>,
    messages: RwLock<Vec<Message>>,
}

impl Session {
    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }
    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    pub fn pull(&self, mut gen_spec: GenSpec) -> Result<TokenPuller, ExecError> {
        if gen_spec.max_tokens.is_none() {
            gen_spec.max_tokens = self.settings.stopping.max_tokens;
        }
        let mut guard = self.state.lock();
        let state = guard
            .take()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;

        let state_slot = Arc::downgrade(&self.state);

        let puller = TokenPuller::new(
            self.id,
            self.hooks.clone(),
            self.bundle.clone(),
            state_slot,
            state,
            gen_spec,
            self.paused.clone(),
            self.stopped.clone(),
        );
        Ok(puller)
    }

    pub(crate) fn new(
        id: SessionId,
        bundle: Arc<ModelBundle>,
        hooks: Arc<HookBus>,
        settings: Settings,
        messages: Vec<Message>,
        persona: Option<&crate::types::Persona>,
    ) -> Result<Self, ExecError> {
        let mut messages = messages;

        let include_meta = settings.prompt.include_meta.unwrap_or(true);
        let meta_prompt = if include_meta {
            crate::session_rt::prompt::build_meta_prompt()
        } else {
            String::new()
        };
        let system_prompt = settings.prompt.system_prompt.as_deref();
        let merged_prompt = merge_prompts(&meta_prompt, system_prompt, persona);

        let has_system = messages.iter().any(|m| m.role == "system");
        if !has_system && !merged_prompt.trim().is_empty() {
            messages.insert(
                0,
                Message {
                    role: "system".into(),
                    body: MessageBody::Content {
                        content: MessageContent::SingleText(merged_prompt),
                    },
                    name: None,
                    tool_call_id: None,
                },
            );
        }

        let chat_template = ChatTemplate::new(
            bundle.chat_template_str.clone(),
            Some(TokenizerConfigToken::String(bundle.bos_str.clone())),
            Some(TokenizerConfigToken::String(bundle.eos_str.clone())),
        );

        let prompt = chat_template
            .apply(messages.clone(), None, None)
            .map_err(ExecError::Other)?;

        let tokens = bundle
            .tokenizer
            .encode(&prompt, true)
            .map_err(ExecError::Other)?;

        let total_tokens = tokens.len();
        hooks.emit(HookEvent::SessionPrefillStart {
            session_id: id,
            prompt_tokens: total_tokens,
        });

        // Build ort input tensors
        let token_ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let attention_mask: Vec<i64> = vec![1i64; total_tokens];
        let position_ids: Vec<i64> = (0..total_tokens as i64).collect();

        let ids_array = Array2::from_shape_vec((1, total_tokens), token_ids)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;
        let mask_array = Array2::from_shape_vec((1, total_tokens), attention_mask)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;
        let pos_array = Array2::from_shape_vec((1, total_tokens), position_ids)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;

        let ids_tensor = ort::value::Tensor::from_array(ids_array)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;
        let mask_tensor = ort::value::Tensor::from_array(mask_array)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;
        let pos_tensor = if bundle.has_position_ids {
            Some(
                ort::value::Tensor::from_array(pos_array)
                    .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?,
            )
        } else {
            None
        };

        // Build empty KV cache inputs for the first run.
        // Shape: [1, num_kv_heads, 0, head_dim] — zero-length sequence dimension.
        let num_kv_heads = bundle.num_kv_heads;
        let head_dim_val = bundle.head_dim;

        // Scope ort session usage so borrows end before moving bundle
        let cache = {
            // Build KV cache tensors as ort Values
            let mut kv_values: Vec<(String, ort::value::Value)> = Vec::new();
            if num_kv_heads > 0 && head_dim_val > 0 {
                for layer in 0..bundle.num_layers {
                    for kind in &["key", "value"] {
                        let empty: ndarray::ArrayD<f32> =
                            ndarray::ArrayD::zeros(ndarray::IxDyn(&[
                                1,
                                num_kv_heads,
                                0,
                                head_dim_val,
                            ]));
                        let tensor = ort::value::Tensor::from_array(empty).map_err(|e| {
                            ExecError::Other(anyhow::anyhow!("ort kv tensor: {}", e))
                        })?;
                        kv_values
                            .push((format!("past_key_values.{}.{}", layer, kind), tensor.into()));
                    }
                }
            }

            // Assemble all inputs
            let mut inputs: Vec<(String, ort::value::Value)> = vec![
                ("input_ids".to_string(), ids_tensor.into()),
                ("attention_mask".to_string(), mask_tensor.into()),
            ];
            if let Some(pt) = pos_tensor {
                inputs.push(("position_ids".to_string(), pt.into()));
            }
            inputs.extend(kv_values);

            let input_refs: Vec<(&str, &ort::value::Value)> =
                inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

            let mut ort_session = bundle.session.lock();
            let outputs = ort_session
                .run(input_refs)
                .map_err(|e| ExecError::Other(anyhow::anyhow!("ort run: {}", e)))?;
            Self::extract_kv_cache(&outputs, bundle.num_layers)?
        };

        hooks.emit(HookEvent::SessionPrefillOk {
            session_id: id,
            prompt_tokens: total_tokens,
        });

        Ok(Self {
            id,
            bundle,
            hooks,
            settings,
            paused: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(Some(DecodeState {
                cache,
                cur_pos: total_tokens,
            }))),
            messages: RwLock::new(messages),
        })
    }

    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        if new_messages.is_empty() {
            return Ok(0);
        }

        {
            let mut msgs = self.messages.write();
            msgs.extend(new_messages.clone());
        }

        let tpl = ChatTemplate::new(
            self.bundle.chat_template_str.clone(),
            Some(TokenizerConfigToken::String(self.bundle.bos_str.clone())),
            Some(TokenizerConfigToken::String(self.bundle.eos_str.clone())),
        );

        let delta_text = tpl
            .apply(new_messages, None, None)
            .map_err(ExecError::Other)?;

        let delta_tokens = self
            .bundle
            .tokenizer
            .encode(&delta_text, false)
            .map_err(ExecError::Other)?;

        if delta_tokens.is_empty() {
            return Ok(0);
        }

        let mut guard = self.state.lock();
        let st = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;

        let delta_len = delta_tokens.len();
        self.hooks.emit(HookEvent::SessionPrefillStart {
            session_id: self.id,
            prompt_tokens: delta_len,
        });

        let token_ids: Vec<i64> = delta_tokens.iter().map(|&t| t as i64).collect();
        let total_seq = st.cur_pos + delta_len;
        let attention_mask: Vec<i64> = vec![1i64; total_seq];

        let ids_array = Array2::from_shape_vec((1, delta_len), token_ids)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;
        let mask_array = Array2::from_shape_vec((1, total_seq), attention_mask)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;

        let ids_tensor = ort::value::Tensor::from_array(ids_array)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;
        let mask_tensor = ort::value::Tensor::from_array(mask_array)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;

        let new_cache = {
            let mut ort_session = self.bundle.session.lock();
            // TODO: pass KV cache as inputs for incremental decode
            let outputs = ort_session
                .run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor,
                ])
                .map_err(|e| ExecError::Other(anyhow::anyhow!("ort run: {}", e)))?;
            Self::extract_kv_cache(&outputs, self.bundle.num_layers)?
        };

        st.cache = new_cache;
        st.cur_pos = total_seq;

        self.hooks.emit(HookEvent::SessionPrefillOk {
            session_id: self.id,
            prompt_tokens: delta_len,
        });

        Ok(0)
    }

    pub(crate) fn extract_kv_cache(
        outputs: &ort::session::SessionOutputs<'_>,
        num_layers: usize,
    ) -> Result<KvCache, ExecError> {
        let mut cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let key_name = format!("present.{}.key", i);
            let val_name = format!("present.{}.value", i);

            let key = outputs
                .get(&key_name)
                .ok_or_else(|| ExecError::Other(anyhow::anyhow!("missing output: {}", key_name)))?
                .try_extract_tensor::<f32>()
                .map_err(|e| ExecError::Other(anyhow::anyhow!("extract {}: {}", key_name, e)))?;
            let val = outputs
                .get(&val_name)
                .ok_or_else(|| ExecError::Other(anyhow::anyhow!("missing output: {}", val_name)))?
                .try_extract_tensor::<f32>()
                .map_err(|e| ExecError::Other(anyhow::anyhow!("extract {}: {}", val_name, e)))?;

            let key_shape: Vec<usize> = key.0.iter().map(|&s| s as usize).collect();
            let val_shape: Vec<usize> = val.0.iter().map(|&s| s as usize).collect();

            let key_arr =
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&key_shape), key.1.to_vec())
                    .map_err(|e| ExecError::Other(anyhow::anyhow!("kv reshape: {}", e)))?;
            let val_arr =
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&val_shape), val.1.to_vec())
                    .map_err(|e| ExecError::Other(anyhow::anyhow!("kv reshape: {}", e)))?;

            cache.push((key_arr, val_arr));
        }
        Ok(cache)
    }
}

// ─── Trait impls (Phase 2) ─────────────────────────────────────────────────

impl crate::backend::traits::BackendSession for Session {
    fn id(&self) -> SessionId {
        self.id
    }
    fn pause(&self) {
        Session::pause(self)
    }
    fn resume(&self) {
        Session::resume(self)
    }
    fn stop(&self) {
        Session::stop(self)
    }
    fn pull(
        &self,
        spec: GenSpec,
    ) -> Result<Box<dyn crate::backend::traits::TokenPullerDyn>, ExecError> {
        let p = Session::pull(self, spec)?;
        Ok(Box::new(p) as Box<dyn crate::backend::traits::TokenPullerDyn>)
    }
    fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        Session::append_messages(self, new_messages)
    }
}
