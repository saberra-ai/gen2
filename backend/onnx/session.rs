//! ONNX inference session — manages KV cache and generation state.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use ndarray::Array2;

use super::bundle::ModelBundle;
use super::puller::TokenPuller;
use crate::gen2::Message;
use crate::gen2::engine::{ExecError, HookBus, HookEvent, Settings};
use crate::gen2::generation::GenSpec;
use crate::gen2::session_rt::prompt::merge_prompts;
use crate::gen2::backend::common::chat_template::ChatTemplate;
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
            crate::gen2::session_rt::prompt::build_meta_prompt()
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
            .map_err(|e| ExecError::Other(e.into()))?;

        let tokens = bundle
            .tokenizer
            .encode(&prompt, true)
            .map_err(|e| ExecError::Other(e))?;

        let total_tokens = tokens.len();
        hooks.emit(HookEvent::SessionPrefillStart {
            session_id: id,
            prompt_tokens: total_tokens,
        });

        // Build ort input tensors
        let token_ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let attention_mask: Vec<i64> = vec![1i64; total_tokens];

        let ids_array = Array2::from_shape_vec((1, total_tokens), token_ids)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;
        let mask_array = Array2::from_shape_vec((1, total_tokens), attention_mask)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ndarray: {}", e)))?;

        let ids_tensor = ort::value::Tensor::from_array(ids_array)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;
        let mask_tensor = ort::value::Tensor::from_array(mask_array)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("ort tensor: {}", e)))?;

        // Scope ort session usage so borrows end before moving bundle
        let cache = {
            let mut ort_session = bundle.session.lock();
            let outputs = ort_session
                .run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor,
                ])
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
            .map_err(|e| ExecError::Other(e.into()))?;

        let delta_tokens = self
            .bundle
            .tokenizer
            .encode(&delta_text, false)
            .map_err(|e| ExecError::Other(e))?;

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

    fn extract_kv_cache(
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
