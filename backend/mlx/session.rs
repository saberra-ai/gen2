//! MLX inference session — manages KV cache and generation state.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::bundle::ModelBundle;
use super::model::KvCache;
use super::puller::TokenPuller;
use crate::gen2::Message;
use crate::gen2::engine::{ExecError, HookBus, HookEvent, Settings};
use crate::gen2::generation::GenSpec;
use crate::gen2::session_rt::prompt::merge_prompts;
use crate::generation::model_runner::chat_template::ChatTemplate;
use crate::generation::model_runner::types::{MessageBody, MessageContent, TokenizerConfigToken};

use parking_lot::{Mutex, RwLock};

pub type SessionId = u64;

/// Decode state for the MLX backend — owns the KV cache.
pub(crate) struct DecodeState {
    pub cache: KvCache,
    pub cur_pos: usize,
}

impl std::fmt::Debug for DecodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeState(MLX)")
            .field("cur_pos", &self.cur_pos)
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

        // Build chat template from tokenizer config
        // For MLX models, we use a simpler approach: check for tokenizer_config.json
        // or use a default Llama3 template
        let bos_str = bundle
            .tokenizer
            .bos_id()
            .map(|id| bundle.tokenizer.decode(&[id]).unwrap_or_default())
            .unwrap_or_default();
        let eos_str = bundle
            .tokenizer
            .eos_id()
            .map(|id| bundle.tokenizer.decode(&[id]).unwrap_or_default())
            .unwrap_or_default();

        // Try to load chat template from tokenizer_config.json in the model dir
        let chat_template_str = Self::load_chat_template_from_dir(&bundle.model_dir)
            .unwrap_or_else(|| Self::default_llama3_template());

        let chat_template = ChatTemplate::new(
            chat_template_str,
            Some(TokenizerConfigToken::String(bos_str)),
            Some(TokenizerConfigToken::String(eos_str)),
        );

        let prompt = chat_template
            .apply(messages.clone(), None, None)
            .map_err(|e| ExecError::Other(e.into()))?;

        // Tokenize with the HF tokenizer
        let tokens = bundle
            .tokenizer
            .encode(&prompt, true)
            .map_err(|e| ExecError::Other(e))?;

        let total_tokens = tokens.len();

        hooks.emit(HookEvent::SessionPrefillStart {
            session_id: id,
            prompt_tokens: total_tokens,
        });

        // Initialize KV cache (one entry per layer)
        let mut cache: KvCache = vec![None; bundle.config.num_hidden_layers];

        // Run prefill: forward pass with all prompt tokens
        let _logits = bundle.model.forward(&tokens, &mut cache, &bundle.rope);

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

    /// Append new messages and prefill the delta.
    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        if new_messages.is_empty() {
            return Ok(0);
        }

        {
            let mut msgs = self.messages.write();
            msgs.extend(new_messages.clone());
        }

        // Tokenize just the new messages
        let bos_str = self
            .bundle
            .tokenizer
            .bos_id()
            .map(|id| self.bundle.tokenizer.decode(&[id]).unwrap_or_default())
            .unwrap_or_default();
        let eos_str = self
            .bundle
            .tokenizer
            .eos_id()
            .map(|id| self.bundle.tokenizer.decode(&[id]).unwrap_or_default())
            .unwrap_or_default();

        let chat_template_str = Self::load_chat_template_from_dir(&self.bundle.model_dir)
            .unwrap_or_else(|| Self::default_llama3_template());

        let tpl = ChatTemplate::new(
            chat_template_str,
            Some(TokenizerConfigToken::String(bos_str)),
            Some(TokenizerConfigToken::String(eos_str)),
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

        self.hooks.emit(HookEvent::SessionPrefillStart {
            session_id: self.id,
            prompt_tokens: delta_tokens.len(),
        });

        // Prefill delta into existing KV cache
        let _logits = self
            .bundle
            .model
            .forward(&delta_tokens, &mut st.cache, &self.bundle.rope);
        st.cur_pos += delta_tokens.len();

        self.hooks.emit(HookEvent::SessionPrefillOk {
            session_id: self.id,
            prompt_tokens: delta_tokens.len(),
        });

        Ok(0)
    }

    /// Load the Jinja2 chat template from `tokenizer_config.json` in the model directory.
    fn load_chat_template_from_dir(model_dir: &std::path::Path) -> Option<String> {
        let config_path = model_dir.join("tokenizer_config.json");
        let content = std::fs::read_to_string(&config_path).ok()?;
        let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
        parsed.get("chat_template")?.as_str().map(|s| s.to_string())
    }

    fn default_llama3_template() -> String {
        // Minimal Llama3 instruct template
        r#"{% for message in messages %}{% if message.role == 'system' %}<|start_header_id|>system<|end_header_id|>

{{ message.content }}<|eot_id|>{% elif message.role == 'user' %}<|start_header_id|>user<|end_header_id|>

{{ message.content }}<|eot_id|>{% elif message.role == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

{{ message.content }}<|eot_id|>{% endif %}{% endfor %}<|start_header_id|>assistant<|end_header_id|>

"#.to_string()
    }
}
