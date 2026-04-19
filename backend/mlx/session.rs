//! MLX inference session — manages KV cache and generation state.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::bundle::ModelBundle;
use super::model::KvCache;
use super::puller::TokenPuller;
use crate::gen2::Message;
use crate::gen2::backend::common::chat_template::ChatTemplate;
use crate::gen2::engine::{ExecError, HookBus, HookEvent, Settings};
use crate::gen2::generation::GenSpec;
use crate::gen2::session_rt::prompt::merge_prompts;
use crate::types::message::{MessageBody, MessageContent, TokenizerConfigToken};

use parking_lot::{Mutex, RwLock};

pub type SessionId = u64;

/// Decode state for the MLX backend — owns the KV cache.
pub(crate) struct DecodeState {
    pub cache: KvCache,
    pub cur_pos: usize,
    /// Logits from the most recent prefill / append_messages call.
    /// The puller consumes this on its first iteration to avoid an extra forward pass.
    pub pending_logits: Option<mlx_rs::Array>,
    /// The last token ID fed into the model (last prompt token after prefill,
    /// then each sampled token). The puller updates this after every sample.
    pub last_token: u32,
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

        // Use chat template + BOS/EOS cached in bundle (loaded once at model load time)
        let chat_template = ChatTemplate::new(
            bundle.chat_template_str.clone(),
            Some(TokenizerConfigToken::String(bundle.bos_str.clone())),
            Some(TokenizerConfigToken::String(bundle.eos_str.clone())),
        );

        let prompt = chat_template
            .apply(messages.clone(), None, None)
            .map_err(|e| ExecError::Other(e.into()))?;

        if std::env::var("PIO_MLX_DEBUG_PROMPT").is_ok() {
            eprintln!(
                "\n── Session::new rendered prompt ({} bytes) ──\n{:?}\n──\n",
                prompt.len(),
                prompt
            );
        }

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

        // Initialize KV cache (Gemma 4 uses num_non_shared slots; Llama uses num_hidden_layers)
        let cache_slots = bundle.model.num_non_shared_layers();
        let mut cache: KvCache = vec![None; cache_slots];

        // Run prefill: forward pass with all prompt tokens (offset 0 — empty cache)
        let prefill_logits = bundle.model.forward(&tokens, 0, &mut cache, &bundle.rope);
        let last_prompt_token = tokens.last().copied().unwrap_or(0);

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
                pending_logits: Some(prefill_logits),
                last_token: last_prompt_token,
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
        let tpl = ChatTemplate::new(
            self.bundle.chat_template_str.clone(),
            Some(TokenizerConfigToken::String(self.bundle.bos_str.clone())),
            Some(TokenizerConfigToken::String(self.bundle.eos_str.clone())),
        );

        let delta_text = tpl
            .apply(new_messages, None, None)
            .map_err(|e| ExecError::Other(e.into()))?;

        if std::env::var("PIO_MLX_DEBUG_PROMPT").is_ok() {
            eprintln!(
                "\n── append_messages delta ({} bytes) ──\n{:?}\n──\n",
                delta_text.len(),
                delta_text
            );
        }

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

        // Prefill delta into existing KV cache at the current true position.
        let delta_logits = self.bundle.model.forward(
            &delta_tokens,
            st.cur_pos,
            &mut st.cache,
            &self.bundle.rope,
        );
        st.cur_pos += delta_tokens.len();
        st.last_token = delta_tokens.last().copied().unwrap_or(st.last_token);
        st.pending_logits = Some(delta_logits);

        self.hooks.emit(HookEvent::SessionPrefillOk {
            session_id: self.id,
            prompt_tokens: delta_tokens.len(),
        });

        Ok(0)
    }
}

// ─── Trait impls (Phase 2) ─────────────────────────────────────────────────

impl crate::gen2::backend::traits::BackendSession for Session {
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
    ) -> Result<Box<dyn crate::gen2::backend::traits::TokenPullerDyn>, ExecError> {
        let p = Session::pull(self, spec)?;
        Ok(Box::new(p) as Box<dyn crate::gen2::backend::traits::TokenPullerDyn>)
    }
    fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        Session::append_messages(self, new_messages)
    }
    // No KV snapshot, no poison detection — defaults apply.
}
