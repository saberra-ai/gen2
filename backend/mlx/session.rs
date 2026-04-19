//! MLX inference session — manages KV cache and generation state.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use mlx_rs::ops::indexing::IndexOp;

use super::bundle::ModelBundle;
use super::model::{KvCache, ModelConfig};
use super::puller::TokenPuller;
use crate::gen2::Message;
use crate::gen2::backend::common::chat_template::ChatTemplate;
use crate::gen2::engine::{ExecError, HookBus, HookEvent, Settings};
use crate::gen2::generation::GenSpec;
use crate::gen2::session_rt::prompt::merge_prompts;
use crate::types::message::{MessageBody, MessageContent, TokenizerConfigToken};

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use parking_lot::{Mutex, RwLock};

pub type SessionId = u64;

// ─── PrefixCacheEntry ─────────────────────────────────────────────────────────

/// Cached KV state for a fixed system prompt + persona.
///
/// `Array::clone()` increments MLX's reference count (O(1), no Metal copy).
/// The forward pass creates new concatenated arrays from these without mutating
/// them, so sharing across sessions is data-race-free.
pub struct PrefixCacheEntry {
    /// FNV-style hash of the rendered prefix text (model + system prompt + persona).
    pub key: u64,
    /// KV state after prefilling the prefix, ready to be cloned into a session.
    pub kv: KvCache,
    /// Number of prefix tokens (= k.shape()[2] for non-evicted cache).
    pub cur_pos: usize,
    pub policy: CachePolicy,
}

impl PrefixCacheEntry {
    /// Clone the KV arrays (shallow MLX refcount increment) into a fresh `DecodeState`.
    pub fn clone_into_state(&self, last_token: u32) -> DecodeState {
        let kv: KvCache = self
            .kv
            .iter()
            .map(|e| e.as_ref().map(|(k, v)| (k.clone(), v.clone())))
            .collect();
        DecodeState {
            cache: kv,
            cur_pos: self.cur_pos,
            cache_len: self.cur_pos,
            policy: CachePolicy {
                evict_trigger: self.policy.evict_trigger,
                evict_to: self.policy.evict_to,
            },
            pending_logits: None,
            last_token,
            evictions: 0,
        }
    }
}

/// Hash a string to a u64 key using the standard `DefaultHasher`.
pub fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

// ─── CachePolicy ──────────────────────────────────────────────────────────────

/// Number of "attention sink" tokens to preserve at the start of the cache.
/// These early tokens attract disproportionate attention mass; dropping them
/// causes quality collapse on long conversations (StreamingLLM, 2023).
const SINK_TOKENS: usize = 4;

/// Engine-level KV cache budget, enforced independently of any model sliding window.
pub(crate) struct CachePolicy {
    /// Eviction triggers at this fill level (80% of hard ceiling).
    pub evict_trigger: usize,
    /// Truncate cache to this many tokens on eviction (60% of ceiling).
    pub evict_to: usize,
}

impl CachePolicy {
    /// Derive a safe budget from the model config and available device RAM.
    /// Uses 20% of physical RAM, capped at max_position_embeddings.
    pub fn compute(config: &ModelConfig, num_cache_slots: usize) -> Self {
        let kv_heads = config.num_key_value_heads.max(1);
        let head_dim = config.head_dim();
        // fp16: 2 bytes × K and V × kv_heads × head_dim per cached token per layer
        let bytes_per_token = (num_cache_slots * 2 * kv_heads * head_dim * 2).max(1);

        let ram = total_ram_bytes();
        let budget_bytes = if ram > 0 {
            ram / 5 // 20% of physical RAM
        } else {
            8u64 * 1024 * 1024 * 1024 / 5 // conservative 8 GB default
        };

        let max_by_ram = (budget_bytes as usize) / bytes_per_token;
        let max_by_model = config.max_position_embeddings;
        let ceiling = max_by_ram.min(max_by_model).max(512);
        // Trigger proactively at 80% — never touch the hard ceiling.
        let evict_trigger = (ceiling * 4) / 5;
        // After eviction keep 60% — gives headroom before the next trigger.
        let evict_to = (ceiling * 3) / 5;

        tracing::debug!(
            ceiling,
            evict_trigger,
            evict_to,
            ram_gb = ram / (1024 * 1024 * 1024),
            "cache policy computed"
        );

        Self {
            evict_trigger,
            evict_to,
        }
    }
}

/// Returns total physical RAM in bytes. Returns 0 on unsupported platforms.
fn total_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut mem: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        unsafe {
            libc::sysctlbyname(
                b"hw.memsize\0".as_ptr() as *const libc::c_char,
                &mut mem as *mut u64 as *mut libc::c_void,
                &mut size as *mut usize,
                std::ptr::null_mut(),
                0,
            );
        }
        mem
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

// ─── DecodeState ──────────────────────────────────────────────────────────────

/// Decode state for the MLX backend — owns the KV cache.
pub(crate) struct DecodeState {
    pub cache: KvCache,
    /// Absolute sequence position for RoPE — never decremented on eviction.
    pub cur_pos: usize,
    /// Tokens actually present in the cache arrays. Decremented on eviction.
    pub cache_len: usize,
    pub policy: CachePolicy,
    /// Logits from the most recent prefill / append_messages call.
    /// The puller consumes this on its first iteration to avoid an extra forward pass.
    pub pending_logits: Option<mlx_rs::Array>,
    /// The last token ID fed into the model (last prompt token after prefill,
    /// then each sampled token). The puller updates this after every sample.
    pub last_token: u32,
    /// Number of engine-level evictions in this session (for observability).
    pub evictions: u32,
}

impl DecodeState {
    /// Evict old cache entries when `cache_len` exceeds the proactive trigger.
    ///
    /// Preserves the first `SINK_TOKENS` entries (attention sinks) and the most
    /// recent `evict_to - SINK_TOKENS` entries. `cur_pos` is never changed —
    /// it tracks the absolute sequence position for RoPE correctness.
    pub fn maybe_evict(&mut self) {
        if self.cache_len < self.policy.evict_trigger {
            return;
        }
        let target = self.policy.evict_to;
        let sink = SINK_TOKENS.min(target);
        let window = target - sink;

        for slot in self.cache.iter_mut() {
            if let Some(kv) = slot {
                let seq = kv.0.shape()[2] as usize;
                if seq <= target {
                    continue;
                }
                let win_start = (seq - window) as i32;
                let sink_end = sink as i32;
                // sinks: [0..sink_end], recent window: [win_start..seq]
                let sk = kv.0.index((.., .., 0..sink_end, ..));
                let sv = kv.1.index((.., .., 0..sink_end, ..));
                let wk = kv.0.index((.., .., win_start.., ..));
                let wv = kv.1.index((.., .., win_start.., ..));
                kv.0 = mlx_rs::ops::concatenate_axis(&[&sk, &wk], 2).expect("mlx op");
                kv.1 = mlx_rs::ops::concatenate_axis(&[&sv, &wv], 2).expect("mlx op");
            }
        }
        tracing::debug!(
            cur_pos = self.cur_pos,
            old_cache_len = self.cache_len,
            new_cache_len = target,
            sink_tokens = sink,
            "kv cache eviction (attention sinks preserved)"
        );
        self.cache_len = target;
        self.evictions += 1;
    }
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

    /// Public constructor — no prefix caching (used by tests).
    pub(crate) fn new(
        id: SessionId,
        bundle: Arc<ModelBundle>,
        hooks: Arc<HookBus>,
        settings: Settings,
        messages: Vec<Message>,
        persona: Option<&crate::types::Persona>,
    ) -> Result<Self, ExecError> {
        let (session, _) =
            Self::new_with_prefix(id, bundle, hooks, settings, messages, persona, 0, None)?;
        Ok(session)
    }

    /// Full constructor used by `Engine::start_session`.
    ///
    /// If `cached_prefix` is Some the first `cached_prefix.cur_pos` tokens are
    /// assumed already in the KV cache; only the delta is prefilled.
    ///
    /// Returns `(Session, Option<PrefixCacheEntry>)`. The entry is Some when a
    /// fresh prefix was prefilled that the engine should store for later hits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_prefix(
        id: SessionId,
        bundle: Arc<ModelBundle>,
        hooks: Arc<HookBus>,
        settings: Settings,
        messages: Vec<Message>,
        persona: Option<&crate::types::Persona>,
        prefix_key: u64,
        cached_prefix: Option<DecodeState>,
    ) -> Result<(Self, Option<PrefixCacheEntry>), ExecError> {
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

        // Tokenize the full prompt (all messages).
        let full_prompt = chat_template
            .apply(messages.clone(), None, None)
            .map_err(|e| ExecError::Other(e.into()))?;

        if std::env::var("PIO_MLX_DEBUG_PROMPT").is_ok() {
            eprintln!(
                "\n── Session::new full prompt ({} bytes) ──\n{:?}\n──\n",
                full_prompt.len(),
                full_prompt
            );
        }

        let full_tokens = bundle
            .tokenizer
            .encode(&full_prompt, true)
            .map_err(ExecError::Other)?;

        let cache_slots = bundle.model.num_non_shared_layers();
        let policy = CachePolicy::compute(&bundle.config, cache_slots);

        // ── Prefix cache HIT ──────────────────────────────────────────────────
        if let Some(mut state) = cached_prefix {
            let prefix_len = state.cur_pos;
            if full_tokens.len() > prefix_len {
                let delta = &full_tokens[prefix_len..];
                let total = full_tokens.len();
                hooks.emit(HookEvent::SessionPrefillStart {
                    session_id: id,
                    prompt_tokens: delta.len(),
                });
                let delta_pos = state.cur_pos;
                let b = &bundle;
                let delta_logits = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    b.model.forward(delta, delta_pos, &mut state.cache, &b.rope)
                }))
                .map_err(|e| ExecError::OutOfMemory(oom_msg(e)))?;

                let last_token = delta.last().copied().unwrap_or(state.last_token);
                state.cur_pos = total;
                state.cache_len = total;
                state.pending_logits = Some(delta_logits);
                state.last_token = last_token;
                state.maybe_evict();

                hooks.emit(HookEvent::SessionPrefillOk {
                    session_id: id,
                    prompt_tokens: delta.len(),
                });
                tracing::debug!(prefix_len, delta = delta.len(), "prefix cache HIT");

                let session = Self {
                    id,
                    bundle,
                    hooks,
                    settings,
                    paused: Arc::new(AtomicBool::new(false)),
                    stopped: Arc::new(AtomicBool::new(false)),
                    state: Arc::new(Mutex::new(Some(state))),
                    messages: RwLock::new(messages),
                };
                return Ok((session, None));
            }
        }

        // ── Prefix cache MISS — attempt two-phase prefill ─────────────────────
        // Extract system messages for the prefix phase.
        let sys_messages: Vec<Message> = messages
            .iter()
            .take_while(|m| m.role == "system")
            .cloned()
            .collect();

        let new_prefix_entry: Option<PrefixCacheEntry> = if !sys_messages.is_empty()
            && prefix_key != 0
        {
            // Render and tokenize just the system prefix. Pass
            // `add_generation_prompt: false` so the prefix is a strict prefix
            // of the full tokenization (templates like Gemma append a
            // `<|turn>model\n` suffix otherwise, breaking the prefix check).
            match chat_template.apply_with_options(sys_messages, None, None, false) {
                Ok(prefix_text) => {
                    match bundle.tokenizer.encode(&prefix_text, true) {
                        Ok(prefix_tokens)
                            if prefix_tokens.len() < full_tokens.len()
                                && full_tokens.starts_with(&prefix_tokens) =>
                        {
                            // Prefill system prefix first.
                            let mut prefix_cache: KvCache = vec![None; cache_slots];
                            let b = &bundle;
                            let prefix_logits =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    b.model
                                        .forward(&prefix_tokens, 0, &mut prefix_cache, &b.rope)
                                }))
                                .map_err(|e| ExecError::OutOfMemory(oom_msg(e)))?;
                            let prefix_len = prefix_tokens.len();
                            let _ = prefix_logits; // not used — we'll run the delta next

                            // Snapshot the KV state for later sessions.
                            let entry_kv: KvCache = prefix_cache
                                .iter()
                                .map(|e| e.as_ref().map(|(k, v)| (k.clone(), v.clone())))
                                .collect();
                            let entry = PrefixCacheEntry {
                                key: prefix_key,
                                kv: entry_kv,
                                cur_pos: prefix_len,
                                policy: CachePolicy {
                                    evict_trigger: policy.evict_trigger,
                                    evict_to: policy.evict_to,
                                },
                            };

                            // Prefill the delta (non-prefix tokens) into the same cache.
                            let delta = &full_tokens[prefix_len..];
                            hooks.emit(HookEvent::SessionPrefillStart {
                                session_id: id,
                                prompt_tokens: full_tokens.len(),
                            });
                            let delta_logits =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    b.model
                                        .forward(delta, prefix_len, &mut prefix_cache, &b.rope)
                                }))
                                .map_err(|e| ExecError::OutOfMemory(oom_msg(e)))?;

                            let last_token = full_tokens.last().copied().unwrap_or(0);
                            let total = full_tokens.len();
                            let mut init_state = DecodeState {
                                cache: prefix_cache,
                                cur_pos: total,
                                cache_len: total,
                                policy,
                                pending_logits: Some(delta_logits),
                                last_token,
                                evictions: 0,
                            };
                            init_state.maybe_evict();
                            hooks.emit(HookEvent::SessionPrefillOk {
                                session_id: id,
                                prompt_tokens: total,
                            });
                            tracing::debug!(
                                prefix_len,
                                delta = delta.len(),
                                "prefix cache MISS — two-phase prefill, entry stored"
                            );
                            let session = Self {
                                id,
                                bundle,
                                hooks,
                                settings,
                                paused: Arc::new(AtomicBool::new(false)),
                                stopped: Arc::new(AtomicBool::new(false)),
                                state: Arc::new(Mutex::new(Some(init_state))),
                                messages: RwLock::new(messages),
                            };
                            return Ok((session, Some(entry)));
                        }
                        _ => None,
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };
        let _ = new_prefix_entry; // unreachable but satisfies type checker

        // ── Single-phase full prefill (no caching possible) ───────────────────
        let total_tokens = full_tokens.len();
        hooks.emit(HookEvent::SessionPrefillStart {
            session_id: id,
            prompt_tokens: total_tokens,
        });

        let mut cache: KvCache = vec![None; cache_slots];
        let b = &bundle;
        let prefill_logits = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.model.forward(&full_tokens, 0, &mut cache, &b.rope)
        }))
        .map_err(|e| ExecError::OutOfMemory(oom_msg(e)))?;

        let last_prompt_token = full_tokens.last().copied().unwrap_or(0);
        hooks.emit(HookEvent::SessionPrefillOk {
            session_id: id,
            prompt_tokens: total_tokens,
        });

        let mut init_state = DecodeState {
            cache,
            cur_pos: total_tokens,
            cache_len: total_tokens,
            policy,
            pending_logits: Some(prefill_logits),
            last_token: last_prompt_token,
            evictions: 0,
        };
        init_state.maybe_evict();

        let session = Self {
            id,
            bundle,
            hooks,
            settings,
            paused: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(Some(init_state))),
            messages: RwLock::new(messages),
        };
        Ok((session, None))
    }

    // ── helpers ───────────────────────────────────────────────────────────────

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

        // Prefill delta — guard against MLX OOM.
        let delta_pos = st.cur_pos;
        let b = &self.bundle;
        let delta_logits = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.model
                .forward(&delta_tokens, delta_pos, &mut st.cache, &b.rope)
        }))
        .map_err(|e| ExecError::OutOfMemory(oom_msg(e)))?;
        st.cur_pos += delta_tokens.len();
        st.cache_len += delta_tokens.len();
        st.last_token = delta_tokens.last().copied().unwrap_or(st.last_token);
        st.pending_logits = Some(delta_logits);
        st.maybe_evict();

        self.hooks.emit(HookEvent::SessionPrefillOk {
            session_id: self.id,
            prompt_tokens: delta_tokens.len(),
        });

        Ok(0)
    }
}

// ─── Trait impls (Phase 2) ─────────────────────────────────────────────────

/// Extract a readable message from a caught panic payload (OOM or other MLX panic).
fn oom_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "MLX forward pass panicked (likely OOM)".to_string()
    }
}

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
