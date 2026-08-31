//! MLX inference session — manages KV cache and generation state.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use mlx_rs::ops::indexing::IndexOp;

use super::bundle::ModelBundle;
use super::model::{KvCache, Model, ModelConfig};
use super::puller::{ArPuller, PrecomputedPuller, TokenPuller};
use crate::Message;
use crate::backend::common::chat_template::ChatTemplate;
use crate::engine::{ExecError, HookBus, HookEvent, Settings};
use crate::generation::GenSpec;
use crate::session_rt::prompt::merge_prompts;
use crate::types::message::{MessageBody, MessageChunk, MessageContent, TokenizerConfigToken};

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
                c"hw.memsize".as_ptr(),
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

        for kv in self.cache.iter_mut().flatten() {
            // Use the true fill (`cache_len`), not the array's seq dim. For
            // the default path these are equal (the slot holds the filled
            // prefix). For the `PIO_MLX_FAST` step-buffer cache the slot may
            // be an over-allocated buffer whose `shape()[2]` is the
            // capacity — clamping by `cache_len` keeps eviction correct for
            // both paths without changing flag-off behaviour.
            let cap = kv.0.shape()[2] as usize;
            let seq = self.cache_len.min(cap);
            if seq <= target {
                continue;
            }
            let win_start = (seq - window) as i32;
            let seq_i = seq as i32;
            let sink_end = sink as i32;
            // sinks: [0..sink_end], recent window: [win_start..seq]. Bound
            // the upper end at `seq` (the true fill) rather than running to
            // the array end — the fast step buffer has zero padding past
            // `seq` that must not be folded into the retained window.
            let sk = kv.0.index((.., .., 0..sink_end, ..));
            let sv = kv.1.index((.., .., 0..sink_end, ..));
            let wk = kv.0.index((.., .., win_start..seq_i, ..));
            let wv = kv.1.index((.., .., win_start..seq_i, ..));
            kv.0 = mlx_rs::ops::concatenate_axis(&[&sk, &wk], 2).expect("mlx op");
            kv.1 = mlx_rs::ops::concatenate_axis(&[&sv, &wv], 2).expect("mlx op");
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
    /// Chat-template `enable_thinking` policy for this session.
    /// Pinned at session start; `append_messages` reuses it so deltas
    /// render with the same policy as the initial prompt. `None` =
    /// template default.
    enable_thinking: Option<bool>,
    /// For DiffusionGemma only: the tokenized prompt, held verbatim instead of
    /// being prefilled into a KV cache (the model is encoder/decoder, not
    /// autoregressive — there is no streaming prefill). `pull()` feeds this to
    /// `diffusion_generate` and emits the result via a `PrecomputedPuller`.
    /// `None` for autoregressive models. Wrapped in a `Mutex` so multi-turn
    /// `append_messages` can re-render the whole conversation (the model has no
    /// reusable KV state to delta-prefill into).
    diffusion_prompt: Mutex<Option<Vec<u32>>>,
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

        // ── DiffusionGemma: non-streaming denoising, sequential emit ──────────
        // The whole 256-token canvas is denoised to completion here, then the
        // resulting ids are streamed one-by-one (decoding text per token)
        // through a `PrecomputedPuller`. Interface-compatible with the AR path:
        // same `TokenEvent::Token{..}` → `Eos` sequence callers already drain.
        let diffusion_prompt = self.diffusion_prompt.lock().clone();
        if let Some(prompt_ids) = diffusion_prompt {
            let Model::DiffusionGemma(model) = &self.bundle.model else {
                return Err(ExecError::Other(anyhow::anyhow!(
                    "session has a diffusion prompt but the model is not DiffusionGemma"
                )));
            };
            let mut params = self.bundle.diffusion_params.clone().unwrap_or_default();
            // Production knob: the user-tunable `LlmConfig.diffusion_denoising_steps`
            // (default 24) rides in on the `GenSpec` and overrides the checkpoint's
            // `max_denoising_steps` (48). 24 is ~2x faster with no measured chat
            // quality loss (20-turn A/B). `None` keeps the checkpoint value.
            if let Some(steps) = gen_spec.diffusion_denoising_steps
                && steps > 0
            {
                params.max_denoising_steps = steps;
            }
            // Test/debug override: `PIO_MLX_DENOISING_STEPS` wins over both the
            // config knob and the checkpoint — it lets a validation pass sweep
            // the step count (e.g. the 24-step A/B) WITHOUT touching config or
            // the checkpoint. Unset in normal operation. Follows the existing
            // `PIO_MLX_*` env-override convention in this backend.
            if let Ok(v) = std::env::var("PIO_MLX_DENOISING_STEPS")
                && let Ok(n) = v.trim().parse::<usize>()
                && n > 0
            {
                params.max_denoising_steps = n;
            }
            // Map GenSpec → denoising params (light touch):
            // - temperature > 0 raises the schedule ceiling (more exploration);
            //   temperature <= 0 / unset keeps the checkpoint's greedy schedule.
            if let Some(t) = gen_spec.temperature
                && t > 0.0
            {
                params.t_max = (params.t_max * t).clamp(params.t_min, 4.0);
            }

            let prompt_len = prompt_ids.len();
            let out_ids = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                model.diffusion_generate(&prompt_ids, &params)
            }))
            .map_err(super::classify_forward_panic)?;

            // `max_tokens` caps the emitted canvas length.
            let out_ids = match gen_spec.max_tokens {
                Some(cap) if out_ids.len() > cap => out_ids[..cap].to_vec(),
                _ => out_ids,
            };

            return Ok(TokenPuller::Precomputed(PrecomputedPuller::new(
                self.id,
                self.hooks.clone(),
                self.bundle.clone(),
                out_ids,
                prompt_len,
            )));
        }

        let mut guard = self.state.lock();
        let state = guard
            .take()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;

        let state_slot = Arc::downgrade(&self.state);

        let puller = TokenPuller::Ar(Box::new(ArPuller::new(
            self.id,
            self.hooks.clone(),
            self.bundle.clone(),
            state_slot,
            state,
            gen_spec,
            self.paused.clone(),
            self.stopped.clone(),
        )));
        Ok(puller)
    }

    /// Public constructor — no prefix caching (used by tests).
    ///
    /// Defaults `enable_thinking=Some(true)` to preserve the prior
    /// test-suite behaviour. New code should prefer `new_with_prefix`
    /// with an explicit `ThinkingMode`.
    #[allow(dead_code)]
    pub(crate) fn new(
        id: SessionId,
        bundle: Arc<ModelBundle>,
        hooks: Arc<HookBus>,
        settings: Settings,
        messages: Vec<Message>,
        persona: Option<&crate::types::Persona>,
    ) -> Result<Self, ExecError> {
        let (session, _) = Self::new_with_prefix(
            id,
            bundle,
            hooks,
            settings,
            messages,
            persona,
            0,
            None,
            Some(true),
        )?;
        Ok(session)
    }

    /// Full constructor used by `Engine::start_session`.
    ///
    /// If `cached_prefix` is Some the first `cached_prefix.cur_pos` tokens are
    /// assumed already in the KV cache; only the delta is prefilled.
    ///
    /// `enable_thinking` is forwarded to the chat template — `Some(true)` /
    /// `Some(false)` override, `None` leaves the template's own default.
    /// See `ThinkingMode::as_enable_thinking` for the mapping.
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
        enable_thinking: Option<bool>,
    ) -> Result<(Self, Option<PrefixCacheEntry>), ExecError> {
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
                },
            );
        }

        let chat_template = ChatTemplate::new(
            bundle.chat_template_str.clone(),
            Some(TokenizerConfigToken::String(bundle.bos_str.clone())),
            Some(TokenizerConfigToken::String(bundle.eos_str.clone())),
        );

        // Tokenize the full prompt (all messages).
        //
        // Thinking mode:
        // - `Some(true)`  → force reasoning channel on. Gemma 4's template
        //   injects the `<|think|>` marker at the top of the system turn
        //   (the actual think-mode trigger) and omits the
        //   `<|channel>thought\n<channel|>` trailer after
        //   `<|turn>model\n`. The mlx-lm default.
        // - `Some(false)` → force reasoning channel off. Gemma 4 drops the
        //   system `<|think|>` marker. Useful when the caller wants direct
        //   answers and doesn't need the "long-form" integrity the
        //   thinking channel brings (short chats, tool-calling, streaming
        //   structured output).
        // - `None` → template's own default. Avoid on Gemma 4 —
        //   this yields the pathological "system has no `<|think|>` but
        //   model turn has a `<|channel>thought` trailer" state that
        //   drives the jargon-loop / `l l l l` regression we saw on
        //   26B/31B long outputs. The enum's `Auto` maps here only for
        //   models where we know the default is coherent; callers
        //   should prefer explicit On/Off on Gemma.
        let full_prompt = chat_template
            .apply(messages.clone(), None, enable_thinking)
            .map_err(ExecError::Other)?;

        if std::env::var("PIO_MLX_DEBUG_PROMPT").is_ok() {
            eprintln!(
                "\n── Session::new full prompt ({} bytes) ──\n{:?}\n──\n",
                full_prompt.len(),
                full_prompt
            );
        }

        // ── Native vision (Gemma 4 VLM) prefill ───────────────────────────────
        // When the bundle carries a vision tower AND the messages contain
        // images, we must (a) preprocess each image to a pixel tensor and its
        // per-image soft-token count, and (b) **expand** each image placeholder
        // in the rendered prompt into `<boi> + image_token×n_soft + <eoi>`
        // BEFORE tokenizing — mirroring `processing_gemma4.py:500-513`. Pio's
        // chat template flattens the image chunk to markdown `![](url)` (via
        // `as_visible_text`), so without this expansion `full_tokens` carries
        // ZERO `image_token_id` rows and the scatter has no target. The number
        // of `image_token` rows per image MUST equal that image's pooled
        // vision-feature row count (the scatter-count invariant that
        // `forward_with_image` asserts) — both are derived from the SAME
        // preprocessing here so they agree per-image. Image prompts are unique,
        // so we bypass the prefix cache. v1: one image, prefill only.
        if let Some(vision) = bundle.vision.as_ref() {
            let image_urls = extract_image_urls(&messages);
            if !image_urls.is_empty() {
                let cache_slots = cache_slots_for(&bundle);
                // Preprocess each image once: pixels (for the tower) + n_soft
                // (for the prompt expansion). Both come from the same processor
                // so the expansion count == the pooled row count per image.
                let proc = super::model::vision_preprocess::Gemma4ImageProcessor::default();
                let mut pixels: Vec<mlx_rs::Array> = Vec::with_capacity(image_urls.len());
                let mut expansions: Vec<(String, String)> = Vec::with_capacity(image_urls.len());
                for url in &image_urls {
                    // Hardened, panic-free decode with dimension/pixel caps —
                    // the single trust boundary for user-supplied image bytes.
                    let img = super::model::vision_preprocess::load_attached_image(url)?;
                    let n_soft = proc.num_soft_tokens(img.width(), img.height());
                    pixels.push(proc.preprocess(&img));
                    // The markdown `as_visible_text` emits for this chunk → its
                    // expansion. Original (un-stripped) URL, matching the prompt.
                    expansions.push((
                        format!("![]({url})"),
                        vision.image_placeholder_expansion(n_soft),
                    ));
                }

                // Replace each image's markdown placeholder, in prompt order, by
                // its expansion. `replacen(.., 1)` consumes one occurrence at a
                // time so repeated identical URLs map to distinct images.
                let mut expanded_prompt = full_prompt.clone();
                for (marker, expansion) in &expansions {
                    expanded_prompt = expanded_prompt.replacen(marker, expansion, 1);
                }

                let full_tokens = bundle
                    .tokenizer
                    .encode(&expanded_prompt, true)
                    .map_err(ExecError::Other)?;

                return Self::prefill_with_images(
                    id,
                    bundle,
                    hooks,
                    settings,
                    messages,
                    enable_thinking,
                    &full_tokens,
                    pixels,
                    cache_slots,
                );
            }
        }

        let full_tokens = bundle
            .tokenizer
            .encode(&full_prompt, true)
            .map_err(ExecError::Other)?;

        // ── DiffusionGemma: no autoregressive prefill ─────────────────────────
        // The model is encoder/decoder block-diffusion; `forward` panics for it.
        // Hold the tokenized prompt verbatim and let `pull()` run the denoising
        // loop, emitting the result through a `PrecomputedPuller`. No KV cache,
        // no prefix caching.
        if bundle.model.is_diffusion() {
            hooks.emit(HookEvent::SessionPrefillStart {
                session_id: id,
                prompt_tokens: full_tokens.len(),
            });
            hooks.emit(HookEvent::SessionPrefillOk {
                session_id: id,
                prompt_tokens: full_tokens.len(),
            });
            let session = Self {
                id,
                bundle,
                hooks,
                settings,
                paused: Arc::new(AtomicBool::new(false)),
                stopped: Arc::new(AtomicBool::new(false)),
                state: Arc::new(Mutex::new(None)),
                messages: RwLock::new(messages),
                enable_thinking,
                diffusion_prompt: Mutex::new(Some(full_tokens)),
            };
            return Ok((session, None));
        }

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
                .map_err(super::classify_forward_panic)?;

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
                    enable_thinking,
                    diffusion_prompt: Mutex::new(None),
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
            match chat_template.apply_with_options(sys_messages, None, enable_thinking, false) {
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
                                .map_err(super::classify_forward_panic)?;
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
                                .map_err(super::classify_forward_panic)?;

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
                                enable_thinking,
                                diffusion_prompt: Mutex::new(None),
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
        .map_err(super::classify_forward_panic)?;

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
            enable_thinking,
            diffusion_prompt: Mutex::new(None),
        };
        Ok((session, None))
    }

    /// Single-phase prefill for an image prompt: encode each pre-processed image
    /// through the vision tower + projector to get `[1, n_soft, text_hidden]`
    /// features per image (concatenated in prompt order), then
    /// `forward_with_image` scatters them into the `image_token_id` rows.
    /// Bypasses the prefix cache.
    ///
    /// `pixels` are the per-image `[1,3,H,W]` tensors produced by the SAME
    /// `Gemma4ImageProcessor` that derived the `n_soft` counts used to expand
    /// `full_tokens`'s image placeholders — so the projected feature rows equal
    /// the `image_token_id` count (the invariant `forward_with_image` asserts).
    #[allow(clippy::too_many_arguments)]
    fn prefill_with_images(
        id: SessionId,
        bundle: Arc<ModelBundle>,
        hooks: Arc<HookBus>,
        settings: Settings,
        messages: Vec<Message>,
        enable_thinking: Option<bool>,
        full_tokens: &[u32],
        pixels: Vec<mlx_rs::Array>,
        cache_slots: usize,
    ) -> Result<(Self, Option<PrefixCacheEntry>), ExecError> {
        let vision = bundle
            .vision
            .as_ref()
            .expect("prefill_with_images called without a vision tower");
        let total_tokens = full_tokens.len();
        hooks.emit(HookEvent::SessionPrefillStart {
            session_id: id,
            prompt_tokens: total_tokens,
        });

        // Encode each image; concat the projected features in prompt order so
        // they map 1:1 to the image-token runs.
        let mut feats: Vec<mlx_rs::Array> = Vec::with_capacity(pixels.len());
        for px in &pixels {
            feats.push(vision.encode_image(px));
        }
        let image_features = if feats.len() == 1 {
            feats.into_iter().next().expect("len==1")
        } else {
            let refs: Vec<&mlx_rs::Array> = feats.iter().collect();
            mlx_rs::ops::concatenate_axis(&refs, 1)
                .map_err(|e| ExecError::Other(anyhow::anyhow!("concat image features: {e}")))?
        };

        // Invariant: the `image_token_id` placeholders in the prompt must equal
        // the pooled vision-feature rows (the scatter has one target row each).
        // Both were derived from the same preprocessing, so a mismatch is a bug.
        let image_token_id = vision.image_token_id;
        let n_img_tokens = full_tokens.iter().filter(|&&t| t == image_token_id).count();
        let n_feat_rows = image_features.shape()[1] as usize;
        if n_img_tokens != n_feat_rows {
            return Err(ExecError::Other(anyhow::anyhow!(
                "vision prompt/feature mismatch: {n_img_tokens} image_token_id rows in \
                 the prompt vs {n_feat_rows} pooled vision rows — the expansion count \
                 and the tower's pooled-row count diverged"
            )));
        }
        let cache_slots_n = cache_slots;
        let policy = CachePolicy::compute(&bundle.config, cache_slots_n);

        let mut cache: KvCache = vec![None; cache_slots_n];
        let b = &bundle;
        let ft = full_tokens.to_vec();
        let prefill_logits = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.model
                .forward_with_image(&ft, &image_features, image_token_id, 0, &mut cache)
        }))
        .map_err(super::classify_forward_panic)?;

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
            enable_thinking,
            diffusion_prompt: Mutex::new(None),
        };
        Ok((session, None))
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Append new messages and prefill the delta.
    ///
    /// The model's previous assistant reply is already in the KV cache as the
    /// actually-sampled tokens — which can include special tokens like
    /// Gemma 4's `<|channel>` that get stripped when the text is decoded for
    /// user display. Re-rendering those replies through the chat template
    /// would produce a token stream missing those specials, corrupting any
    /// attempt to reuse the KV cache via LCP diffing.
    ///
    /// Instead, we:
    /// 1. Keep assistant messages in `self.messages` for bookkeeping only.
    /// 2. Render ONLY the new user (+system/tool) messages through the chat
    ///    template to produce a clean "continuation" delta.
    /// 3. Strip the leading `<bos>` that Gemma's template unconditionally
    ///    prepends (would otherwise inject mid-stream).
    /// 4. Prepend `<turn|>` if the model's last sampled token wasn't an
    ///    end-of-turn marker — needed to close the previous assistant turn
    ///    cleanly when generation stopped via loop detector or max_tokens.
    /// 5. Prefill the result as a delta into the existing KV cache.
    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        if new_messages.is_empty() {
            return Ok(0);
        }

        {
            let mut msgs = self.messages.write();
            msgs.extend(new_messages.clone());
        }

        // ── DiffusionGemma: re-render the whole conversation ──────────────────
        // The model is non-autoregressive — there is no KV cache to delta-prefill
        // into, so every turn re-tokenizes the full message chain. The next
        // `pull()` denoises a fresh canvas conditioned on the updated prompt.
        if self.diffusion_prompt.lock().is_some() {
            let msgs = self.messages.read().clone();
            let tpl = ChatTemplate::new(
                self.bundle.chat_template_str.clone(),
                Some(TokenizerConfigToken::String(self.bundle.bos_str.clone())),
                Some(TokenizerConfigToken::String(self.bundle.eos_str.clone())),
            );
            let full_prompt = tpl
                .apply(msgs, None, self.enable_thinking)
                .map_err(ExecError::Other)?;
            let full_tokens = self
                .bundle
                .tokenizer
                .encode(&full_prompt, true)
                .map_err(ExecError::Other)?;
            let n = full_tokens.len();
            *self.diffusion_prompt.lock() = Some(full_tokens);
            return Ok(n);
        }

        // Drop assistant messages — their content is already in cache as
        // sampled tokens. Rendering them as text would strip special tokens
        // (like `<|channel>`) that were actually sampled, breaking the cache
        // continuation.
        let to_render: Vec<Message> = new_messages
            .into_iter()
            .filter(|m| m.role != "assistant")
            .collect();

        if to_render.is_empty() {
            return Ok(0);
        }

        let tpl = ChatTemplate::new(
            self.bundle.chat_template_str.clone(),
            Some(TokenizerConfigToken::String(self.bundle.bos_str.clone())),
            Some(TokenizerConfigToken::String(self.bundle.eos_str.clone())),
        );

        let delta_text = tpl
            .apply(to_render, None, self.enable_thinking)
            .map_err(ExecError::Other)?;

        // Gemma's chat template starts with `{{ bos_token }}` unconditionally.
        // Mid-stream BOS would corrupt the attention pattern.
        let bos = self.bundle.bos_str.as_str();
        let delta_text = delta_text.strip_prefix(bos).unwrap_or(&delta_text);

        // Gemma 4's template also auto-inserts an empty system-turn stub
        // (`<|turn>system\n<|think|>\n<turn|>\n`) when enable_thinking=true
        // and no explicit system message is passed. Appearing once at
        // session start it sets the reasoning mode; appearing again on
        // every subsequent turn's delta confuses the model — we saw
        // post-answer drift into `wiadomości` and eventual regurgitation
        // on 26B turns 7-9. The real system message is already in cache
        // from the initial prefill, so strip these stubs from deltas.
        let empty_sys_stub_re = [
            "<|turn>system\n<|think|>\n<turn|>\n",
            "<|turn>system\n<turn|>\n",
        ];
        let mut delta_text: String = delta_text.to_string();
        for stub in empty_sys_stub_re {
            if delta_text.starts_with(stub) {
                delta_text = delta_text[stub.len()..].to_string();
                break;
            }
        }
        let delta_text: &str = &delta_text;

        if std::env::var("PIO_MLX_DEBUG_PROMPT").is_ok() {
            eprintln!(
                "\n── append_messages delta ({} bytes) ──\n{:?}\n──\n",
                delta_text.len(),
                delta_text
            );
        }

        let mut delta_tokens = self
            .bundle
            .tokenizer
            .encode(delta_text, false)
            .map_err(ExecError::Other)?;

        let mut guard = self.state.lock();
        let st = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;

        // If the previous turn ended without a natural `<turn|>` (model hit
        // max_tokens or the loop detector truncated), prepend one so the
        // delta opens a clean turn boundary.
        let eot_candidates: [&str; 2] = ["<turn|>", "<end_of_turn>"];
        let eot_id: Option<u32> = eot_candidates.iter().find_map(|name| {
            self.bundle
                .tokenizer
                .encode(name, false)
                .ok()
                .and_then(|ids| (ids.len() == 1).then(|| ids[0]))
        });
        if let Some(eot) = eot_id
            && st.last_token != eot
        {
            delta_tokens.insert(0, eot);
        }

        if delta_tokens.is_empty() {
            return Ok(0);
        }

        self.hooks.emit(HookEvent::SessionPrefillStart {
            session_id: self.id,
            prompt_tokens: delta_tokens.len(),
        });

        let delta_pos = st.cur_pos;
        let b = &self.bundle;
        let delta_logits = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.model
                .forward(&delta_tokens, delta_pos, &mut st.cache, &b.rope)
        }))
        .map_err(super::classify_forward_panic)?;
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

/// Extract image **URLs** from the message stream, in order, preserving the
/// original (un-stripped) URL. The URL is what `as_visible_text` renders into
/// the prompt as `![](url)` (types/message.rs), so the caller can match the
/// markdown placeholder to expand it; the `file://` prefix is stripped only at
/// decode time. Mirrors the llama backend's gather (`llama/session.rs`): walk
/// `MessageContent::MultipleChunks` → `MessageChunk::ImageUrl`.
fn extract_image_urls(messages: &[Message]) -> Vec<String> {
    let mut out = Vec::new();
    for m in messages {
        if let MessageBody::Content { content } = &m.body
            && let MessageContent::MultipleChunks(chunks) = content
        {
            for ch in chunks {
                if let MessageChunk::ImageUrl { image_url } = ch {
                    out.push(image_url.url.clone());
                }
            }
        }
    }
    out
}

/// Number of KV-cache slots the model needs (= non-shared layer count).
fn cache_slots_for(bundle: &ModelBundle) -> usize {
    bundle.model.num_non_shared_layers()
}

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
    // No KV snapshot, no poison detection — defaults apply.
}
