use std::num::NonZeroU32;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::bundle::ModelBundle;
use super::puller::TokenPuller;
use crate::gen2::Message;
use crate::gen2::backend::common::chat_template::ChatTemplate;
use crate::gen2::engine::{ExecError, HookBus, HookEvent, Settings};
use crate::gen2::generation::{GenSpec, TokenEvent};
use crate::gen2::kv::{
    KvLoadReport, KvLoadSpec, KvMeta, KvSaveSpec, KvSnapshot, build_blob, parse_blob,
    read_from_path, write_to_path,
};
use crate::gen2::session_rt::media_util::messages_have_images;
use crate::gen2::session_rt::prompt::merge_prompts;
use crate::types::message::{MessageBody, MessageChunk, MessageContent, TokenizerConfigToken};
use chrono::Utc;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdInputText, mtmd_default_marker};
use llama_cpp_2::sampling::LlamaSampler;
use parking_lot::{Mutex, RwLock};
use rand::Rng;
use self_cell::self_cell;
use std::fmt;
use std::ops::{Deref, DerefMut};

pub type SessionId = u64;

// Wrapper so self_cell can accept an ident for the dependent type
pub struct DepCtx<'a>(pub LlamaContext<'a>);
impl<'a> Deref for DepCtx<'a> {
    type Target = LlamaContext<'a>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a> DerefMut for DepCtx<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

self_cell! {
    pub(crate) struct SessionCtxCell {
        owner: Arc<ModelBundle>,
        #[covariant]
        dependent: DepCtx,
    }
}

impl fmt::Debug for SessionCtxCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionCtxCell").finish()
    }
}

pub struct Session {
    pub id: SessionId,
    pub bundle: Arc<ModelBundle>,
    hooks: Arc<HookBus>,
    settings: Settings,
    chat_template: ChatTemplate,
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    state: Arc<Mutex<Option<DecodeState>>>,
    messages: RwLock<Vec<Message>>,
    /// Number of old messages dropped during session creation due to context overflow.
    initial_messages_dropped: usize,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("settings", &self.settings)
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct DecodeState {
    pub ctx_cell: SessionCtxCell,
    pub cur_pos: i32,
    pub logits_i: i32,
    /// Timestamp (ggml_time_us) when prefill started, for accurate TTFT.
    pub prefill_start_us: u64,
}

impl Session {
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Release);
    }
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
    }

    /// Messages dropped during initial session creation due to context overflow.
    pub fn initial_messages_dropped(&self) -> usize {
        self.initial_messages_dropped
    }

    /// Returns true if the session's decode state was lost (e.g. due to an FFI
    /// panic in the puller). When poisoned the session cannot generate further
    /// tokens and must be discarded.
    pub fn is_poisoned(&self) -> bool {
        // If the state slot is None, it means either a puller is active (normal)
        // or the puller was dropped without restoring state (poisoned).
        // We check the stopped flag as a proxy: if stopped was set, we know
        // no puller should be outstanding.
        let guard = self.state.lock();
        guard.is_none() && self.stopped.load(Ordering::Acquire)
    }

    pub fn pull(&self, mut gen_spec: GenSpec) -> Result<TokenPuller, ExecError> {
        if gen_spec.max_tokens.is_none() {
            gen_spec.max_tokens = self.settings.stopping.max_tokens;
        }
        let mut guard = self.state.lock();
        let state = guard
            .take()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;

        // Weak link back to this session’s state slot
        let state_slot = Arc::downgrade(&self.state);

        let pre_events = self.build_media_events();
        let puller = TokenPuller::new_from_session(
            self.id,
            self.hooks.clone(),
            self.bundle.clone(),
            state_slot,
            state,
            Self::sampler_from_settings(&self.settings),
            gen_spec,
            self.paused.clone(),
            self.stopped.clone(),
            pre_events,
        );
        Ok(puller)
    }

    pub fn save_cache(&self, dst: KvSaveSpec) -> Result<KvSnapshot, ExecError> {
        let guard = self.state.lock();
        let state = guard
            .as_ref()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;
        let sz = state.ctx_cell.with_dependent(|_, ctx| ctx.get_state_size());
        if sz == 0 {
            return Err(ExecError::Other(anyhow::anyhow!("no state available")));
        }
        let mut buf = vec![0u8; sz];
        // SAFETY: buf is allocated with exactly `sz` bytes, matching get_state_size().
        // copy_state_data writes at most `sz` bytes and returns the actual count.
        let written = state
            .ctx_cell
            .with_dependent(|_, ctx| unsafe { ctx.copy_state_data(buf.as_mut_ptr()) });
        if written == 0 {
            return Err(ExecError::Other(anyhow::anyhow!("failed to copy state")));
        }
        buf.truncate(written);
        let pos_max = state
            .ctx_cell
            .with_dependent(|_, ctx| ctx.kv_cache_seq_pos_max(0));
        let tokens_covered = (pos_max + 1).max(0) as usize;

        let meta = self.build_kv_meta(&self.bundle)?;
        let blob = build_blob(meta.clone(), &buf).map_err(ExecError::Other)?;

        match dst {
            KvSaveSpec::InMemory => Ok(KvSnapshot {
                tokens_covered,
                bytes: blob.clone(),
                meta,
            }),
            KvSaveSpec::ToPath(path) => {
                write_to_path(&path, &blob).map_err(|e| ExecError::Io(e.to_string()))?;
                Ok(KvSnapshot {
                    tokens_covered,
                    bytes: blob,
                    meta,
                })
            }
        }
    }

    pub fn load_cache(&self, src: KvLoadSpec) -> Result<KvLoadReport, ExecError> {
        let mut guard = self.state.lock();
        let state = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;
        let path = match &src {
            KvLoadSpec::Strict(p) | KvLoadSpec::Lenient(p) => p,
        };
        let blob = read_from_path(path).map_err(|e| ExecError::Io(e.to_string()))?;
        let (hdr, payload) = parse_blob(&blob).map_err(|e| ExecError::KvCorrupt(e.to_string()))?;

        // Validate meta compatibility (strict checks). Also verify tokenizer/template.
        let cur = &self.bundle.meta;
        let mut incompatible_reasons: Vec<String> = Vec::new();
        if hdr.meta.model_uuid != cur.model_uuid {
            incompatible_reasons.push("model_uuid".into());
        }
        if hdr.meta.n_ctx != cur.n_ctx {
            incompatible_reasons.push("n_ctx".into());
        }
        if hdr.meta.n_layer != cur.n_layer {
            incompatible_reasons.push("n_layer".into());
        }
        let expected_meta = self.build_kv_meta(&self.bundle)?;
        if hdr.meta.tokenizer_digest != expected_meta.tokenizer_digest {
            incompatible_reasons.push("tokenizer_digest".into());
        }
        if hdr.meta.template_fingerprint != expected_meta.template_fingerprint {
            incompatible_reasons.push("template_fingerprint".into());
        }
        if !incompatible_reasons.is_empty() {
            let reason = format!("incompatible: {}", incompatible_reasons.join(","));
            return match src {
                KvLoadSpec::Strict(_) => Err(ExecError::KvIncompatible(reason)),
                KvLoadSpec::Lenient(_) => Ok(KvLoadReport {
                    loaded: false,
                    reason: Some(reason),
                    tokens_covered: 0,
                }),
            };
        }

        // SAFETY: payload was produced by copy_state_data from a compatible context
        // (validated by the meta checks above). set_state_data restores KV cache state.
        state
            .ctx_cell
            .with_dependent_mut(|_, ctx| unsafe { ctx.set_state_data(payload) });
        let pos_max = state
            .ctx_cell
            .with_dependent(|_, ctx| ctx.kv_cache_seq_pos_max(0));
        let tokens_covered = (pos_max + 1).max(0) as usize;
        Ok(KvLoadReport {
            loaded: true,
            reason: None,
            tokens_covered,
        })
    }

    fn build_kv_meta(&self, bundle: &Arc<ModelBundle>) -> Result<KvMeta, ExecError> {
        // Digests are pre-computed in ModelMeta at model load time (see loader.rs)
        Ok(KvMeta {
            model_uuid: bundle.meta.model_uuid.clone(),
            n_ctx: bundle.meta.n_ctx,
            n_layer: bundle.meta.n_layer,
            tokenizer_digest: bundle.meta.tokenizer_digest,
            template_fingerprint: bundle.meta.template_fingerprint,
            created_at_us: Utc::now().timestamp_micros(),
        })
    }
}

impl Session {
    fn sampler_from_settings(settings: &Settings) -> LlamaSampler {
        // Order: penalties → top_k → min_p → top_p → temp → dist
        let mut chain = Vec::new();

        if let Some(penalties) = Self::penalties_sampler(settings) {
            tracing::debug!("Sampling penalties: {:?}", penalties);
            chain.push(penalties);
        }
        if let Some(k) = settings.sampling.top_k {
            tracing::debug!("Sampling top_k: {}", k);
            chain.push(LlamaSampler::top_k(k));
        }
        // min_p: default 0.05 when neither min_p nor top_p is explicitly set
        let min_p = settings.sampling.min_p.or_else(|| {
            if settings.sampling.top_p.is_none() {
                Some(0.05)
            } else {
                None
            }
        });
        if let Some(mp) = min_p {
            tracing::debug!("Sampling min_p: {}", mp);
            chain.push(LlamaSampler::min_p(mp, 1));
        }
        if let Some(tp) = settings.sampling.top_p {
            tracing::debug!("Sampling top_p: {}", tp);
            chain.push(LlamaSampler::top_p(tp, 0));
        }
        if let Some(t) = settings.sampling.temperature {
            tracing::debug!("Sampling temperature: {}", t);
            chain.push(LlamaSampler::temp(t));
        }

        let seed = settings
            .sampling
            .seed
            .unwrap_or_else(|| rand::rng().random());
        chain.push(LlamaSampler::dist(seed));

        LlamaSampler::chain_simple(chain)
    }

    fn penalties_sampler(settings: &Settings) -> Option<LlamaSampler> {
        let sampling = &settings.sampling;
        let penalty_last_n = sampling.penalty_last_n.unwrap_or(0);
        let penalty_repeat = sampling.penalty_repeat.unwrap_or(1.0);
        let penalty_freq = sampling.penalty_freq.unwrap_or(0.0);
        let penalty_present = sampling.penalty_present.unwrap_or(0.0);

        let repeat_is_default = (penalty_repeat - 1.0).abs() <= f32::EPSILON;
        if penalty_last_n == 0 && repeat_is_default && penalty_freq == 0.0 && penalty_present == 0.0
        {
            None
        } else {
            Some(LlamaSampler::penalties(
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            ))
        }
    }
}

impl Session {
    #[allow(clippy::arc_with_non_send_sync)]
    pub(crate) fn new(
        id: SessionId,
        bundle: Arc<ModelBundle>,
        backend: Arc<LlamaBackend>,
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

        tracing::debug!("Session prompt messages: {:?}", messages);
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

        // Build chat template
        let mut bos_decoder = encoding_rs::UTF_8.new_decoder();
        let mut eos_decoder = encoding_rs::UTF_8.new_decoder();
        let chat_template = ChatTemplate::new(
            bundle
                .model
                .chat_template(None)
                .map_err(|e| ExecError::Other(e.into()))?
                .to_string()
                .map_err(|e| ExecError::Other(e.into()))?,
            Some(TokenizerConfigToken::String(
                bundle
                    .model
                    .token_to_piece(bundle.model.token_bos(), &mut bos_decoder, true, None)
                    .map_err(|e| ExecError::Other(e.into()))?,
            )),
            Some(TokenizerConfigToken::String(
                bundle
                    .model
                    .token_to_piece(bundle.model.token_eos(), &mut eos_decoder, true, None)
                    .map_err(|e| ExecError::Other(e.into()))?,
            )),
        );

        let prompt = chat_template
            .apply(messages.clone(), None, None)
            .map_err(ExecError::Other)?;

        // Tokenize prompt
        let mut tokens_list = bundle
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| ExecError::Other(e.into()))?;

        // Create context
        let ctx_size = settings
            .system
            .ctx_size
            .unwrap_or(bundle.meta.n_ctx.max(128));

        // Context overflow: truncate old messages until conversation fits
        let original_message_count = messages.len();
        let gen_reserve = crate::gen2::session_rt::prompt::generation_reserve(
            ctx_size as usize,
            settings.stopping.max_tokens,
        );
        let ctx_limit = (ctx_size as usize).saturating_sub(gen_reserve);
        if tokens_list.len() > ctx_limit && messages.len() > 2 {
            tracing::warn!(
                "initial context overflow: {} tokens > {} limit, truncating conversation",
                tokens_list.len(),
                ctx_limit
            );
            // Estimate how many messages to batch-remove based on avg tokens/msg
            let first_is_system = messages
                .first()
                .map(|m| m.role == "system")
                .unwrap_or(false);
            let removable = if first_is_system {
                messages.len() - 2
            } else {
                messages.len() - 1
            };
            let avg_per_msg = tokens_list.len() / messages.len().max(1);
            let excess = tokens_list.len().saturating_sub(ctx_limit);
            let est_remove = (excess / avg_per_msg.max(1)).min(removable);
            let remove_idx = if first_is_system { 1 } else { 0 };
            // Batch remove estimated messages
            for _ in 0..est_remove {
                if messages.len() <= 2 {
                    break;
                }
                messages.remove(remove_idx);
            }
            // Re-tokenize and fine-tune with per-message loop
            let truncated_prompt = chat_template
                .apply(messages.clone(), None, None)
                .map_err(ExecError::Other)?;
            tokens_list = bundle
                .model
                .str_to_token(&truncated_prompt, AddBos::Always)
                .map_err(|e| ExecError::Other(e.into()))?;
            while tokens_list.len() > ctx_limit && messages.len() > 2 {
                messages.remove(remove_idx);
                let truncated_prompt = chat_template
                    .apply(messages.clone(), None, None)
                    .map_err(ExecError::Other)?;
                tokens_list = bundle
                    .model
                    .str_to_token(&truncated_prompt, AddBos::Always)
                    .map_err(|e| ExecError::Other(e.into()))?;
            }
        }
        let batch_size = settings.system.batch_size.unwrap_or(128);
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_size))
            .with_n_batch(batch_size);
        if let Some(n) = settings.system.threads {
            ctx_params = ctx_params.with_n_threads(n as i32);
        }
        if let Some(n) = settings.system.threads_batch {
            ctx_params = ctx_params.with_n_threads_batch(n as i32);
        }
        // Flash attention: default to AUTO (let llama.cpp decide based on model)
        if settings.system.flash_attn.unwrap_or(true) {
            ctx_params =
                ctx_params.with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO);
        }
        let mut ctx_cell = SessionCtxCell::try_new(bundle.clone(), |owner| {
            owner.model.new_context(&backend, ctx_params).map(DepCtx)
        })
        .map_err(|e| ExecError::Other(e.into()))?;

        // Optional MTMD (images) prefill path
        {
            if let Some(mtmd_ctx) = bundle
                .mtmd_ctx
                .as_ref()
                .filter(|_| messages_have_images(&messages))
            {
                let marker = bundle
                    .mtmd_marker
                    .clone()
                    .unwrap_or_else(|| mtmd_default_marker().to_string());
                // Count images and build file list

                let mut img_paths: Vec<String> = Vec::new();
                for m in &messages {
                    if let MessageBody::Content { content } = &m.body
                        && let MessageContent::MultipleChunks(chunks) = content
                    {
                        for ch in chunks {
                            if let MessageChunk::ImageUrl { image_url } = ch {
                                let u = image_url.url.clone();
                                let path = if let Some(rest) = u.strip_prefix("file://") {
                                    rest.to_string()
                                } else {
                                    u
                                };
                                img_paths.push(path);
                            }
                        }
                    }
                }
                if !img_paths.is_empty() {
                    // Ensure prompt has enough markers
                    let mut prompt_mm = prompt.clone();
                    let have = prompt_mm.matches(&marker).count();
                    for _ in have..img_paths.len() {
                        prompt_mm.push_str(&marker);
                    }

                    // Load bitmaps
                    let mut bitmaps: Vec<MtmdBitmap> = Vec::with_capacity(img_paths.len());
                    for p in img_paths {
                        let bmp = MtmdBitmap::from_file(mtmd_ctx, &p)
                            .map_err(|e| ExecError::Other(e.into()))?;
                        bitmaps.push(bmp);
                    }
                    let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
                    let input = MtmdInputText {
                        text: prompt_mm,
                        add_special: true,
                        parse_special: true,
                    };
                    let chunks = mtmd_ctx
                        .tokenize(input, &refs)
                        .map_err(|e| ExecError::Other(e.into()))?;
                    // Evaluate chunks to prefill
                    let n_past = ctx_cell
                        .with_dependent_mut(|_, ctx| {
                            chunks.eval_chunks(mtmd_ctx, ctx, 0, 0, batch_size as i32, true)
                        })
                        .map_err(|e| ExecError::Other(e.into()))?;
                    hooks.emit(HookEvent::SessionPrefillStart {
                        session_id: id,
                        prompt_tokens: n_past as usize,
                    });
                    hooks.emit(HookEvent::SessionPrefillOk {
                        session_id: id,
                        prompt_tokens: n_past as usize,
                    });

                    // Build sampler
                    let _sampler = Self::sampler_from_settings(&settings);

                    return Ok(Self {
                        id,
                        bundle,
                        hooks,
                        settings,
                        chat_template,
                        paused: Arc::new(AtomicBool::new(false)),
                        stopped: Arc::new(AtomicBool::new(false)),
                        // For MTMD, start sampling from last logits (-1)
                        state: Arc::from(Mutex::new(Some(DecodeState {
                            ctx_cell,
                            cur_pos: n_past,
                            logits_i: (n_past - 1).max(0),
                            prefill_start_us: llama_cpp_2::ggml_time_us() as u64,
                        }))),
                        messages: RwLock::new(messages),
                        initial_messages_dropped: 0, // MTMD path has no truncation
                    });
                }
            }
        }

        // tracing::debug!("session.prefill.start", id=%id);
        // Prefill prompt tokens
        let prefill_start_us = llama_cpp_2::ggml_time_us() as u64;
        let mut batch = LlamaBatch::new(batch_size as usize, 1);
        let total_tokens = tokens_list.len() as i32;
        let mut cur_pos = 0_i32;
        let mut remaining = tokens_list;
        hooks.emit(HookEvent::SessionPrefillStart {
            session_id: id,
            prompt_tokens: total_tokens as usize,
        });
        let mut last_batch_tokens: i32 = 0;
        while !remaining.is_empty() {
            let chunk_size = remaining.len().min(batch_size as usize);
            let chunk: Vec<_> = remaining.drain(..chunk_size).collect();
            batch.clear();
            for (i, token) in chunk.into_iter().enumerate() {
                let absolute = cur_pos + i as i32;
                let is_last = absolute == (total_tokens - 1);
                batch
                    .add(token, absolute, &[0], is_last)
                    .map_err(|e| ExecError::Other(e.into()))?;
            }
            ctx_cell
                .with_dependent_mut(|_, ctx| ctx.decode(&mut batch))
                .map_err(|e| ExecError::Other(e.into()))?;
            cur_pos += chunk_size as i32;
            last_batch_tokens = batch.n_tokens();
        }
        // tracing::debug!("session.prefill.ok", id=%id, total_tokens=%total_tokens);
        hooks.emit(HookEvent::SessionPrefillOk {
            session_id: id,
            prompt_tokens: total_tokens as usize,
        });

        // Build sampler
        let _sampler = Self::sampler_from_settings(&settings);

        Ok(Self {
            id,
            bundle,
            hooks,
            settings,
            chat_template,
            paused: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(Some(DecodeState {
                ctx_cell,
                cur_pos: total_tokens,
                logits_i: (last_batch_tokens - 1),
                prefill_start_us,
            }))),
            initial_messages_dropped: original_message_count.saturating_sub(messages.len()),
            messages: RwLock::new(messages),
        })
    }
}

impl Session {
    fn build_media_events(&self) -> std::collections::VecDeque<TokenEvent> {
        use crate::gen2::generation::MediaBoundary;
        use crate::types::message::{MessageBody, MessageChunk, MessageContent};
        let mut out = std::collections::VecDeque::new();
        let mut idx = 0usize;
        let msgs = self.messages.read();

        for m in msgs.iter() {
            if let MessageBody::Content { content } = &m.body
                && let MessageContent::MultipleChunks(chunks) = content
            {
                for ch in chunks {
                    if matches!(ch, MessageChunk::ImageUrl { .. }) {
                        out.push_back(TokenEvent::MediaBoundary(MediaBoundary::BeginImage { idx }));
                        out.push_back(TokenEvent::MediaBoundary(MediaBoundary::EndImage { idx }));
                        idx += 1;
                    }
                }
            }
        }
        out
    }
}

impl Session {
    /// Append new messages and prefill only the delta into the KV.
    /// Returns the number of old messages dropped due to context overflow (0 = no truncation).
    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        if new_messages.is_empty() {
            return Ok(0);
        }

        // 1) Extend transcript safely
        {
            let mut msgs = self.messages.write();
            msgs.extend(new_messages.clone());
        }

        // 2) Render the FULL conversation with the cached template and compute token delta.
        //    This is correct for ALL chat templates (including those that format
        //    differently based on message index or role transitions).
        let all_messages = self.messages.read().clone();
        let tpl = &self.chat_template;

        let full_prompt = tpl
            .apply(all_messages, None, None)
            .map_err(ExecError::Other)?;

        // 3) Tokenize full prompt, then take only the delta beyond what's already in KV
        let full_tokens = self
            .bundle
            .model
            .str_to_token(&full_prompt, AddBos::Always)
            .map_err(|e| ExecError::Other(e.into()))?;

        // 4) Context overflow check — if the full conversation exceeds the context
        //    window, clear KV cache, truncate old messages, and re-encode from scratch.
        let ctx_size = self
            .settings
            .system
            .ctx_size
            .unwrap_or(self.bundle.meta.n_ctx.max(128)) as usize;
        let gen_reserve = crate::gen2::session_rt::prompt::generation_reserve(
            ctx_size,
            self.settings.stopping.max_tokens,
        );
        let ctx_limit = ctx_size.saturating_sub(gen_reserve);

        let mut guard = self.state.lock();
        let st = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;

        if full_tokens.len() > ctx_limit {
            // Context overflow — reset and re-encode with truncated conversation
            tracing::warn!(
                "context overflow: {} tokens > {} limit, truncating conversation",
                full_tokens.len(),
                ctx_limit
            );

            // Clear KV cache
            st.ctx_cell
                .with_dependent_mut(|_, ctx| ctx.clear_kv_cache());
            st.cur_pos = 0;

            // Drop oldest non-system messages until conversation fits
            let mut msgs = self.messages.write();
            let mut dropped = 0_usize;
            while msgs.len() > 2 {
                // Keep at least system prompt (if any) + last user message
                let first_is_system = msgs.first().map(|m| m.role == "system").unwrap_or(false);
                let remove_idx = if first_is_system { 1 } else { 0 };
                msgs.remove(remove_idx);
                dropped += 1;

                // Re-render and re-tokenize to check fit
                let check_msgs = msgs.clone();
                drop(msgs);
                let check_prompt = tpl
                    .apply(check_msgs, None, None)
                    .map_err(ExecError::Other)?;
                let check_tokens = self
                    .bundle
                    .model
                    .str_to_token(&check_prompt, AddBos::Always)
                    .map_err(|e| ExecError::Other(e.into()))?;
                if check_tokens.len() <= ctx_limit {
                    // Fits — prefill the truncated conversation from scratch
                    let remaining = check_tokens;
                    let batch_size = self.settings.system.batch_size.unwrap_or(128) as usize;
                    let mut batch = LlamaBatch::new(batch_size, 1);
                    let mut to_process = remaining;

                    self.hooks.emit(HookEvent::SessionPrefillStart {
                        session_id: self.id,
                        prompt_tokens: to_process.len(),
                    });

                    let mut last_batch_tokens: i32 = 0;
                    while !to_process.is_empty() {
                        let chunk_size = to_process.len().min(batch_size);
                        let chunk: Vec<_> = to_process.drain(..chunk_size).collect();
                        batch.clear();
                        for (i, token) in chunk.into_iter().enumerate() {
                            let absolute = st.cur_pos + i as i32;
                            let is_last = (i + 1 == chunk_size) && to_process.is_empty();
                            batch.add(token, absolute, &[0], is_last).map_err(|_| {
                                ExecError::Other(anyhow::anyhow!("batch add error"))
                            })?;
                        }
                        st.ctx_cell
                            .with_dependent_mut(|_, ctx| ctx.decode(&mut batch))
                            .map_err(|_| {
                                ExecError::Other(anyhow::anyhow!(
                                    "decode failed after context truncation"
                                ))
                            })?;
                        st.cur_pos += chunk_size as i32;
                        last_batch_tokens = batch.n_tokens();
                    }
                    st.logits_i = (last_batch_tokens - 1).max(0);

                    self.hooks.emit(HookEvent::SessionPrefillOk {
                        session_id: self.id,
                        prompt_tokens: st.cur_pos as usize,
                    });
                    return Ok(dropped);
                }
                msgs = self.messages.write();
            }
            return Err(ExecError::Other(anyhow::anyhow!(
                "conversation too long even after truncation (ctx_size={})",
                ctx_size
            )));
        }

        // Normal path — no overflow, prefill only the delta
        let kv_pos = st.cur_pos as usize;
        let mut remaining: Vec<_> = if kv_pos < full_tokens.len() {
            full_tokens[kv_pos..].to_vec()
        } else {
            Vec::new()
        };

        if remaining.is_empty() {
            return Ok(0);
        }

        let batch_size = self.settings.system.batch_size.unwrap_or(128) as usize;
        let mut batch = LlamaBatch::new(batch_size, 1);

        let delta_len = remaining.len() as i32;
        self.hooks.emit(HookEvent::SessionPrefillStart {
            session_id: self.id,
            prompt_tokens: delta_len as usize,
        });

        let mut last_batch_tokens: i32 = 0;

        while !remaining.is_empty() {
            let chunk_size = remaining.len().min(batch_size);
            let chunk: Vec<_> = remaining.drain(..chunk_size).collect();
            batch.clear();

            for (i, token) in chunk.into_iter().enumerate() {
                let absolute = st.cur_pos + i as i32;
                let is_last = (i + 1 == chunk_size) && remaining.is_empty(); // last item of last chunk
                batch
                    .add(token, absolute, &[0], is_last)
                    .map_err(|_e| ExecError::Other(anyhow::anyhow!("batch error")))?;
            }

            st.ctx_cell
                .with_dependent_mut(|_, ctx| ctx.decode(&mut batch))
                .map_err(|_e| ExecError::Other(anyhow::anyhow!("batch error")))?;
            st.cur_pos += chunk_size as i32;
            last_batch_tokens = batch.n_tokens();
        }

        st.logits_i = (last_batch_tokens - 1).max(0);

        self.hooks.emit(HookEvent::SessionPrefillOk {
            session_id: self.id,
            prompt_tokens: delta_len as usize,
        });

        Ok(0)
    }
}
