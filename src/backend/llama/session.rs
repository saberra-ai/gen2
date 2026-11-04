use std::num::NonZeroU32;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

mod microphone;

use self::microphone::MicrophoneCapture;
use super::bundle::ModelBundle;
use super::media::MediaPrompt;
use super::puller::TokenPuller;
use crate::Message;
use crate::backend::common::chat_template::ChatTemplate;
use crate::backend::common::grammar::{GrammarMatcher, GrammarVocab};
use crate::engine::{Capabilities, ExecError, HookBus, HookEvent, Settings};
use crate::generation::{GenSpec, TokenEvent};
use crate::kv::{
    KvLoadReport, KvLoadSpec, KvMeta, KvSaveSpec, KvSnapshot, build_blob, parse_blob,
    read_from_path, write_to_path,
};
use crate::session_rt::media_util::{media_parts, with_media_markers};
use crate::session_rt::prompt::merge_prompts;
use crate::types::message::{MessageBody, MessageChunk, MessageContent, TokenizerConfigToken};
use chrono::Utc;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::mtmd_default_marker;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use parking_lot::{Mutex, RwLock};
use rand::RngExt;
use self_cell::self_cell;
use std::fmt;

/// Fallback prefill batch size when settings carry none. Normal app paths
/// fill `batch_size` from the device profile (`default_batch_size()`); this
/// only fires for bare-Settings callers (tests, embedded users).
const DEFAULT_BATCH_SIZE: u32 = 512;
use std::ops::{Deref, DerefMut};

pub type SessionId = u64;

// Wrapper so self_cell can accept an ident for the dependent type
pub enum DepCtx<'a> {
    Plain(LlamaContext<'a>),
    #[cfg(feature = "llama-dflash2")]
    DFlash2(llama_cpp_2::speculative::MtpSpeculative<'a>),
}
impl<'a> Deref for DepCtx<'a> {
    type Target = LlamaContext<'a>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Plain(ctx) => ctx,
            #[cfg(feature = "llama-dflash2")]
            Self::DFlash2(spec) => spec.target_context(),
        }
    }
}
impl<'a> DerefMut for DepCtx<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Plain(ctx) => ctx,
            #[cfg(feature = "llama-dflash2")]
            Self::DFlash2(spec) => spec.target_context_mut(),
        }
    }
}

impl<'a> DepCtx<'a> {
    fn create(
        owner: &'a ModelBundle,
        backend: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<Self, ExecError> {
        #[cfg(feature = "llama-dflash2")]
        if let Some(draft) = &owner.draft {
            let target = owner
                .model
                .new_context(
                    backend,
                    params
                        .clone()
                        .with_n_rs_seq(draft.config.max_tokens as u32)
                        .with_output_limits(
                            draft.config.max_tokens as u32 + 1,
                            draft.config.max_tokens as u32 + 1,
                        ),
                )
                .map_err(|e| ExecError::Other(e.into()))?;
            let draft_ctx = draft
                .model
                .new_context_with_ctx_other(
                    backend,
                    // Match the server: draft KV is F16 independently of target KV.
                    params
                        .with_n_rs_seq(0)
                        .with_type_k(llama_cpp_2::context::params::KvCacheType::F16)
                        .with_type_v(llama_cpp_2::context::params::KvCacheType::F16)
                        .with_output_limits(
                            draft.config.max_tokens as u32 + 1,
                            draft.config.max_tokens as u32 + 1,
                        ),
                    &target,
                )
                .map_err(|e| ExecError::Other(e.into()))?;
            return llama_cpp_2::speculative::MtpSpeculative::new_dflash2(
                target,
                draft_ctx,
                draft.config.max_tokens,
            )
            .map(Self::DFlash2)
            .map_err(|e| ExecError::Other(e.into()));
        }
        owner
            .model
            .new_context(backend, params.with_output_limits(1, 1))
            .map(Self::Plain)
            .map_err(|e| ExecError::Other(e.into()))
    }

    /// DFlash2 consumes target features after every physical prefill/decode batch.
    pub fn decode(&mut self, batch: &mut LlamaBatch<'_>) -> Result<(), ExecError> {
        self.deref_mut()
            .decode(batch)
            .map_err(|e| ExecError::Other(e.into()))?;
        #[cfg(feature = "llama-dflash2")]
        if let Self::DFlash2(spec) = self {
            spec.process(batch).map_err(|e| {
                ExecError::Other(anyhow::Error::new(e).context("DFlash2 feature injection failed"))
            })?;
        }
        Ok(())
    }

    /// Compaction restarts both sequences, never just the target's transcript.
    pub fn clear_kv_cache(&mut self) {
        self.deref_mut().clear_kv_cache();
        #[cfg(feature = "llama-dflash2")]
        if let Self::DFlash2(spec) = self {
            spec.draft_context_mut().clear_kv_cache();
        }
    }
}

fn kv_params(mut params: LlamaContextParams, settings: &Settings) -> LlamaContextParams {
    use llama_cpp_2::context::params::KvCacheType;
    if settings.system.kv_cache_type.as_deref() == Some("q8_0") {
        params = params
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);
    }
    params
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
    /// Context window this session's llama context was created with —
    /// after the fit clamp, so callers can observe what actually loaded.
    ctx_size: u32,
    /// Tool names enabled for this session — arms the output parser's
    /// name-gate on every pull (rehearsal text stays text).
    enabled_tool_names: Option<std::collections::HashSet<String>>,
    /// The chat template's `enable_thinking`, pinned when the session
    /// started. Every later render — the continuation delta, the overflow
    /// probe — passes the same value, so a follow-up turn is prompted the
    /// way the first one was: a Qwen3 continuation rendered without it
    /// fell back to the template's default and started thinking.
    enable_thinking: bool,
    strict_request: Option<GenSpec>,
    request_context: Option<crate::types::RequestContext>,
    prepared_sampling: Mutex<Option<(LlamaSampler, Option<GrammarMatcher>)>>,
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
    microphone: Option<MicrophoneCapture>,
    /// Timestamp (ggml_time_us) when prefill started, for accurate TTFT.
    pub prefill_start_us: u64,
}

impl DecodeState {
    pub(super) fn ready(
        ctx_cell: SessionCtxCell,
        cur_pos: i32,
        logits_i: i32,
        prefill_start_us: u64,
    ) -> Self {
        Self {
            ctx_cell,
            cur_pos,
            logits_i,
            prefill_start_us,
            microphone: None,
        }
    }

    fn prefill_media(&mut self, prompt: MediaPrompt, batch_size: u32) -> Result<usize, ExecError> {
        if self.microphone.is_some() {
            return Err(ExecError::InvalidArg("microphone is already capturing"));
        }
        let prefix_positions = prompt.prefix.total_positions();
        let microphone = prompt
            .microphone_suffix
            .as_deref()
            .map(|suffix| {
                self.ctx_cell.with_dependent(|bundle, _| {
                    let suffix = tokenize_chat_prompt(&bundle.model, suffix)?;
                    if suffix.is_empty() {
                        return Err(ExecError::InvalidArg("empty microphone prompt suffix"));
                    }
                    Ok(MicrophoneCapture {
                        suffix,
                        pcm: Vec::new(),
                    })
                })
            })
            .transpose()?;
        let suffix_positions = microphone
            .as_ref()
            .map_or(0, |capture| capture.suffix.len() as i32);
        let available = self.ctx_cell.with_dependent(|_, ctx| ctx.n_ctx()) as i32;
        if prefix_positions <= 0
            || self
                .cur_pos
                .saturating_add(prefix_positions)
                .saturating_add(suffix_positions)
                >= available
        {
            return Err(ExecError::ContextOverflow(
                "media turn exceeds the remaining context".into(),
            ));
        }

        self.cur_pos = self.ctx_cell.with_dependent_mut(|bundle, ctx| {
            let mtmd = bundle
                .mtmd_ctx
                .as_ref()
                .ok_or(ExecError::FeatureUnsupported("media input"))?;
            let end = prompt
                .prefix
                .eval_chunks(
                    mtmd,
                    ctx,
                    self.cur_pos,
                    0,
                    batch_size as i32,
                    microphone.is_none(),
                )
                .map_err(|err| ExecError::Other(err.into()))?;
            if microphone.is_some() {
                ctx.start_mtmd_audio_input(mtmd)
                    .map_err(microphone::native_error)?;
            }
            Ok::<_, ExecError>(end)
        })?;
        // MTMD computes logits for the final row only.
        self.logits_i = -1;
        self.microphone = microphone;
        Ok(prefix_positions as usize)
    }
}

/// Build a [`GrammarVocab`] straight from the GGUF's embedded vocab so
/// grammar-constrained decoding works without a `tokenizer.json`. Mirrors
/// `llama_cpp_2::llguidance_sampler::build_tok_env`: a token's normal byte
/// form, else its special form prefixed with the `0xFF` toktrie marker,
/// else an empty entry. Called lazily once per loaded model bundle; temporary
/// token-byte vectors are discarded after preparing its immutable token trie.
fn grammar_vocab_from_model(model: &LlamaModel) -> GrammarVocab {
    let n_vocab = model.n_vocab();
    let mut words: Vec<Vec<u8>> = Vec::with_capacity(n_vocab.max(0) as usize);
    for i in 0..n_vocab {
        let token = LlamaToken(i);
        let normal = model
            .token_to_piece_bytes(token, 32, false, None)
            .unwrap_or_default();
        if !normal.is_empty() {
            words.push(normal);
            continue;
        }
        let special = model
            .token_to_piece_bytes(token, 32, true, None)
            .unwrap_or_default();
        if special.is_empty() {
            words.push(Vec::new());
        } else {
            let mut marked = Vec::with_capacity(special.len() + 1);
            marked.push(0xFF);
            marked.extend(special);
            words.push(marked);
        }
    }
    GrammarVocab {
        words,
        eos: model.token_eos().0 as u32,
        bos: Some(model.token_bos().0 as u32),
    }
}

/// Tokenize a chat-template-rendered prompt for the llama.cpp backend.
///
/// Always uses [`AddBos::Never`]. The chat template is constructed with
/// `bos_token` and renders `{{ bos_token }}` as a literal `<bos>` string at
/// the start of the prompt; with `parse_special = true` (the default in
/// `str_to_token`), that string resolves back to the BOS token id.
/// `AddBos::Always` would prepend a *second* BOS — llama.cpp logs
/// `check_double_bos_eos` when this happens, and Gemma 4 IT models respond
/// to `[BOS, BOS, ...]` by emitting their EOS (token 106 `<turn|>`) within
/// 7-8 sampled tokens, as if the conversation were already closed.
///
/// All chat-prompt tokenization in this backend MUST go through this
/// helper rather than calling `str_to_token` directly, otherwise the
/// double-BOS bug regresses silently.
pub(crate) fn tokenize_chat_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    prompt: &str,
) -> Result<Vec<llama_cpp_2::token::LlamaToken>, ExecError> {
    model
        .str_to_token(prompt, AddBos::Never)
        .map_err(|e| ExecError::Other(e.into()))
}

impl Session {
    pub fn pause(&self) {
        if self.strict_request.is_some() {
            return;
        }
        self.paused.store(true, Ordering::Release);
    }
    pub fn resume(&self) {
        if self.strict_request.is_some() {
            return;
        }
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
        if gen_spec.require_full_context && self.strict_request.is_none() {
            return Err(ExecError::FeatureUnsupported(
                "strict pull requires strict preparation",
            ));
        }
        if gen_spec.output_mode != crate::generation::OutputMode::Text {
            if !self
                .bundle
                .capabilities
                .contains(crate::engine::Capabilities::SPEECH)
            {
                return Err(ExecError::FeatureUnsupported("speech output"));
            }
            if gen_spec.grammar.is_some() || gen_spec.require_full_context {
                return Err(ExecError::InvalidArg(
                    "speech output requires unconstrained decoding",
                ));
            }
        }
        if gen_spec.max_tokens.is_none() {
            gen_spec.max_tokens = self.settings.stopping.max_tokens;
        }
        // Apply per-pull GenSpec sampling overrides on top of engine
        // Settings. Without this, the sampler chain below reads from
        // `settings.sampling.*` directly and ignores GenSpec — which
        // silently drops `recommended_sampling(model_id)` values that
        // the matrix harness and any per-call temp/top_p override try
        // to pass.
        let effective_settings = self.settings.with_gen_spec_overrides(&gen_spec);
        #[cfg(feature = "llama-dflash2")]
        let use_dflash = gen_spec.output_mode == crate::generation::OutputMode::Text
            && self.bundle.draft.is_some()
            && effective_settings.sampling.temperature == Some(0.0)
            && gen_spec.grammar.is_none()
            && !matches!(
                gen_spec.speculative,
                Some(crate::backend::common::speculative::SpeculativeMode::Off)
            );
        let (sampler, grammar) = if let Some(request) = &self.strict_request {
            if &gen_spec != request {
                return Err(ExecError::InvalidArg(
                    "strict request generation settings are frozen",
                ));
            }
            self.prepared_sampling
                .lock()
                .take()
                .ok_or(ExecError::FeatureUnsupported("strict session pause/resume"))?
        } else {
            Self::build_sampler_and_grammar(&effective_settings, &gen_spec, &self.bundle)?
        };
        let mut guard = self.state.lock();
        if guard
            .as_ref()
            .is_some_and(|state| state.microphone.is_some())
        {
            return Err(ExecError::InvalidArg(
                "commit microphone input before generation",
            ));
        }
        let state = guard
            .take()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;
        let state_slot = Arc::downgrade(&self.state);

        let pre_events = self.build_media_events();
        let mut puller = TokenPuller::new_from_session(
            self.id,
            self.hooks.clone(),
            self.bundle.clone(),
            state_slot,
            state,
            sampler,
            grammar,
            gen_spec,
            self.paused.clone(),
            self.stopped.clone(),
            pre_events,
        );
        puller.set_request_context(self.request_context.clone());
        if let Some(tools) = &self.enabled_tool_names {
            puller.arm_enabled_tools(tools.clone());
        }
        #[cfg(feature = "llama-dflash2")]
        puller.enable_dflash2(use_dflash);
        Ok(puller)
    }

    /// Build the base sampler chain plus an optional [`GrammarMatcher`]
    /// from `gen_spec.grammar`. The matcher masks logits in the puller
    /// (`TokenPuller::sample_one`) using the SAME llguidance engine the
    /// MLX backend uses — fed the GGUF's embedded vocab via
    /// [`grammar_vocab_from_model`]. This replaces llama.cpp's built-in
    /// `LlamaSampler::llguidance`, whose runtime Matcher rejected the
    /// opening token at the pinned dep rev and silently fell back to
    /// unconstrained output. On matcher-build failure we generate
    /// unconstrained for legacy requests; exact-context requests fail closed.
    fn build_sampler_and_grammar(
        settings: &Settings,
        gen_spec: &GenSpec,
        bundle: &ModelBundle,
    ) -> Result<(LlamaSampler, Option<GrammarMatcher>), ExecError> {
        let Some(spec) = gen_spec.grammar.clone() else {
            return Ok((
                Self::sampler_from_settings(settings, bundle, gen_spec.output_mode),
                None,
            ));
        };
        match bundle
            .grammar
            .matcher(|| grammar_vocab_from_model(&bundle.model), spec)
        {
            Ok(matcher) => {
                // Grammar masking forces *structural* validity but the schema
                // still permits arbitrary inter-token whitespace, and a small
                // model can wedge there — looping on `\n`+indent until
                // max_tokens, producing truncated JSON. Prepend a mild
                // repeat/presence penalty (independent of user settings, which
                // default to no penalty) so repeated whitespace gets damped and
                // generation makes progress to the closing brace. NO frequency
                // penalty here: grammar-constrained JSON legitimately repeats
                // content ids and a count-proportional penalty poisons them
                // (see GRAMMAR_ANTILOOP_PENALTIES).
                let sampler = Self::sampler_from_settings_antiloop(settings, bundle);
                Ok((sampler, Some(matcher)))
            }
            Err(e) => {
                if gen_spec.require_full_context {
                    return Err(ExecError::Coded {
                        code: "strict_grammar_construction_failed".into(),
                        message: "strict grammar construction failed".into(),
                    });
                }
                tracing::warn!(
                    ?e,
                    "llama grammar matcher build failed; generating unconstrained"
                );
                Ok((
                    Self::sampler_from_settings(settings, bundle, gen_spec.output_mode),
                    None,
                ))
            }
        }
    }

    /// Grammar-path antiloop penalty parameters:
    /// `(last_n, repeat, freq, present)` for [`LlamaSampler::penalties`].
    ///
    /// CONSTRAINT — `freq` MUST stay 0.0 on the grammar path.
    /// Grammar-constrained JSON legitimately repeats content ids (e.g.
    /// `"collection":"people"` once per op in a semantic-ops array); a
    /// frequency penalty grows with every occurrence and by ~op 4 buries the
    /// true id's logits, so the model escapes to `""` — observed as
    /// start-strong-decay-to-empty at both 1.2B and 8B, with the
    /// antiloop-free MLX path proving the counterfactual (R09 §5.2).
    /// The mild repeat + one-shot presence penalty keep the original
    /// whitespace-run/rambling protection without count-proportional
    /// distortion; schemas additionally ship
    /// `"x-guidance": {"whitespace_flexible": false}`, which removed the
    /// original wedge motivation for the strong frequency component.
    const GRAMMAR_ANTILOOP_PENALTIES: (i32, f32, f32, f32) = (256, 1.05, 0.0, 0.8);

    /// Like [`Self::sampler_from_settings`] but guarantees a repeat /
    /// presence penalty is present (prepended) to break degenerate
    /// repetition loops under grammar-constrained decoding. Only used on the
    /// grammar path; unconstrained generation keeps the user's exact chain.
    /// Parameters (and the no-frequency-penalty constraint) live in
    /// [`Self::GRAMMAR_ANTILOOP_PENALTIES`].
    fn sampler_from_settings_antiloop(settings: &Settings, bundle: &ModelBundle) -> LlamaSampler {
        let (last_n, repeat, freq, present) = Self::GRAMMAR_ANTILOOP_PENALTIES;
        let antiloop =
            LlamaSampler::penalties(bundle.model.n_vocab(), last_n, repeat, freq, present);
        LlamaSampler::chain_simple(vec![
            antiloop,
            Self::sampler_from_settings(settings, bundle, crate::generation::OutputMode::Text),
        ])
    }

    pub fn save_cache(&self, dst: KvSaveSpec) -> Result<KvSnapshot, ExecError> {
        if self.strict_request.is_some() {
            return Err(ExecError::FeatureUnsupported("strict session KV cache"));
        }
        #[cfg(feature = "llama-dflash2")]
        if self.bundle.draft.is_some() {
            return Err(ExecError::FeatureUnsupported(
                "DFlash2 persistent KV snapshots",
            ));
        }
        let guard = self.state.lock();
        let state = guard
            .as_ref()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;
        if state.microphone.is_some() {
            return Err(ExecError::FeatureUnsupported(
                "uncommitted microphone KV cache",
            ));
        }
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

        let mut meta = self.build_kv_meta(&self.bundle)?;
        meta.kv_token_count = state.cur_pos.max(0) as u64;
        meta.transcript_sha256 = transcript_sha256(&self.messages.read());
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
        if self.strict_request.is_some() {
            return Err(ExecError::FeatureUnsupported("strict session KV cache"));
        }
        #[cfg(feature = "llama-dflash2")]
        if self.bundle.draft.is_some() {
            return Err(ExecError::FeatureUnsupported(
                "DFlash2 persistent KV snapshots",
            ));
        }
        let mut guard = self.state.lock();
        let state = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;
        if state.microphone.is_some() {
            return Err(ExecError::FeatureUnsupported(
                "uncommitted microphone KV cache",
            ));
        }
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
            kv_token_count: 0,
            transcript_sha256: [0u8; 32],
        })
    }
}

/// Hash of a transcript (roles + bodies + names) for keepwarm identity.
/// Computed AFTER meta/persona injection so save and restore hash the
/// same effective message list.
pub(crate) fn transcript_sha256(messages: &[Message]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for m in messages {
        h.update(m.role.as_bytes());
        h.update([0u8]);
        if let Ok(body) = serde_json::to_vec(&m.body) {
            h.update(&body);
        }
        h.update([0u8]);
        if let Some(name) = &m.name {
            h.update(name.as_bytes());
        }
        h.update([0xFFu8]);
    }
    h.finalize().into()
}

impl Session {
    fn sampler_from_settings(
        settings: &Settings,
        bundle: &ModelBundle,
        output_mode: crate::generation::OutputMode,
    ) -> LlamaSampler {
        // Order: penalties → arch_bias → top_k → min_p → top_p → temp → dist
        let mut chain = Vec::new();

        if let Some(penalties) = Self::penalties_sampler(settings, bundle) {
            tracing::debug!("Sampling penalties: {:?}", penalties);
            chain.push(penalties);
        }
        if let Some(bias) = Self::architecture_logit_bias(bundle, output_mode) {
            chain.push(bias);
        }

        // Temperature zero means "take the most likely token", and the
        // truncation samplers below cannot change which token that is: they
        // discard tails, and the maximum is never in a tail. Running them
        // anyway sorts a vocabulary — 151k entries for Qwen3 — once per token,
        // for a result that is decided before they start. Penalties and the
        // architecture bias stay, because those rewrite logits and so can move
        // the maximum.
        if settings.sampling.temperature == Some(0.0) {
            chain.push(LlamaSampler::greedy());
            return LlamaSampler::chain_simple(chain);
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

    /// Per-architecture logit bias to harden sampling against quantization
    /// pathologies. Currently:
    ///
    /// - **Gemma 1 / 2 / 3**: suppress `<start_of_turn>`. Llama.cpp marks
    ///   both `<start_of_turn>` and `<end_of_turn>` as EOG in the gemma
    ///   vocab; under aggressive quants (IQ2_M, Q2_K) the model occasionally
    ///   picks `<start_of_turn>` mid-reply, producing a sentence cut at a
    ///   random position. Suppressing it lets the model continue until it
    ///   either picks `<end_of_turn>` or a continuation token.
    ///
    /// - **Gemma 4**: no-op. Gemma 4 collapses both turn markers into a
    ///   single `<turn|>` token (id 106) which IS the model's legitimate
    ///   EOS. Suppressing it would break end-of-turn entirely. If your
    ///   Gemma 4 replies are getting cut mid-sentence, that's quantization
    ///   noise causing premature EOS sampling — the fix is a higher-
    ///   precision quant (Q4_K_M minimum), not a sampler change. The scan
    ///   below returns None for Gemma 4 because the vocab has no
    ///   `<start_of_turn>` token to suppress.
    fn architecture_logit_bias(
        bundle: &ModelBundle,
        output_mode: crate::generation::OutputMode,
    ) -> Option<LlamaSampler> {
        use llama_cpp_2::token::LlamaToken;
        use llama_cpp_2::token::logit_bias::LlamaLogitBias;
        let arch = bundle.meta.architecture.as_deref()?;
        if arch == "lfm3" && output_mode == crate::generation::OutputMode::Text {
            let bias: Vec<_> = ["<|audio_start|>", "<|mixed_start|>"]
                .into_iter()
                .filter_map(|piece| {
                    let tokens = bundle.model.str_to_token(piece, AddBos::Never).ok()?;
                    match tokens.as_slice() {
                        [token] => Some(LlamaLogitBias::new(*token, f32::NEG_INFINITY)),
                        _ => None,
                    }
                })
                .collect();
            return (!bias.is_empty())
                .then(|| LlamaSampler::logit_bias(bundle.model.n_vocab(), &bias));
        }
        let family = crate::zoo::ModelFamily::detect(Some(arch), None);
        if !matches!(
            family,
            crate::zoo::ModelFamily::Gemma2
                | crate::zoo::ModelFamily::Gemma3
                | crate::zoo::ModelFamily::Gemma4,
        ) {
            return None;
        }
        // The literal string `<start_of_turn>` doesn't tokenize to a single
        // id via `str_to_token` (the Gemma SentencePiece tokenizer treats
        // it as plain text and breaks it into 7 pieces). Instead, scan the
        // start of the vocab — Gemma reserves `<start_of_turn>` /
        // `<end_of_turn>` as control tokens in the first ~256 ids — and
        // match by piece text directly.
        let n_vocab = bundle.model.n_vocab();
        let scan_end = 256.min(n_vocab);
        let mut found: Option<i32> = None;
        for id in 0..scan_end {
            let bytes = bundle
                .model
                .token_to_piece_bytes(LlamaToken(id), 32, true, None)
                .unwrap_or_default();
            if bytes.as_slice() == b"<start_of_turn>" {
                found = Some(id);
                break;
            }
        }
        let Some(tok_id) = found else {
            tracing::warn!(
                target: "pio::gen2::llama::sampler",
                arch = %arch,
                "could not locate <start_of_turn> in vocab[0..{scan_end}]; skipping gemma logit bias"
            );
            return None;
        };
        let biases = vec![LlamaLogitBias::new(LlamaToken(tok_id), f32::NEG_INFINITY)];
        tracing::info!(
            target: "pio::gen2::llama::sampler",
            arch = %arch,
            token_id = tok_id,
            "gemma logit bias: suppressing <start_of_turn>"
        );
        Some(LlamaSampler::logit_bias(n_vocab, &biases))
    }

    fn penalties_sampler(settings: &Settings, bundle: &ModelBundle) -> Option<LlamaSampler> {
        let sampling = &settings.sampling;
        // `-1` has always meant "the whole context" to callers, and the
        // settings validator still accepts it. llama.cpp stopped reading it
        // that way (negative values now clamp to 0, which disables the
        // penalty), so the translation happens here instead of silently
        // flipping the setting off.
        let penalty_last_n = match sampling.penalty_last_n.unwrap_or(0) {
            -1 => i32::try_from(bundle.meta.n_ctx).unwrap_or(i32::MAX),
            n => n,
        };
        let penalty_repeat = sampling.penalty_repeat.unwrap_or(1.0);
        let penalty_freq = sampling.penalty_freq.unwrap_or(0.0);
        let penalty_present = sampling.penalty_present.unwrap_or(0.0);

        let repeat_is_default = (penalty_repeat - 1.0).abs() <= f32::EPSILON;
        if penalty_last_n == 0 && repeat_is_default && penalty_freq == 0.0 && penalty_present == 0.0
        {
            None
        } else {
            Some(LlamaSampler::penalties(
                bundle.model.n_vocab(),
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            ))
        }
    }
}

impl Session {
    /// Keepwarm restore: rebuild a session from a saved KV blob, skipping
    /// the prefill of everything the blob covers. Returns `Ok(None)` on a
    /// lenient miss (caller falls through to the cold path); `Err` only
    /// for strict-mode mismatches or real failures after commit.
    ///
    /// Semantics are exact-state resumption: the blob's transcript hash
    /// must equal the new transcript minus its final (delta) message.
    /// There is no token-prefix LCP on purpose — KV holds SAMPLED
    /// assistant tokens that a re-render can't reproduce (task #91), so
    /// partial matches against a fresh render would lie about KV contents.
    #[allow(clippy::too_many_arguments, clippy::arc_with_non_send_sync)]
    fn try_restore(
        enabled_tool_names: &Option<std::collections::HashSet<String>>,
        enable_thinking: bool,
        id: SessionId,
        bundle: &Arc<ModelBundle>,
        backend: &Arc<LlamaBackend>,
        hooks: &Arc<HookBus>,
        settings: &Settings,
        chat_template: &ChatTemplate,
        messages: &[Message],
        spec: &KvLoadSpec,
    ) -> Result<Option<Self>, ExecError> {
        let (path, strict) = match spec {
            KvLoadSpec::Strict(p) => (p, true),
            KvLoadSpec::Lenient(p) => (p, false),
        };
        macro_rules! miss {
            ($reason:expr) => {{
                let reason: String = $reason;
                if strict {
                    return Err(ExecError::KvIncompatible(reason));
                }
                tracing::info!(target: "pio::gen2::kv::keepwarm", %reason, "restore miss — cold path");
                return Ok(None);
            }};
        }

        let blob = match read_from_path(path) {
            Ok(b) => b,
            Err(e) => miss!(format!("kv blob unreadable: {e}")),
        };
        let (hdr, payload) = match parse_blob(&blob) {
            Ok(v) => v,
            Err(e) => miss!(format!("kv blob corrupt: {e}")),
        };
        let cur = &bundle.meta;
        if hdr.meta.model_uuid != cur.model_uuid
            || hdr.meta.n_ctx != cur.n_ctx
            || hdr.meta.n_layer != cur.n_layer
            || hdr.meta.tokenizer_digest != cur.tokenizer_digest
            || hdr.meta.template_fingerprint != cur.template_fingerprint
        {
            miss!("model identity mismatch".to_string());
        }
        if hdr.meta.kv_token_count == 0 {
            miss!("pre-keepwarm blob (no token count)".to_string());
        }
        if messages.len() < 2 {
            miss!("transcript too short for resume".to_string());
        }
        // Two valid split points: the blob was saved either after the
        // reply was appended to the transcript (delta = [user]) or right
        // after generation, before the continuation append (delta =
        // [assistant, user]) — the sampled assistant tokens are already
        // IN the KV, and assistant turns never re-render (same invariant
        // as the live append path).
        let mut split = None;
        for k in [messages.len() - 1, messages.len().saturating_sub(2)] {
            if k == 0 {
                continue;
            }
            if k == messages.len().saturating_sub(2) && messages[k].role != "assistant" {
                continue;
            }
            if transcript_sha256(&messages[..k]) == hdr.meta.transcript_sha256 {
                split = Some(k);
                break;
            }
        }
        let Some(k) = split else {
            miss!("transcript divergence".to_string());
        };
        let delta = &messages[k..];
        // Delta renders like the warm append path: assistant turns are
        // already in KV as sampled tokens and never re-render.
        let delta_msgs: Vec<Message> = delta
            .iter()
            .filter(|m| m.role != "assistant")
            .cloned()
            .collect();
        if delta_msgs.is_empty() {
            miss!("empty delta after restore".to_string());
        }
        let delta_prompt = chat_template
            .apply(delta_msgs, None, None)
            .map_err(ExecError::Other)?;
        let delta_tokens = bundle
            .model
            .str_to_token(&delta_prompt, AddBos::Never)
            .map_err(|e| ExecError::Other(e.into()))?;
        if delta_tokens.is_empty() {
            miss!("delta rendered to zero tokens".to_string());
        }

        // Context setup mirrors the cold path (incl. the fit clamp).
        let ctx_size = settings.system.ctx_size.unwrap_or_else(|| {
            use crate::bundle::gguf::fit_context;
            let hw = crate::hardware::HardwareProfile::cached();
            let kv = bundle.context_kv_bytes_per_token;
            fit_context(
                hw.inference_budget_bytes(),
                bundle.model.size(),
                kv,
                bundle.meta.n_ctx.max(128),
                Some(hw.tier_context_cap()),
            )
            .max(128)
        });
        // `kv_token_count` comes off the blob header. It is inside the header
        // digest now, so it is no longer attacker-chosen, but a wrong number
        // here should decline the cache rather than wrap and silently accept a
        // state that does not fit.
        let restored_n = hdr.meta.kv_token_count as usize;
        let needed = restored_n
            .saturating_add(delta_tokens.len())
            .saturating_add(64);
        if needed >= ctx_size as usize {
            miss!(format!(
                "restored state ({restored_n} tokens) + delta ({}) won't fit ctx {ctx_size}",
                delta_tokens.len()
            ));
        }

        let batch_size = settings.system.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        // n_ubatch must track n_batch: llama.cpp caps physical prefill
        // micro-batches at n_ubatch (default 512), so raising n_batch alone
        // does nothing above 512.
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_size))
            .with_n_batch(batch_size)
            .with_n_ubatch(batch_size);
        if let Some(n) = settings.system.threads {
            ctx_params = ctx_params.with_n_threads(n as i32);
        }
        if let Some(n) = settings.system.threads_batch {
            ctx_params = ctx_params.with_n_threads_batch(n as i32);
        }
        if settings.system.flash_attn.unwrap_or(true) {
            ctx_params =
                ctx_params.with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO);
        }
        let mut ctx_cell = SessionCtxCell::try_new(bundle.clone(), |owner| {
            DepCtx::create(owner, backend, kv_params(ctx_params, settings))
        })
        .map_err(|e| ExecError::Other(e.into()))?;

        // SAFETY: payload came from copy_state_data on a context validated
        // compatible by the identity checks above.
        ctx_cell.with_dependent_mut(|_, ctx| unsafe { ctx.set_state_data(payload) });
        let pos_max = ctx_cell.with_dependent(|_, ctx| ctx.kv_cache_seq_pos_max(0));
        if ((pos_max + 1).max(0) as usize) < restored_n {
            miss!(format!(
                "restored KV covers {} tokens, header claims {restored_n}",
                (pos_max + 1).max(0)
            ));
        }

        // Prefill only the delta, starting at the restored position.
        let prefill_start_us = llama_cpp_2::ggml_time_us() as u64;
        hooks.emit(HookEvent::SessionPrefillStart {
            session_id: id,
            prompt_tokens: delta_tokens.len(),
        });
        let mut batch = LlamaBatch::new(batch_size as usize, 1);
        let mut cur_pos = restored_n as i32;
        let mut remaining = delta_tokens;
        let total_delta = remaining.len();
        let mut done = 0usize;
        let mut last_batch_tokens: i32 = 0;
        while !remaining.is_empty() {
            let chunk_size = remaining.len().min(batch_size as usize);
            let chunk: Vec<_> = remaining.drain(..chunk_size).collect();
            batch.clear();
            for (i, token) in chunk.into_iter().enumerate() {
                let is_last = done + i + 1 == total_delta;
                batch
                    .add(token, cur_pos + i as i32, &[0], is_last)
                    .map_err(|e| ExecError::Other(e.into()))?;
            }
            ctx_cell
                .with_dependent_mut(|_, ctx| ctx.decode(&mut batch))
                .map_err(|e| ExecError::Other(e.into()))?;
            cur_pos += chunk_size as i32;
            done += chunk_size;
            last_batch_tokens = batch.n_tokens();
        }
        hooks.emit(HookEvent::SessionPrefillOk {
            session_id: id,
            prompt_tokens: total_delta,
        });
        tracing::info!(
            target: "pio::gen2::kv::keepwarm",
            restored_tokens = restored_n,
            delta_tokens = total_delta,
            "session restored from saved KV — prefill skipped"
        );

        Ok(Some(Self {
            id,
            bundle: bundle.clone(),
            hooks: hooks.clone(),
            settings: settings.clone(),
            chat_template: chat_template.clone(),
            paused: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(Some(DecodeState {
                ctx_cell,
                microphone: None,
                cur_pos,
                logits_i: (last_batch_tokens - 1).max(0),
                prefill_start_us,
            }))),
            messages: RwLock::new(messages.to_vec()),
            initial_messages_dropped: 0,
            ctx_size,
            enabled_tool_names: enabled_tool_names.clone(),
            enable_thinking,
            strict_request: None,
            request_context: None,
            prepared_sampling: Mutex::new(None),
        }))
    }

    #[allow(clippy::too_many_arguments, clippy::arc_with_non_send_sync)]
    pub(crate) fn new(
        id: SessionId,
        bundle: Arc<ModelBundle>,
        backend: Arc<LlamaBackend>,
        hooks: Arc<HookBus>,
        settings: Settings,
        messages: Vec<Message>,
        persona: Option<&crate::types::Persona>,
        cache: Option<KvLoadSpec>,
        tools: Option<(Vec<crate::types::message::ToolSpec>, String)>,
        thinking: crate::generation::ThinkingMode,
        strict_request: Option<GenSpec>,
    ) -> Result<Self, ExecError> {
        let has_media = media_parts(&messages).next().is_some();
        let has_microphone =
            media_parts(&messages).any(|part| matches!(part, MessageChunk::Microphone));
        if has_microphone && !bundle.capabilities.contains(Capabilities::MICROPHONE) {
            return Err(ExecError::FeatureUnsupported("microphone input"));
        }
        if let Some(request) = &strict_request {
            if !request.require_full_context || !request.max_tokens.is_some_and(|n| n > 0) {
                return Err(ExecError::InvalidArg(
                    "strict request needs a positive output budget",
                ));
            }
            if cache.is_some()
                || has_media
                || !matches!(
                    request.speculative,
                    Some(crate::backend::common::speculative::SpeculativeMode::Off)
                )
            {
                return Err(ExecError::FeatureUnsupported(
                    "strict request cache/media/speculation",
                ));
            }
            #[cfg(feature = "llama-dflash2")]
            if bundle.draft.is_some() {
                return Err(ExecError::FeatureUnsupported("strict DFlash2 request"));
            }
        }
        let enabled_tool_names: Option<std::collections::HashSet<String>> = tools
            .as_ref()
            .map(|(ts, _)| ts.iter().map(|t| t.function.name.clone()).collect());
        // The chat template's `enable_thinking`. An explicit policy wins;
        // `Auto` is the family's default — see
        // `ModelFamily::default_enable_thinking`, the single owner of that
        // flag: Gemma 4 IT gates its thinking block on it and, without it,
        // emits `<turn|>` inside markdown bold instead of finishing answers,
        // so the family mirrors llama-cli's `--jinja` default of `true`.
        let enable_thinking = match thinking {
            crate::generation::ThinkingMode::On => true,
            crate::generation::ThinkingMode::Off => false,
            crate::generation::ThinkingMode::Auto => {
                crate::zoo::ModelFamily::detect(bundle.meta.architecture.as_deref(), None)
                    .default_enable_thinking()
            }
        };
        let mut messages = messages;

        let include_meta = settings.prompt.include_meta.unwrap_or(true)
            && std::env::var("PIO_DISABLE_META_PROMPT").ok().as_deref() != Some("1");
        let meta_prompt = if include_meta {
            crate::session_rt::prompt::build_meta_prompt()
        } else {
            String::new()
        };
        let system_prompt = settings.prompt.system_prompt.as_deref();

        let merged_prompt = merge_prompts(&meta_prompt, system_prompt, persona);

        if strict_request.is_none() {
            tracing::debug!(message_count = messages.len(), "preparing session prompt");
        }

        // Build chat template FIRST so the system-injection logic below
        // can probe whether the template accepts a `system` role at
        // message[0]. Gemma 2's template raises an exception if it
        // sees one — surfaced by the 20-turn zoo matrix on gemma-2-2b
        // as `syntax error: System role not supported`. The probe lets
        // us route around that by folding system into the first user
        // message instead.
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

        let has_system = messages.iter().any(|m| m.role == "system");
        if !has_system && !merged_prompt.trim().is_empty() {
            if chat_template.supports_system_role() {
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
            } else if let Some(first_user_idx) = messages.iter().position(|m| m.role == "user")
                && let MessageBody::Content {
                    content: MessageContent::SingleText(text),
                } = &mut messages[first_user_idx].body
            {
                *text = format!("{}\n\n{}", merged_prompt, text);
                // For non-SingleText content (multimodal etc.) we leave
                // the message alone — folding into a chunked content is
                // an open question per modality and gets handled in the
                // template anyway via the user-role path.
            }
            // else: no user message to fold into, no system support;
            // skip injection entirely. The model will just get whatever
            // messages it was passed.
        }

        // ── Keepwarm: exact-state resume attempt before any prefill.
        // Media sessions take the MTMD branch below and are excluded.
        if let Some(spec) = &cache
            && !has_media
        {
            #[cfg(feature = "llama-dflash2")]
            if bundle.draft.is_some() {
                return Err(ExecError::FeatureUnsupported(
                    "DFlash2 persistent KV restore",
                ));
            }
            match Self::try_restore(
                &enabled_tool_names,
                enable_thinking,
                id,
                &bundle,
                &backend,
                &hooks,
                &settings,
                &chat_template,
                &messages,
                spec,
            ) {
                Ok(Some(session)) => return Ok(session),
                Ok(None) => {} // lenient miss — fall through to cold path
                Err(e) => return Err(e),
            }
        }

        let template_messages = if bundle.mtmd_ctx.is_some() && has_media {
            with_media_markers(
                &messages,
                bundle
                    .mtmd_marker
                    .as_deref()
                    .unwrap_or(mtmd_default_marker()),
            )
        } else {
            messages.clone()
        };
        let prompt = chat_template
            .apply(template_messages, tools.clone(), Some(enable_thinking))
            .map_err(ExecError::Other)?;
        if strict_request.is_none() {
            tracing::info!(target:"pio::gen2::llama::prompt", len=prompt.len(), enable_thinking, prompt=%prompt, "rendered chat prompt");
        }
        let mut tokens_list = tokenize_chat_prompt(&bundle.model, &prompt)?;
        if strict_request.is_none() {
            tracing::info!(target:"pio::gen2::llama::prompt", n_tokens=tokens_list.len(), "tokenized chat prompt");
        }

        // Create context. An explicit user setting wins; otherwise clamp
        // the model's training context to what actually fits this host —
        // n_ctx_train can be 262144, whose KV cache alone would blow the
        // inference budget the fit/residency layers were built around.
        let ctx_size = settings.system.ctx_size.unwrap_or_else(|| {
            use crate::bundle::gguf::fit_context;
            let hw = crate::hardware::HardwareProfile::cached();
            let kv = bundle.context_kv_bytes_per_token;
            let fitted = fit_context(
                hw.inference_budget_bytes(),
                bundle.model.size(),
                kv,
                bundle.meta.n_ctx.max(128),
                Some(hw.tier_context_cap()),
            );
            if fitted < bundle.meta.n_ctx {
                tracing::info!(
                    target: "pio::gen2::llama::ctx_fit",
                    n_ctx_train = bundle.meta.n_ctx,
                    fitted,
                    kv_bytes_per_token = kv,
                    model_bytes = bundle.model.size(),
                    budget_bytes = hw.inference_budget_bytes(),
                    "clamped default context to fit host memory"
                );
            }
            fitted.max(128)
        });

        // Context overflow: truncate old messages until conversation fits.
        // Generic driver in session_rt::truncate — Phase 3 refactor.
        let original_message_count = messages.len();
        if strict_request.is_none() && !has_microphone {
            let tokenizer = Arc::new(super::tokenizer_adapter::LlamaSessionTokenizer {
                bundle: bundle.clone(),
                chat_template: chat_template.clone(),
            }) as Arc<dyn crate::backend::traits::SessionTokenizer>;
            let outcome = crate::session_rt::ColdStart::apply(
                tokenizer,
                &settings,
                ctx_size as usize,
                messages,
            )?;
            messages = outcome.messages;
            // Only re-tokenize if truncation actually dropped something; otherwise
            // `tokens_list` from the initial tokenization above is still valid.
            if outcome.dropped > 0 {
                let final_prompt = chat_template
                    .apply(messages.clone(), tools.clone(), Some(enable_thinking))
                    .map_err(ExecError::Other)?;
                tokens_list = tokenize_chat_prompt(&bundle.model, &final_prompt)?;
            }
        }
        let batch_size = settings.system.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        // n_ubatch must track n_batch (see the restore path above).
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_size))
            .with_n_batch(batch_size)
            .with_n_ubatch(batch_size);
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
            DepCtx::create(owner, &backend, kv_params(ctx_params, &settings))
        })
        .map_err(|e| ExecError::Other(e.into()))?;

        // Query the actual single-sequence allocation before ANY prefill. The
        // same frozen rendered token vector below is checked and then consumed.
        let actual_context = ctx_cell.with_dependent(|_, ctx| ctx.n_ctx());
        let request_context = strict_request
            .as_ref()
            .map(|request| {
                exact_request_context(
                    tokens_list.len(),
                    request.max_tokens.unwrap(),
                    actual_context,
                )
            })
            .transpose()?;
        let prepared_sampling = strict_request
            .as_ref()
            .map(|request| {
                Self::build_sampler_and_grammar(
                    &settings.with_gen_spec_overrides(request),
                    request,
                    &bundle,
                )
            })
            .transpose()?;

        if has_media {
            let mtmd_ctx = bundle
                .mtmd_ctx
                .as_ref()
                .ok_or(ExecError::FeatureUnsupported("media input"))?;
            let marker = bundle
                .mtmd_marker
                .as_deref()
                .unwrap_or(mtmd_default_marker());
            let prompt = super::media::tokenize(mtmd_ctx, prompt, marker, &messages, true)?;
            let mut state = DecodeState {
                ctx_cell,
                cur_pos: 0,
                logits_i: -1,
                microphone: None,
                prefill_start_us: llama_cpp_2::ggml_time_us() as u64,
            };
            let prompt_tokens = state.prefill_media(prompt, batch_size)?;
            if state.microphone.is_none() {
                hooks.emit(HookEvent::SessionPrefillStart {
                    session_id: id,
                    prompt_tokens,
                });
                hooks.emit(HookEvent::SessionPrefillOk {
                    session_id: id,
                    prompt_tokens,
                });
            }

            return Ok(Self {
                id,
                bundle,
                hooks,
                settings,
                chat_template,
                paused: Arc::new(AtomicBool::new(false)),
                stopped: Arc::new(AtomicBool::new(false)),
                state: Arc::new(Mutex::new(Some(state))),
                messages: RwLock::new(messages),
                initial_messages_dropped: 0, // MTMD path has no truncation
                ctx_size,
                enabled_tool_names,
                enable_thinking,
                strict_request: None,
                request_context: None,
                prepared_sampling: Mutex::new(None),
            });
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
                microphone: None,
                cur_pos: total_tokens,
                logits_i: (last_batch_tokens - 1),
                prefill_start_us,
            }))),
            initial_messages_dropped: original_message_count.saturating_sub(messages.len()),
            messages: RwLock::new(messages),
            ctx_size,
            enabled_tool_names,
            enable_thinking,
            strict_request,
            request_context,
            prepared_sampling: Mutex::new(prepared_sampling),
        })
    }
}

impl Session {
    fn build_media_events(&self) -> std::collections::VecDeque<TokenEvent> {
        use crate::generation::MediaBoundary;
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
    fn append_lfm3_omni(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        let Some(mtmd) = self.bundle.mtmd_ctx.as_ref() else {
            return Err(ExecError::FeatureUnsupported("LFM3-Omni projector"));
        };
        if let Some(media) = crate::session_rt::media_util::unsupported_media(&new_messages, || {
            self.bundle.capabilities
        }) {
            return Err(ExecError::FeatureUnsupported(media));
        }

        let delta: Vec<_> = new_messages
            .iter()
            .filter(|m| m.role != "assistant")
            .cloned()
            .collect();
        if delta.is_empty() {
            self.messages.write().extend(new_messages);
            return Ok(0);
        }
        let marker = self
            .bundle
            .mtmd_marker
            .as_deref()
            .unwrap_or(mtmd_default_marker());
        let prompt = self
            .chat_template
            .apply(
                with_media_markers(&delta, marker),
                None,
                Some(self.enable_thinking),
            )
            .map_err(ExecError::Other)?;
        // Both Gen2's text decoder and the LFM3 codec stop before decoding the
        // sampled end-of-turn token. Close that assistant turn and strip the
        // explicit BOS that the template puts at the start of a fresh chat.
        let continuation = format!(
            "<|im_end|>\n{}",
            prompt.strip_prefix("<|startoftext|>").unwrap_or(&prompt)
        );
        let prompt = super::media::tokenize(mtmd, continuation, marker, &delta, false)?;
        let mut guard = self.state.lock();
        let st = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;
        let batch_size = self
            .settings
            .system
            .batch_size
            .unwrap_or(DEFAULT_BATCH_SIZE);
        st.prefill_start_us = llama_cpp_2::ggml_time_us() as u64;
        let prompt_tokens = match st.prefill_media(prompt, batch_size) {
            Ok(positions) => positions,
            Err(err) => {
                *guard = None;
                self.stopped.store(true, Ordering::Release);
                return Err(err);
            }
        };
        if st.microphone.is_none() {
            self.hooks.emit(HookEvent::SessionPrefillStart {
                session_id: self.id,
                prompt_tokens,
            });
            self.hooks.emit(HookEvent::SessionPrefillOk {
                session_id: self.id,
                prompt_tokens,
            });
        }
        self.messages.write().extend(new_messages);

        Ok(0)
    }

    /// Append new messages and prefill only the delta into the KV.
    /// Returns the number of old messages dropped due to context overflow (0 = no truncation).
    ///
    /// **KV continuation semantics**: the model's prior-turn assistant
    /// reply is already in KV as *sampled* tokens, which may include
    /// special tokens (`<|eot_id|>` on Llama, `<|im_end|>` on Qwen) that
    /// the chat template wouldn't emit if we re-rendered the assistant
    /// turn from text. So we NEVER re-tokenize the full conversation
    /// and diff against cur_pos — the template-rendered token stream
    /// and the sampled token stream diverge at the assistant boundary,
    /// and the diff arithmetic breaks silently (cur_pos > full_tokens.len()
    /// → empty delta → `prompt_tokens: 0` → immediate EOS, the exact
    /// symptom on Llama-3.2-3B GGUF multi-turn before this fix).
    ///
    /// Instead we render ONLY the *new* non-assistant messages (user,
    /// system) through the template, strip any leading BOS, prepend a
    /// turn-boundary token if the last sampled token wasn't an EOT, and
    /// prefill that delta. Mirrors the MLX path in `mlx/session.rs`.
    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        if self.strict_request.is_some() {
            return Err(ExecError::FeatureUnsupported("strict session append"));
        }
        if new_messages.is_empty() {
            return Ok(0);
        }

        if self
            .bundle
            .capabilities
            .contains(crate::engine::Capabilities::VIDEO)
        {
            return self.append_lfm3_omni(new_messages);
        }

        // 1) Extend the transcript so future calls (and the compaction
        //    path below) see the full history. Assistant messages are
        //    kept here for bookkeeping even though they don't re-render.
        {
            let mut msgs = self.messages.write();
            msgs.extend(new_messages.clone());
        }

        let tpl = &self.chat_template;

        // 2) Context overflow probe. We still need to know the FULL
        //    token count to decide whether to compact; overflow handling
        //    resets KV and re-prefills from scratch, so the drift between
        //    sampled and re-rendered tokens doesn't matter there.
        let all_messages = self.messages.read().clone();
        let full_prompt = tpl
            .apply(all_messages, None, Some(self.enable_thinking))
            .map_err(ExecError::Other)?;
        let full_tokens = tokenize_chat_prompt(&self.bundle.model, &full_prompt)?;

        let ctx_size = self
            .settings
            .system
            .ctx_size
            .unwrap_or(self.bundle.meta.n_ctx.max(128)) as usize;
        let gen_reserve = crate::session_rt::prompt::generation_reserve(
            ctx_size,
            self.settings.stopping.max_tokens,
        );
        let ctx_limit = ctx_size.saturating_sub(gen_reserve);

        let mut guard = self.state.lock();
        let st = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;

        if full_tokens.len() > ctx_limit {
            // Context overflow — reset and re-encode with truncated conversation.
            // Prefer Tier-1 algorithmic compaction (same as cold-start `maybe_compact`) so we
            // preserve intent via `<compact-summary>` instead of silently dropping turns.
            tracing::warn!(
                "context overflow: {} tokens > {} limit, compacting or truncating conversation",
                full_tokens.len(),
                ctx_limit
            );

            // Clear KV cache
            st.ctx_cell
                .with_dependent_mut(|_, ctx| ctx.clear_kv_cache());
            st.cur_pos = 0;

            let mut msgs = self.messages.write();

            // Generic driver in session_rt::truncate — Phase 3 refactor.
            let tokenizer = Arc::new(super::tokenizer_adapter::LlamaSessionTokenizer {
                bundle: self.bundle.clone(),
                chat_template: tpl.clone(),
            }) as Arc<dyn crate::backend::traits::SessionTokenizer>;
            let outcome = crate::session_rt::WarmStart::apply(
                tokenizer,
                &self.settings,
                ctx_size,
                msgs.clone(),
            )?;
            let working = outcome.messages;
            let dropped = outcome.dropped;

            // Re-tokenize the final message list for prefill.
            let remaining = {
                let p = tpl
                    .apply(working.clone(), None, Some(self.enable_thinking))
                    .map_err(ExecError::Other)?;
                tokenize_chat_prompt(&self.bundle.model, &p)?
            };

            *msgs = working;
            drop(msgs);

            // Fits — prefill the truncated conversation from scratch
            let mut to_process = remaining;
            let batch_size = self
                .settings
                .system
                .batch_size
                .unwrap_or(DEFAULT_BATCH_SIZE) as usize;
            let mut batch = LlamaBatch::new(batch_size, 1);

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
                    batch
                        .add(token, absolute, &[0], is_last)
                        .map_err(|_| ExecError::Other(anyhow::anyhow!("batch add error")))?;
                }
                if let Err(error) = st
                    .ctx_cell
                    .with_dependent_mut(|_, ctx| ctx.decode(&mut batch))
                {
                    // A successful target decode followed by failed draft
                    // injection must never leave warm state available for reuse.
                    *guard = None;
                    return Err(error);
                }
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

        // Normal path — render only the NEW non-assistant messages
        // and prefill them as a delta. Dropping assistant messages is
        // critical: their content is already in KV as sampled tokens,
        // and re-rendering them through the template would produce a
        // different token sequence (template strips / re-wraps text
        // differently than sampling produced it). Ignoring that drift
        // was the root of task #91: on Llama-3.2-3B GGUF the sampled
        // assistant reply ends with `<|eot_id|>` (id 128009) which the
        // template re-render does emit, but header wrapping differs,
        // and cur_pos ended up >= full_tokens.len() → empty delta →
        // `prompt_tokens: 0` → immediate EOS.
        let to_render: Vec<Message> = new_messages
            .into_iter()
            .filter(|m| m.role != "assistant")
            .collect();

        if to_render.is_empty() {
            // Pure assistant message append — nothing to prefill, but
            // the transcript was already extended above so future calls
            // see it.
            return Ok(0);
        }

        // No tools here: the tool block is already in KV from the first
        // prefill, and a delta rendered with tools would repeat it. The
        // thinking flag is not — it shapes the generation prompt itself.
        let delta_prompt = tpl
            .apply(to_render, None, Some(self.enable_thinking))
            .map_err(ExecError::Other)?;

        // Tokenize WITHOUT the BOS token — we've already emitted BOS
        // during the initial prefill and it must not reappear mid-stream.
        let remaining = self
            .bundle
            .model
            .str_to_token(&delta_prompt, AddBos::Never)
            .map_err(|e| ExecError::Other(e.into()))?;

        if remaining.is_empty() {
            return Ok(0);
        }
        let mut remaining = remaining;

        let batch_size = self
            .settings
            .system
            .batch_size
            .unwrap_or(DEFAULT_BATCH_SIZE) as usize;
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

            if let Err(error) = st
                .ctx_cell
                .with_dependent_mut(|_, ctx| ctx.decode(&mut batch))
            {
                *guard = None;
                return Err(error);
            }
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

// ─── Trait impls (Phase 2) ─────────────────────────────────────────────────
//
// Forward to existing inherent methods; dispatch enum remains in charge.

use crate::backend::traits::{BackendSession, KvSnapshot as KvSnapshotTrait, TokenPullerDyn};

impl BackendSession for Session {
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
    fn pull(&self, spec: GenSpec) -> Result<Box<dyn TokenPullerDyn>, ExecError> {
        let p = Session::pull(self, spec)?;
        Ok(Box::new(p) as Box<dyn TokenPullerDyn>)
    }
    fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        Session::append_messages(self, new_messages)
    }
    fn is_capturing_microphone(&self) -> bool {
        Session::is_capturing_microphone(self)
    }
    fn push_microphone(&self, samples: &[f32]) -> Result<usize, ExecError> {
        Session::push_microphone(self, samples)
    }
    fn commit_microphone(&self) -> Result<Vec<f32>, ExecError> {
        Session::commit_microphone(self)
    }
    fn as_kv_snapshot(&self) -> Option<&dyn KvSnapshotTrait> {
        if self.strict_request.is_some() {
            return None;
        }
        #[cfg(feature = "llama-dflash2")]
        if self.bundle.draft.is_some() {
            return None;
        }
        Some(self)
    }
    fn initial_messages_dropped(&self) -> usize {
        self.initial_messages_dropped
    }

    fn ctx_size(&self) -> u32 {
        self.ctx_size
    }
    fn is_poisoned(&self) -> bool {
        Session::is_poisoned(self)
    }
}

impl KvSnapshotTrait for Session {
    fn save_cache(&self, dst: KvSaveSpec) -> Result<KvSnapshot, ExecError> {
        Session::save_cache(self, dst)
    }
    fn load_cache(&self, src: KvLoadSpec) -> Result<KvLoadReport, ExecError> {
        Session::load_cache(self, src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_cpp_2::model::params::LlamaModelParams;
    use std::path::PathBuf;

    /// Regression pin for R09 §5.2: the grammar-path antiloop sampler must
    /// NOT apply a frequency penalty. Grammar-constrained JSON legitimately
    /// repeats content ids (`"collection":"people"` per op); a
    /// count-proportional penalty buries the true id by ~op 4 and the model
    /// escapes to `""`. A built `LlamaSampler` chain is an opaque FFI
    /// handle, so the observable seam is the parameter tuple the chain is
    /// constructed from — this pins it (the real end-to-end proof is a live
    /// 8B creation through the grammar path, ticket 03's exit criterion).
    #[test]
    fn grammar_antiloop_penalties_omit_frequency_component() {
        let (last_n, repeat, freq, present) = Session::GRAMMAR_ANTILOOP_PENALTIES;
        assert_eq!(
            freq, 0.0,
            "frequency penalty poisons repeated content ids in grammar-constrained JSON (R09 §5.2)"
        );
        // The rest of the antiloop protection stays: mild repeat + one-shot
        // presence penalty over a bounded window.
        assert_eq!(last_n, 256);
        assert!(repeat > 1.0);
        assert!(present > 0.0);
    }

    /// Locate a Gemma-family GGUF for the test, preferring `PIO_TEST_GGUF`
    /// then falling back to the dev model paths used during the bug
    /// investigation. Returns `None` if no candidate exists — caller
    /// should `eprintln!` + return early so the suite stays green
    /// without a model bundled in CI.
    fn find_gemma_gguf() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("PIO_TEST_GGUF") {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }
        for candidate in [
            "/Users/victor/pio-test-models/gemma-4-E2B-it-Q4_K_M.gguf",
            "/Users/victor/pio-test-models/gemma-4-E2B-it-UD-IQ2_M.gguf",
        ] {
            let path = PathBuf::from(candidate);
            if path.exists() {
                return Some(path);
            }
        }
        None
    }

    /// Regression: the chat template is constructed with `bos_token`, so
    /// `chat_template.apply()` renders a string that starts with a literal
    /// `<bos>`. With `parse_special=true`, `str_to_token` resolves that
    /// back to the BOS token id; combining that with `AddBos::Always`
    /// doubles the BOS token at position 0.
    ///
    /// Empirically, Gemma 4 IT models respond to `[BOS, BOS, ...]` by
    /// emitting their EOS (token 106 `<turn|>`) within 7-8 sampled tokens
    /// — the model treats the doubled start-of-sequence as "conversation
    /// already closed" and tries to terminate. Verified across E2B-IQ2_M,
    /// E2B-Q4_K_M, and 31B-Q4_K_XL — all truncate identically on factual
    /// prompts ("What is the capital of France?" → "The capital of France
    /// is **<turn|>").
    ///
    /// This test asserts the rendered + tokenized prompt has exactly one
    /// BOS at the start. Gated on a Gemma GGUF being available; skipped
    /// otherwise so the unit suite stays green.
    #[test]
    fn chat_template_tokenization_does_not_double_bos() {
        let Some(model_path) = find_gemma_gguf() else {
            eprintln!(
                "skipping: no Gemma GGUF found (set PIO_TEST_GGUF or place one in /Users/victor/pio-test-models/)"
            );
            return;
        };

        // Not `LlamaBackend::init()`: it may only succeed once per process,
        // and any test that ran first has already claimed it. This test failed
        // for exactly that reason in every configuration that compiles the
        // llama backend and runs the unit suite — which nothing did.
        let backend = crate::backend::llama::engine::get_backend().expect("llama backend");
        let params = LlamaModelParams::default();
        let model = llama_cpp_2::model::LlamaModel::load_from_file(&backend, &model_path, &params)
            .expect("load model");

        let mut bos_dec = encoding_rs::UTF_8.new_decoder();
        let mut eos_dec = encoding_rs::UTF_8.new_decoder();
        let template_str = model
            .chat_template(None)
            .expect("chat template present")
            .to_string()
            .expect("chat template utf-8");
        let bos_str = model
            .token_to_piece(model.token_bos(), &mut bos_dec, true, None)
            .expect("decode BOS");
        let eos_str = model
            .token_to_piece(model.token_eos(), &mut eos_dec, true, None)
            .expect("decode EOS");

        let chat_template = ChatTemplate::new(
            template_str,
            Some(TokenizerConfigToken::String(bos_str.clone())),
            Some(TokenizerConfigToken::String(eos_str)),
        );

        // Render a single-user-message chat the same way Session::new does.
        let user_msg = Message {
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText("What is the capital of France?".into()),
            },
            name: None,
            tool_call_id: None,
        };

        let prompt = chat_template
            .apply(vec![user_msg], None, Some(true))
            .expect("render chat template");

        // Sanity: the template should have placed the BOS string at the
        // very start. If this fails, the bug is in the template rather
        // than the tokenizer call — the test below would be misleading.
        assert!(
            prompt.starts_with(&bos_str),
            "chat template should render the BOS string at start; got: {:?}",
            prompt.chars().take(40).collect::<String>()
        );

        // The actual contract: route through the production helper and
        // assert exactly one BOS at position 0.
        let tokens = tokenize_chat_prompt(&model, &prompt).expect("tokenize prompt");

        let bos_id = model.token_bos();
        let leading_bos = tokens.iter().take_while(|t| **t == bos_id).count();

        assert_eq!(
            leading_bos,
            1,
            "rendered chat prompt was tokenized with {leading_bos} leading BOS tokens (expected 1). \
             Token ids: {:?}",
            tokens.iter().take(8).collect::<Vec<_>>()
        );
    }
}

/// Unlike conversational truncation, the full requested output is reserved.
fn exact_request_context(
    prompt: usize,
    output: usize,
    capacity: u32,
) -> Result<crate::types::RequestContext, ExecError> {
    if output == 0
        || prompt
            .checked_add(output)
            .is_none_or(|total| total > capacity as usize)
    {
        return Err(ExecError::ContextOverflow(format!(
            "exact prompt={prompt} output={output} capacity={capacity}; input preserved"
        )));
    }
    Ok(crate::types::RequestContext {
        prompt_tokens: prompt as u32,
        output_tokens: output as u32,
        capacity_tokens: capacity,
    })
}
#[cfg(test)]
mod strict_context_tests {
    use super::*;
    #[test]
    fn exact_budget_uses_full_output_reserve_and_rejects_overflow() {
        let report = exact_request_context(1, 127, 128).unwrap();
        assert_eq!(report.output_tokens, 127);
        assert!(exact_request_context(2, 127, 128).is_err());
        assert!(exact_request_context(128, 1, 128).is_err());
        assert!(exact_request_context(1, 0, 128).is_err());
        assert!(exact_request_context(1, usize::MAX, 128).is_err());
    }
}
