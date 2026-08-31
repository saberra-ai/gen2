//! [`BackendSession`] for the mlxcel backend.
//!
//! A session pins the prompt (built from `SessionSpec.messages`) and a `stop`
//! flag. `pull()` maps `GenSpec`+`Settings` into a full `SamplingConfig` (see
//! [`build_sampling_config`]), opens a bounded token
//! channel, hands a [`GenRequest`](super::worker::GenRequest) to the worker, and
//! returns a [`MlxcelTokenPuller`](super::puller::MlxcelTokenPuller) that drains
//! it. The heavy MLX work runs on the worker thread; the session/puller only
//! move `Send` data (strings, token tuples) across channels.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;

use mlxcel::SamplingConfig;

use crate::backend::common::chat_template::ChatTemplate;
use crate::engine::{ExecError, Settings};
use crate::generation::GenSpec;
use crate::types::message::{Message, MessageBody, MessageContent, TokenizerConfigToken};

use super::puller::MlxcelTokenPuller;
use super::worker::ModelWorker;

/// Chat-template `enable_thinking` policy for mlxcel sessions.
///
/// Mirrors mlx's default (`mlx::Session::new`,
/// `pio-core/src/gen2/backend/mlx/session.rs:378` passes `Some(true)`). We force
/// `Some(true)` rather than `None`: Gemma 4's template is sensitive to this — the
/// `None` (template-default) path yields the pathological "system has no
/// `<|think|>` but the model turn has a `<|channel>thought` trailer" state that
/// drives the jargon-loop / `l l l l` degeneration mlx documents at
/// `pio-core/src/gen2/backend/mlx/session.rs:450-456`. Explicit `Some(true)` is
/// the mlx-lm default and the coherent choice for gemma-4-coder.
const ENABLE_THINKING: Option<bool> = Some(true);

/// Bounded capacity for the worker→puller token channel. Small: it applies
/// backpressure to the decode loop if the consumer stalls, bounding memory. The
/// controller drains promptly, so this is rarely the limiting factor.
const TOKEN_CHANNEL_CAP: usize = 256;

pub(crate) struct MlxcelSession {
    id: u64,
    worker: Arc<ModelWorker>,
    settings: Settings,
    /// Conversation so far. `pull()` renders it into the prompt; `append_messages`
    /// extends it between turns.
    messages: parking_lot::RwLock<Vec<Message>>,
    /// The model's REAL Jinja chat template, built ONCE at session construction.
    ///
    /// LEAK GUARD: `ChatTemplate::new` leaks the parsed template + environment
    /// via `Box::leak` (`pio-core/src/gen2/backend/common/chat_template.rs:70-72`).
    /// `build_prompt` runs per-`pull()` in the agent loop (many calls per
    /// session); building a fresh `ChatTemplate` each call would leak unbounded
    /// for the process lifetime. So we build it ONCE here and reuse it —
    /// `ChatTemplate` is `Send + Sync` (only a `Template<'static,'static>` +
    /// `Option<String>` + `bool`), so it lives directly inside the
    /// `Arc<dyn BackendSession>` shared across threads. `None` when the model
    /// shipped no chat template — `build_prompt` then falls back to the naive
    /// concat with a loud warn.
    chat_template: Option<ChatTemplate>,
    /// Set by `stop()`; the worker's `on_token` callback checks it to halt the
    /// decode loop mid-stream. Recreated per `pull()`.
    stopped: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
}

impl std::fmt::Debug for MlxcelSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxcelSession")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl MlxcelSession {
    pub(crate) fn new(
        id: u64,
        worker: Arc<ModelWorker>,
        settings: Settings,
        messages: Vec<Message>,
        chat_template: Option<String>,
        bos_str: Option<String>,
        eos_str: Option<String>,
    ) -> Self {
        // LEAK GUARD: build the ChatTemplate ONCE here (it leaks internally via
        // `Box::leak`), never per-`pull()`. See the `chat_template` field doc.
        // Mirrors mlx's construction (`pio-core/src/gen2/backend/mlx/session.rs:431`).
        let chat_template = chat_template.map(|tpl| {
            ChatTemplate::new(
                tpl,
                bos_str.map(TokenizerConfigToken::String),
                eos_str.map(TokenizerConfigToken::String),
            )
        });

        Self {
            id,
            worker,
            settings,
            messages: parking_lot::RwLock::new(messages),
            chat_template,
            stopped: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Build the prompt string from the conversation.
    ///
    /// S6b: when the model ships a real chat template, render THROUGH it
    /// (gemma `<start_of_turn>` etc.) via [`render_with_template`] — mirroring
    /// `mlx::Session` (`pio-core/src/gen2/backend/mlx/session.rs:457-459`:
    /// `chat_template.apply(messages, None, enable_thinking)`). When no template
    /// is present we fall back to the naive role-tagged concat, but LOUDLY —
    /// never a silent quality regression.
    fn build_prompt(&self) -> String {
        let msgs = self.messages.read();
        let system_prompt = self
            .settings
            .prompt
            .system_prompt
            .as_deref()
            .filter(|s| !s.trim().is_empty());

        match self.chat_template.as_ref() {
            Some(tpl) => render_with_template(tpl, &msgs, system_prompt, ENABLE_THINKING),
            None => {
                tracing::warn!(
                    "mlxcel: model has no chat_template; naive prompt — quality degraded"
                );
                render_naive(&msgs, system_prompt)
            }
        }
    }
}

/// Render the conversation through the model's REAL chat template.
///
/// Mirrors `mlx::Session` (`pio-core/src/gen2/backend/mlx/session.rs:431-459`):
/// prepends a system-role message (preserving the old `build_prompt`'s
/// system-prompt injection — mlx does the same at
/// `pio-core/src/gen2/backend/mlx/session.rs:418-428`), then calls
/// `ChatTemplate::apply(messages, None, enable_thinking)`. The template appends
/// its own generation prompt (e.g. gemma's `<start_of_turn>model`), so unlike
/// the naive path we must NOT prime an "Assistant:" tail ourselves.
///
/// On render failure (a malformed template) we fall back to the naive concat
/// with a loud warn rather than returning an empty/garbage prompt — fail-loud,
/// never silent.
fn render_with_template(
    tpl: &ChatTemplate,
    msgs: &[Message],
    system_prompt: Option<&str>,
    enable_thinking: Option<bool>,
) -> String {
    let mut messages: Vec<Message> = Vec::with_capacity(msgs.len() + 1);
    if let Some(sys) = system_prompt {
        messages.push(Message {
            role: "system".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(sys.trim().to_string()),
            },
            name: None,
        });
    }
    messages.extend(msgs.iter().cloned());

    match tpl.apply(messages, None, enable_thinking) {
        Ok(prompt) => prompt,
        Err(e) => {
            tracing::warn!(
                "mlxcel: chat_template render failed ({e}); falling back to naive prompt — \
                 quality degraded"
            );
            render_naive(msgs, system_prompt)
        }
    }
}

/// Naive role-tagged concat — the legacy S2 tracer-bullet path. Retained ONLY
/// for the (rare) no-template fallback. NOT the model's real chat template.
fn render_naive(msgs: &[Message], system_prompt: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = system_prompt {
        out.push_str("System: ");
        out.push_str(sys.trim());
        out.push_str("\n\n");
    }
    for m in msgs.iter() {
        let text = match &m.body {
            MessageBody::Content { content } => content.as_visible_text(),
            // Tool-call messages have no visible text in this fallback path.
            MessageBody::Tool { .. } => String::new(),
        };
        let role = match m.role.as_str() {
            "assistant" => "Assistant",
            "system" => "System",
            _ => "User",
        };
        out.push_str(role);
        out.push_str(": ");
        out.push_str(&text);
        out.push('\n');
    }
    // Prime the model to continue as the assistant.
    out.push_str("Assistant: ");
    out
}

impl crate::backend::traits::BackendSession for MlxcelSession {
    fn id(&self) -> u64 {
        self.id
    }

    fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    fn pull(
        &self,
        spec: GenSpec,
    ) -> Result<Box<dyn crate::backend::traits::TokenPullerDyn>, ExecError> {
        // Fresh stop flag per generation so a prior `stop()` doesn't leak into
        // the next `pull`.
        self.stopped.store(false, Ordering::SeqCst);

        let max_tokens = spec
            .max_tokens
            .or(self.settings.stopping.max_tokens)
            .unwrap_or(512);

        // S4b: full GenSpec + Settings → SamplingConfig (temp/top_p/top_k/min_p/
        // seed/penalties/DRY all propagate). Applies on the FAST text path; the
        // grammar path is greedy-argmax by design (deterministic tool calls) so
        // these params intentionally don't apply there.
        let sampling = build_sampling_config(&spec, &self.settings);

        let prompt = self.build_prompt();

        // Grammar-constrained decode (S4): when set, the worker diverts off the
        // fast `generate_streaming` path onto the manual masked loop. `None`
        // (the common case) keeps the fast path.
        let grammar = spec.grammar.clone();

        let (tokens_tx, tokens_rx) = sync_channel(TOKEN_CHANNEL_CAP);
        let _prompt_len = self.worker.start_generation_blocking(
            prompt,
            max_tokens,
            sampling,
            grammar,
            self.stopped.clone(),
            tokens_tx,
        )?;

        Ok(Box::new(MlxcelTokenPuller::new(tokens_rx))
            as Box<dyn crate::backend::traits::TokenPullerDyn>)
    }

    fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        let mut msgs = self.messages.write();
        msgs.extend(new_messages);
        Ok(0)
    }
}

/// Map gen2's per-generation [`GenSpec`] over the backend-level [`Settings`]
/// sampling defaults into mlxcel's [`SamplingConfig`].
///
/// Precedence: a `GenSpec` field wins when `Some`; else the `Settings.sampling`
/// default; else mlxcel's own default (greedy-safe: temp `0.0` → argmax).
///
/// mlxcel supports every knob gen2 exposes **except** `xtc_probability`/
/// `xtc_threshold` (no mlxcel field) and `eot_bias` (needs the worker's EOS
/// id) — those are dropped with a loud `warn!` rather than silently, so the gap
/// is honest (doctrine: no silent skip). The agent's anti-loop **DRY** damping
/// and temperature-sampling therefore survive on the fast text path.
fn build_sampling_config(spec: &GenSpec, settings: &Settings) -> SamplingConfig {
    let s = &settings.sampling;
    let mut c = SamplingConfig {
        // Scalar sampling — GenSpec overrides the Settings default, else mlxcel's
        // greedy-safe fallbacks (temp 0.0, top_k 0/off, top_p 1.0/off, min_p 0.0).
        temperature: spec.temperature.or(s.temperature).unwrap_or(0.0),
        top_k: spec.top_k.or(s.top_k).unwrap_or(0),
        top_p: spec.top_p.or(s.top_p).unwrap_or(1.0),
        min_p: spec.min_p.or(s.min_p).unwrap_or(0.0),
        seed: spec.seed.or_else(|| s.seed.map(u64::from)),
        // History penalties — gen2 carries these on Settings (not GenSpec).
        repetition_penalty: s.penalty_repeat.unwrap_or(1.0),
        frequency_penalty: s.penalty_freq.unwrap_or(0.0),
        presence_penalty: s.penalty_present.unwrap_or(0.0),
        ..SamplingConfig::default()
    };

    // DRY anti-loop damping — GenSpec (CoreAgentInference sets these to stop a
    // small model repeat-looping). Keep mlxcel's default base/allowed_length
    // (1.75 / 2) unless overridden.
    if let Some(m) = spec.dry_multiplier {
        c.dry_multiplier = m;
    }
    if let Some(b) = spec.dry_base {
        c.dry_base = b;
    }
    if let Some(n) = spec.dry_allowed_length {
        c.dry_allowed_length = n;
    }
    if let Some(last_n) = s.penalty_last_n {
        c.dry_penalty_last_n = last_n.max(0) as usize;
    }

    // Honest gaps: mlxcel has no XTC field and eot_bias needs the worker EOS id.
    // Warn loudly if a caller actually requested them (never a silent drop).
    if spec.xtc_probability.is_some_and(|p| p > 0.0) || spec.xtc_threshold.is_some() {
        tracing::warn!(
            "mlxcel backend: XTC sampling (xtc_probability/xtc_threshold) is unsupported \
             and will be ignored"
        );
    }
    if spec.eot_bias.is_some_and(|b| b != 0.0) {
        tracing::warn!("mlxcel backend: eot_bias is unsupported and will be ignored");
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::SamplingSettings;

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.into()),
            },
            name: None,
        }
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: "assistant".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.into()),
            },
            name: None,
        }
    }

    /// A minimal but representative gemma-style chat template: emits the real
    /// `<start_of_turn>{role}` markers and a trailing generation prompt. This is
    /// the shape of the template that gemma-4-coder ships (the exact template
    /// the go/no-go model uses through backend-mlx).
    const GEMMA_TEMPLATE: &str = r#"{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m.role }}
{{ m.content }}<end_of_turn>
{% endfor %}{% if add_generation_prompt %}<start_of_turn>model
{% endif %}"#;

    /// S6b core: rendering through the REAL gemma template produces the model's
    /// actual turn markers (`<start_of_turn>user`) and NOT the naive
    /// `\nUser: ` / `\nAssistant: ` concat. This is the whole bug fix — the old
    /// `build_prompt` handed the model a malformed prompt.
    #[test]
    fn render_uses_real_template_not_naive_concat() {
        let tpl = ChatTemplate::new(
            GEMMA_TEMPLATE.to_string(),
            Some(TokenizerConfigToken::String("<bos>".into())),
            Some(TokenizerConfigToken::String("<eos>".into())),
        );
        let msgs = vec![user_msg("rename foo to bar"), assistant_msg("ok, doing it")];
        let out = render_with_template(&tpl, &msgs, Some("You are a coder."), ENABLE_THINKING);

        // Real template markers must be present.
        assert!(
            out.contains("<start_of_turn>user"),
            "must render the real gemma turn marker, got: {out:?}"
        );
        assert!(
            out.contains("<start_of_turn>system"),
            "system prompt must be injected as a system-role turn, got: {out:?}"
        );
        assert!(
            out.contains("You are a coder."),
            "system prompt text must survive, got: {out:?}"
        );
        assert!(
            out.contains("rename foo to bar"),
            "user content must survive, got: {out:?}"
        );
        // BOS must expand (decode-keep-specials contract).
        assert!(
            out.starts_with("<bos>"),
            "bos_token must expand, got: {out:?}"
        );

        // The naive markers must be GONE — this is the regression guard.
        assert!(
            !out.contains("\nUser: "),
            "naive `User: ` marker must NOT appear, got: {out:?}"
        );
        assert!(
            !out.contains("\nAssistant: "),
            "naive `Assistant: ` marker must NOT appear, got: {out:?}"
        );
    }

    /// The no-template fallback still produces the naive concat (so a model with
    /// no chat_template is not left prompt-less). The loud warn is emitted at the
    /// call site in `build_prompt`.
    #[test]
    fn naive_fallback_when_no_template() {
        let msgs = vec![user_msg("hello")];
        let out = render_naive(&msgs, Some("sys"));
        assert!(out.contains("\nUser: hello") || out.contains("User: hello"));
        assert!(out.ends_with("Assistant: "));
        assert!(out.starts_with("System: sys"));
    }

    /// LEAK GUARD compile-time proof: `ChatTemplate` is `Send + Sync`, so the
    /// session can hold ONE prebuilt template directly inside the shared
    /// `Arc<dyn BackendSession>` and reuse it across `pull()` calls — never
    /// rebuilding (and re-leaking) per generation.
    #[test]
    fn chat_template_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ChatTemplate>();
        assert_send_sync::<MlxcelSession>();
    }

    /// S4b: GenSpec overrides land in the right SamplingConfig fields, and
    /// DRY/temperature (what the agent relies on) survive the mapping.
    #[test]
    fn genspec_overrides_propagate() {
        let mut settings = Settings::default();
        settings.sampling.penalty_repeat = Some(1.3); // Settings-only knob
        let spec = GenSpec {
            temperature: Some(0.3),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            seed: Some(1234),
            dry_multiplier: Some(0.8),
            dry_base: Some(1.75),
            dry_allowed_length: Some(3),
            ..GenSpec::default()
        };
        let c = build_sampling_config(&spec, &settings);
        assert_eq!(c.temperature, 0.3, "temperature must propagate");
        assert_eq!(c.top_p, 0.9);
        assert_eq!(c.top_k, 40);
        assert_eq!(c.min_p, 0.05);
        assert_eq!(c.seed, Some(1234));
        assert_eq!(c.repetition_penalty, 1.3, "Settings penalty must propagate");
        assert_eq!(c.dry_multiplier, 0.8, "agent DRY damping must survive");
        assert_eq!(c.dry_allowed_length, 3);
    }

    /// Settings.sampling supplies the default; GenSpec `None` doesn't clobber it.
    #[test]
    fn settings_defaults_apply_when_genspec_unset() {
        let settings = Settings {
            sampling: SamplingSettings {
                temperature: Some(0.7),
                top_p: Some(0.95),
                ..SamplingSettings::default()
            },
            ..Settings::default()
        };
        let c = build_sampling_config(&GenSpec::default(), &settings);
        assert_eq!(c.temperature, 0.7);
        assert_eq!(c.top_p, 0.95);
    }

    /// Nothing set anywhere → greedy-safe (temp 0.0 → mlxcel argmax path).
    #[test]
    fn empty_is_greedy_safe() {
        let c = build_sampling_config(&GenSpec::default(), &Settings::default());
        assert_eq!(c.temperature, 0.0);
        assert_eq!(c.top_p, 1.0);
        assert_eq!(c.min_p, 0.0);
        assert_eq!(c.repetition_penalty, 1.0);
        assert_eq!(c.dry_multiplier, 0.0, "no DRY unless requested");
    }
}
