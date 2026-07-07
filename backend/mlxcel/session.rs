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

use crate::gen2::engine::{ExecError, Settings};
use crate::gen2::generation::GenSpec;
use crate::types::message::Message;

use super::puller::MlxcelTokenPuller;
use super::worker::ModelWorker;

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
    ) -> Self {
        Self {
            id,
            worker,
            settings,
            messages: parking_lot::RwLock::new(messages),
            stopped: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Build the greedy prompt string from the conversation.
    ///
    /// Tracer-bullet (S2): a simple role-tagged join, NOT the model's real chat
    /// template. Full Jinja chat-template rendering is a later slice; this is
    /// enough to drive a real greedy stream for the capability proof.
    fn build_prompt(&self) -> String {
        let msgs = self.messages.read();
        let mut out = String::new();
        if let Some(sys) = self.settings.prompt.system_prompt.as_deref()
            && !sys.trim().is_empty()
        {
            out.push_str("System: ");
            out.push_str(sys.trim());
            out.push_str("\n\n");
        }
        for m in msgs.iter() {
            let text = match &m.body {
                crate::types::message::MessageBody::Content { content } => {
                    content.as_visible_text()
                }
                // Tool-call messages have no visible text in this tracer-bullet
                // prompt (structured tool syntax is a later slice).
                crate::types::message::MessageBody::Tool { .. } => String::new(),
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
}

impl crate::gen2::backend::traits::BackendSession for MlxcelSession {
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
    ) -> Result<Box<dyn crate::gen2::backend::traits::TokenPullerDyn>, ExecError> {
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
            as Box<dyn crate::gen2::backend::traits::TokenPullerDyn>)
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
    use crate::gen2::engine::SamplingSettings;

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
