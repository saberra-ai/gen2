//! [`BackendSession`] for the mlxcel backend.
//!
//! A session pins the prompt (built from `SessionSpec.messages`) and a `stop`
//! flag. `pull()` builds the greedy `SamplingConfig`, opens a bounded token
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

        // Tracer-bullet: greedy unless a temperature was explicitly requested.
        // (Full sampling-config plumbing — top_p/min_p/penalties — is a later
        // slice; S2 proves the FAST greedy stream.)
        let mut sampling = match spec.temperature {
            Some(t) if t > 0.0 => SamplingConfig::with_temperature(t),
            _ => SamplingConfig::greedy(),
        };
        if let Some(seed) = spec.seed {
            sampling.seed = Some(seed);
        }

        let prompt = self.build_prompt();

        let (tokens_tx, tokens_rx) = sync_channel(TOKEN_CHANNEL_CAP);
        let _prompt_len = self.worker.start_generation_blocking(
            prompt,
            max_tokens,
            sampling,
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
