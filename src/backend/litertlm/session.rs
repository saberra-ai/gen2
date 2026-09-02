//! One conversation, with gen2 holding the transcript.
//!
//! LiteRT-LM's `Conversation` is stateful and keeps its own prefilled KV, so
//! this is a genuine two-sided arrangement: gen2 owns what the conversation
//! *is*, and LiteRT-LM owns what it has already computed. The rule that keeps
//! them honest is that only newly appended messages are ever sent. When the
//! caller edits or clears history, the controller rebuilds the session — which
//! discards this object and its conversation together — so LiteRT-LM can never
//! be left holding a turn gen2 has deleted.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::backend::common::stop_matcher::StopMatcher;
use crate::backend::common::tool_calls::Protocol;
use crate::backend::facade::SessionId;
use crate::backend::{BackendSession, TokenPullerDyn};
use crate::engine::{ExecError, Settings};
use crate::generation::GenSpec;
use crate::types::message::Message;

use super::convert;
use super::ffi::{OwnedConversation, OwnedEngine, OwnedOptionalArgs, StreamSink};
use super::puller::{LiteRtLmPuller, StreamHandle};

pub(super) struct LiteRtLmSession {
    id: SessionId,
    /// Declared before `engine`, so it is destroyed first. LiteRT-LM says
    /// "EngineAdvancedImpl destructed with 1 living sessions" when a
    /// conversation outlives the engine that owns it, and struct fields drop
    /// in declaration order.
    ///
    /// By `Arc` because a [`Canceller`] handed to a puller holds one too: the
    /// conversation a generation might still be cancelling cannot be freed
    /// while that handle exists.
    ///
    /// `None` until the first turn. LiteRT-LM fixes the sampler when a
    /// conversation is created and `start_session` does not know what the
    /// first turn will ask for, so opening one eagerly meant opening a second
    /// immediately — two conversations per turn, each with its own KV. On the
    /// CPU that was waste; on the GPU it was slow enough to look like a hang.
    /// Waiting until the sampler is known costs nothing and opens one.
    conversation: Mutex<Option<Arc<OwnedConversation>>>,
    /// Held so the conversation can be rebuilt when a turn needs context
    /// LiteRT-LM cannot be given incrementally, and so the engine outlives
    /// every conversation opened on it.
    engine: Arc<OwnedEngine>,
    settings: Settings,
    tools_json: Option<String>,
    /// Messages LiteRT-LM has already been given. Held so a turn sends only
    /// what is new — the whole point of a stateful conversation.
    delivered: Mutex<Vec<Message>>,
    /// Appended since the last generation, and not yet sent.
    pending: Mutex<Vec<Message>>,
    /// The reasoning-channel policy this session was opened with. Fixed at
    /// conversation-creation time by LiteRT-LM, so it travels with every
    /// rebuild.
    thinking: Option<bool>,
    /// The sampler the live conversation is actually configured with.
    ///
    /// LiteRT-LM attaches a sampler to the conversation, not to a request, so
    /// this is the only place a per-turn `.temperature()` or `.seed()` can be
    /// noticed. Without it, every turn ran under whatever the conversation was
    /// opened with and the caller's request was silently ignored.
    sampler: Mutex<Option<convert::Sampler>>,
    /// The context the conversation was configured with, or 0 when the caller
    /// did not set one.
    ctx_size: u32,
    stopped: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
}

impl std::fmt::Debug for LiteRtLmSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiteRtLmSession")
            .field("id", &self.id)
            .field("delivered", &self.delivered.lock().len())
            .field("pending", &self.pending.lock().len())
            .finish_non_exhaustive()
    }
}

impl LiteRtLmSession {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        id: SessionId,
        engine: Arc<OwnedEngine>,
        settings: Settings,
        tools_json: Option<String>,
        thinking: Option<bool>,
        sampler: Option<convert::Sampler>,
        delivered: Vec<Message>,
        pending: Vec<Message>,
        ctx_size: u32,
    ) -> Self {
        Self {
            id,
            conversation: Mutex::new(None),
            engine,
            settings,
            tools_json,
            thinking,
            sampler: Mutex::new(sampler),
            delivered: Mutex::new(delivered),
            pending: Mutex::new(pending),
            ctx_size,
            stopped: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Rebuild the conversation so it holds exactly `history`.
    ///
    /// LiteRT-LM takes one message per send and has no way to add context
    /// without generating from it, so a turn that appends several messages at
    /// once cannot be delivered incrementally. Rebuilding is the honest
    /// answer: it costs the prefill, and it keeps the model's view identical
    /// to gen2's, which is the invariant the whole crate rests on.
    fn rebuild(
        &self,
        history: &[Message],
        spec: &GenSpec,
    ) -> Result<Arc<OwnedConversation>, ExecError> {
        let wanted = convert::sampler_of(&self.settings, spec);
        let fresh = super::engine::open_conversation(
            &self.engine,
            self.tools_json.as_deref(),
            self.thinking,
            wanted.clone(),
            spec.grammar.is_some(),
            &self.settings,
            history,
        )?;
        #[allow(clippy::arc_with_non_send_sync)]
        let fresh = Arc::new(fresh);
        // The old conversation is dropped here, before the caller can start a
        // generation on the new one — so a rebuild never has two alive at once.
        *self.conversation.lock() = Some(Arc::clone(&fresh));
        *self.sampler.lock() = wanted;
        Ok(fresh)
    }

    /// The conversation to run this turn on, opened or reopened as needed.
    ///
    /// LiteRT-LM fixes the sampler when a conversation is created, so the only
    /// way to honour a per-turn `.temperature()`, `.top_k()`, `.top_p()` or
    /// `.seed()` is to open a new one. That costs the prefill, which is why it
    /// happens only when the sampler actually differs — a conversation whose
    /// sampling never changes is opened once and kept for the session.
    fn conversation_for(
        &self,
        spec: &GenSpec,
        history: &[Message],
    ) -> Result<Arc<OwnedConversation>, ExecError> {
        let wanted = convert::sampler_of(&self.settings, spec);
        let current = self.conversation.lock().clone();
        match current {
            Some(conversation) if *self.sampler.lock() == wanted => Ok(conversation),
            _ => self.rebuild(history, spec),
        }
    }
}

impl BackendSession for LiteRtLmSession {
    fn id(&self) -> SessionId {
        self.id
    }

    fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }

    fn resume(&self) {
        self.paused.store(false, Ordering::Release);
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
    }

    fn pull(&self, spec: GenSpec) -> Result<Box<dyn TokenPullerDyn>, ExecError> {
        // A new generation starts un-stopped: the flag belongs to the turn
        // that set it, and carrying it forward would make one cancellation
        // silently cancel the next turn too.
        self.stopped.store(false, Ordering::Release);
        self.paused.store(false, Ordering::Release);

        // Refuse sampling this backend cannot honour rather than quietly
        // producing different output than the caller asked for.
        if let Some(reason) = convert::unsupported_sampling(&self.settings, &spec) {
            return Err(ExecError::FeatureUnsupported(reason));
        }

        let mut options = OwnedOptionalArgs::new(Arc::clone(self.engine.runtime()))?;
        if let Some(max) = spec.max_tokens.or(self.settings.stopping.max_tokens) {
            options.set_max_output_tokens(max.min(i32::MAX as usize) as i32);
        }
        if let Some(grammar) = &spec.grammar {
            let (kind, body) = convert::constraint_of(grammar)?;
            options.set_constraint(kind, &body)?;
        }
        if let Some((repeat, freq, present)) = convert::penalties_of(&self.settings, &spec) {
            options.set_penalties(
                repeat,
                freq,
                present,
                convert::penalty_window(&self.settings, &spec),
            )?;
        }

        let turn = std::mem::take(&mut *self.pending.lock());
        let Some((last, earlier)) = turn.split_last() else {
            return Err(ExecError::InvalidArg(
                "no new message to generate from — the conversation already \
                 holds everything gen2 has sent",
            ));
        };
        let mut history = self.delivered.lock().clone();
        history.extend_from_slice(earlier);
        let conversation = if earlier.is_empty() {
            // The usual case: one new message, and the live conversation is
            // reused unless this turn asks for different sampling.
            self.conversation_for(&spec, &history)?
        } else {
            // Everything before the final message is context rather than a
            // prompt, and LiteRT-LM has no way to take context without
            // generating from it, so the conversation is reopened holding it.
            self.rebuild(&history, &spec)?
        };

        // The message says which call it answers whenever the model gave one
        // an id; the backward search is only the fallback for results recorded
        // without one.
        let call_id = last.tool_call_id.clone().unwrap_or_else(|| {
            self.delivered
                .lock()
                .iter()
                .rev()
                .find_map(|m| match &m.body {
                    crate::types::message::MessageBody::Tool { tool_calls } => {
                        tool_calls.last().map(|c| c.id.clone())
                    }
                    _ => None,
                })
                .unwrap_or_default()
        });
        let message_json = convert::message_json(last, &call_id).to_string();

        let runtime = Arc::clone(self.engine.runtime());
        let (sink, chunks, finished) = StreamSink::new(runtime);
        conversation.send_stream(&message_json, &options, sink)?;

        // Delivered only once the runtime has taken it. A message recorded as
        // delivered before the send succeeded would be skipped on the retry.
        self.delivered.lock().extend(turn.iter().cloned());

        let canceller = super::ffi::Canceller::of(&conversation);
        let stream = StreamHandle {
            chunks,
            finished,
            cancel: Box::new(move || canceller.cancel()),
        };

        Ok(Box::new(LiteRtLmPuller::new(
            stream,
            Arc::clone(&self.stopped),
            Arc::clone(&self.paused),
            Protocol::Auto,
            // Stopwords are gen2's to enforce here: LiteRT-LM's C API exposes
            // no stop-sequence setting, so a backend that neither applied nor
            // refused them would let a caller's `.stop()` do nothing.
            StopMatcher::from_strings(self.settings.stopping.stopwords.clone()),
        )))
    }

    fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        self.pending.lock().extend(new_messages);
        // Nothing is dropped here. LiteRT-LM manages its own context, and
        // reporting a truncation gen2 did not perform would be a lie the
        // caller acts on.
        Ok(0)
    }

    fn ctx_size(&self) -> u32 {
        self.ctx_size
    }
}
