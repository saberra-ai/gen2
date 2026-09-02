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
    conversation: Mutex<Arc<OwnedConversation>>,
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
        conversation: OwnedConversation,
        settings: Settings,
        tools_json: Option<String>,
        delivered: Vec<Message>,
        pending: Vec<Message>,
        ctx_size: u32,
    ) -> Self {
        // Shared with the `Canceller` a generation hands to its puller, all on
        // the one thread the controller confines this backend to.
        #[allow(clippy::arc_with_non_send_sync)]
        let conversation = Mutex::new(Arc::new(conversation));
        Self {
            id,
            conversation,
            engine,
            settings,
            tools_json,
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
    fn rebuild(&self, history: &[Message], spec: &GenSpec) -> Result<(), ExecError> {
        let fresh = super::engine::open_conversation(
            &self.engine,
            &self.settings,
            spec,
            self.tools_json.as_deref(),
            history,
        )?;
        #[allow(clippy::arc_with_non_send_sync)]
        let fresh = Arc::new(fresh);
        *self.conversation.lock() = fresh;
        Ok(())
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
            options.set_penalties(repeat, freq, present)?;
        }

        let turn = std::mem::take(&mut *self.pending.lock());
        let Some((last, earlier)) = turn.split_last() else {
            return Err(ExecError::InvalidArg(
                "no new message to generate from — the conversation already \
                 holds everything gen2 has sent",
            ));
        };
        // Everything before the final message is context rather than a prompt,
        // and LiteRT-LM has no way to take context without generating from it.
        // Rare in practice — a turn is normally one message — so the rebuild
        // buys correctness at a cost almost nothing pays.
        if !earlier.is_empty() {
            let mut history = self.delivered.lock().clone();
            history.extend_from_slice(earlier);
            self.rebuild(&history, &spec)?;
        }

        let call_id = self
            .delivered
            .lock()
            .iter()
            .rev()
            .find_map(|m| match &m.body {
                crate::types::message::MessageBody::Tool { tool_calls } => {
                    tool_calls.last().map(|c| c.id.clone())
                }
                _ => None,
            })
            .unwrap_or_default();
        let message_json = convert::message_json(last, &call_id).to_string();

        let runtime = Arc::clone(self.engine.runtime());
        let (sink, chunks, finished) = StreamSink::new(runtime);
        {
            let conversation = self.conversation.lock();
            conversation.send_stream(&message_json, &options, sink)?;
        }

        // Delivered only once the runtime has taken it. A message recorded as
        // delivered before the send succeeded would be skipped on the retry.
        self.delivered.lock().extend(turn.iter().cloned());

        let canceller = super::ffi::Canceller::of(&self.conversation.lock());
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
