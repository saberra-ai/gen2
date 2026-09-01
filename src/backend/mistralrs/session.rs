//! One conversation, held by gen2.
//!
//! The transcript lives here rather than in mistral.rs. gen2 owns what the
//! model has seen — that is the invariant the whole crate rests on — so
//! mistral.rs's own session handling, agent loop and tool execution are all
//! unused. Every turn sends the conversation gen2 says it is.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mistralrs::RequestBuilder;
use mistralrs::blocking::BlockingModel;
use parking_lot::RwLock;

use crate::backend::facade::SessionId;
use crate::backend::{BackendSession, TokenPullerDyn};
use crate::engine::{ExecError, Settings};
use crate::generation::GenSpec;
use crate::types::message::{Message, ToolSpec};

use super::convert;
use super::puller::MistralRsPuller;

pub(super) struct MistralRsSession {
    id: SessionId,
    model: Arc<BlockingModel>,
    settings: Settings,
    messages: RwLock<Vec<Message>>,
    tools: Vec<ToolSpec>,
    stopped: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
}

impl std::fmt::Debug for MistralRsSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralRsSession")
            .field("id", &self.id)
            .field("messages", &self.messages.read().len())
            .finish_non_exhaustive()
    }
}

impl MistralRsSession {
    pub(super) fn new(
        id: SessionId,
        model: Arc<BlockingModel>,
        settings: Settings,
        messages: Vec<Message>,
        tools: Vec<ToolSpec>,
    ) -> Self {
        Self {
            id,
            model,
            settings,
            messages: RwLock::new(messages),
            tools,
            stopped: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl BackendSession for MistralRsSession {
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

        // Refuse a seed this backend cannot honour, rather than generating
        // unreproducibly and letting the caller believe otherwise.
        if convert::unsupported_seed(&self.settings, &spec) {
            return Err(ExecError::FeatureUnsupported(
                "seed: mistral.rs exposes no per-request seed; use greedy() for \
                 deterministic output, or a backend that does",
            ));
        }

        let mut request = RequestBuilder::new();
        request = convert::messages_into(request, &self.messages.read());
        if !self.tools.is_empty() {
            request = request.set_tools(convert::tools_into(&self.tools));
        }
        if let Some(grammar) = &spec.grammar {
            request = request.set_constraint(convert::constraint_of(grammar));
        }
        request = convert::sampling_into(request, &self.settings, &spec);

        let stream = self
            .model
            .stream_chat_request(request)
            .map_err(|e| ExecError::Other(anyhow::anyhow!("mistral.rs request failed: {e}")))?;

        Ok(Box::new(MistralRsPuller::new(
            stream,
            Arc::clone(&self.stopped),
            Arc::clone(&self.paused),
        )))
    }

    fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        self.messages.write().extend(new_messages);
        // Nothing is dropped here. mistral.rs manages its own context, and
        // reporting a truncation gen2 did not perform would be a lie the
        // caller acts on.
        Ok(0)
    }
}
