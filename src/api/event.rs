//! Semantic streaming (api_spec.md §15) and cancellation (§16).
//!
//! A backend streams tokenizer fragments; a harness wants to know what the
//! model is *doing* — thinking, answering, or asking for a tool. [`EventStream`]
//! turns the one into the other as it goes, so a terminal, a Tauri window, or
//! a websocket bridge never has to know that Qwen3 thinks inside `<think>` and
//! Gemma 4 inside `<|channel>thought`, or how a tool call is spelled.
//!
//! The same assembly produces the [`Response`] a blocking [`Turn::run`] returns,
//! so `stream()` and `run()` cannot disagree about what was said.
//!
//! [`Turn::run`]: super::turn::Turn::run

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::generation::{ChannelMarkers, ReplyStateMachine, StreamEmission};
use crate::types::message::{FunctionDefinition, Message, ToolCall as MessageToolCall};

use super::error::{Error, Result};
use super::model::Model;
use super::output::{
    AssistantMessage, FinishReason, GenerationStats, OutputPart, ToolCall, ToolCallId, Usage,
};
use super::response::Response;
use super::session::{MessageId, Session};
use super::stream::{Event as TokenEvent, Finish, TokenStream};

#[cfg(feature = "tokio")]
pub use super::async_turn::AsyncEventStream;

/// One thing the model did, as a stream sees it.
///
/// `#[non_exhaustive]`: match with a trailing `_ =>`. The §28.10 shape:
///
/// ```no_run
/// # let model = gen2::load("m.gguf")?;
/// # let mut session = gen2::Session::new();
/// use gen2::event::Event;
///
/// let mut stream = model.turn(&mut session).user("Inspect the parser").stream()?;
/// while let Some(event) = stream.next() {
///     match event? {
///         Event::TextDelta(text) => print!("{text}"),
///         Event::ReasoningDelta(text) => eprint!("{text}"),
///         Event::ToolCallStart { name, .. } => eprintln!("calling {name}"),
///         _ => {}
///     }
/// }
/// let response = stream.finish()?;
/// # Ok::<(), gen2::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Event {
    /// A fragment of the visible reply.
    TextDelta(String),
    /// A fragment of the reasoning channel.
    ReasoningDelta(String),
    /// The model began a tool call.
    ToolCallStart {
        /// Stable within the session; the id a result is pushed under.
        id: ToolCallId,
        /// The tool named.
        name: String,
    },
    /// A fragment of a tool call's arguments, as the model wrote them.
    ToolCallArgumentsDelta {
        /// The call being argued.
        id: ToolCallId,
        /// Raw argument text.
        delta: String,
    },
    /// A tool call is complete.
    ToolCallEnd {
        /// The call, arguments parsed.
        call: ToolCall,
    },
    /// Token accounting, once the backend reported it.
    Usage(Usage),
    /// The generation ended. Always the last event.
    Finished(FinishReason),
}

/// A turn in progress, as semantic [`Event`]s.
///
/// Built by [`Turn::stream`](super::turn::Turn::stream). Iterate it to the end,
/// then [`EventStream::finish`] for the [`Response`] — the same one
/// [`Turn::run`](super::turn::Turn::run) would have returned, with the
/// assistant message already appended to the session.
///
/// Dropping the stream early abandons the events; the generation runs on
/// until it ends on its own. Use [`EventStream::canceller`] to stop it.
pub struct EventStream<'a> {
    core: StreamCore,
    session: SessionSlot<'a>,
}

/// The session a stream writes its outcome to.
///
/// A turn borrows the caller's; a one-shot generation owns an ephemeral one
/// (boxed, so the slot is a pointer either way) and forgets it when the
/// stream is over.
pub(crate) enum SessionSlot<'a> {
    Borrowed(&'a mut Session),
    Owned(Box<Session>),
}

impl SessionSlot<'_> {
    pub(crate) fn get(&mut self) -> &mut Session {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(s) => s,
        }
    }

    pub(crate) fn id(&self) -> &super::session::SessionId {
        match self {
            Self::Borrowed(s) => s.id(),
            Self::Owned(s) => s.id(),
        }
    }

    pub(crate) fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }
}

/// Everything a stream is apart from the session it settles into.
///
/// `Send` and `'static`, so it can go to a blocking worker while the caller
/// keeps the `&mut Session`: the async surface pulls events here, and writes
/// the outcome to the session back on the caller's side.
pub(crate) struct StreamCore {
    inner: TokenStream,
    model: Model,
    assembler: Assembler,
    queue: VecDeque<Event>,
    /// The inner stream has ended, one way or another.
    done: bool,
    /// The outcome has been written to the session.
    settled: bool,
    /// Why the inner stream failed, when it did, for a `finish()` after the
    /// error was already yielded.
    failed: Option<String>,
    response: Option<Response>,
    cancel: Arc<AtomicBool>,
    /// Close the conversation once the turn is over: a per-turn prefix
    /// (system or tools override) must not be continued by the next turn.
    close_after: bool,
}

/// What a finished stream leaves behind for the session.
pub(crate) struct Settled {
    pub(crate) max_tokens: Option<usize>,
    pub(crate) close_after: bool,
}

impl StreamCore {
    pub(crate) fn new(
        inner: TokenStream,
        next_message_id: MessageId,
        model: Model,
        cancel: Arc<AtomicBool>,
        settled: Settled,
    ) -> Self {
        Self {
            inner,
            model,
            assembler: Assembler::new(CallIds::new(next_message_id), settled.max_tokens),
            queue: VecDeque::new(),
            done: false,
            settled: false,
            failed: None,
            response: None,
            cancel,
            close_after: settled.close_after,
        }
    }

    /// A stream that ended before it began: cancelled ahead of dispatch.
    pub(crate) fn cancelled_before_start(
        next_message_id: MessageId,
        model: Model,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        let (_, rx) = std::sync::mpsc::sync_channel(1);
        let mut core = Self::new(
            TokenStream::new(rx),
            next_message_id,
            model,
            cancel,
            Settled {
                max_tokens: None,
                close_after: false,
            },
        );
        core.done = true;
        core.settled = true;
        core.queue
            .push_back(Event::Finished(FinishReason::Cancelled));
        core.response = Some(Response::assembled(
            AssistantMessage::default(),
            FinishReason::Cancelled,
            GenerationStats::default(),
            None,
        ));
        core
    }

    /// The next event while the backend is still producing them.
    ///
    /// `None` once the inner stream has ended — by finishing or by failing —
    /// at which point [`StreamCore::complete`] writes the outcome and queues
    /// the closing events, unless it already did.
    pub(crate) fn pull(&mut self) -> Option<Result<Event>> {
        loop {
            if let Some(event) = self.queue.pop_front() {
                return Some(Ok(event));
            }
            if self.done {
                return None;
            }
            match self.inner.next() {
                Some(Ok(event)) => self.queue.extend(self.assembler.push(event)),
                Some(Err(e)) => {
                    self.done = true;
                    self.failed = Some(e.to_string());
                    return Some(Err(e));
                }
                None => {
                    self.done = true;
                    return None;
                }
            }
        }
    }

    /// Whether the outcome still has to be written to the session.
    pub(crate) fn needs_completion(&self) -> bool {
        self.done && !self.settled && self.failed.is_none()
    }

    /// The inner stream is over: record the outcome in the session and queue
    /// the closing events (usage, finish reason).
    pub(crate) fn complete(&mut self, session: &mut Session) {
        self.settled = true;
        let finish = self.inner.finish().unwrap_or_default();
        let engine = self.model.engine();
        super::chat::settle(engine, session, &self.inner);
        let cancelled = matches!(finish, Finish::Stopped);

        self.queue.extend(self.assembler.finish(finish));
        let (message, finish_reason, stats, shed) = self.assembler.take();
        session.note_shed(shed);

        // A cancelled turn that produced nothing has nothing to record.
        let worth_recording = !cancelled
            || !message.text().is_empty()
            || message.reasoning().is_some()
            || !message.tool_calls().is_empty();
        let message_id = worth_recording.then(|| session.push(record_of(&message)));

        // Stopping evicts the backend's runtime, and a per-turn prefix must
        // not be continued: either way the next turn rebuilds.
        if cancelled || self.close_after {
            session.opened = false;
        }
        self.response = Some(Response::assembled(
            message,
            finish_reason,
            stats,
            message_id,
        ));
    }

    /// A closing event queued by [`StreamCore::complete`], if any is left.
    /// Only the async bridge drains the queue this way; the sync iterator
    /// reads it directly.
    #[cfg(feature = "tokio")]
    pub(crate) fn pop_queued(&mut self) -> Option<Event> {
        self.queue.pop_front()
    }

    #[cfg(feature = "tokio")]
    pub(crate) fn has_queued(&self) -> bool {
        !self.queue.is_empty()
    }

    /// The outcome, once everything has been drained.
    pub(crate) fn take_response(&mut self) -> Result<Response> {
        match self.response.take() {
            Some(response) => Ok(response),
            None => Err(Error::Generation {
                code: "stream_failed".into(),
                message: self
                    .failed
                    .take()
                    .unwrap_or_else(|| "the stream ended without an outcome".into()),
            }),
        }
    }

    pub(crate) fn canceller(&self, chat_id: String) -> Canceller {
        Canceller {
            model: self.model.clone(),
            chat_id,
            flag: Arc::clone(&self.cancel),
        }
    }

    pub(crate) fn model(&self) -> &Model {
        &self.model
    }

    pub(crate) fn is_done(&self) -> bool {
        self.done
    }
}

impl<'a> EventStream<'a> {
    pub(crate) fn new(core: StreamCore, session: SessionSlot<'a>) -> Self {
        Self { core, session }
    }

    /// A handle that stops this turn from another thread.
    pub fn canceller(&self) -> Canceller {
        self.core.canceller(self.session.id().to_string())
    }

    /// Drain whatever is left and return the outcome.
    ///
    /// The same [`Response`] shape as [`Turn::run`](super::turn::Turn::run):
    /// text, reasoning, tool calls, finish reason, usage, and the id of the
    /// assistant message now in the session.
    pub fn finish(mut self) -> Result<Response> {
        for event in self.by_ref() {
            event?;
        }
        let response = self.core.take_response()?;
        Ok(if self.session.is_owned() {
            response.detached()
        } else {
            response
        })
    }
}

impl Iterator for EventStream<'_> {
    type Item = Result<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(event) = self.core.pull() {
                return Some(event);
            }
            if self.core.needs_completion() {
                self.core.complete(self.session.get());
                continue;
            }
            return None;
        }
    }
}

impl Drop for EventStream<'_> {
    fn drop(&mut self) {
        // An ephemeral session is over with its stream, however it ended; the
        // engine must not keep bookkeeping for it.
        if let SessionSlot::Owned(session) = &self.session {
            self.core.model().engine().forget(session);
        }
    }
}

impl std::iter::FusedIterator for EventStream<'_> {}

impl std::fmt::Debug for EventStream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream")
            .field("session", &self.session.id())
            .field("done", &self.core.is_done())
            .finish_non_exhaustive()
    }
}

/// Stops a turn. Cheap to clone and safe to send to another thread — the
/// thread reading the stream is busy reading it.
///
/// ```no_run
/// # let model = gen2::load("m.gguf")?;
/// # let mut session = gen2::Session::new();
/// let mut stream = model.turn(&mut session).user("Write a novel").stream()?;
/// let cancel = stream.canceller();
/// std::thread::spawn(move || cancel.cancel());
/// let response = stream.finish()?;
/// // `response.finish_reason()` is `Cancelled`; the partial text is kept.
/// # Ok::<(), gen2::Error>(())
/// ```
#[derive(Clone)]
pub struct Canceller {
    model: Model,
    chat_id: String,
    flag: Arc<AtomicBool>,
}

impl Canceller {
    pub(crate) fn new(model: Model, chat_id: String, flag: Arc<AtomicBool>) -> Self {
        Self {
            model,
            chat_id,
            flag,
        }
    }

    /// Stop the turn. The stream ends with [`Event::Finished`]
    /// carrying [`FinishReason::Cancelled`]; what was generated is kept.
    ///
    /// Before the turn has been dispatched, this makes it not run at all.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        // A controller that is gone ends the stream by itself.
        let _ = self.model.engine().stop(self.chat_id.clone());
    }

    /// Whether [`Canceller::cancel`] has been called.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for Canceller {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Canceller")
            .field("chat_id", &self.chat_id)
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

// ── Assembly ────────────────────────────────────────────────────────────────

/// Mints tool-call ids that are unique for the life of a session.
///
/// Built from the id the assistant message is about to get, which comes from
/// the session's event log and so survives a serialize/replay round trip;
/// two turns can never mint the same id because every recorded message moves
/// the counter.
#[derive(Debug, Clone)]
pub(crate) struct CallIds {
    message: MessageId,
    next: usize,
}

impl CallIds {
    pub(crate) fn new(message: MessageId) -> Self {
        Self { message, next: 0 }
    }

    fn mint(&mut self) -> ToolCallId {
        let id = ToolCallId(format!("call_{}_{}", self.message.as_u64(), self.next));
        self.next += 1;
        id
    }
}

/// Which channel a run of text belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Text,
    Reasoning,
}

/// Turns token events into semantic events and, at the end, a message.
///
/// The reasoning split is the crate's own [`ReplyStateMachine`] over every
/// scaffold it knows (`<think>`, Gemma 4's thought channel): a model that
/// uses none of them yields only text. Tool calls arrive from the backends
/// already parsed, so one call becomes `Start`, one `ArgumentsDelta` carrying
/// the whole argument text, and `End` — the sequence a backend streaming
/// arguments piecemeal would produce, so a consumer handles both alike.
pub(crate) struct Assembler {
    splitter: ReplyStateMachine,
    parts: Vec<OutputPart>,
    /// The channel the open run is in, if a run is open.
    open: Option<Channel>,
    /// Trailing whitespace of the open reasoning run, held back until real
    /// content follows it: the newline before `</think>` is the scaffold's.
    held: String,
    /// Whether a reasoning scaffold has been seen — the template puts
    /// newlines between it and the answer, which are not the answer.
    scaffolded: bool,
    calls: Vec<ToolCall>,
    call_ids: CallIds,
    stats: Option<crate::types::ExecutionStats>,
    shed: usize,
    max_tokens: Option<usize>,
    finish_reason: Option<FinishReason>,
}

impl Assembler {
    pub(crate) fn new(call_ids: CallIds, max_tokens: Option<usize>) -> Self {
        Self {
            splitter: ReplyStateMachine::new(all_markers()),
            parts: Vec::new(),
            open: None,
            held: String::new(),
            scaffolded: false,
            calls: Vec::new(),
            call_ids,
            stats: None,
            shed: 0,
            max_tokens,
            finish_reason: None,
        }
    }

    /// Absorb one token event; return the semantic events it amounts to.
    pub(crate) fn push(&mut self, event: TokenEvent) -> Vec<Event> {
        let mut out = Vec::new();
        match event {
            TokenEvent::Token(text) => {
                for emission in self.splitter.push_emit(&text) {
                    self.emit(emission, &mut out);
                }
            }
            TokenEvent::ToolCall(call) => {
                let id = match call.id {
                    Some(id) if !id.is_empty() => ToolCallId(id),
                    _ => self.call_ids.mint(),
                };
                out.push(Event::ToolCallStart {
                    id: id.clone(),
                    name: call.name.clone(),
                });
                out.push(Event::ToolCallArgumentsDelta {
                    id: id.clone(),
                    delta: call.arguments.clone(),
                });
                let call = ToolCall {
                    id,
                    name: call.name,
                    // Kept as a string when it was not JSON: a malformed
                    // call is still visible rather than dropped.
                    arguments: serde_json::from_str(&call.arguments)
                        .unwrap_or_else(|_| serde_json::Value::String(call.arguments)),
                };
                self.calls.push(call.clone());
                out.push(Event::ToolCallEnd { call });
            }
            TokenEvent::Stats(stats) => self.stats = Some(stats),
            TokenEvent::ContextTruncated { dropped } => self.shed += dropped,
            TokenEvent::ContextCompacted { compacted, .. } => self.shed += compacted,
            _ => {}
        }
        out
    }

    /// The stream ended: flush, account, and name the reason.
    pub(crate) fn finish(&mut self, finish: Finish) -> Vec<Event> {
        let mut out = Vec::new();
        for emission in self.splitter.flush_pending() {
            self.emit(emission, &mut out);
        }
        self.close_run();

        let stats = GenerationStats::from_reported(self.stats.clone());
        let usage = stats.usage();
        if stats.reported() {
            out.push(Event::Usage(usage));
        }

        let reason = if !self.calls.is_empty() && !matches!(finish, Finish::Stopped) {
            FinishReason::ToolCall
        } else {
            match finish {
                Finish::Eos => {
                    // The backends report end-of-sequence for both; the
                    // budget is the only way to tell them apart.
                    let at_budget = stats.reported()
                        && self
                            .max_tokens
                            .is_some_and(|max| usage.completion_tokens as usize >= max);
                    if at_budget {
                        FinishReason::Length
                    } else {
                        FinishReason::Stop
                    }
                }
                Finish::Stopped => FinishReason::Cancelled,
                other => FinishReason::Other(format!("{other:?}")),
            }
        };
        out.push(Event::Finished(reason.clone()));
        self.finish_reason = Some(reason);
        out
    }

    /// The outcome, once [`Assembler::finish`] has run.
    pub(crate) fn take(&mut self) -> (AssistantMessage, FinishReason, GenerationStats, usize) {
        let mut content = std::mem::take(&mut self.parts);
        let calls = std::mem::take(&mut self.calls);
        // An answer that is only tool calls has no text part; an empty text
        // part would suggest the model said something and said nothing.
        // A reply with no parts at all is one empty text part, so `text()`
        // is always the visible reply.
        if calls.is_empty() && content.is_empty() {
            content.push(OutputPart::Text(String::new()));
        }
        content.extend(calls.into_iter().map(OutputPart::ToolCall));
        (
            AssistantMessage { content },
            self.finish_reason
                .take()
                .unwrap_or(FinishReason::Other("unfinished".into())),
            GenerationStats::from_reported(self.stats.take()),
            std::mem::take(&mut self.shed),
        )
    }

    /// Route one channel-tagged slice into the parts and the event list.
    fn emit(&mut self, emission: StreamEmission, out: &mut Vec<Event>) {
        let (channel, text) = match emission {
            StreamEmission::Content(t) => (Channel::Text, t),
            StreamEmission::Reasoning(t) => (Channel::Reasoning, t),
        };
        if channel == Channel::Reasoning {
            self.scaffolded = true;
        }
        if self.open != Some(channel) {
            self.close_run();
        }
        let fresh = self.open.is_none();
        // A run's leading whitespace is the template's, not the model's:
        // Qwen3 opens with `<think>\n`, and puts `\n\n` between `</think>`
        // and the answer. Held back until the run has real content, so a
        // delta never carries what the message will not.
        let text: &str = if fresh {
            match channel {
                Channel::Reasoning => text.trim_start(),
                Channel::Text if self.scaffolded => text.trim_start_matches('\n'),
                Channel::Text => &text,
            }
        } else {
            &text
        };
        if text.is_empty() {
            return;
        }
        // Likewise the newline before `</think>`: whitespace at the end of
        // a reasoning run waits for what follows it.
        let mut chunk = std::mem::take(&mut self.held);
        chunk.push_str(text);
        if channel == Channel::Reasoning {
            let body = chunk.trim_end().len();
            self.held = chunk[body..].to_string();
            chunk.truncate(body);
            if chunk.is_empty() {
                self.open = Some(channel);
                return;
            }
        }
        self.open = Some(channel);
        match (channel, self.parts.last_mut()) {
            (Channel::Text, Some(OutputPart::Text(buf))) if !fresh => buf.push_str(&chunk),
            (Channel::Reasoning, Some(OutputPart::Reasoning(buf))) if !fresh => {
                buf.push_str(&chunk)
            }
            (Channel::Text, _) => self.parts.push(OutputPart::Text(chunk.clone())),
            (Channel::Reasoning, _) => self.parts.push(OutputPart::Reasoning(chunk.clone())),
        }
        out.push(match channel {
            Channel::Text => Event::TextDelta(chunk),
            Channel::Reasoning => Event::ReasoningDelta(chunk),
        });
    }

    /// End the open run: whitespace held at its end was the scaffold's.
    fn close_run(&mut self) {
        self.open = None;
        self.held.clear();
    }
}

/// Every reasoning scaffold the crate knows, so the split does not depend on
/// knowing the model's family up front. Longer markers first, as the
/// scanner wants.
fn all_markers() -> ChannelMarkers {
    let gemma = ChannelMarkers::gemma4();
    let qwen = ChannelMarkers::qwen3_deepseek();
    ChannelMarkers {
        open: gemma.open.into_iter().chain(qwen.open).collect(),
        close: gemma.close.into_iter().chain(qwen.close).collect(),
    }
}

/// The transcript message an assistant turn becomes.
///
/// A turn that asked for tools is recorded as its calls, under the ids the
/// response carries, so a result pushed with [`ToolCall::id`] answers a call
/// the transcript makes. Otherwise the visible text and the reasoning, kept
/// apart so a chat template can strip prior thinking as it prefers.
fn record_of(message: &AssistantMessage) -> Message {
    let calls = message.tool_calls();
    if calls.is_empty() {
        return Message::assistant_structured(message.text(), message.reasoning());
    }
    Message::assistant_tool_calls(
        calls
            .into_iter()
            .map(|c| MessageToolCall {
                id: c.id.to_string(),
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    description: None,
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                },
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::ToolCall as StreamToolCall;
    use crate::types::ExecutionStats;

    fn assembler() -> Assembler {
        Assembler::new(CallIds::new(MessageId::for_test(7)), None)
    }

    fn run(tokens: &[&str], finish: Finish) -> (Vec<Event>, AssistantMessage, FinishReason) {
        let mut a = assembler();
        let mut events = Vec::new();
        for t in tokens {
            events.extend(a.push(TokenEvent::Token((*t).to_string())));
        }
        events.extend(a.finish(finish));
        let (message, reason, _, _) = a.take();
        (events, message, reason)
    }

    #[test]
    fn plain_tokens_are_text_deltas_then_finished() {
        let (events, message, reason) = run(&["Hel", "lo"], Finish::Eos);
        assert_eq!(
            events,
            vec![
                Event::TextDelta("Hel".into()),
                Event::TextDelta("lo".into()),
                Event::Finished(FinishReason::Stop),
            ]
        );
        assert_eq!(message.text(), "Hello");
        assert_eq!(message.reasoning(), None);
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn a_think_scaffold_becomes_reasoning_deltas_and_a_reasoning_part() {
        let (events, message, _) = run(
            &[
                "<think>",
                "\n",
                "weigh",
                " it",
                "\n</think>",
                "\n\n",
                "Yes",
                ".",
            ],
            Finish::Eos,
        );
        assert_eq!(
            events,
            vec![
                Event::ReasoningDelta("weigh".into()),
                Event::ReasoningDelta(" it".into()),
                Event::TextDelta("Yes".into()),
                Event::TextDelta(".".into()),
                Event::Finished(FinishReason::Stop),
            ],
            "the scaffold's own newlines are not deltas"
        );
        assert_eq!(message.text(), "Yes.");
        assert_eq!(message.reasoning().as_deref(), Some("weigh it"));
        assert_eq!(
            message.content,
            vec![
                OutputPart::Reasoning("weigh it".into()),
                OutputPart::Text("Yes.".into())
            ]
        );
    }

    #[test]
    fn a_marker_split_across_tokens_is_still_a_marker() {
        let (events, message, _) = run(&["<th", "ink>", "hm", "</th", "ink>", "ok"], Finish::Eos);
        assert_eq!(message.reasoning().as_deref(), Some("hm"));
        assert_eq!(message.text(), "ok");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::TextDelta(t) if t.contains('<'))),
            "no marker fragment leaks as text: {events:?}"
        );
    }

    #[test]
    fn gemma_thought_channel_is_recognised_too() {
        let (_, message, _) = run(
            &["<|channel>thought\n", "hmm", "\n<channel|>", "yes"],
            Finish::Eos,
        );
        assert_eq!(message.reasoning().as_deref(), Some("hmm"));
        assert_eq!(message.text(), "yes");
    }

    #[test]
    fn an_unclosed_scaffold_is_all_reasoning_and_no_reply() {
        // The budget ran out mid-thought. Reporting the thought as the answer
        // would hand a harness the model's scratchpad as if it were the reply.
        let (_, message, _) = run(&["<think>", "still going"], Finish::Eos);
        assert_eq!(message.text(), "");
        assert_eq!(message.reasoning().as_deref(), Some("still going"));
        assert_eq!(
            message.content,
            vec![OutputPart::Reasoning("still going".into())]
        );
    }

    #[test]
    fn an_empty_scaffold_leaves_no_reasoning_and_no_reasoning_events() {
        // Qwen3 with thinking off emits `<think>\n\n</think>\n\n` before the reply.
        let (events, message, _) = run(&["<think>\n\n</think>\n\n", "fine"], Finish::Eos);
        assert_eq!(message.content, vec![OutputPart::Text("fine".into())]);
        assert_eq!(
            events,
            vec![
                Event::TextDelta("fine".into()),
                Event::Finished(FinishReason::Stop)
            ]
        );
    }

    #[test]
    fn an_empty_reply_is_one_empty_text_part() {
        let (_, message, _) = run(&[], Finish::Eos);
        assert_eq!(message.content, vec![OutputPart::Text(String::new())]);
        assert_eq!(message.text(), "");
    }

    #[test]
    fn a_complete_tool_call_is_start_arguments_end() {
        let mut a = assembler();
        let events = a.push(TokenEvent::ToolCall(StreamToolCall {
            id: None,
            name: "get_weather".into(),
            arguments: r#"{"city":"Paris"}"#.into(),
        }));
        let id = ToolCallId("call_7_0".into());
        assert_eq!(
            events,
            vec![
                Event::ToolCallStart {
                    id: id.clone(),
                    name: "get_weather".into()
                },
                Event::ToolCallArgumentsDelta {
                    id: id.clone(),
                    delta: r#"{"city":"Paris"}"#.into()
                },
                Event::ToolCallEnd {
                    call: ToolCall {
                        id,
                        name: "get_weather".into(),
                        arguments: serde_json::json!({"city": "Paris"}),
                    }
                },
            ]
        );
        let end = a.finish(Finish::Eos);
        assert_eq!(end.last(), Some(&Event::Finished(FinishReason::ToolCall)));
        let (message, reason, _, _) = a.take();
        assert_eq!(reason, FinishReason::ToolCall);
        assert_eq!(
            message.text(),
            "",
            "no text part is invented for a tool-only turn"
        );
        assert_eq!(message.tool_calls().len(), 1);
    }

    #[test]
    fn ids_are_minted_per_message_and_provider_ids_are_kept() {
        let mut a = assembler();
        a.push(TokenEvent::ToolCall(StreamToolCall {
            id: None,
            name: "a".into(),
            arguments: "{}".into(),
        }));
        a.push(TokenEvent::ToolCall(StreamToolCall {
            id: Some("abc".into()),
            name: "b".into(),
            arguments: "not json".into(),
        }));
        a.push(TokenEvent::ToolCall(StreamToolCall {
            id: None,
            name: "c".into(),
            arguments: "{}".into(),
        }));
        a.finish(Finish::Eos);
        let (message, _, _, _) = a.take();
        let calls = message.tool_calls();
        assert_eq!(calls[0].id().as_str(), "call_7_0");
        assert_eq!(calls[1].id().as_str(), "abc", "a provider id is kept");
        assert_eq!(calls[2].id().as_str(), "call_7_1");
        assert_eq!(
            calls[1].arguments(),
            &serde_json::Value::String("not json".into()),
            "malformed arguments survive as a string"
        );
    }

    #[test]
    fn a_stop_is_cancelled_with_the_partial_text_kept() {
        let (events, message, reason) = run(&["partial"], Finish::Stopped);
        assert_eq!(reason, FinishReason::Cancelled);
        assert_eq!(message.text(), "partial");
        assert_eq!(
            events.last(),
            Some(&Event::Finished(FinishReason::Cancelled))
        );
    }

    #[test]
    fn hitting_the_budget_is_length_and_usage_is_reported_first() {
        let mut a = Assembler::new(CallIds::new(MessageId::for_test(1)), Some(8));
        a.push(TokenEvent::Token("and then".into()));
        a.push(TokenEvent::Stats(ExecutionStats {
            prompt_tokens: 10,
            decode_tokens: 8,
            ..Default::default()
        }));
        let end = a.finish(Finish::Eos);
        assert_eq!(
            end,
            vec![
                Event::Usage(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 8
                }),
                Event::Finished(FinishReason::Length),
            ]
        );
        // Same stream under a larger budget: the model chose to stop.
        let mut a = Assembler::new(CallIds::new(MessageId::for_test(1)), Some(64));
        a.push(TokenEvent::Stats(ExecutionStats {
            decode_tokens: 8,
            ..Default::default()
        }));
        let (_, reason, _, _) = {
            a.finish(Finish::Eos);
            a.take()
        };
        assert_eq!(reason, FinishReason::Stop);
        // No token count means no grounds to claim the budget was hit.
        let (_, _, reason) = run(&["x"], Finish::Eos);
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn the_recorded_message_carries_the_calls_under_their_ids() {
        let mut a = assembler();
        a.push(TokenEvent::ToolCall(StreamToolCall {
            id: None,
            name: "read".into(),
            arguments: r#"{"path":"x"}"#.into(),
        }));
        a.finish(Finish::Eos);
        let (message, _, _, _) = a.take();
        let record = record_of(&message);
        let crate::types::message::MessageBody::Tool { tool_calls } = &record.body else {
            panic!("a tool turn is recorded as its calls");
        };
        assert_eq!(tool_calls[0].id, "call_7_0");
        assert_eq!(tool_calls[0].function.arguments["path"], "x");

        let plain = record_of(&AssistantMessage {
            content: vec![
                OutputPart::Reasoning("why".into()),
                OutputPart::Text("because".into()),
            ],
        });
        assert_eq!(plain.role, "assistant");
        assert_eq!(plain.text(), "because");
    }
}
