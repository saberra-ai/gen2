//! LiteRT-LM's streamed reply, as gen2 [`TokenEvent`]s.
//!
//! The contract is the one every backend keeps: exactly one terminal event,
//! nothing after it, and no text lost on the way. LiteRT-LM streams plain
//! text, so tool calls are recovered from it by gen2's own cross-backend
//! parser — the same one the llama and MLX backends use, which is why a model
//! that calls tools behaves identically whichever backend is under it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use crate::backend::TokenPullerDyn;
use crate::backend::common::stop_matcher::{StopMatcher, StopState};
use crate::backend::common::tool_calls::{ParserOutput, Protocol, ToolCallParser};
use crate::engine::ExecError;
use crate::generation::{Token, TokenEvent};

use super::convert::{Part, decode_chunk};
use super::ffi::Chunk;

/// How long to wait on the channel before checking the stop flag again.
///
/// The wait exists only so a cancellation is noticed while the model is
/// thinking. It is not a per-token cost: a chunk that has already arrived
/// returns immediately.
const POLL: Duration = Duration::from_millis(50);

/// Everything the puller needs from a running stream.
///
/// Stated as data and closures rather than as FFI handles, so the mapping from
/// chunks to [`TokenEvent`]s — which is where this backend's contract actually
/// lives — can be tested without a runtime or a 600MB model in the room.
pub(super) struct StreamHandle {
    /// Chunks, as LiteRT-LM's callback thread produces them.
    pub chunks: Receiver<Chunk>,
    /// Set by the callback once the stream is over.
    pub finished: Arc<AtomicBool>,
    /// Ask the runtime to stop generating.
    pub cancel: Box<dyn Fn()>,
}

pub(super) struct LiteRtLmPuller {
    chunks: Receiver<Chunk>,
    finished: Arc<AtomicBool>,
    cancel: Box<dyn Fn()>,
    stopped: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    parser: ToolCallParser,
    /// The caller's stop sequences, applied here because LiteRT-LM's C API has
    /// no setting for them.
    stops: StopMatcher,
    /// How much of the stop matcher's accumulated buffer has already gone
    /// downstream. The matcher reports offsets into the whole reply, so this
    /// is what turns those into the next slice to emit.
    emitted: usize,
    /// Emitted one at a time, in the order the model asked for them.
    ready: Vec<TokenEvent>,
    /// The stream ended; the terminal event is owed once `ready` is drained.
    ending: Option<TokenEvent>,
    /// The terminal event has gone out. Nothing follows it.
    done: bool,
    /// Whether the callback's reference was already released by a final chunk.
    saw_final: bool,
}

impl LiteRtLmPuller {
    pub(super) fn new(
        stream: StreamHandle,
        stopped: Arc<AtomicBool>,
        paused: Arc<AtomicBool>,
        protocol: Protocol,
        stops: StopMatcher,
    ) -> Self {
        Self {
            chunks: stream.chunks,
            finished: stream.finished,
            cancel: stream.cancel,
            stopped,
            paused,
            parser: ToolCallParser::new(protocol),
            stops,
            emitted: 0,
            ready: Vec::new(),
            ending: None,
            done: false,
            saw_final: false,
        }
    }

    /// Turn parser output into events, in order.
    fn absorb(&mut self, outputs: Vec<ParserOutput>) {
        for output in outputs {
            match output {
                ParserOutput::Text(text) if text.is_empty() => {}
                ParserOutput::Text(text) => self.ready.push(TokenEvent::Token(Token {
                    // LiteRT-LM streams detokenised text, not ids. Reporting a
                    // fabricated id would be worse than admitting there isn't
                    // one; consumers read `text`.
                    id: 0,
                    text,
                    logprob: None,
                })),
                ParserOutput::ToolCall(call) => self.ready.push(TokenEvent::ToolCall(call)),
            }
        }
    }

    /// Text safe to emit, and whether a stop sequence ended the turn.
    ///
    /// The matcher accumulates the whole reply and answers in offsets into it,
    /// because a stop sequence can straddle a chunk boundary. Skipped entirely
    /// when the caller set no stop sequences, so an ordinary reply is not
    /// buffered in full for nothing.
    fn apply_stops(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        let (emit_to, hit) = match self.stops.push(text) {
            StopState::Clean => (self.stops.buffer().len(), false),
            StopState::Partial { hold } => (self.stops.buffer().len() - hold, false),
            StopState::Full { emit_at, .. } => (emit_at, true),
        };
        let buffer = self.stops.buffer();
        // A stop pattern can sit inside text already emitted, which would make
        // `emit_to` run backwards. Clamped rather than trusted: slicing a
        // string on a bad range is a panic, and a panic here would take the
        // controller down mid-reply.
        let from = self.emitted.min(buffer.len());
        let to = emit_to.clamp(from, buffer.len());
        let out = if buffer.is_char_boundary(from) && buffer.is_char_boundary(to) {
            buffer[from..to].to_string()
        } else {
            String::new()
        };
        self.emitted = to;
        (out, hit)
    }

    /// End the stream after whatever is already queued.
    fn end_with(&mut self, event: TokenEvent) {
        // Anything the parser was still holding is real output; dropping it
        // would silently truncate the last few characters of every reply.
        let tail = self.parser.flush();
        self.absorb(tail);
        self.ending = Some(event);
    }
}

impl TokenPullerDyn for LiteRtLmPuller {
    fn next_event(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        loop {
            if self.done {
                return None;
            }
            if !self.ready.is_empty() {
                return Some(Ok(self.ready.remove(0)));
            }
            if let Some(event) = self.ending.take() {
                self.done = true;
                return Some(Ok(event));
            }

            // A caller who asked to stop gets one cancel and then waits for
            // the runtime to wind down, so the terminal event still reflects
            // what actually happened rather than being asserted from here.
            if self.stopped.load(Ordering::Acquire) && !self.saw_final {
                (self.cancel)();
            }
            if self.paused.load(Ordering::Acquire) {
                return Some(Ok(TokenEvent::Paused));
            }

            match self.chunks.recv_timeout(POLL) {
                Ok(Chunk::Text(raw)) => {
                    // Each chunk is a JSON message, not a token. What comes
                    // out of it is either text — which still goes through the
                    // tool-call parser, since a model may write a call as
                    // markup inside its prose — or a call the runtime already
                    // parsed, which needs no second opinion.
                    for part in decode_chunk(&raw) {
                        match part {
                            Part::Text(text) => {
                                // Stop sequences are checked before the
                                // tool-call parser so a stop that lands
                                // mid-marker still ends the turn, and the text
                                // the matcher holds back is never emitted.
                                let (visible, hit) = self.apply_stops(&text);
                                if !visible.is_empty() {
                                    let outputs = self.parser.push(&visible);
                                    self.absorb(outputs);
                                }
                                if hit {
                                    // The caller asked for generation to end
                                    // here, so the runtime is told to stop and
                                    // the turn ends as a normal completion —
                                    // a stop sequence is a finish, not a
                                    // cancellation.
                                    (self.cancel)();
                                    self.end_with(TokenEvent::Eos);
                                    break;
                                }
                            }
                            Part::Call(call) => self.ready.push(TokenEvent::ToolCall(call)),
                        }
                    }
                }
                Ok(Chunk::Failed(message)) => {
                    self.saw_final = true;
                    self.done = true;
                    return Some(Err(ExecError::Generation(format!(
                        "LiteRT-LM generation failed: {message}"
                    ))));
                }
                Ok(Chunk::Done) => {
                    self.saw_final = true;
                    // A stop the caller asked for is a stop, even though the
                    // runtime ends the stream the same way it ends a natural
                    // one. Reporting `Eos` here would tell the caller their
                    // cancellation did not take.
                    let terminal = if self.stopped.load(Ordering::Acquire) {
                        TokenEvent::Stopped
                    } else {
                        TokenEvent::Eos
                    };
                    self.end_with(terminal);
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Still generating. Loop round so a stop or pause set in
                    // the meantime is seen.
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Every sender is gone without a final chunk, so the
                    // callback will never run again. Ending is the only honest
                    // answer — waiting would hang the controller — but *how*
                    // it ends matters: reporting a stop would dress a runtime
                    // failure up as a cancellation the caller made, and a
                    // truncated answer would be shown as a complete one.
                    self.saw_final = true;
                    self.done = true;
                    if self.stopped.load(Ordering::Acquire) {
                        return Some(Ok(TokenEvent::Stopped));
                    }
                    return Some(Err(ExecError::Generation(
                        "LiteRT-LM stopped streaming without ending the turn; \
                         the reply is incomplete"
                            .to_string(),
                    )));
                }
            }
        }
    }
}

impl Drop for LiteRtLmPuller {
    /// Stop a generation nobody is reading any more.
    ///
    /// Dropping the puller is already safe: the callback owns the only
    /// reference to its sink and releases it on the final chunk, and a send
    /// into the closed channel just fails. What is left is waste — the model
    /// would keep decoding into nothing — so the runtime is told to stop.
    /// Deliberately without waiting: a `Drop` that blocks the controller for
    /// seconds is a worse problem than a few tokens of wasted compute.
    fn drop(&mut self) {
        if self.saw_final || self.finished.load(Ordering::Acquire) {
            return;
        }
        (self.cancel)();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives the puller with chunks, without a runtime in the room.
    ///
    /// The mapping from chunks to events is where this backend's contract
    /// lives, and it is worth proving without a 600MB model.
    fn drain(chunks: Vec<Chunk>, stopped: bool) -> Vec<Result<TokenEvent, ExecError>> {
        drain_with_stops(chunks, stopped, &[])
    }

    fn drain_with_stops(
        chunks: Vec<Chunk>,
        stopped: bool,
        stops: &[&str],
    ) -> Vec<Result<TokenEvent, ExecError>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let ends = chunks
            .iter()
            .any(|c| matches!(c, Chunk::Done | Chunk::Failed(_)));
        for chunk in chunks {
            tx.send(chunk).expect("the receiver is alive");
        }
        // Dropped when the script ends, so a stream with no final chunk
        // disconnects exactly as a dead callback thread would.
        drop(tx);

        let stream = StreamHandle {
            chunks: rx,
            finished: Arc::new(AtomicBool::new(ends)),
            cancel: Box::new(|| {}),
        };
        let mut puller = LiteRtLmPuller::new(
            stream,
            Arc::new(AtomicBool::new(stopped)),
            Arc::new(AtomicBool::new(false)),
            Protocol::Auto,
            StopMatcher::from_strings(stops.iter().map(|s| s.to_string()).collect()),
        );
        let mut out = Vec::new();
        while let Some(event) = puller.next_event() {
            out.push(event);
            assert!(out.len() <= 64, "the puller never terminated");
        }
        out
    }

    /// A chunk in the shape LiteRT-LM actually sends: a whole JSON message
    /// carrying one token.
    fn message(text: &str) -> String {
        serde_json::json!({
            "role": "assistant",
            "content": [{ "type": "text", "text": text }],
        })
        .to_string()
    }

    fn texts(events: &[Result<TokenEvent, ExecError>]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                Ok(TokenEvent::Token(t)) => Some(t.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_stream_ends_with_exactly_one_terminal_event_and_nothing_after_it() {
        let events = drain(
            vec![
                Chunk::Text(message("Hello")),
                Chunk::Text(message(" world")),
                Chunk::Done,
            ],
            false,
        );

        assert_eq!(texts(&events), "Hello world");
        let terminals = events
            .iter()
            .filter(|e| matches!(e, Ok(TokenEvent::Eos | TokenEvent::Stopped)))
            .count();
        assert_eq!(
            terminals, 1,
            "a backend that emits two terminal events makes every consumer's \
             state machine wrong: {events:?}"
        );
        assert!(
            matches!(events.last(), Some(Ok(TokenEvent::Eos))),
            "the terminal event must be last, got {events:?}"
        );
    }

    #[test]
    fn no_text_is_lost_when_the_final_chunk_carries_some() {
        // The trap every streaming backend falls into: the chunk that says
        // "done" also carries the last of the reply, and a puller that treats
        // the flag as "stop now" drops it.
        let events = drain(vec![Chunk::Text(message("answer")), Chunk::Done], false);
        assert_eq!(texts(&events), "answer");
    }

    #[test]
    fn a_cancelled_generation_reports_a_stop_and_not_an_end_of_stream() {
        // The runtime winds a cancelled stream down exactly like a finished
        // one. Reporting `Eos` would tell the caller their stop did not take,
        // and a UI would show a truncated reply as a complete one.
        let events = drain(vec![Chunk::Text(message("par")), Chunk::Done], true);
        assert!(
            matches!(events.last(), Some(Ok(TokenEvent::Stopped))),
            "expected a stop, got {events:?}"
        );
    }

    #[test]
    fn a_runtime_error_ends_the_stream_as_an_error() {
        let events = drain(
            vec![
                Chunk::Text(message("half")),
                Chunk::Failed("out of memory".into()),
            ],
            false,
        );
        let last = events.last().expect("the stream should have ended");
        assert!(
            matches!(last, Err(ExecError::Generation(m)) if m.contains("out of memory")),
            "the runtime's own message should survive, got {last:?}"
        );
    }

    #[test]
    fn a_tool_call_in_the_text_stream_comes_out_as_a_tool_call() {
        // LiteRT-LM streams tool calls as text. If they stayed text, gen2's
        // agent loop would never fire and the model's call would be shown to
        // the user as markup.
        let events = drain(
            vec![
                Chunk::Text(message("<tool_call>{\"name\": \"get_weather\", ")),
                Chunk::Text(message("\"arguments\": {\"city\": \"Paris\"}}</tool_call>")),
                Chunk::Done,
            ],
            false,
        );
        let call = events.iter().find_map(|e| match e {
            Ok(TokenEvent::ToolCall(c)) => Some(c),
            _ => None,
        });
        let call = call.unwrap_or_else(|| panic!("no tool call was parsed out of {events:?}"));
        assert_eq!(call.name, "get_weather");
        assert!(call.arguments.contains("Paris"));
    }

    /// A stream that dies must not look like a cancellation.
    ///
    /// The runtime crashing, the callback vanishing, the protocol breaking —
    /// all reach the puller as a closed channel with no final chunk. Reporting
    /// `Stopped` dressed every one of those up as a stop the caller made, so a
    /// UI would show a truncated answer as a finished one and nothing would
    /// retry.
    #[test]
    fn a_stream_that_dies_without_a_final_chunk_is_an_error_not_a_cancellation() {
        let events = drain(vec![Chunk::Text(message("cut off"))], false);
        assert_eq!(
            texts(&events),
            "cut off",
            "what did arrive should still reach the caller"
        );
        let last = events.last().expect("the stream should have ended");
        assert!(
            matches!(last, Err(ExecError::Generation(m)) if m.contains("without ending the turn")),
            "expected a generation failure, got {last:?}"
        );
    }

    /// But when the caller *did* stop, the same disconnect is a stop.
    #[test]
    fn a_stream_the_caller_stopped_still_reports_a_stop_when_it_dies() {
        let events = drain(vec![Chunk::Text(message("par"))], true);
        assert!(
            matches!(events.last(), Some(Ok(TokenEvent::Stopped))),
            "expected a stop, got {events:?}"
        );
    }

    /// Stop sequences are gen2's to enforce here.
    ///
    /// LiteRT-LM's C API has no stop-sequence setting, so a backend that
    /// neither applied nor refused them would let `.stop("END")` do nothing at
    /// all — the caller would get the whole reply and believe it had been cut.
    #[test]
    fn a_stop_sequence_ends_the_turn_and_is_not_itself_emitted() {
        let events = drain_with_stops(
            vec![
                Chunk::Text(message("keep this")),
                Chunk::Text(message("END")),
                Chunk::Text(message(" and not this")),
                Chunk::Done,
            ],
            false,
            &["END"],
        );
        assert_eq!(
            texts(&events),
            "keep this",
            "text at or after the stop sequence must not reach the caller: {events:?}"
        );
        assert!(
            matches!(events.last(), Some(Ok(TokenEvent::Eos))),
            "a stop sequence is a normal finish, not a cancellation: {events:?}"
        );
    }

    /// A stop sequence split across chunks still fires.
    ///
    /// The reason the matcher buffers at all: LiteRT-LM streams a token at a
    /// time, so a two-character stop word routinely arrives in two pieces.
    #[test]
    fn a_stop_sequence_split_across_chunks_is_still_caught() {
        let events = drain_with_stops(
            vec![
                Chunk::Text(message("answer")),
                Chunk::Text(message("<|")),
                Chunk::Text(message("end|>")),
                Chunk::Text(message(" trailing")),
                Chunk::Done,
            ],
            false,
            &["<|end|>"],
        );
        assert_eq!(
            texts(&events),
            "answer",
            "a stop sequence straddling two chunks was missed: {events:?}"
        );
    }

    /// And a reply with no stop sequences is passed through untouched.
    #[test]
    fn text_is_not_held_back_when_the_caller_set_no_stop_sequences() {
        let events = drain(
            vec![
                Chunk::Text(message("all")),
                Chunk::Text(message(" of it")),
                Chunk::Done,
            ],
            false,
        );
        assert_eq!(texts(&events), "all of it");
    }
}
