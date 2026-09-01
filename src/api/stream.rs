//! What comes back from a generation.

use std::sync::mpsc::Receiver;

use crate::controller::ControllerEvent;
use crate::generation::{MediaBoundary, ToolCall};
use crate::types::ExecutionStats;

use super::error::{Error, Result};

/// One thing that happened during a generation.
///
/// Terminal events (`Eos`, `Stopped`) end the stream rather than being yielded,
/// and a failure arrives as `Err` from the iterator — so a loop over
/// [`TokenStream`] sees only the events it can act on.
///
/// `#[non_exhaustive]`: new event kinds are added over time, so match with a
/// trailing `_ =>`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Event {
    /// A fragment of generated text. Concatenate these to get the reply.
    Token(String),
    /// The model asked to call a tool.
    ToolCall(ToolCall),
    /// A media segment boundary in a multimodal reply.
    MediaBoundary(MediaBoundary),
    /// Timing and token counts for the finished generation.
    Stats(ExecutionStats),
    /// The conversation didn't fit the context window; `dropped` messages were
    /// removed to make room.
    ContextTruncated { dropped: usize },
    /// The conversation didn't fit; `compacted` messages were replaced by a
    /// summary instead of being dropped outright.
    ContextCompacted { compacted: usize, strategy: String },
}

/// Why a generation ended.
///
/// Not `Copy`: the agent variants carry which budget or which tool, and that
/// detail is what makes the reason actionable.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Finish {
    /// The model emitted end-of-sequence, or hit the token budget.
    #[default]
    Eos,
    /// Stopped on request.
    Stopped,
    /// The tool loop hit its depth limit with the model still asking for more
    /// tools. See [`Chat::tool_depth`](super::Chat::tool_depth).
    ToolDepthReached,
    /// An agent ran out of a budget — which one is in the payload.
    OutOfBudget(Budget),
    /// An agent stopped because it was making no progress.
    GaveUp(Struggle),
}

/// Which limit an agent reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Budget {
    /// Rounds of generate-and-call.
    Steps,
    /// Total tokens generated across the run.
    Tokens,
    /// Wall-clock.
    Deadline,
}

/// Why an agent was judged to be going nowhere.
///
/// Depth alone doesn't catch these: a model calling the same tool with the same
/// arguments seven times has burned its budget without doing anything, and
/// that is the characteristic small-model failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Struggle {
    /// The same call, with the same arguments, repeatedly.
    RepeatingCall { tool: String, times: usize },
    /// One tool failing over and over.
    ToolKeepsFailing { tool: String, times: usize },
}

/// A generation in progress.
///
/// Iterates [`Event`]s until the generation ends. The stream is fused: once it
/// has finished or failed it keeps returning `None`.
///
/// Dropping it early is fine — that abandons the events, though it does not by
/// itself stop the generation. Use [`Engine::stop`](super::Engine::stop) with
/// the chat id for that.
pub struct TokenStream {
    rx: Receiver<ControllerEvent>,
    finish: Option<Finish>,
    done: bool,
}

impl TokenStream {
    pub(crate) fn new(rx: Receiver<ControllerEvent>) -> Self {
        Self {
            rx,
            finish: None,
            done: false,
        }
    }

    /// How the generation ended, once it has. `None` while still running, or
    /// if it failed.
    pub fn finish(&self) -> Option<Finish> {
        self.finish.clone()
    }

    /// Run to completion, concatenating every text fragment.
    ///
    /// The common case: you want the reply, not the play-by-play. Non-text
    /// events are discarded; a failure anywhere aborts with that error.
    pub fn text(mut self) -> Result<String> {
        let mut out = String::new();
        for event in &mut self {
            if let Event::Token(t) = event? {
                out.push_str(&t);
            }
        }
        Ok(out)
    }

    /// Run to completion, invoking `on_token` for each text fragment as it
    /// arrives, and return the full text.
    pub fn text_streaming(self, on_token: impl FnMut(&str)) -> Result<String> {
        Ok(self.complete_streaming(on_token)?.text)
    }

    /// Run to completion and return everything that happened: the text, how it
    /// ended, timing, any tool calls, and whether context had to be dropped.
    ///
    /// Use this instead of matching the raw event stream when you want the
    /// outcome rather than the play-by-play — it does the accumulating that
    /// every caller would otherwise write by hand.
    pub fn complete(self) -> Result<Completion> {
        self.complete_streaming(|_| {})
    }

    /// [`Self::complete`], with `on_token` called for each fragment as it
    /// arrives.
    pub fn complete_streaming(mut self, mut on_token: impl FnMut(&str)) -> Result<Completion> {
        let mut done = Completion::default();
        for event in &mut self {
            match event? {
                Event::Token(t) => {
                    on_token(&t);
                    done.text.push_str(&t);
                }
                Event::ToolCall(c) => done.tool_calls.push(c),
                Event::Stats(s) => done.stats = Some(s),
                Event::ContextTruncated { dropped } => done.dropped += dropped,
                Event::ContextCompacted { compacted, .. } => done.compacted += compacted,
                _ => {}
            }
        }
        // Set last: `finish` is only known once the stream has drained, and a
        // stream that drained without erroring always has one.
        done.finish = self.finish.clone().unwrap_or(Finish::Eos);
        Ok(done)
    }

    /// The text fragments alone, as an iterator.
    ///
    /// For when you want to stream tokens but not match on event kinds:
    ///
    /// ```no_run
    /// # use gen2::Engine;
    /// # let engine = Engine::load("m.gguf")?;
    /// for token in engine.infer("hi").tokens()? {
    ///     print!("{}", token?);
    /// }
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn tokens(self) -> Tokens {
        Tokens { inner: self }
    }
}

/// Everything a finished generation produced.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Completion {
    /// The reply text, with every fragment concatenated.
    pub text: String,
    /// How the generation ended.
    pub finish: Finish,
    /// Timing and token counts, when the backend reported them.
    pub stats: Option<ExecutionStats>,
    /// Tool calls the model asked for.
    pub tool_calls: Vec<ToolCall>,
    /// Old messages dropped outright to fit the context window.
    pub dropped: usize,
    /// Old messages replaced by a summary to fit the context window.
    pub compacted: usize,
    /// How many rounds of tool calls ran before the model answered.
    ///
    /// `0` for an ordinary turn. Only the tool loop raises it — see
    /// [`Chat::on_tool`](super::Chat::on_tool).
    pub tool_rounds: usize,
}

impl Completion {
    /// Whether context had to be shed — either dropped or compacted — to fit
    /// this turn.
    pub fn context_was_shed(&self) -> bool {
        self.dropped > 0 || self.compacted > 0
    }
}

/// The text fragments of a generation. See [`TokenStream::tokens`].
pub struct Tokens {
    inner: TokenStream,
}

impl Tokens {
    /// How the generation ended, once it has.
    pub fn finish(&self) -> Option<Finish> {
        self.inner.finish()
    }
}

impl Iterator for Tokens {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Skip non-text events rather than ending: a tool call or a stats
        // frame mid-stream must not look like the end of the text.
        loop {
            match self.inner.next()? {
                Ok(Event::Token(t)) => return Some(Ok(t)),
                Ok(_) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

impl std::iter::FusedIterator for Tokens {}

impl std::fmt::Debug for Tokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokens").finish_non_exhaustive()
    }
}

impl Iterator for TokenStream {
    type Item = Result<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // The sender closing without a terminal event means the controller died
        // mid-generation. That is a failure, not a normal end — a caller that
        // saw `None` here would read a truncated reply as complete.
        let Ok(event) = self.rx.recv() else {
            self.done = true;
            return match self.finish {
                Some(_) => None,
                None => Some(Err(Error::ControllerGone)),
            };
        };

        Some(Ok(match event {
            ControllerEvent::Token(t) => Event::Token(t),
            ControllerEvent::ToolCall(c) => Event::ToolCall(c),
            ControllerEvent::MediaBoundary(b) => Event::MediaBoundary(b),
            ControllerEvent::FinalStats(s) => Event::Stats(s),
            ControllerEvent::ContextTruncated(dropped) => Event::ContextTruncated { dropped },
            ControllerEvent::ContextCompacted {
                compacted,
                strategy,
            } => Event::ContextCompacted {
                compacted,
                strategy,
            },
            ControllerEvent::Error { code, message } => {
                self.done = true;
                return Some(Err(Error::Generation { code, message }));
            }
            ControllerEvent::Eos => {
                self.finish = Some(Finish::Eos);
                self.done = true;
                return None;
            }
            ControllerEvent::Stopped => {
                self.finish = Some(Finish::Stopped);
                self.done = true;
                return None;
            }
        }))
    }
}

impl std::iter::FusedIterator for TokenStream {}

impl std::fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenStream")
            .field("finish", &self.finish)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    fn stream_of(events: Vec<ControllerEvent>) -> TokenStream {
        let (tx, rx) = sync_channel(events.len().max(1));
        for e in events {
            tx.send(e).unwrap();
        }
        drop(tx);
        TokenStream::new(rx)
    }

    #[test]
    fn text_concatenates_fragments_and_stops_at_eos() {
        let s = stream_of(vec![
            ControllerEvent::Token("Hel".into()),
            ControllerEvent::Token("lo".into()),
            ControllerEvent::Eos,
            ControllerEvent::Token(" ignored".into()),
        ]);
        assert_eq!(s.text().unwrap(), "Hello");
    }

    #[test]
    fn a_generation_error_surfaces_as_err_not_a_short_read() {
        let s = stream_of(vec![
            ControllerEvent::Token("partial".into()),
            ControllerEvent::Error {
                code: "context_overflow".into(),
                message: "too long".into(),
            },
        ]);
        let err = s.text().unwrap_err();
        assert_eq!(err.code(), Some("context_overflow"));
    }

    #[test]
    fn a_dropped_controller_mid_stream_is_an_error() {
        // No terminal event: the sender just went away. Returning the partial
        // text as if complete is the bug this guards.
        let s = stream_of(vec![ControllerEvent::Token("partial".into())]);
        assert!(matches!(s.text(), Err(Error::ControllerGone)));
    }

    #[test]
    fn finish_records_how_the_generation_ended() {
        let mut s = stream_of(vec![
            ControllerEvent::Token("x".into()),
            ControllerEvent::Stopped,
        ]);
        assert_eq!(s.finish(), None, "not known until the stream is drained");
        while s.next().is_some() {}
        assert_eq!(s.finish(), Some(Finish::Stopped));
    }

    #[test]
    fn complete_gathers_the_whole_outcome_in_one_value() {
        let s = stream_of(vec![
            ControllerEvent::Token("Hel".into()),
            ControllerEvent::ContextTruncated(3),
            ControllerEvent::Token("lo".into()),
            ControllerEvent::FinalStats(ExecutionStats {
                decode_tokens: 2,
                ..Default::default()
            }),
            ControllerEvent::Eos,
        ]);
        let done = s.complete().unwrap();
        assert_eq!(done.text, "Hello");
        assert_eq!(done.finish, Finish::Eos);
        assert_eq!(done.stats.as_ref().unwrap().decode_tokens, 2);
        assert_eq!(done.dropped, 3);
        assert!(done.context_was_shed());
    }

    #[test]
    fn complete_records_a_stop_as_the_finish_reason() {
        let s = stream_of(vec![
            ControllerEvent::Token("partial".into()),
            ControllerEvent::Stopped,
        ]);
        let done = s.complete().unwrap();
        assert_eq!(done.finish, Finish::Stopped);
        assert_eq!(done.text, "partial");
        assert!(!done.context_was_shed());
    }

    #[test]
    fn complete_streaming_sees_every_fragment_in_order() {
        let s = stream_of(vec![
            ControllerEvent::Token("a".into()),
            ControllerEvent::Token("b".into()),
            ControllerEvent::Eos,
        ]);
        let mut seen = Vec::new();
        let done = s.complete_streaming(|t| seen.push(t.to_string())).unwrap();
        assert_eq!(seen, ["a", "b"]);
        assert_eq!(done.text, "ab");
    }

    #[test]
    fn tokens_skips_non_text_events_instead_of_ending() {
        // A stats frame between two fragments must not truncate the text.
        let s = stream_of(vec![
            ControllerEvent::Token("a".into()),
            ControllerEvent::FinalStats(ExecutionStats::default()),
            ControllerEvent::Token("b".into()),
            ControllerEvent::Eos,
        ]);
        let text: Result<String> = s.tokens().collect();
        assert_eq!(text.unwrap(), "ab");
    }

    #[test]
    fn tokens_propagates_a_generation_error() {
        let s = stream_of(vec![
            ControllerEvent::Token("a".into()),
            ControllerEvent::Error {
                code: "boom".into(),
                message: "bad".into(),
            },
        ]);
        let text: Result<String> = s.tokens().collect();
        assert_eq!(text.unwrap_err().code(), Some("boom"));
    }

    #[test]
    fn the_stream_is_fused_after_finishing() {
        let mut s = stream_of(vec![ControllerEvent::Eos]);
        assert!(s.next().is_none());
        assert!(s.next().is_none(), "a fused stream stays exhausted");
    }
}
