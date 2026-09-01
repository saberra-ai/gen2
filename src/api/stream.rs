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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Finish {
    /// The model emitted end-of-sequence, or hit the token budget.
    Eos,
    /// Stopped on request.
    Stopped,
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
        self.finish
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
    pub fn text_streaming(mut self, mut on_token: impl FnMut(&str)) -> Result<String> {
        let mut out = String::new();
        for event in &mut self {
            if let Event::Token(t) = event? {
                on_token(&t);
                out.push_str(&t);
            }
        }
        Ok(out)
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
    fn the_stream_is_fused_after_finishing() {
        let mut s = stream_of(vec![ControllerEvent::Eos]);
        assert!(s.next().is_none());
        assert!(s.next().is_none(), "a fused stream stays exhausted");
    }
}
