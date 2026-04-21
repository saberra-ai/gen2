//! Per-backend-shared streaming output filter.
//!
//! Routes freshly-sampled tokens through the [`StopMatcher`] with proper
//! hold-queue semantics (partial-match → hold, full-match → truncate +
//! stop, clean → release). Every backend's puller delegates here so
//! zero-garbage chat behaviour is the same across MLX, llama, ONNX, and
//! any future backend — the matcher, held queue, and pending-event
//! plumbing all live in one place.

use std::collections::VecDeque;

use super::stop_matcher::{StopMatcher, StopPattern, StopState};
use super::tool_calls::{ParserOutput, Protocol as ToolProtocol, ToolCallParser};
use crate::gen2::engine::ExecError;
use crate::gen2::generation::{Token, TokenEvent};

/// Filter + buffer that sits between a backend's sampled-token emission
/// and the downstream consumer. Owns the stop matcher, the held-token
/// queue, and a pending-event queue.
pub struct OutputFilter {
    stop: StopMatcher,
    /// Tokens whose text is provisionally buffered while the stop
    /// matcher reports a partial-suffix match. Released as Token events
    /// on Clean; truncated-and-dropped on Full; dropped entirely on
    /// `finalize` (terminal stop from some other signal).
    held: VecDeque<(u32, String)>,
    /// Streaming tool-call parser — chunks of clean text (released from
    /// the stop-matcher hold queue) flow through here and are split
    /// into plain Text spans and structured ToolCall events.
    tool_parser: Option<ToolCallParser>,
    /// Events ready to be returned to the iterator caller. `pop` drains
    /// this.
    pending: VecDeque<Result<TokenEvent, ExecError>>,
    done: bool,
}

impl OutputFilter {
    pub fn new(patterns: Vec<StopPattern>) -> Self {
        Self {
            stop: StopMatcher::new(patterns),
            held: VecDeque::new(),
            tool_parser: Some(ToolCallParser::new(ToolProtocol::Auto)),
            pending: VecDeque::new(),
            done: false,
        }
    }

    pub fn with_matcher(matcher: StopMatcher) -> Self {
        Self {
            stop: matcher,
            held: VecDeque::new(),
            tool_parser: Some(ToolCallParser::new(ToolProtocol::Auto)),
            pending: VecDeque::new(),
            done: false,
        }
    }

    /// Disable tool-call parsing for this filter (e.g. backends that
    /// already do their own tool-call handling, or tests that want raw
    /// text events).
    pub fn without_tool_calls(mut self) -> Self {
        self.tool_parser = None;
        self
    }

    /// Override the tool-call protocol (defaults to `Auto`).
    pub fn with_tool_protocol(mut self, protocol: ToolProtocol) -> Self {
        self.tool_parser = Some(ToolCallParser::new(protocol));
        self
    }

    /// Emit a plain text chunk as a Token event, splitting tool-call
    /// markers into structured `TokenEvent::ToolCall` events where the
    /// tool-call parser recognises them.
    ///
    /// Invariant: the Token's `id` field is the id of the LAST token
    /// whose text contributed to the chunk. A single chunk may span
    /// multiple original tokens (when text accumulates across holds)
    /// and a single original token's text may be split into multiple
    /// chunks (Text + ToolCall + Text). This mapping is lossy in the
    /// sense that id-per-char isn't preserved — consumers that need
    /// exact token attribution should disable tool-call parsing.
    fn emit_text_chunk(&mut self, id: u32, text: String) {
        let Some(parser) = self.tool_parser.as_mut() else {
            self.pending.push_back(Ok(TokenEvent::Token(Token {
                id,
                text,
                logprob: None,
            })));
            return;
        };
        for out in parser.push(&text) {
            match out {
                ParserOutput::Text(s) => {
                    if !s.is_empty() {
                        self.pending.push_back(Ok(TokenEvent::Token(Token {
                            id,
                            text: s,
                            logprob: None,
                        })));
                    }
                }
                ParserOutput::ToolCall(tc) => {
                    self.pending.push_back(Ok(TokenEvent::ToolCall(tc)));
                }
            }
        }
    }

    /// Feed a sampled token + its decoded text through the filter.
    ///
    /// Returns:
    ///   - `true`  ⇒ at least one event is now ready in `pending` (Token
    ///     release, or Eos on Full), OR `done == true`. Caller should
    ///     drain via `pop()`.
    ///   - `false` ⇒ the token was held (partial-match). Caller should
    ///     sample another token and push it.
    ///
    /// Matches llama.cpp's `server_slot` stream behaviour exactly:
    /// partial matches suppress emission until the next token resolves.
    pub fn push_token(&mut self, token_id: u32, text: String) -> bool {
        if self.done {
            // Defensive: refuse to buffer after a terminal stop. Drop.
            return false;
        }
        if self.stop.is_empty() {
            // Fast path: no stop patterns — text goes straight to the
            // tool-call parser (if any) and then to pending.
            self.emit_text_chunk(token_id, text);
            return true;
        }
        let state = self.stop.push(&text);
        self.held.push_back((token_id, text));
        match state {
            StopState::Clean => {
                let held_snapshot: Vec<(u32, String)> = self.held.drain(..).collect();
                for (tid, t) in held_snapshot {
                    self.emit_text_chunk(tid, t);
                }
                self.stop.reset();
                true
            }
            StopState::Partial { .. } => false,
            StopState::Full { emit_at, .. } => {
                // Emit held text up to `emit_at` (= pattern start +
                // keep_prefix). Straddling tokens are truncated at the
                // last char-boundary that stays within the safe prefix.
                let mut cum = 0usize;
                let held_snapshot: Vec<(u32, String)> = self.held.drain(..).collect();
                for (tid, t) in held_snapshot {
                    if cum >= emit_at {
                        break;
                    }
                    let remain = emit_at - cum;
                    if t.len() <= remain {
                        self.emit_text_chunk(tid, t.clone());
                        cum += t.len();
                    } else {
                        let mut cut = remain;
                        while cut > 0 && !t.is_char_boundary(cut) {
                            cut -= 1;
                        }
                        if cut > 0 {
                            self.emit_text_chunk(tid, t[..cut].to_string());
                        }
                        break;
                    }
                }
                // Flush tool-call parser — any buffered partial tag
                // gets released as literal text.
                if let Some(parser) = self.tool_parser.as_mut() {
                    for out in parser.flush() {
                        match out {
                            ParserOutput::Text(s) => {
                                if !s.is_empty() {
                                    self.pending.push_back(Ok(TokenEvent::Token(Token {
                                        id: 0,
                                        text: s,
                                        logprob: None,
                                    })));
                                }
                            }
                            ParserOutput::ToolCall(tc) => {
                                self.pending.push_back(Ok(TokenEvent::ToolCall(tc)));
                            }
                        }
                    }
                }
                self.pending.push_back(Ok(TokenEvent::Eos));
                self.done = true;
                self.stop.reset();
                true
            }
        }
    }

    /// Terminal stop from outside the matcher (EOG token, explicit stop
    /// flag, max_tokens, loop detector). Drops the held queue (held
    /// tokens were partial stop-pattern matches; resolving differently
    /// means we can't safely emit them — usually they'd be the garbage
    /// the partial was pointing at). Flushes the tool-call parser so any
    /// buffered partial-tag tail is released as literal text. Pushes
    /// `extra` as the terminal event.
    pub fn finalize(&mut self, extra: TokenEvent) {
        self.held.clear();
        self.stop.reset();
        if let Some(parser) = self.tool_parser.as_mut() {
            for out in parser.flush() {
                match out {
                    ParserOutput::Text(s) => {
                        if !s.is_empty() {
                            self.pending.push_back(Ok(TokenEvent::Token(Token {
                                id: 0,
                                text: s,
                                logprob: None,
                            })));
                        }
                    }
                    ParserOutput::ToolCall(tc) => {
                        self.pending.push_back(Ok(TokenEvent::ToolCall(tc)));
                    }
                }
            }
        }
        self.pending.push_back(Ok(extra));
        self.done = true;
    }

    /// Pop the next ready event. Returns None if nothing's queued yet.
    pub fn pop(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        self.pending.pop_front()
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// `true` when the filter has no patterns configured — callers can
    /// use this to skip the push/pop dance when token emission is raw.
    pub fn is_passthrough(&self) -> bool {
        self.stop.is_empty()
    }

    /// Push a non-Token event directly into pending (e.g. Paused). Does
    /// NOT flush held, so partial-match state is preserved across
    /// pause/resume.
    pub fn push_event(&mut self, ev: TokenEvent) {
        self.pending.push_back(Ok(ev));
    }

    /// Push an error into pending. Sets done — further pushes are dropped.
    pub fn push_err(&mut self, err: ExecError) {
        self.pending.push_back(Err(err));
        self.done = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(patterns: &[&str]) -> OutputFilter {
        OutputFilter::new(
            patterns
                .iter()
                .map(|s| StopPattern::new(*s))
                .collect(),
        )
    }

    #[test]
    fn clean_text_passes_through() {
        let mut f = mk(&["\nuser\n"]);
        assert!(f.push_token(1, "hello".into()));
        let ev = f.pop().unwrap().unwrap();
        match ev {
            TokenEvent::Token(t) => assert_eq!(t.text, "hello"),
            _ => panic!("expected Token"),
        }
    }

    #[test]
    fn partial_then_full_drops_garbage() {
        let mut f = mk(&["\nuser\n"]);
        assert!(f.push_token(1, "hello.".into())); // clean
        let _ = f.pop();
        assert!(!f.push_token(2, "\n".into())); // partial, hold
        assert!(!f.push_token(3, "us".into())); // partial, hold
        assert!(!f.push_token(4, "er".into())); // partial, hold
        assert!(f.push_token(5, "\n".into())); // FULL
        // Only Eos should come out — held tokens (\n, us, er, \n) are
        // all part of the stop pattern and dropped.
        let ev = f.pop().unwrap().unwrap();
        assert!(matches!(ev, TokenEvent::Eos));
        assert!(f.is_done());
    }

    #[test]
    fn partial_resolved_clean_releases_held() {
        let mut f = mk(&["\nuser\n"]);
        assert!(!f.push_token(1, "\n".into())); // partial
        assert!(!f.push_token(2, "us".into())); // partial
        // Next token invalidates the partial — release everything.
        assert!(f.push_token(3, " something".into()));
        let mut texts = vec![];
        while let Some(ev) = f.pop() {
            if let TokenEvent::Token(t) = ev.unwrap() {
                texts.push(t.text);
            }
        }
        assert_eq!(texts.join(""), "\nus something");
    }

    #[test]
    fn finalize_drops_held() {
        let mut f = mk(&["\nuser\n"]);
        assert!(!f.push_token(1, "\nuser".into())); // partial
        f.finalize(TokenEvent::Eos);
        let ev = f.pop().unwrap().unwrap();
        assert!(matches!(ev, TokenEvent::Eos));
        // Held "\nuser" never shows up.
        assert!(f.pop().is_none());
    }

    #[test]
    fn tool_call_extracted_from_clean_stream() {
        // No stop patterns, just tool-call parsing.
        let mut f = OutputFilter::new(vec![]);
        assert!(f.push_token(1, "Answer: ".into()));
        assert!(f.push_token(2, r#"<tool_call>{"name":"x","arguments":{}}</tool_call>"#.into()));
        assert!(f.push_token(3, " done".into()));
        let mut events: Vec<TokenEvent> = Vec::new();
        while let Some(ev) = f.pop() {
            events.push(ev.unwrap());
        }
        let tc = events.iter().find(|e| matches!(e, TokenEvent::ToolCall(_)));
        assert!(tc.is_some(), "expected a ToolCall in {events:?}");
        // There should be text events surrounding the tool call too.
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                TokenEvent::Token(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect();
        assert!(text.contains("Answer:"));
        assert!(text.contains("done"));
    }

    #[test]
    fn keep_prefix_preserves_punctuation_on_full() {
        // `.user` with keep_prefix=1 — the "." should be emitted, "user" dropped.
        let mut f = OutputFilter::new(vec![StopPattern::new(".user").keep(1)]);
        assert!(f.push_token(1, "quicksort".into())); // clean
        let _ = f.pop();
        assert!(!f.push_token(2, ".".into())); // partial
        assert!(f.push_token(3, "user".into())); // FULL at emit_at=1
        let mut out = String::new();
        while let Some(ev) = f.pop() {
            if let Ok(TokenEvent::Token(t)) = ev {
                out.push_str(&t.text);
            }
        }
        // The "." should survive, the "user" should not.
        assert_eq!(out, ".");
    }
}
