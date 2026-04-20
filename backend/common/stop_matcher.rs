//! Multi-character stop-sequence detection, mirroring llama.cpp's
//! `server_find_stopping_strings` logic.
//!
//! Token-level stop ids only fire when the model emits the exact EOT token.
//! Under aggressive quantization (Gemma 4 26B 4bit) the EOT's logit is
//! sometimes beaten by a plain-text token like `\n` or the word `user`, so
//! the model ends a turn by literally typing `\nuser\n` in prose instead of
//! emitting `<turn|>`. A character-level matcher catches these.
//!
//! Behaviour per llama.cpp server.cpp:
//! - **Full match**: pattern appears in the accumulated buffer → stop,
//!   trim output at the start of the match.
//! - **Partial suffix match**: the buffer ends with a prefix of some
//!   pattern → hold back that suffix from streaming; if a later token
//!   completes the pattern we stop cleanly with no leak, if not we
//!   release the held bytes.

/// A stop pattern plus how much of its leading text to *keep* in the
/// emitted output on a full match. Keeping a prefix is useful for patterns
/// anchored on punctuation (`.user`, `!system`) where the punctuation is
/// legitimate content but the role word that follows isn't.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopPattern {
    pub text: String,
    pub keep_prefix: usize,
}

impl StopPattern {
    pub fn new<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            keep_prefix: 0,
        }
    }

    pub fn keep(mut self, bytes: usize) -> Self {
        self.keep_prefix = bytes;
        self
    }

    fn len(&self) -> usize {
        self.text.len()
    }
}

/// Outcome of feeding a chunk into [`StopMatcher::push`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopState {
    /// No match, safe to emit all buffered text downstream.
    Clean,
    /// Buffer ends with a prefix of some pattern. `hold` trailing bytes
    /// should be withheld until the next `push` resolves them (into Full,
    /// back to Clean, or a larger hold).
    Partial { hold: usize },
    /// Pattern was found. `emit_at` is the byte offset in the buffer up
    /// to which text should be emitted (exclusive); everything from there
    /// to the end of the matched pattern is dropped. `emit_at = at +
    /// pattern.keep_prefix`.
    Full {
        at: usize,
        pattern_len: usize,
        emit_at: usize,
    },
}

/// Accumulates decoded output text; checks for stop patterns after each push.
pub struct StopMatcher {
    patterns: Vec<StopPattern>,
    buf: String,
}

impl StopMatcher {
    pub fn new(patterns: Vec<StopPattern>) -> Self {
        Self {
            patterns,
            buf: String::new(),
        }
    }

    pub fn from_strings(patterns: Vec<String>) -> Self {
        Self::new(patterns.into_iter().map(StopPattern::new).collect())
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Append `text` to the accumulator and classify the new state.
    ///
    /// Full-match detection runs on the entire buffer (not just the new
    /// chunk) because a pattern may straddle a token boundary across
    /// multiple pushes. Partial-match detection looks at the buffer's
    /// suffix: the longest prefix of any pattern that matches the tail.
    pub fn push(&mut self, text: &str) -> StopState {
        self.buf.push_str(text);

        // Full match — pick the EARLIEST occurrence across all patterns.
        // Llama.cpp trims at the earliest start so later output beyond a
        // stop pattern is always dropped.
        let mut earliest: Option<(usize, &StopPattern)> = None;
        for p in &self.patterns {
            if p.text.is_empty() {
                continue;
            }
            if let Some(pos) = self.buf.find(p.text.as_str())
                && earliest.map(|(at, _)| pos < at).unwrap_or(true)
            {
                earliest = Some((pos, p));
            }
        }
        if let Some((at, pattern)) = earliest {
            return StopState::Full {
                at,
                pattern_len: pattern.len(),
                emit_at: at + pattern.keep_prefix,
            };
        }

        // Partial suffix match — longest k such that buf.ends_with(pattern[..k]).
        // We want the LONGEST hold so any ambiguous suffix is withheld long
        // enough for the next token to either complete the pattern or
        // invalidate it.
        let mut longest_hold = 0usize;
        for p in &self.patterns {
            if p.text.is_empty() {
                continue;
            }
            let max = p.len().saturating_sub(1).min(self.buf.len());
            for k in (1..=max).rev() {
                // `text[..k]` might fall on a non-char-boundary if the
                // pattern contains multibyte chars; check before slicing.
                if !p.text.is_char_boundary(k) {
                    continue;
                }
                if self.buf.ends_with(&p.text[..k]) {
                    if k > longest_hold {
                        longest_hold = k;
                    }
                    break;
                }
            }
        }

        if longest_hold > 0 {
            StopState::Partial {
                hold: longest_hold,
            }
        } else {
            StopState::Clean
        }
    }

    /// Immutable view of the accumulator (e.g. for debug logging).
    pub fn buffer(&self) -> &str {
        &self.buf
    }

    /// Reset the accumulator between sessions.
    pub fn reset(&mut self) {
        self.buf.clear();
    }

    /// Default stop patterns for a Gemma 4 chat session. Covers the
    /// quantization failure modes we observed on 26B-4bit where the model
    /// writes the next turn's header in plain text instead of emitting the
    /// `<turn|>` special token.
    ///
    /// Two pattern families:
    ///   1. Role-word + newline: `user\n`, `system\n`, `assistant\n`.
    ///      Catches `...answer.\nuser\n` — the standard failure mode where
    ///      the model types the next turn header on a new line.
    ///   2. Punctuation + role-word (NO space): `.user`, `!user`, `?user`,
    ///      `:user`, `;user`, and the same for `system`/`assistant`. Never
    ///      legitimate English — in real prose "." is always followed by
    ///      space before "user". This catches the no-newline variant we
    ///      see on 26B when the model goes `...quicksort.user<garbage>`.
    ///   3. Special-token fallbacks: `<|turn>`, `<start_of_turn>`,
    ///      `<end_of_turn>` — if the model ever emits these as literal
    ///      text (rather than the tokenizer's single-id form) it's
    ///      definitively a turn boundary.
    pub fn gemma4_chat_defaults() -> Vec<StopPattern> {
        let mut v = Vec::new();
        for role in ["user", "system", "assistant"] {
            // Role + newline: common fake-turn-on-newline pattern.
            v.push(StopPattern::new(format!("{role}\n")));
            // Punctuation + role (no space): quantization-specific "no
            // newline, just goes straight into the next turn" pattern.
            // `keep_prefix=1` preserves the punctuation in the emitted
            // output — so `"...quicksort.user..."` becomes `"...quicksort."`
            // instead of `"...quicksort"`.
            for p in [".", "!", "?", ":", ";"] {
                v.push(StopPattern::new(format!("{p}{role}")).keep(1));
            }
        }
        v.push(StopPattern::new("<|turn>"));
        v.push(StopPattern::new("<start_of_turn>"));
        v.push(StopPattern::new("<end_of_turn>"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_match_detected_mid_stream() {
        let mut m = StopMatcher::from_strings(vec!["\nuser\n".to_string()]);
        assert_eq!(m.push("hello world."), StopState::Clean);
        let state = m.push("\nuser\n");
        match state {
            StopState::Full {
                at,
                pattern_len,
                emit_at,
            } => {
                assert_eq!(&m.buffer()[at..at + pattern_len], "\nuser\n");
                assert_eq!(emit_at, at); // keep_prefix = 0
            }
            _ => panic!("expected Full, got {state:?}"),
        }
    }

    #[test]
    fn partial_match_hold_grows_and_resolves() {
        let mut m = StopMatcher::from_strings(vec!["\nuser\n".to_string()]);
        assert_eq!(m.push("hello."), StopState::Clean);
        assert_eq!(m.push("\n"), StopState::Partial { hold: 1 });
        assert_eq!(m.push("us"), StopState::Partial { hold: 3 });
        assert_eq!(m.push("er"), StopState::Partial { hold: 5 });
        match m.push("\n") {
            StopState::Full { .. } => {}
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn partial_match_invalidated_by_unrelated_next() {
        let mut m = StopMatcher::from_strings(vec!["\nuser\n".to_string()]);
        m.push("hello.\nus");
        assert!(matches!(
            m.push("\n"),
            StopState::Clean | StopState::Partial { .. }
        ));
        // "\nus\n" isn't a prefix of "\nuser\n", so clean.
        assert_eq!(m.push("something"), StopState::Clean);
    }

    #[test]
    fn earliest_of_multiple_matches_wins() {
        let mut m = StopMatcher::from_strings(vec![
            "\nassistant\n".to_string(),
            "\nuser\n".to_string(),
        ]);
        let state = m.push("hi\nuser\nand \nassistant\n");
        match state {
            StopState::Full { at, .. } => {
                assert_eq!(&m.buffer()[at..at + "\nuser\n".len()], "\nuser\n");
            }
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn multibyte_pattern_does_not_panic_on_char_boundary() {
        let mut m = StopMatcher::from_strings(vec!["…stop".to_string()]);
        m.push("abc…s");
        m.push("top");
    }

    #[test]
    fn keep_prefix_preserves_punctuation() {
        // `.user` matches, but keep_prefix=1 tells us to emit the period
        // and drop only the `user`. Essential for the Gemma 4 quant
        // failure where the model writes `"...quicksort.user<garbage>"`.
        let mut m = StopMatcher::new(vec![StopPattern::new(".user").keep(1)]);
        let state = m.push("quicksort.user");
        match state {
            StopState::Full {
                at,
                pattern_len,
                emit_at,
            } => {
                assert_eq!(&m.buffer()[at..at + pattern_len], ".user");
                // emit_at should be AT the start of "user", preserving "."
                assert_eq!(emit_at, at + 1);
                assert_eq!(&m.buffer()[..emit_at], "quicksort.");
            }
            other => panic!("expected Full, got {other:?}"),
        }
    }
}
