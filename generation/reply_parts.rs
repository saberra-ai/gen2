//! Split a streaming assistant reply into structured parts (reasoning vs.
//! visible content) as text arrives.
//!
//! Phase 2a of the inference-robustness plan. Feeds the `ReplyShape`
//! telemetry and is the groundwork for Phase 2b's structured
//! [`MessageContent::Assistant`] variant.
//!
//! The state machine is intentionally cheap: it scans decoded text (not
//! raw token IDs) for a small set of channel-boundary markers. Token
//! counts are approximated from byte length (`len / 4` — the standard
//! rough heuristic). Exact counts require hooking the backend's output
//! filter, which is a bigger change and not load-bearing for the
//! telemetry signal — the ratio of thinking to visible is what matters,
//! and approximation preserves that.
//!
//! Correctness: the scanner handles markers that land across multiple
//! `push` calls (token boundaries inside a marker). Everything else is
//! a simple two-state machine.

use serde::{Deserialize, Serialize};

use crate::gen2::generation::ToolCall;

/// Structured form of an assistant reply.
///
/// Mirrors the fields the chat templates for Gemma-4, Qwen3-Thinking,
/// DeepSeek-R1, and GPT-oss consume. `reasoning` is `None` for models
/// that don't expose a channel, or when the state machine didn't see
/// an open marker in the stream.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ReplyParts {
    /// What the end-user sees. Everything emitted outside a reasoning
    /// channel.
    pub content: String,
    /// The model's reasoning block, if any. Rendered as
    /// `<|channel>thought\n{reasoning}\n<channel|>` (Gemma-4) or
    /// `<think>{reasoning}</think>` (Qwen3 / DeepSeek-R1) when the
    /// chat template re-emits the turn.
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Tool calls extracted from the stream. Phase 2a leaves this
    /// empty — tool-call extraction is already handled by
    /// [`crate::gen2::backend::common::tool_calls`] and will flow in
    /// when the state machine subscribes to backend tool events in
    /// a later phase.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// Channel-boundary markers for one model family. Open/close strings
/// are matched against incoming decoded text. Order matters — the
/// scanner picks the first match; put longer / more-specific markers
/// first if there's any ambiguity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelMarkers {
    /// Strings that, when seen in the stream, transition from visible
    /// content into reasoning. e.g. `"<|channel>thought\n"`, `"<think>"`.
    pub open: Vec<String>,
    /// Strings that transition back to visible content. e.g.
    /// `"\n<channel|>"`, `"</think>"`.
    pub close: Vec<String>,
}

impl ChannelMarkers {
    /// Empty markers — state machine stays in `Visible` forever.
    /// Use for models without a reasoning channel (plain Llama, etc.).
    pub fn none() -> Self {
        Self::default()
    }

    /// Gemma-4 markers. The chat template emits
    /// `<|channel>thought\n{reasoning}\n<channel|>` when
    /// `enable_thinking=true`.
    pub fn gemma4() -> Self {
        Self {
            open: vec!["<|channel>thought\n".into(), "<|channel>thought".into()],
            close: vec!["\n<channel|>".into(), "<channel|>".into()],
        }
    }

    /// Qwen3-Thinking / DeepSeek-R1 markers — both families use the
    /// same `<think>` / `</think>` text form.
    pub fn qwen3_deepseek() -> Self {
        Self {
            open: vec!["<think>".into()],
            close: vec!["</think>".into()],
        }
    }

    /// Best-effort marker detection from a model id string. When the
    /// model's name indicates its family, return its markers; otherwise
    /// `none()`. Callers that know the model precisely should construct
    /// markers directly.
    pub fn from_model_hint(model_id: &str) -> Self {
        let m = model_id.to_ascii_lowercase();
        if m.contains("gemma-4") || m.contains("gemma4") {
            Self::gemma4()
        } else if m.contains("qwen3") || m.contains("deepseek") || m.contains("r1") {
            Self::qwen3_deepseek()
        } else {
            Self::none()
        }
    }

    /// Best-effort marker detection from a backend's `bundle_architecture()`
    /// (lowercased GGUF `general.architecture` or HF `model_type`). Returns
    /// `none()` when arch is unknown — callers can fall back to model-id
    /// hint detection if they have one. Sourced from
    /// [`crate::gen2::backend::traits::Backend::bundle_architecture`].
    pub fn from_architecture(arch: Option<&str>) -> Self {
        match arch {
            Some(a) if a.starts_with("gemma4") => Self::gemma4(),
            Some("qwen3") | Some("qwen3moe") | Some("deepseek2") => Self::qwen3_deepseek(),
            _ => Self::none(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelState {
    Visible,
    InReasoning,
}

/// One emission slice from the streaming state machine — a substring of
/// the input chunk, tagged with the channel it belongs to. Produced by
/// [`ReplyStateMachine::push_emit`] and consumed by the OAI streaming
/// layer so clients receive separate `content` vs `reasoning_content`
/// deltas (vLLM streaming convention).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEmission {
    /// Substring that lands in the visible reply.
    Content(String),
    /// Substring that lands in the reasoning channel.
    Reasoning(String),
}

/// Streaming splitter. `push` accepts arbitrary chunks of decoded text
/// (typically one `ControllerEvent::Token(text)` worth at a time);
/// `finish` returns the accumulated [`ReplyParts`].
#[derive(Debug, Clone)]
pub struct ReplyStateMachine {
    markers: ChannelMarkers,
    state: ChannelState,
    reasoning: String,
    content: String,
    /// Bytes carried over from the previous `push` because they might
    /// be the leading prefix of a marker. Flushed to the active buffer
    /// at `finish` time if no match materialises.
    pending: String,
    /// Count of literal special-token *substrings* we spotted that
    /// didn't match any known open/close marker. Non-zero here is the
    /// signal that the template or model leaked a special that our
    /// channel logic doesn't know about — a Phase-3 concern.
    leaked_specials: u32,
    /// Exact per-channel token counts driven by [`Self::push_emit`].
    /// Each `push_emit` call = one backend token; the count gets
    /// attributed to the channel that was active *at arrival time*
    /// (so a closing-marker token is counted against the channel it
    /// terminated). `None` means no token-paced calls were made;
    /// [`Self::summary`] then falls back to a byte-length approximation.
    explicit_thinking_tokens: Option<u32>,
    explicit_content_tokens: Option<u32>,
}

impl ReplyStateMachine {
    pub fn new(markers: ChannelMarkers) -> Self {
        Self {
            markers,
            state: ChannelState::Visible,
            reasoning: String::new(),
            content: String::new(),
            pending: String::new(),
            leaked_specials: 0,
            explicit_thinking_tokens: None,
            explicit_content_tokens: None,
        }
    }

    /// Feed a chunk of decoded text and return each channel-tagged
    /// slice that's ready to forward to a streaming client. Partial
    /// markers at the tail are held back and won't appear in the
    /// return value until the next call completes them.
    ///
    /// Use this when a client needs *incremental* channel separation
    /// (OAI SSE streaming). For batch use, [`Self::push`] + [`Self::finish`]
    /// is simpler.
    ///
    /// **Token accounting.** Every call to `push_emit` is treated as
    /// exactly one backend-sampled token (each `ControllerEvent::Token`
    /// event in the upstream stream corresponds to one decode step).
    /// The token is attributed to the channel that was active at the
    /// start of this call — a token whose decoded text straddles a
    /// close-marker boundary still counts against the ending channel,
    /// matching the model's own "what state was I in when I sampled
    /// this" semantics. Callers should invoke once per
    /// `ControllerEvent::Token` for correctness.
    pub fn push_emit(&mut self, chunk: &str) -> Vec<StreamEmission> {
        // Attribute this token to the current channel BEFORE consuming.
        // `get_or_insert(0)` promotes the Option to Some on first use;
        // once we've seen any push_emit call, `summary()` stops using
        // the byte/4 approximation and reads the exact counters.
        match self.state {
            ChannelState::Visible => {
                let c = self.explicit_content_tokens.get_or_insert(0);
                *c = c.saturating_add(1);
                // Ensure the reasoning counter is also materialised so
                // `summary()` reports 0 (not a fallback byte estimate).
                self.explicit_thinking_tokens.get_or_insert(0);
            }
            ChannelState::InReasoning => {
                let t = self.explicit_thinking_tokens.get_or_insert(0);
                *t = t.saturating_add(1);
                self.explicit_content_tokens.get_or_insert(0);
            }
        }
        let mut emissions = Vec::new();
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(chunk);

        let bytes = buf.as_bytes();
        let mut i = 0;
        // `run_start` marks the start of the current channel's
        // contiguous byte range inside `buf`; we flush one emission
        // whenever the channel transitions.
        let mut run_start = 0;
        let mut run_channel = self.state;

        while i < bytes.len() {
            let patterns = match self.state {
                ChannelState::Visible => &self.markers.open,
                ChannelState::InReasoning => &self.markers.close,
            };
            if let Some(mlen) = match_any(&buf[i..], patterns) {
                // Don't commit to the match yet if a longer pattern
                // shares this prefix and the buffer doesn't rule it
                // out. In batch mode `match_any` already picks the
                // longest match; in streaming mode we may have only
                // seen enough bytes to match the shorter form. Holding
                // here preserves streaming/batch equivalence — without
                // this, e.g. `<|channel>thought` commits before the
                // `\n` arrives and `<|channel>thought\n` can no longer
                // win, leaving a stray `\n` at the start of reasoning.
                if longer_pattern_could_extend(&buf[i..], mlen, patterns) {
                    if run_start < i {
                        let slice = buf[run_start..i].to_string();
                        emissions.push(match run_channel {
                            ChannelState::Visible => StreamEmission::Content(slice),
                            ChannelState::InReasoning => StreamEmission::Reasoning(slice),
                        });
                    }
                    self.pending = buf[i..].to_string();
                    self.mirror_into_buffers(&emissions);
                    return emissions;
                }
                // Flush whatever came before the marker in the old channel.
                if run_start < i {
                    let slice = buf[run_start..i].to_string();
                    emissions.push(match run_channel {
                        ChannelState::Visible => StreamEmission::Content(slice),
                        ChannelState::InReasoning => StreamEmission::Reasoning(slice),
                    });
                }
                // Markers are never emitted to the client.
                i += mlen;
                self.state = match self.state {
                    ChannelState::Visible => ChannelState::InReasoning,
                    ChannelState::InReasoning => ChannelState::Visible,
                };
                run_start = i;
                run_channel = self.state;
                continue;
            }
            if could_start_any(&buf[i..], patterns) {
                // Flush everything up to here in the current channel.
                if run_start < i {
                    let slice = buf[run_start..i].to_string();
                    emissions.push(match run_channel {
                        ChannelState::Visible => StreamEmission::Content(slice),
                        ChannelState::InReasoning => StreamEmission::Reasoning(slice),
                    });
                }
                // Hold the rest back for the next push_emit call.
                self.pending = buf[i..].to_string();
                // Accumulate into the internal buffers so summary/finish
                // stay consistent.
                self.mirror_into_buffers(&emissions);
                return emissions;
            }
            let step = utf8_char_len(bytes, i);
            if self.state == ChannelState::Visible
                && bytes[i] == b'<'
                && has_special_token_shape(&buf[i..])
            {
                self.leaked_specials = self.leaked_specials.saturating_add(1);
            }
            i += step;
        }
        // Flush the tail of the last run.
        if run_start < bytes.len() {
            let slice = buf[run_start..].to_string();
            emissions.push(match run_channel {
                ChannelState::Visible => StreamEmission::Content(slice),
                ChannelState::InReasoning => StreamEmission::Reasoning(slice),
            });
        }
        self.mirror_into_buffers(&emissions);
        emissions
    }

    /// Apply each emission to the internal content/reasoning buffers
    /// so [`Self::summary`] and [`Self::finish`] see the same state the
    /// streaming client saw. Separate from the emit loop so the emit
    /// and mirror concerns stay readable.
    fn mirror_into_buffers(&mut self, emissions: &[StreamEmission]) {
        for e in emissions {
            match e {
                StreamEmission::Content(s) => self.content.push_str(s),
                StreamEmission::Reasoning(s) => self.reasoning.push_str(s),
            }
        }
    }

    /// Feed a chunk of decoded text. Runs the state machine across
    /// `pending + chunk`. Partial markers at the tail are held back
    /// in `pending` for the next call.
    pub fn push(&mut self, chunk: &str) {
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(chunk);

        // Walk `buf` left-to-right. At each position, check whether
        // any marker for the current state matches at this offset.
        let bytes = buf.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            match self.state {
                ChannelState::Visible => {
                    if let Some(mlen) = match_any(&buf[i..], &self.markers.open) {
                        self.state = ChannelState::InReasoning;
                        i += mlen;
                        continue;
                    }
                    if could_start_any(&buf[i..], &self.markers.open) {
                        // Hold back — this might complete on the next push.
                        self.pending = buf[i..].to_string();
                        return;
                    }
                    // Non-marker byte: accumulate. Use char boundaries.
                    let step = utf8_char_len(bytes, i);
                    self.content.push_str(&buf[i..i + step]);
                    // Detect stray `<|...>` that we didn't recognise as
                    // a marker — purely a metric signal, content is
                    // still passed through unchanged.
                    if bytes[i] == b'<' && has_special_token_shape(&buf[i..]) {
                        self.leaked_specials = self.leaked_specials.saturating_add(1);
                    }
                    i += step;
                }
                ChannelState::InReasoning => {
                    if let Some(mlen) = match_any(&buf[i..], &self.markers.close) {
                        self.state = ChannelState::Visible;
                        i += mlen;
                        continue;
                    }
                    if could_start_any(&buf[i..], &self.markers.close) {
                        self.pending = buf[i..].to_string();
                        return;
                    }
                    let step = utf8_char_len(bytes, i);
                    self.reasoning.push_str(&buf[i..i + step]);
                    i += step;
                }
            }
        }
    }

    /// Streaming counterpart to [`Self::finish`]: drain any held-back
    /// `pending` bytes (a partial-marker prefix that never resolved)
    /// into a single emission for the channel that was active when
    /// they were buffered. Use this on terminal events to flush the
    /// tail without consuming the state machine.
    ///
    /// Returns an empty vec when `pending` is empty.
    pub fn flush_pending(&mut self) -> Vec<StreamEmission> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let s = std::mem::take(&mut self.pending);
        let emission = match self.state {
            ChannelState::Visible => StreamEmission::Content(s),
            ChannelState::InReasoning => StreamEmission::Reasoning(s),
        };
        // Mirror into the internal buffers so `finish` / `summary`
        // remain consistent with what the streaming caller saw.
        self.mirror_into_buffers(std::slice::from_ref(&emission));
        vec![emission]
    }

    /// Consume the state machine and return the split. Any unflushed
    /// `pending` bytes go to the active buffer — we can't hold them
    /// back indefinitely once the stream is over.
    pub fn finish(mut self) -> ReplyParts {
        if !self.pending.is_empty() {
            match self.state {
                ChannelState::Visible => self.content.push_str(&self.pending),
                ChannelState::InReasoning => self.reasoning.push_str(&self.pending),
            }
        }
        ReplyParts {
            content: self.content,
            reasoning: if self.reasoning.is_empty() {
                None
            } else {
                Some(self.reasoning)
            },
            tool_calls: Vec::new(),
        }
    }

    /// Non-consuming summary for the Phase-0 [`ReplyShape`] telemetry.
    /// Returns `(thinking_tokens, content_tokens, special_token_count)`.
    ///
    /// Token counts come from the explicit per-call counters populated
    /// by [`Self::push_emit`] — each call increments by one,
    /// corresponding to one `ControllerEvent::Token` (one backend
    /// decode step). Callers that only use [`Self::push`] fall back to
    /// a byte-length approximation (`bytes / 4`), which is coarse but
    /// bounded — good enough for non-streaming paths that don't have
    /// per-token granularity.
    pub fn summary(&self) -> (u32, u32, u32) {
        let thinking = self
            .explicit_thinking_tokens
            .unwrap_or((self.reasoning.len() / 4) as u32);
        let content = self
            .explicit_content_tokens
            .unwrap_or((self.content.len() / 4) as u32);
        (thinking, content, self.leaked_specials)
    }
}

// ── helpers ────────────────────────────────────────────────────────────────

/// Return the length in bytes of the longest marker from `patterns` that
/// matches at the start of `s`, or `None` if none match.
fn match_any(s: &str, patterns: &[String]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for p in patterns {
        if s.starts_with(p.as_str()) {
            let len = p.len();
            best = Some(best.map_or(len, |b| b.max(len)));
        }
    }
    best
}

/// True when `s` could be the *prefix* of any pattern — i.e. we might
/// need more bytes before we can decide. Used to defer emission across
/// push boundaries.
fn could_start_any(s: &str, patterns: &[String]) -> bool {
    if s.is_empty() {
        return false;
    }
    for p in patterns {
        if p.starts_with(s) && p.len() > s.len() {
            return true;
        }
    }
    false
}

/// True when a strictly-longer pattern than `matched_len` shares the
/// matched prefix AND hasn't been ruled out by `s`'s current length.
/// In batch mode `match_any` already picks the longest; this helper
/// defers streaming commits until the longest possible match is known.
///
/// Example: with patterns `["<|channel>thought", "<|channel>thought\n"]`,
/// input `s = "<|channel>thought"` returns true — the `\n` variant
/// might still complete. Input `s = "<|channel>thoughtX"` returns
/// false — the `X` rules out the longer variant so committing the
/// short match is safe.
fn longer_pattern_could_extend(s: &str, matched_len: usize, patterns: &[String]) -> bool {
    for p in patterns {
        if p.len() <= matched_len {
            continue;
        }
        // The longer pattern must (a) share the matched prefix and
        // (b) still be a possible extension of `s` as-is.
        if p.as_bytes().get(..matched_len) == s.as_bytes().get(..matched_len) && p.starts_with(s) {
            return true;
        }
    }
    false
}

/// Best-effort UTF-8 char-length at byte offset. Falls back to 1 if the
/// leading byte is mid-sequence (should not happen on valid input but
/// defends against malformed streams).
fn utf8_char_len(bytes: &[u8], i: usize) -> usize {
    let b = bytes[i];
    if b < 0x80 {
        1
    } else if b & 0xE0 == 0xC0 {
        2
    } else if b & 0xF0 == 0xE0 {
        3
    } else if b & 0xF8 == 0xF0 {
        4
    } else {
        1
    }
}

/// Cheap heuristic: does `s` look like it starts with a `<|...>` or
/// `<...>` special-token literal? Used only to count leakage, not to
/// alter content — false positives are fine.
fn has_special_token_shape(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'<') {
        return false;
    }
    // `<|...>` shape — has pipe after <
    if bytes.get(1) == Some(&b'|') {
        return true;
    }
    // `<channel|>` or similar — has `|>` before next whitespace
    for (i, b) in bytes.iter().enumerate().skip(1).take(32) {
        if *b == b'|' && bytes.get(i + 1) == Some(&b'>') {
            return true;
        }
        if *b == b'>' || b.is_ascii_whitespace() {
            return false;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemma() -> ChannelMarkers {
        ChannelMarkers::gemma4()
    }

    #[test]
    fn plain_text_no_markers_goes_to_content() {
        let mut sm = ReplyStateMachine::new(ChannelMarkers::none());
        sm.push("hello world");
        let parts = sm.finish();
        assert_eq!(parts.content, "hello world");
        assert_eq!(parts.reasoning, None);
    }

    #[test]
    fn gemma4_splits_thought_and_visible() {
        let mut sm = ReplyStateMachine::new(gemma());
        // Typical Gemma-4 stream shape: thought channel, then answer.
        // `\n<channel|>` is the close marker so the trailing `\n` of
        // the reasoning block is consumed by the marker, not by the
        // reasoning buffer.
        sm.push("<|channel>thought\nlet me think about this\n<channel|>");
        sm.push("The answer is 42.");
        let parts = sm.finish();
        assert_eq!(parts.content, "The answer is 42.");
        assert_eq!(parts.reasoning.as_deref(), Some("let me think about this"));
    }

    #[test]
    fn qwen3_deepseek_think_tags() {
        let mut sm = ReplyStateMachine::new(ChannelMarkers::qwen3_deepseek());
        sm.push("<think>internal monologue</think>final answer");
        let parts = sm.finish();
        assert_eq!(parts.content, "final answer");
        assert_eq!(parts.reasoning.as_deref(), Some("internal monologue"));
    }

    #[test]
    fn marker_split_across_push_boundary() {
        let mut sm = ReplyStateMachine::new(gemma());
        sm.push("<|chan");
        sm.push("nel>thought\nreason\n<channel|>");
        sm.push("done");
        let parts = sm.finish();
        assert_eq!(parts.content, "done");
        assert_eq!(parts.reasoning.as_deref(), Some("reason"));
    }

    #[test]
    fn close_marker_split_across_push_boundary() {
        let mut sm = ReplyStateMachine::new(gemma());
        sm.push("<|channel>thought\nthink\n<chan");
        sm.push("nel|>visible");
        let parts = sm.finish();
        assert_eq!(parts.reasoning.as_deref(), Some("think"));
        assert_eq!(parts.content, "visible");
    }

    #[test]
    fn no_open_marker_everything_is_content() {
        let mut sm = ReplyStateMachine::new(gemma());
        sm.push("just a direct answer, no thinking");
        let parts = sm.finish();
        assert_eq!(parts.content, "just a direct answer, no thinking");
        assert_eq!(parts.reasoning, None);
    }

    #[test]
    fn unclosed_thought_drains_to_reasoning_at_finish() {
        // Model truncated mid-thought (max_tokens hit). The reasoning
        // buffer keeps what was emitted so it can be persisted and
        // replayed — the next turn's session-cache re-key captures it.
        let mut sm = ReplyStateMachine::new(gemma());
        sm.push("<|channel>thought\nhalf a thought");
        let parts = sm.finish();
        assert_eq!(parts.reasoning.as_deref(), Some("half a thought"));
        assert_eq!(parts.content, "");
    }

    #[test]
    fn utf8_multibyte_content_survives_split() {
        let mut sm = ReplyStateMachine::new(ChannelMarkers::none());
        sm.push("héllo ");
        sm.push("wörld 🌍");
        let parts = sm.finish();
        assert_eq!(parts.content, "héllo wörld 🌍");
    }

    #[test]
    fn summary_approximates_tokens_from_bytes() {
        // `push` (batch) has no per-token signal — we fall back to
        // byte/4. Kept for the non-streaming path.
        let mut sm = ReplyStateMachine::new(gemma());
        sm.push("<|channel>thought\n");
        sm.push("aaaa"); // 4 bytes of reasoning
        sm.push("<channel|>");
        sm.push("bbbbbbbb"); // 8 bytes of content
        let (thinking, content, _) = sm.summary();
        assert_eq!(thinking, 1); // 4 / 4 = 1
        assert_eq!(content, 2); // 8 / 4 = 2
    }

    #[test]
    fn push_emit_tracks_exact_token_counts() {
        // Each push_emit call represents one ControllerEvent::Token,
        // so summary() should report exact counts independent of
        // decoded byte length.
        let mut sm = ReplyStateMachine::new(gemma());
        // 1 token = open marker (single backend token, tokenizer-dependent)
        sm.push_emit("<|channel>thought\n");
        // 3 reasoning tokens
        sm.push_emit("one");
        sm.push_emit("two");
        sm.push_emit("three");
        // Close marker (1 token).
        sm.push_emit("<channel|>");
        // 2 visible tokens
        sm.push_emit("hello");
        sm.push_emit("world");
        let (thinking, content, _) = sm.summary();
        // The tokens that consumed the markers stay attributed to the
        // pre-state channel: open-marker token sampled in Visible,
        // close-marker token sampled in Reasoning. So:
        // - content_tokens = 1 (open marker) + 2 (hello, world) = 3
        // - thinking_tokens = 3 (one/two/three) + 1 (close marker) = 4
        assert_eq!(thinking, 4);
        assert_eq!(content, 3);
    }

    #[test]
    fn push_emit_exact_counts_ignore_byte_lengths() {
        // A single push_emit call with 1000 bytes of text still counts
        // as exactly one token — byte length is irrelevant once
        // explicit counting is engaged.
        let mut sm = ReplyStateMachine::new(ChannelMarkers::none());
        sm.push_emit(&"x".repeat(1000));
        let (thinking, content, _) = sm.summary();
        assert_eq!(thinking, 0);
        assert_eq!(content, 1);
    }

    #[test]
    fn push_only_sticks_to_byte_approximation() {
        // If the caller never calls push_emit, summary() must stay on
        // the legacy byte/4 estimate. Zero-length push doesn't engage
        // the explicit counters.
        let mut sm = ReplyStateMachine::new(ChannelMarkers::none());
        sm.push(&"a".repeat(40));
        let (_, content, _) = sm.summary();
        assert_eq!(content, 10); // 40 bytes / 4
    }

    #[test]
    fn leaked_specials_counted_not_stripped() {
        // If the model emits a special-token literal outside any known
        // channel, we still forward the text (users see it) but bump
        // the leak counter so Phase 5 can alert on it.
        let mut sm = ReplyStateMachine::new(ChannelMarkers::none());
        sm.push("hello <|unknown_special|> world");
        let (_, _, specials) = sm.summary();
        let parts = sm.finish();
        assert!(specials >= 1, "expected at least one leaked-special hit");
        assert!(parts.content.contains("unknown_special"));
    }

    #[test]
    fn push_emit_tags_chunks_by_channel() {
        let mut sm = ReplyStateMachine::new(gemma());
        let e1 = sm.push_emit("<|channel>thought\n");
        assert!(e1.is_empty(), "open marker alone emits nothing");
        let e2 = sm.push_emit("reasoning here\n");
        // The trailing `\n` is held back because it could be the start
        // of the `\n<channel|>` close marker; only "reasoning here"
        // is emitted this call.
        assert_eq!(e2, vec![StreamEmission::Reasoning("reasoning here".into())]);
        let e3 = sm.push_emit("<channel|>visible answer");
        // The pending `\n` + incoming `<channel|>` forms the close
        // marker, which is stripped. Rest is visible content.
        assert_eq!(e3, vec![StreamEmission::Content("visible answer".into())]);
        let parts = sm.finish();
        assert_eq!(parts.content, "visible answer");
        assert_eq!(parts.reasoning.as_deref(), Some("reasoning here"));
    }

    #[test]
    fn push_emit_plain_content_no_markers() {
        let mut sm = ReplyStateMachine::new(ChannelMarkers::none());
        let e = sm.push_emit("hello world");
        assert_eq!(e, vec![StreamEmission::Content("hello world".into())]);
    }

    #[test]
    fn push_emit_holds_partial_marker() {
        let mut sm = ReplyStateMachine::new(gemma());
        // First chunk ends with a partial open marker; nothing should
        // emit yet because we don't know if the next chunk completes it.
        let e1 = sm.push_emit("visible <|chan");
        assert_eq!(e1, vec![StreamEmission::Content("visible ".into())]);
        let e2 = sm.push_emit("nel>thought\n");
        assert!(e2.is_empty(), "still inside completed marker");
        let e3 = sm.push_emit("my reasoning<channel|>rest");
        // Within the partial: two slices (reasoning, then content).
        assert_eq!(
            e3,
            vec![
                StreamEmission::Reasoning("my reasoning".into()),
                StreamEmission::Content("rest".into()),
            ]
        );
    }

    #[test]
    fn push_emit_mirrors_to_internal_buffers() {
        // finish() must reflect what push_emit consumed. Since each
        // push_emit call = one token (attributed to pre-state), the
        // token counts here are exact-call counts, not byte-derived:
        // - call 1: entire Gemma turn in one chunk, pre-state = Visible → 1 content
        // - call 2: "done", pre-state = Visible (we exited reasoning
        //   inside call 1) → 1 content, 0 thinking
        //
        // The reasoning *text* is still preserved correctly; only the
        // token counts reflect the caller's use pattern.
        let mut sm = ReplyStateMachine::new(gemma());
        sm.push_emit("<|channel>thought\nthinking\n<channel|>");
        sm.push_emit("done");
        let (thinking, content, _) = sm.summary();
        let parts = sm.finish();
        assert_eq!(parts.reasoning.as_deref(), Some("thinking"));
        assert_eq!(parts.content, "done");
        assert_eq!(thinking, 0);
        assert_eq!(content, 2);
    }

    #[test]
    fn from_model_hint_matches_common_families() {
        assert_eq!(
            ChannelMarkers::from_model_hint("gemma-4-e2b-it-4bit"),
            ChannelMarkers::gemma4()
        );
        assert_eq!(
            ChannelMarkers::from_model_hint("Qwen3-0.6B"),
            ChannelMarkers::qwen3_deepseek()
        );
        assert_eq!(
            ChannelMarkers::from_model_hint("deepseek-r1-distill"),
            ChannelMarkers::qwen3_deepseek()
        );
        assert_eq!(
            ChannelMarkers::from_model_hint("llama-3.2-3b"),
            ChannelMarkers::none()
        );
    }
}
