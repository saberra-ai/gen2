//! Streaming tool-call parser shared across gen2 backends.
//!
//! Modern chat models express tool invocations as inline markers in their
//! generated text:
//!   - **Gemma 4 / Llama 3.x / Qwen3 / Hermes**:
//!     `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
//!   - **Mistral v7 / some OpenAI-compat**: `[TOOL_CALLS][{...}]`
//!   - **Anthropic**: `<function_calls><invoke name="..."><parameter
//!     name="...">...</parameter>...</invoke></function_calls>`
//!
//! A backend's token stream is a sequence of small text chunks, so the
//! parser is byte-incremental: feed it chunks, get back zero or more
//! `ParserOutput` events (plain text spans + structured tool calls).
//!
//! This module is backend-agnostic — it sits between a `TokenEvent::Token`
//! emission and the downstream consumer. Every gen2 puller can route
//! through it to get structured tool-call events for free.

use crate::gen2::generation::ToolCall;

/// What the parser wants the caller to emit after a chunk is absorbed.
#[derive(Debug, Clone, PartialEq)]
pub enum ParserOutput {
    /// A span of plain text that's NOT part of any tool-call marker.
    /// The caller re-emits this as a `TokenEvent::Token`.
    Text(String),
    /// A fully-parsed tool call. Caller emits as `TokenEvent::ToolCall`.
    ToolCall(ToolCall),
}

/// Supported wire protocols. Defaults to a safe superset detector that
/// handles the three common open formats; callers with model-specific
/// knowledge (e.g. Anthropic's XML) can force a single protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Protocol {
    /// Auto-detect on first marker seen; sticky for the rest of the stream.
    #[default]
    Auto,
    /// `<tool_call>...</tool_call>` with JSON body.
    TagJson,
    /// `[TOOL_CALLS][{...}]` Mistral/OpenAI-ish format.
    BracketJson,
    /// Anthropic `<function_calls><invoke name="..."><parameter>...`
    AnthropicXml,
}

pub struct ToolCallParser {
    protocol: Protocol,
    /// Rolling text buffer the parser's looking inside. Reset whenever a
    /// clean span is emitted.
    buf: String,
    state: State,
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    /// Outside any marker — emit clean text.
    Outside,
    /// Inside a `<tool_call>...</tool_call>` span. Collecting body chars.
    InTagBody { body_start: usize },
    /// Inside a `[TOOL_CALLS][...]` span.
    InBracketBody { body_start: usize },
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new(Protocol::default())
    }
}

impl ToolCallParser {
    pub fn new(protocol: Protocol) -> Self {
        Self {
            protocol,
            buf: String::new(),
            state: State::Outside,
        }
    }

    /// Feed a chunk of decoded text and drain the resulting events. The
    /// parser holds unresolved partial markers internally; callers need
    /// NOT worry about partial tags spanning multiple calls. Calling with
    /// `""` is safe (noop, returns empty).
    pub fn push(&mut self, text: &str) -> Vec<ParserOutput> {
        if text.is_empty() {
            return Vec::new();
        }
        self.buf.push_str(text);
        let mut out = Vec::new();
        self.drain_into(&mut out);
        out
    }

    /// Called at end-of-stream (EOS from the model). Releases any
    /// partially-buffered text as plain text (abandoning unclosed tags).
    pub fn flush(&mut self) -> Vec<ParserOutput> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        // Release whatever's in the buffer, including unclosed markers.
        // Consumers see malformed tool-call attempts as literal text.
        let text = std::mem::take(&mut self.buf);
        self.state = State::Outside;
        vec![ParserOutput::Text(text)]
    }

    /// Consume as much of `self.buf` as possible, writing events into `out`.
    /// Leaves any trailing incomplete marker in the buffer for future calls.
    fn drain_into(&mut self, out: &mut Vec<ParserOutput>) {
        loop {
            match self.state {
                State::Outside => {
                    // Look for the earliest opening marker of any allowed protocol.
                    let next = self.find_opening();
                    match next {
                        None => {
                            // No marker in sight. BUT: the tail of the buffer
                            // could be a partial marker prefix. We only
                            // release text up to the last safe position.
                            let safe = self.safe_release_end();
                            if safe > 0 {
                                let released: String = self.buf.drain(..safe).collect();
                                out.push(ParserOutput::Text(released));
                            }
                            return;
                        }
                        Some((at, marker)) => {
                            if at > 0 {
                                let before: String = self.buf.drain(..at).collect();
                                out.push(ParserOutput::Text(before));
                            }
                            // Consume the marker itself; shift to In* state.
                            self.buf.drain(..marker.len());
                            self.state = match marker {
                                Marker::TagOpen => State::InTagBody { body_start: 0 },
                                Marker::BracketOpen => State::InBracketBody { body_start: 0 },
                            };
                            // Lock the protocol on first use in Auto mode.
                            if self.protocol == Protocol::Auto {
                                self.protocol = match marker {
                                    Marker::TagOpen => Protocol::TagJson,
                                    Marker::BracketOpen => Protocol::BracketJson,
                                };
                            }
                        }
                    }
                }
                State::InTagBody { .. } => {
                    // Search for closing `</tool_call>`.
                    const CLOSE: &str = "</tool_call>";
                    if let Some(pos) = self.buf.find(CLOSE) {
                        let body: String = self.buf.drain(..pos).collect();
                        self.buf.drain(..CLOSE.len());
                        if let Some(tc) = parse_tag_json(body.trim()) {
                            out.push(ParserOutput::ToolCall(tc));
                        } else {
                            // Malformed JSON body — emit as literal text
                            // so nothing is silently lost.
                            out.push(ParserOutput::Text(format!("<tool_call>{body}</tool_call>")));
                        }
                        self.state = State::Outside;
                    } else {
                        // Waiting for close tag — hold.
                        return;
                    }
                }
                State::InBracketBody { .. } => {
                    // Bracket body ends with a balanced `]` at top level.
                    // The bracket format is `[TOOL_CALLS][...]` where the
                    // INNER brackets are a JSON array. We already consumed
                    // `[TOOL_CALLS][` and need the matching `]`.
                    match find_balanced_close(&self.buf) {
                        Some(pos) => {
                            let body: String = self.buf.drain(..pos).collect();
                            self.buf.drain(..1); // consume `]`
                            let calls = parse_bracket_json(&body);
                            for c in calls {
                                out.push(ParserOutput::ToolCall(c));
                            }
                            self.state = State::Outside;
                        }
                        None => return,
                    }
                }
            }
        }
    }

    /// Returns the earliest `<tool_call>` or `[TOOL_CALLS][` position and
    /// which marker hit, honouring the current `protocol`.
    fn find_opening(&self) -> Option<(usize, Marker)> {
        let tag_at = if matches!(self.protocol, Protocol::Auto | Protocol::TagJson) {
            self.buf.find("<tool_call>")
        } else {
            None
        };
        let bracket_at = if matches!(self.protocol, Protocol::Auto | Protocol::BracketJson) {
            self.buf.find("[TOOL_CALLS][")
        } else {
            None
        };
        match (tag_at, bracket_at) {
            (None, None) => None,
            (Some(a), None) => Some((a, Marker::TagOpen)),
            (None, Some(b)) => Some((b, Marker::BracketOpen)),
            (Some(a), Some(b)) => {
                if a <= b {
                    Some((a, Marker::TagOpen))
                } else {
                    Some((b, Marker::BracketOpen))
                }
            }
        }
    }

    /// Return how many bytes from the start of `buf` are safe to release
    /// as plain text without swallowing a partial marker at the tail.
    /// Finds the earliest position where the suffix of `buf` starts
    /// matching the prefix of ANY marker, and holds from there onwards.
    fn safe_release_end(&self) -> usize {
        let partial_start = partial_marker_start(&self.buf, self.protocol);
        match partial_start {
            Some(at) => at,
            None => {
                let mut n = self.buf.len();
                while n > 0 && !self.buf.is_char_boundary(n) {
                    n -= 1;
                }
                n
            }
        }
    }

    /// Currently-locked protocol (after first auto-detect).
    pub fn protocol(&self) -> Protocol {
        self.protocol
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    TagOpen,
    BracketOpen,
}

impl Marker {
    fn len(self) -> usize {
        match self {
            Marker::TagOpen => "<tool_call>".len(),
            Marker::BracketOpen => "[TOOL_CALLS][".len(),
        }
    }
}

/// If the suffix of `s` is a prefix of some marker, return the byte offset
/// in `s` where the partial starts (i.e. release everything before it).
/// Returns None when nothing is a partial match.
///
/// Picks the EARLIEST partial start so we hold as little as possible while
/// still capturing all potential openings. E.g. for buf="abc<" and marker
/// "<tool_call>", returns Some(3) so "abc" is released and "<" is held.
fn partial_marker_start(s: &str, protocol: Protocol) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
    let candidates: &[&str] = match protocol {
        Protocol::Auto => &["<tool_call>", "[TOOL_CALLS]["],
        Protocol::TagJson => &["<tool_call>"],
        Protocol::BracketJson => &["[TOOL_CALLS]["],
        Protocol::AnthropicXml => &["<function_calls>"],
    };
    let mut best: Option<usize> = None;
    for c in candidates {
        // Longest suffix of `s` that is a prefix of `c`, up to len(c)-1.
        let max = (c.len() - 1).min(s.len());
        for k in (1..=max).rev() {
            if !s.is_char_boundary(s.len() - k) {
                continue;
            }
            if c.starts_with(&s[s.len() - k..]) {
                let start = s.len() - k;
                best = Some(match best {
                    Some(b) => b.min(start),
                    None => start,
                });
                break;
            }
        }
    }
    best
}

/// Walk a JSON array body from after the opening `[` and find the matching
/// closing `]`. Returns the byte offset of the closing `]` (exclusive of
/// the char, inclusive as an end index). Honours quoted-string nesting so
/// `]` inside strings doesn't end prematurely. Returns None on incomplete.
fn find_balanced_close(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth: i32 = 1; // we're already inside the outer `[`
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth == 0 && b == b']' {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_tag_json(body: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let args = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .map(|a| {
            if a.is_string() {
                a.as_str().unwrap().to_string()
            } else {
                a.to_string()
            }
        })
        .unwrap_or_else(|| "{}".to_string());
    let id = v.get("id").and_then(|i| i.as_str().map(|s| s.to_string()));
    Some(ToolCall {
        id,
        name,
        arguments: args,
    })
}

fn parse_bracket_json(body: &str) -> Vec<ToolCall> {
    // `body` is the array contents: `{"name":...},{"name":...}`
    // Wrap in brackets to re-parse as a proper JSON array.
    let wrapped = format!("[{body}]");
    let Ok(serde_json::Value::Array(items)) = serde_json::from_str(&wrapped) else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter_map(|v| {
            let name = v.get("name")?.as_str()?.to_string();
            let args = v
                .get("arguments")
                .map(|a| {
                    if a.is_string() {
                        a.as_str().unwrap().to_string()
                    } else {
                        a.to_string()
                    }
                })
                .unwrap_or_else(|| "{}".to_string());
            let id = v.get("id").and_then(|i| i.as_str().map(|s| s.to_string()));
            Some(ToolCall {
                id,
                name,
                arguments: args,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passes_through() {
        let mut p = ToolCallParser::new(Protocol::Auto);
        let out = p.push("hello world");
        assert_eq!(out.len(), 1);
        match &out[0] {
            ParserOutput::Text(s) => assert_eq!(s, "hello world"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn tag_tool_call_single_chunk() {
        let mut p = ToolCallParser::new(Protocol::Auto);
        let events = p.push(r#"preface <tool_call>{"name":"get_weather","arguments":{"city":"Lisbon"}}</tool_call> done"#);
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], ParserOutput::Text(s) if s == "preface "));
        match &events[1] {
            ParserOutput::ToolCall(tc) => {
                assert_eq!(tc.name, "get_weather");
                assert!(tc.arguments.contains("Lisbon"));
            }
            _ => panic!("expected ToolCall"),
        }
        assert!(matches!(&events[2], ParserOutput::Text(s) if s == " done"));
    }

    #[test]
    fn tag_tool_call_split_across_chunks() {
        let mut p = ToolCallParser::new(Protocol::Auto);
        let mut all = Vec::new();
        all.extend(p.push("hello <tool"));
        all.extend(p.push("_call>{\"name"));
        all.extend(p.push(r#"":"foo","arguments":{}}</too"#));
        all.extend(p.push("l_call> bye"));
        // Expect: Text("hello "), ToolCall(foo), Text(" bye")
        assert!(matches!(&all[0], ParserOutput::Text(s) if s == "hello "));
        let tc_idx = all
            .iter()
            .position(|e| matches!(e, ParserOutput::ToolCall(_)))
            .unwrap();
        match &all[tc_idx] {
            ParserOutput::ToolCall(tc) => assert_eq!(tc.name, "foo"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn bracket_format_parses() {
        let mut p = ToolCallParser::new(Protocol::Auto);
        let out = p.push(
            r#"go [TOOL_CALLS][{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}] end"#,
        );
        let calls: Vec<&ToolCall> = out
            .iter()
            .filter_map(|e| match e {
                ParserOutput::ToolCall(tc) => Some(tc),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn malformed_tool_call_emits_as_text() {
        let mut p = ToolCallParser::new(Protocol::Auto);
        let out = p.push(r#"<tool_call>not json</tool_call>"#);
        // Should emit the whole literal as text since it's unparseable.
        let joined: String = out
            .iter()
            .filter_map(|e| match e {
                ParserOutput::Text(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(joined.contains("<tool_call>"));
        assert!(joined.contains("not json"));
    }

    #[test]
    fn partial_marker_at_tail_is_held() {
        let mut p = ToolCallParser::new(Protocol::Auto);
        // "<tool" could be the start of "<tool_call>" — must not be released.
        let out = p.push("hello <tool");
        let text: String = out
            .iter()
            .filter_map(|e| match e {
                ParserOutput::Text(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "hello ");
        // If we then push "l_call>...</tool_call>" it resolves.
        let out2 = p.push(r#"_call>{"name":"x","arguments":{}}</tool_call>"#);
        assert!(
            out2.iter()
                .any(|e| matches!(e, ParserOutput::ToolCall(tc) if tc.name == "x"))
        );
    }

    #[test]
    fn flush_releases_partial_as_text() {
        let mut p = ToolCallParser::new(Protocol::Auto);
        let _ = p.push("hello <tool");
        let flushed = p.flush();
        let text: String = flushed
            .iter()
            .filter_map(|e| match e {
                ParserOutput::Text(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        // EOS with unclosed tag: release as literal text.
        assert!(text.contains("<tool"));
    }
}
