//! Streaming tool-call parser shared across gen2 backends.
//!
//! Modern chat models express tool invocations as inline markers in their
//! generated text. This parser recognises (and where needed *heals*, see
//! `healing.rs`) the formats our curated catalog's models actually emit:
//!
//!   - **Llama 3.x / Qwen3 / Hermes / Gemma-templated**:
//!     `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
//!   - **Gemma 4 native**: `<|tool_call>call:name{key:value}<tool_call|>`
//!     with `<|"|>` quote tokens, bare keys, and bare array elements
//!   - **Qwen3-coder / function-XML**:
//!     `<function=name><parameter=k>v</parameter></function>`
//!   - **Mistral v7+**: `[TOOL_CALLS][{...}{...}]` (comma-less arrays
//!     healed) and the `[TOOL_CALLS]name[CALL_ID]id[ARGS]{json}` form
//!   - **Reasoning-model rehearsal**: `name[ARGS]{json}` — parsed only
//!     when `name` is a caller-enabled tool (see [`ToolCallParser::with_enabled_tools`])
//!
//! Tool-call syntax inside `<think>`/`[THINK]`/Gemma thought-channel
//! blocks is a *rehearsal*, not a call: it passes through as text so the
//! downstream reasoning splitter ([`crate::gen2::generation::reply_parts`])
//! keeps it in the reasoning channel. The marker literals here mirror
//! `reply_parts.rs::ChannelMarkers` — change them together.
//!
//! A backend's token stream is a sequence of small text chunks, so the
//! parser is byte-incremental: feed it chunks, get back zero or more
//! `ParserOutput` events (plain text spans + structured tool calls).
//! Format-repair semantics ported from Unsloth Studio
//! `studio/backend/core/tool_healing.py`.

use std::collections::HashSet;

use super::healing;
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

/// Supported wire protocols. `Auto` (the default) detects every format
/// concurrently — a model may rehearse in one format and call in another,
/// so unlike the pre-healing parser it does NOT lock onto the first
/// format seen. Callers with model-specific knowledge can force one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Protocol {
    /// Detect all formats for the whole stream.
    #[default]
    Auto,
    /// `<tool_call>...</tool_call>` with JSON body only.
    TagJson,
    /// `[TOOL_CALLS]` array/name forms only.
    BracketJson,
    /// Anthropic `<function_calls>` XML (recognised, not parsed).
    AnthropicXml,
    /// Gemma 4 native `<|tool_call>call:name{...}<tool_call|>` only.
    GemmaNative,
    /// `<function=name><parameter=k>v</parameter></function>` only.
    FuncXml,
}

/// Which reasoning-block close marker ends the current think span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkKind {
    /// `<think>` ... `</think>` (Qwen3 / DeepSeek-R1).
    Xml,
    /// `[THINK]` ... `[/THINK]`.
    Bracket,
    /// `<|channel>thought` ... `<channel|>` (Gemma 4).
    Channel,
}

impl ThinkKind {
    fn close(self) -> &'static str {
        match self {
            ThinkKind::Xml => "</think>",
            ThinkKind::Bracket => "[/THINK]",
            ThinkKind::Channel => "<channel|>",
        }
    }
}

const TAG_OPEN: &str = "<tool_call>";
const TAG_CLOSE: &str = "</tool_call>";
const BRACKET_TAG: &str = "[TOOL_CALLS]";
const BRACKET_CLOSER: &str = "[/TOOL_CALLS]";
const GEMMA_OPEN: &str = "<|tool_call>";
const GEMMA_CLOSE: &str = "<tool_call|>";
const FUNC_OPEN: &str = "<function=";
const FUNC_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";
const ARGS_TAG: &str = "[ARGS]";
const CALL_ID_TAG: &str = "[CALL_ID]";
/// Longest name-run held at the buffer tail so a split `name[ARGS]{..}`
/// rehearsal still sees its name. Only applies when rehearsal parsing is
/// enabled; MCP-style dashed names stay under this comfortably.
const MAX_REHEARSAL_NAME_HOLD: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trigger {
    Tag,
    Bracket,
    Gemma,
    Func,
    Rehearsal,
    ThinkOpen(ThinkKind),
}

/// How a completed call was decoded — the honest healing-telemetry
/// label (unsloth-adoption 12). `CleanJson` = the protocol's normal
/// decoder accepted the body without repair. The other variants are the
/// repair-ish paths worth watching per model in the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallDialect {
    CleanJson,
    /// Gemma 4 `call:name{key:value}` decode (quote-token + bare-key
    /// normalisation is this dialect's normal shape — counted separately
    /// so a NON-gemma arch emitting it is visible).
    GemmaDialect,
    /// `[{...}{...}]` multi-call array decoded comma-tolerantly where a
    /// strict parse failed.
    CommalessArray,
}

/// Per-parser running outcome counts. Read at end of generation and
/// surfaced through gen2 telemetry (`HookEvent::ToolCallOutcomes`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ToolCallTally {
    pub clean: u32,
    pub gemma_dialect: u32,
    pub commaless_array: u32,
    /// Call markup that matched a trigger but failed decode and was
    /// released as text — the honest-failure channel.
    pub fell_through: u32,
}

impl ToolCallTally {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

enum Resolution {
    /// Trigger present but the call may still be completing — release
    /// text before `hold_from` and wait for more chunks.
    NeedMore { hold_from: usize },
    /// Not a call after all — release `buf[..emit_end]` as plain text
    /// and rescan from there. `emit_end` is always > 0 (progress).
    /// `malformed` marks bodies that matched a call trigger but failed
    /// decode (counted as fall-through), vs. text that merely resembled
    /// a trigger (not counted).
    NotACall { emit_end: usize, malformed: bool },
    /// Parsed one or more calls: release `buf[..text_end]` as text,
    /// discard `text_end..consumed_end` (the markup), emit the calls.
    Complete {
        text_end: usize,
        consumed_end: usize,
        dialect: CallDialect,
        calls: Vec<ToolCall>,
    },
    /// A reasoning block opened at the marker ending at `marker_end`.
    EnterThink { kind: ThinkKind, marker_end: usize },
}

pub struct ToolCallParser {
    protocol: Protocol,
    /// Rolling text buffer the parser's looking inside. Reset whenever a
    /// clean span is emitted.
    buf: String,
    /// `Some` while inside a reasoning block: tool markers are rehearsal
    /// text until the block's close marker.
    in_think: Option<ThinkKind>,
    /// When `Some`, the bare `name[ARGS]{json}` rehearsal form parses as
    /// a call for enabled names (and ONLY enabled names — prose like
    /// `foo[ARGS]{..}` for an unknown `foo` stays text). `None` disables
    /// the rehearsal form entirely.
    enabled_tools: Option<HashSet<String>>,
    tally: ToolCallTally,
    /// Set during `flush()`: bodies that balanced but never saw their
    /// close tag resolve as calls instead of waiting forever.
    at_eos: bool,
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
            in_think: None,
            enabled_tools: None,
            tally: ToolCallTally::default(),
            at_eos: false,
        }
    }

    /// Enable the name-gated rehearsal form for this set of tool names.
    /// Running outcome counts for this parser (healing telemetry).
    pub fn tally(&self) -> &ToolCallTally {
        &self.tally
    }

    pub fn with_enabled_tools(mut self, tools: HashSet<String>) -> Self {
        self.enabled_tools = Some(tools);
        self
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

    /// Called at end-of-stream (EOS from the model). A call whose body is
    /// balanced but whose close tag never arrived resolves as a call;
    /// anything else partially-buffered is released as plain text so
    /// malformed attempts are never silently lost.
    pub fn flush(&mut self) -> Vec<ParserOutput> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        self.at_eos = true;
        let mut out = Vec::new();
        self.drain_into(&mut out);
        self.at_eos = false;
        if !self.buf.is_empty() {
            let text = std::mem::take(&mut self.buf);
            out.push(ParserOutput::Text(text));
        }
        self.in_think = None;
        out
    }

    /// Currently-configured protocol.
    pub fn protocol(&self) -> Protocol {
        self.protocol
    }

    fn drain_into(&mut self, out: &mut Vec<ParserOutput>) {
        loop {
            // Inside a reasoning block: everything through the close
            // marker is rehearsal text.
            if let Some(kind) = self.in_think {
                let close = kind.close();
                match self.buf.find(close) {
                    Some(at) => {
                        let end = at + close.len();
                        let released: String = self.buf.drain(..end).collect();
                        out.push(ParserOutput::Text(released));
                        self.in_think = None;
                        continue;
                    }
                    None => {
                        let safe = safe_release_end(&self.buf, &[close]);
                        if safe > 0 {
                            let released: String = self.buf.drain(..safe).collect();
                            out.push(ParserOutput::Text(released));
                        }
                        return;
                    }
                }
            }

            let Some((at, trigger)) = self.find_trigger() else {
                let literals = self.hold_literals();
                let mut safe = safe_release_end(&self.buf, &literals);
                // A rehearsal name may precede a split `[ARGS]` marker —
                // hold the trailing name-run too (bounded).
                if self.enabled_tools.is_some() {
                    safe = safe.min(name_hold_start(&self.buf));
                }
                if safe > 0 {
                    let released: String = self.buf.drain(..safe).collect();
                    out.push(ParserOutput::Text(released));
                }
                return;
            };

            match self.resolve(at, trigger) {
                Resolution::NeedMore { hold_from } => {
                    if hold_from > 0 {
                        let released: String = self.buf.drain(..hold_from).collect();
                        out.push(ParserOutput::Text(released));
                    }
                    return;
                }
                Resolution::NotACall {
                    emit_end,
                    malformed,
                } => {
                    debug_assert!(emit_end > 0, "NotACall must make progress");
                    if malformed {
                        self.tally.fell_through += 1;
                        tracing::info!(
                            target: "pio::gen2::tool_healing",
                            snippet = %self.buf[..emit_end.min(self.buf.len()).min(120)].escape_debug(),
                            "tool-call markup fell through as text"
                        );
                    }
                    let released: String = self.buf.drain(..emit_end).collect();
                    out.push(ParserOutput::Text(released));
                    continue;
                }
                Resolution::Complete {
                    dialect,
                    text_end,
                    consumed_end,
                    calls,
                } => {
                    let n = calls.len() as u32;
                    match dialect {
                        CallDialect::CleanJson => self.tally.clean += n,
                        CallDialect::GemmaDialect => self.tally.gemma_dialect += n,
                        CallDialect::CommalessArray => self.tally.commaless_array += n,
                    }
                    if text_end > 0 {
                        let released: String = self.buf.drain(..text_end).collect();
                        out.push(ParserOutput::Text(released));
                    }
                    self.buf.drain(..consumed_end - text_end);
                    out.extend(calls.into_iter().map(ParserOutput::ToolCall));
                    continue;
                }
                Resolution::EnterThink { kind, marker_end } => {
                    let released: String = self.buf.drain(..marker_end).collect();
                    out.push(ParserOutput::Text(released));
                    self.in_think = Some(kind);
                    continue;
                }
            }
        }
    }

    /// Trigger literals active under the current protocol, for
    /// partial-prefix holding at the buffer tail.
    fn hold_literals(&self) -> Vec<&'static str> {
        let mut lits: Vec<&'static str> = vec!["<think>", "[THINK]", "<|channel>thought"];
        match self.protocol {
            Protocol::Auto => {
                lits.extend([TAG_OPEN, BRACKET_TAG, GEMMA_OPEN, FUNC_OPEN, ARGS_TAG]);
            }
            Protocol::TagJson => lits.push(TAG_OPEN),
            Protocol::BracketJson => lits.push(BRACKET_TAG),
            Protocol::AnthropicXml => lits.push("<function_calls>"),
            Protocol::GemmaNative => lits.push(GEMMA_OPEN),
            Protocol::FuncXml => lits.push(FUNC_OPEN),
        }
        lits
    }

    /// Earliest trigger occurrence in the buffer, honouring the protocol.
    fn find_trigger(&self) -> Option<(usize, Trigger)> {
        let mut best: Option<(usize, Trigger)> = None;
        let mut consider = |pos: Option<usize>, t: Trigger| {
            if let Some(p) = pos
                && best.is_none_or(|(bp, _)| p < bp)
            {
                best = Some((p, t));
            }
        };
        // Think openers are recognised in every protocol so rehearsals
        // never execute regardless of forced format.
        consider(self.buf.find("<think>"), Trigger::ThinkOpen(ThinkKind::Xml));
        consider(
            self.buf.find("[THINK]"),
            Trigger::ThinkOpen(ThinkKind::Bracket),
        );
        consider(
            self.buf.find("<|channel>thought"),
            Trigger::ThinkOpen(ThinkKind::Channel),
        );
        let auto = self.protocol == Protocol::Auto;
        if auto || self.protocol == Protocol::TagJson {
            consider(self.buf.find(TAG_OPEN), Trigger::Tag);
        }
        if auto || self.protocol == Protocol::BracketJson {
            consider(self.buf.find(BRACKET_TAG), Trigger::Bracket);
        }
        if auto || self.protocol == Protocol::GemmaNative {
            consider(self.buf.find(GEMMA_OPEN), Trigger::Gemma);
        }
        if auto || self.protocol == Protocol::FuncXml {
            consider(self.buf.find(FUNC_OPEN), Trigger::Func);
        }
        if auto && self.enabled_tools.is_some() {
            consider(self.buf.find(ARGS_TAG), Trigger::Rehearsal);
        }
        best
    }

    fn resolve(&self, at: usize, trigger: Trigger) -> Resolution {
        match trigger {
            Trigger::ThinkOpen(kind) => {
                let open_len = match kind {
                    ThinkKind::Xml => "<think>".len(),
                    ThinkKind::Bracket => "[THINK]".len(),
                    ThinkKind::Channel => "<|channel>thought".len(),
                };
                Resolution::EnterThink {
                    kind,
                    marker_end: at + open_len,
                }
            }
            Trigger::Tag => self.resolve_tag(at),
            Trigger::Bracket => self.resolve_bracket(at),
            Trigger::Gemma => self.resolve_gemma(at),
            Trigger::Func => self.resolve_func(at),
            Trigger::Rehearsal => self.resolve_rehearsal(at),
        }
    }

    /// `<tool_call>{json}</tool_call>` — also unwraps the XML function
    /// form nested inside the tags (`<tool_call><function=..>..`).
    fn resolve_tag(&self, at: usize) -> Resolution {
        let body_start = at + TAG_OPEN.len();
        let Some(rel_close) = self.buf[body_start..].find(TAG_CLOSE) else {
            return Resolution::NeedMore { hold_from: at };
        };
        let body = self.buf[body_start..body_start + rel_close].trim();
        let consumed_end = body_start + rel_close + TAG_CLOSE.len();
        if let Some(tc) = parse_tag_json(body) {
            return Resolution::Complete {
                dialect: CallDialect::CleanJson,
                text_end: at,
                consumed_end,
                calls: vec![tc],
            };
        }
        // Wrapped XML form: delegate the body to the function parser.
        if body.starts_with(FUNC_OPEN)
            && let Some(tc) = parse_wrapped_func(body)
        {
            return Resolution::Complete {
                dialect: CallDialect::CleanJson,
                text_end: at,
                consumed_end,
                calls: vec![tc],
            };
        }
        // Malformed body — emit the whole literal as text so nothing is
        // silently lost.
        Resolution::NotACall {
            emit_end: consumed_end,
            malformed: true,
        }
    }

    /// `[TOOL_CALLS][{...}]` array form and `[TOOL_CALLS]name{...}` name
    /// form (incl. v11 `[CALL_ID]id` / `[ARGS]` metadata).
    fn resolve_bracket(&self, at: usize) -> Resolution {
        let after_tag = at + BRACKET_TAG.len();
        let rest = &self.buf[after_tag..];
        let ws = rest.len() - rest.trim_start().len();
        let body_at = after_tag + ws;
        let Some(&first) = self.buf.as_bytes().get(body_at) else {
            return Resolution::NeedMore { hold_from: at };
        };
        if first == b'[' {
            // Array form. Balanced end, then comma-tolerant decode.
            let Some(close) = healing::balanced_bracket_end(&self.buf, body_at) else {
                return Resolution::NeedMore { hold_from: at };
            };
            let items = healing::decode_array_items(&self.buf[body_at + 1..close]);
            // Strict-parse probe: if serde accepts the whole array the
            // comma-tolerant decode did no repair work.
            let strict_ok =
                serde_json::from_str::<Vec<serde_json::Value>>(&self.buf[body_at..=close]).is_ok();
            let calls: Vec<ToolCall> = items.iter().filter_map(call_from_value).collect();
            let after = match self.consume_bracket_closer(close + 1) {
                Ok(end) => end,
                Err(()) => return Resolution::NeedMore { hold_from: at },
            };
            if calls.is_empty() {
                return Resolution::NotACall {
                    emit_end: after,
                    malformed: true,
                };
            }
            return Resolution::Complete {
                dialect: if strict_ok {
                    CallDialect::CleanJson
                } else {
                    CallDialect::CommalessArray
                },
                text_end: at,
                consumed_end: after,
                calls,
            };
        }
        // Name form: name, optional [CALL_ID]id, optional [ARGS], ws, {json}.
        let name_end = body_at + name_run_len(&self.buf[body_at..]);
        if name_end == body_at {
            // Not a call shape — release the tag and move on.
            return Resolution::NotACall {
                emit_end: after_tag,
                malformed: false,
            };
        }
        if name_end == self.buf.len() {
            return Resolution::NeedMore { hold_from: at };
        }
        let name = self.buf[body_at..name_end].to_string();
        let mut cursor = name_end;
        if self.buf[cursor..].starts_with(CALL_ID_TAG) {
            let id_start = cursor + CALL_ID_TAG.len();
            let run = name_run_len(&self.buf[id_start..]);
            if run == 0 {
                // Tag complete but no id yet: hold if the id may still be
                // arriving; otherwise it's not a call shape.
                return if id_start == self.buf.len() {
                    Resolution::NeedMore { hold_from: at }
                } else {
                    Resolution::NotACall {
                        emit_end: after_tag,
                        malformed: false,
                    }
                };
            }
            cursor = id_start + run;
        } else if could_extend_meta(&self.buf, cursor, CALL_ID_TAG) {
            return Resolution::NeedMore { hold_from: at };
        }
        if self.buf[cursor..].starts_with(ARGS_TAG) {
            cursor += ARGS_TAG.len();
        } else if could_extend_meta(&self.buf, cursor, ARGS_TAG) {
            return Resolution::NeedMore { hold_from: at };
        }
        let rest = &self.buf[cursor..];
        let ws2 = rest.len() - rest.trim_start().len();
        cursor += ws2;
        match self.buf.as_bytes().get(cursor) {
            None => Resolution::NeedMore { hold_from: at },
            Some(b'{') => {
                let Some(brace_end) = healing::balanced_brace_end(&self.buf, cursor, false) else {
                    return Resolution::NeedMore { hold_from: at };
                };
                let body = &self.buf[cursor..=brace_end];
                let Ok(args) = serde_json::from_str::<serde_json::Value>(body) else {
                    return Resolution::NotACall {
                        emit_end: brace_end + 1,
                        malformed: true,
                    };
                };
                if !args.is_object() {
                    return Resolution::NotACall {
                        emit_end: brace_end + 1,
                        malformed: true,
                    };
                }
                let after = match self.consume_bracket_closer(brace_end + 1) {
                    Ok(end) => end,
                    Err(()) => return Resolution::NeedMore { hold_from: at },
                };
                Resolution::Complete {
                    dialect: CallDialect::CleanJson,
                    text_end: at,
                    consumed_end: after,
                    calls: vec![ToolCall {
                        id: None,
                        name,
                        arguments: args.to_string(),
                    }],
                }
            }
            Some(_) => Resolution::NotACall {
                emit_end: after_tag,
                malformed: false,
            },
        }
    }

    /// Consume an optional `\s*[/TOOL_CALLS]` after `from`. `Ok(end)` on
    /// resolution (with or without closer); `Err(())` while the tail
    /// could still be completing the closer and more input is needed.
    fn consume_bracket_closer(&self, from: usize) -> Result<usize, ()> {
        let rest = &self.buf[from..];
        let ws = rest.len() - rest.trim_start().len();
        let tail = &rest[ws..];
        if tail.starts_with(BRACKET_CLOSER) {
            return Ok(from + ws + BRACKET_CLOSER.len());
        }
        if !self.at_eos
            && (tail.is_empty()
                || (tail.len() < BRACKET_CLOSER.len() && BRACKET_CLOSER.starts_with(tail)))
        {
            return Err(());
        }
        Ok(from)
    }

    /// Gemma 4 native `<|tool_call>call:name{...}<tool_call|>` with
    /// quote-token / bare-key healing.
    fn resolve_gemma(&self, at: usize) -> Resolution {
        let after_tag = at + GEMMA_OPEN.len();
        let head = &self.buf[after_tag..];
        // Expect `\s*call\s*:\s*` then the name.
        let mut cursor = after_tag + (head.len() - head.trim_start().len());
        let need_more = Resolution::NeedMore { hold_from: at };
        let not_a_call = Resolution::NotACall {
            emit_end: after_tag,
            malformed: false,
        };
        if !self.buf[cursor..].starts_with("call") {
            let seen = &self.buf[cursor..];
            return if "call".starts_with(seen) {
                need_more
            } else {
                not_a_call
            };
        }
        cursor += "call".len();
        let rest = &self.buf[cursor..];
        cursor += rest.len() - rest.trim_start().len();
        match self.buf.as_bytes().get(cursor) {
            None => return need_more,
            Some(b':') => cursor += 1,
            Some(_) => return not_a_call,
        }
        let rest = &self.buf[cursor..];
        cursor += rest.len() - rest.trim_start().len();
        let name_end = cursor + gemma_name_run_len(&self.buf[cursor..]);
        if name_end == cursor {
            return if cursor == self.buf.len() {
                need_more
            } else {
                not_a_call
            };
        }
        if name_end == self.buf.len() {
            return need_more;
        }
        let name = self.buf[cursor..name_end].to_string();
        let rest = &self.buf[name_end..];
        let brace_at = name_end + (rest.len() - rest.trim_start().len());
        match self.buf.as_bytes().get(brace_at) {
            None => return need_more,
            Some(b'{') => {}
            Some(_) => return not_a_call,
        }
        let Some(brace_end) = healing::balanced_brace_end(&self.buf, brace_at, true) else {
            return need_more;
        };
        // Optional `\s*<tool_call|>` close.
        let mut consumed_end = brace_end + 1;
        let rest = &self.buf[consumed_end..];
        let ws = rest.len() - rest.trim_start().len();
        let tail = &rest[ws..];
        if tail.starts_with(GEMMA_CLOSE) {
            consumed_end += ws + GEMMA_CLOSE.len();
        } else if !self.at_eos
            && (tail.is_empty()
                || (tail.len() < GEMMA_CLOSE.len() && GEMMA_CLOSE.starts_with(tail)))
        {
            return need_more;
        }
        let Some(args) = healing::gemma_arguments_to_json(&self.buf[brace_at + 1..brace_end])
        else {
            return Resolution::NotACall {
                emit_end: consumed_end,
                malformed: true,
            };
        };
        Resolution::Complete {
            dialect: CallDialect::GemmaDialect,
            text_end: at,
            consumed_end,
            calls: vec![ToolCall {
                id: None,
                name,
                arguments: args.to_string(),
            }],
        }
    }

    /// `<function=name><parameter=k>v</parameter>...</function>`.
    fn resolve_func(&self, at: usize) -> Resolution {
        let name_start = at + FUNC_OPEN.len();
        let name_end = name_start + name_run_len(&self.buf[name_start..]);
        match self.buf.as_bytes().get(name_end) {
            None => return Resolution::NeedMore { hold_from: at },
            Some(b'>') if name_end > name_start => {}
            Some(_) => {
                return Resolution::NotACall {
                    emit_end: name_start,
                    malformed: false,
                };
            }
        }
        let body_start = name_end + 1;
        let Some(close_rel) = func_close_index(&self.buf, body_start, self.at_eos) else {
            return Resolution::NeedMore { hold_from: at };
        };
        let body = &self.buf[body_start..body_start + close_rel];
        let consumed_end = body_start + close_rel + FUNC_CLOSE.len();
        let name = self.buf[name_start..name_end].to_string();
        match parse_parameters(body) {
            Some(args) => Resolution::Complete {
                dialect: CallDialect::CleanJson,
                text_end: at,
                consumed_end,
                calls: vec![ToolCall {
                    id: None,
                    name,
                    arguments: args.to_string(),
                }],
            },
            None => Resolution::NotACall {
                emit_end: consumed_end,
                malformed: false,
            },
        }
    }

    /// Bare `name[ARGS]{json}` rehearsal — a call ONLY when `name` is an
    /// enabled tool; prose mentioning `foo[ARGS]{..}` stays text.
    fn resolve_rehearsal(&self, at: usize) -> Resolution {
        let args_tag_end = at + ARGS_TAG.len();
        // Name is the word-run immediately before the marker; a
        // preceding [CALL_ID] means this [ARGS] is v11 metadata already
        // handled by the bracket resolver.
        let name_start = self.buf[..at]
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            .map(|p| p + 1)
            .unwrap_or(0);
        let name = &self.buf[name_start..at];
        let preceded_by_call_id = self.buf[..name_start].ends_with(CALL_ID_TAG);
        let enabled = self
            .enabled_tools
            .as_ref()
            .is_some_and(|set| set.contains(name));
        if name.is_empty() || preceded_by_call_id || !enabled {
            return Resolution::NotACall {
                emit_end: args_tag_end,
                malformed: false,
            };
        }
        let rest = &self.buf[args_tag_end..];
        let ws = rest.len() - rest.trim_start().len();
        let brace_at = args_tag_end + ws;
        match self.buf.as_bytes().get(brace_at) {
            None => Resolution::NeedMore {
                hold_from: name_start,
            },
            Some(b'{') => {
                let Some(brace_end) = healing::balanced_brace_end(&self.buf, brace_at, false)
                else {
                    return Resolution::NeedMore {
                        hold_from: name_start,
                    };
                };
                let body = &self.buf[brace_at..=brace_end];
                match serde_json::from_str::<serde_json::Value>(body) {
                    Ok(args) if args.is_object() => Resolution::Complete {
                        dialect: CallDialect::CleanJson,
                        text_end: name_start,
                        consumed_end: brace_end + 1,
                        calls: vec![ToolCall {
                            id: None,
                            name: name.to_string(),
                            arguments: args.to_string(),
                        }],
                    },
                    _ => Resolution::NotACall {
                        emit_end: brace_end + 1,
                        malformed: false,
                    },
                }
            }
            Some(_) => Resolution::NotACall {
                emit_end: args_tag_end,
                malformed: false,
            },
        }
    }
}

/// Length of the leading `[\w-]+` run (ASCII identifier + dash).
fn name_run_len(s: &str) -> usize {
    s.bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
        .count()
}

/// Gemma tool names also allow dots (`[\w.\-]+`).
fn gemma_name_run_len(s: &str) -> usize {
    s.bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-' || *b == b'.')
        .count()
}

/// True when the tail from `cursor` is a proper prefix of `tag` — more
/// input could still turn it into that metadata marker.
fn could_extend_meta(buf: &str, cursor: usize, tag: &str) -> bool {
    let tail = &buf[cursor..];
    !tail.is_empty() && tail.len() < tag.len() && tag.starts_with(tail)
}

/// Rehearsal support: earliest start of a trailing name-run that could
/// precede a split `[ARGS]` marker (bounded hold).
fn name_hold_start(buf: &str) -> usize {
    let bytes = buf.as_bytes();
    let mut start = buf.len();
    while start > 0
        && buf.len() - start < MAX_REHEARSAL_NAME_HOLD
        && (bytes[start - 1].is_ascii_alphanumeric() || matches!(bytes[start - 1], b'_' | b'-'))
    {
        start -= 1;
    }
    start
}

/// Return how many bytes from the start of `buf` are safe to release as
/// plain text without swallowing a partial marker at the tail: the
/// earliest position where the buffer's suffix is a prefix of any
/// candidate literal.
fn safe_release_end(buf: &str, literals: &[&str]) -> usize {
    if buf.is_empty() {
        return 0;
    }
    let mut best: Option<usize> = None;
    for c in literals {
        let max = (c.len() - 1).min(buf.len());
        for k in (1..=max).rev() {
            if !buf.is_char_boundary(buf.len() - k) {
                continue;
            }
            if c.starts_with(&buf[buf.len() - k..]) {
                let start = buf.len() - k;
                best = Some(best.map_or(start, |b: usize| b.min(start)));
                break;
            }
        }
    }
    match best {
        Some(at) => at,
        None => {
            let mut n = buf.len();
            while n > 0 && !buf.is_char_boundary(n) {
                n -= 1;
            }
            n
        }
    }
}

fn parse_tag_json(body: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    // `parameters` covers Llama-3.2 drift inside Hermes templates;
    // `args` covers models that shorten the key.
    let args = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .or_else(|| v.get("args"))
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

/// One structured call from a decoded bracket-array element. Non-object
/// `arguments` (e.g. `null`) become `{}` rather than a bogus string.
fn call_from_value(v: &serde_json::Value) -> Option<ToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    let args = match v.get("arguments").or_else(|| v.get("parameters")) {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(a) if a.is_object() => a.to_string(),
        _ => "{}".to_string(),
    };
    let id = v.get("id").and_then(|i| i.as_str().map(|s| s.to_string()));
    Some(ToolCall {
        id,
        name,
        arguments: args,
    })
}

/// Parse a complete `<function=name>...</function>` (or close-less)
/// span that arrived wrapped in `<tool_call>` tags.
fn parse_wrapped_func(body: &str) -> Option<ToolCall> {
    let name_start = FUNC_OPEN.len();
    let name_len = name_run_len(&body[name_start..]);
    if name_len == 0 || body.as_bytes().get(name_start + name_len) != Some(&b'>') {
        return None;
    }
    let name = &body[name_start..name_start + name_len];
    let inner_start = name_start + name_len + 1;
    let inner_end = match func_close_index(body, inner_start, true) {
        Some(rel) => inner_start + rel,
        None => body.len(),
    };
    let args = parse_parameters(&body[inner_start..inner_end])?;
    Some(ToolCall {
        id: None,
        name: name.to_string(),
        arguments: args.to_string(),
    })
}

/// Body-relative index of the first `</function>` that is not argument
/// data (i.e. not inside an open `<parameter=...>` value), or `None`.
fn func_close_index(buf: &str, body_start: usize, at_eos: bool) -> Option<usize> {
    let body = &buf[body_start..];
    let mut search_from = 0;
    while let Some(rel) = body[search_from..].find(FUNC_CLOSE) {
        let idx = search_from + rel;
        if !inside_open_parameter(body, idx, at_eos) {
            return Some(idx);
        }
        search_from = idx + 1;
    }
    None
}

/// True when `pos` (body-relative) falls inside an unclosed parameter
/// value. The parameter's OWN close tag decides: if it closes after
/// `pos`, the position is argument data. Mid-stream (`!at_eos`) a
/// parameter with no close tag yet defers the decision — the close may
/// simply not have arrived, and accepting a `</function>` early would
/// truncate a value that legitimately contains that literal.
fn inside_open_parameter(body: &str, pos: usize, at_eos: bool) -> bool {
    let mut last_param_start = None;
    let mut from = 0;
    while let Some(rel) = body[from..pos].find(PARAM_OPEN) {
        last_param_start = Some(from + rel);
        from = from + rel + PARAM_OPEN.len();
        if from >= pos {
            break;
        }
    }
    let Some(param_start) = last_param_start else {
        return false;
    };
    match body[param_start..].find(PARAM_CLOSE) {
        Some(rel) => param_start + rel > pos,
        None if !at_eos => true,
        None => match body[param_start..].find(FUNC_CLOSE) {
            None => true,
            Some(rel) => pos < param_start + rel,
        },
    }
}

/// Parse `<parameter=k>value</parameter>` pairs into a JSON object. The
/// single wrapping newline the chat template adds around each value is
/// trimmed; interior indentation (code/diff arguments) is preserved.
fn parse_parameters(body: &str) -> Option<serde_json::Value> {
    let mut args = serde_json::Map::new();
    let mut cursor = 0;
    while let Some(rel) = body[cursor..].find(PARAM_OPEN) {
        let name_start = cursor + rel + PARAM_OPEN.len();
        let name_len = name_run_len(&body[name_start..]);
        if name_len == 0 || body.as_bytes().get(name_start + name_len) != Some(&b'>') {
            return None;
        }
        let name = &body[name_start..name_start + name_len];
        let mut val_start = name_start + name_len + 1;
        // Skip horizontal whitespace after the tag (keep a newline for
        // the wrapping-newline trim below).
        while body
            .as_bytes()
            .get(val_start)
            .is_some_and(|b| *b == b' ' || *b == b'\t')
        {
            val_start += 1;
        }
        let val_end_rel = body[val_start..].find(PARAM_OPEN);
        let val_end = val_start + val_end_rel.unwrap_or(body.len() - val_start);
        let mut val = &body[val_start..val_end];
        val = val.trim_end();
        val = val.strip_suffix(PARAM_CLOSE).unwrap_or(val);
        let val = trim_param_value(val);
        args.insert(name.to_string(), serde_json::Value::String(val.to_string()));
        cursor = val_end;
    }
    Some(serde_json::Value::Object(args))
}

/// Trim the single wrapping newline the chat template adds around an XML
/// parameter value (a full trim would destroy code/diff indentation).
fn trim_param_value(val: &str) -> &str {
    let val = val.strip_prefix('\n').unwrap_or(val);
    let val = val.strip_suffix('\n').unwrap_or(val);
    // The close-tag strip above may leave trailing spaces from `\s*</parameter>`.
    val.trim_end_matches([' ', '\t'])
}

#[cfg(test)]
mod tests;
