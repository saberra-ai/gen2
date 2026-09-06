//! What a model produced: structured assistant output, usage, and timing.
//!
//! The canonical output is structured (spec §9.3, §14): text, reasoning, and
//! tool calls are separate parts, so a harness never parses model-protocol
//! syntax out of a string to find out what the model asked for.

use std::time::Duration;

use crate::types::ExecutionStats;

/// An assistant turn, as parts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantMessage {
    /// The parts, in the order the model produced them.
    pub content: Vec<OutputPart>,
}

impl AssistantMessage {
    /// Every text part, concatenated. Reasoning and tool calls are excluded.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| match p {
                OutputPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Every reasoning part, concatenated. `None` when the model produced no
    /// reasoning channel.
    pub fn reasoning(&self) -> Option<String> {
        let mut out = String::new();
        let mut any = false;
        for p in &self.content {
            if let OutputPart::Reasoning(r) = p {
                out.push_str(r);
                any = true;
            }
        }
        any.then_some(out)
    }

    /// The tool calls, in order.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.content
            .iter()
            .filter_map(|p| match p {
                OutputPart::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect()
    }
}

/// One piece of assistant output.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OutputPart {
    /// Visible reply text.
    Text(String),
    /// The reasoning channel, when the model exposes one.
    Reasoning(String),
    /// A request to call a tool. The caller runs it, or does not.
    ToolCall(ToolCall),
}

/// A tool the model asked to call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Distinct within a response, so a result can be matched to its call
    /// even when several were asked for at once.
    pub id: ToolCallId,
    /// The tool's name, as declared.
    pub name: String,
    /// The arguments. A string when the model emitted something that was not
    /// JSON, so a malformed call is still visible rather than dropped.
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// The id a result is pushed under —
    /// [`Session::push_tool_result`](crate::Session::push_tool_result).
    pub fn id(&self) -> &ToolCallId {
        &self.id
    }

    /// The tool's name, as declared.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The arguments, as JSON.
    pub fn arguments(&self) -> &serde_json::Value {
        &self.arguments
    }

    /// The arguments, decoded into the tool's argument type.
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> serde_json::Result<T> {
        serde_json::from_value(self.arguments.clone())
    }
}

/// Identifies one tool call within a session.
///
/// Providers that number their calls supply the id; models using native tool
/// syntax do not, and one is minted from the session's message counter
/// (`call_<message>_<n>`), so parallel calls stay distinguishable and no two
/// turns of one session ever reuse an id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolCallId(pub String);

impl ToolCallId {
    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ToolCallId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why a generation ended.
///
/// Backend-specific reasons are normalised where possible; what could not be
/// normalised arrives as [`FinishReason::Other`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// The model finished its reply.
    Stop,
    /// The token budget ran out first.
    Length,
    /// The model asked for one or more tools and is waiting on the results.
    ToolCall,
    /// Stopped on request. Whatever was produced before is kept.
    Cancelled,
    /// The provider withheld the output.
    ContentFilter,
    /// The generation failed after it had begun.
    Error,
    /// Something this enum has no name for. The payload is the raw reason.
    Other(String),
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stop => f.write_str("stop"),
            Self::Length => f.write_str("length"),
            Self::ToolCall => f.write_str("tool_call"),
            Self::Cancelled => f.write_str("cancelled"),
            Self::ContentFilter => f.write_str("content_filter"),
            Self::Error => f.write_str("error"),
            Self::Other(s) => write!(f, "other({s})"),
        }
    }
}

/// Token accounting for one generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Tokens in the prompt, including anything the backend prepended.
    pub prompt_tokens: u32,
    /// Tokens the model generated.
    pub completion_tokens: u32,
}

impl Usage {
    /// Prompt plus completion.
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Timing and throughput for one generation.
///
/// A view over the engine's [`ExecutionStats`], which is what every backend
/// already reports; nothing is measured twice. Zero everywhere when the
/// backend reported nothing, which [`GenerationStats::reported`] says.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationStats {
    inner: ExecutionStats,
    reported: bool,
}

impl GenerationStats {
    pub(crate) fn from_reported(stats: Option<ExecutionStats>) -> Self {
        match stats {
            Some(inner) => Self {
                inner,
                reported: true,
            },
            None => Self::default(),
        }
    }

    /// Whether the backend reported any statistics at all.
    pub fn reported(&self) -> bool {
        self.reported
    }

    /// Time from the request to the first generated token.
    pub fn time_to_first_token(&self) -> Option<Duration> {
        self.reported
            .then(|| Duration::from_micros(self.inner.first_token_us))
    }

    /// Average decode throughput, tokens per second.
    pub fn tokens_per_second(&self) -> f32 {
        self.inner.avg_tps
    }

    /// Wall-clock time the reasoning channel streamed before visible content,
    /// when the transport measured one.
    pub fn reasoning_time(&self) -> Option<Duration> {
        self.inner.reasoning_ms.map(Duration::from_millis)
    }

    /// Token counts, as [`Usage`].
    pub fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.inner.prompt_tokens,
            completion_tokens: self.inner.decode_tokens,
        }
    }

    /// The engine's own record, for what the accessors above do not cover
    /// (cache occupancy, speculative-decoding counters).
    pub fn execution(&self) -> &ExecutionStats {
        &self.inner
    }
}

impl From<ExecutionStats> for GenerationStats {
    fn from(inner: ExecutionStats) -> Self {
        Self::from_reported(Some(inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_report_nothing_when_the_backend_did_not() {
        let s = GenerationStats::from_reported(None);
        assert!(!s.reported());
        assert_eq!(s.time_to_first_token(), None);
        assert_eq!(s.usage(), Usage::default());
        let s = GenerationStats::from(ExecutionStats {
            prompt_tokens: 3,
            decode_tokens: 2,
            first_token_us: 1500,
            ..Default::default()
        });
        assert!(s.reported());
        assert_eq!(s.usage().total_tokens(), 5);
        assert_eq!(s.time_to_first_token(), Some(Duration::from_micros(1500)));
    }
}
