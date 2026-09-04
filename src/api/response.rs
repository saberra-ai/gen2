//! [`Response`] — everything one generation produced.

use super::output::{
    AssistantMessage, FinishReason, GenerationStats, OutputPart, ToolCall, ToolCallId, Usage,
    split_reasoning,
};
use super::stream::{Completion, Finish};

/// The outcome of a generation: the structured message, why it ended, and
/// what it cost.
///
/// ```no_run
/// # let model = gen2::load("m.gguf")?;
/// let response = model.generate("Write a haiku").max_tokens(64).run()?;
/// println!("{}", response.text());
/// println!("{:?}", response.usage());
/// # Ok::<(), gen2::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Response {
    /// What the model said, as parts.
    pub message: AssistantMessage,
    /// Why it stopped.
    pub finish_reason: FinishReason,
    /// Token accounting.
    pub usage: Usage,
    /// Timing and throughput.
    pub stats: GenerationStats,
}

impl Response {
    /// The visible reply text — every text part, concatenated. Reasoning and
    /// tool calls are not in it.
    pub fn text(&self) -> String {
        self.message.text()
    }

    /// The reasoning channel, when the model produced one.
    pub fn reasoning(&self) -> Option<String> {
        self.message.reasoning()
    }

    /// The tool calls the model asked for, in order. Empty when it answered.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.message.tool_calls()
    }

    /// Why the generation ended.
    pub fn finish_reason(&self) -> &FinishReason {
        &self.finish_reason
    }

    /// Token accounting.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Timing and throughput.
    pub fn stats(&self) -> &GenerationStats {
        &self.stats
    }

    /// Build from what the engine's turn produced.
    ///
    /// `max_tokens` is the budget the turn ran under, which is how a stream
    /// that ended at the budget is told apart from one the model finished —
    /// the engine reports both as end-of-sequence.
    pub(crate) fn from_completion(done: Completion, max_tokens: Option<usize>) -> Self {
        let stats = GenerationStats::from_reported(done.stats);
        let usage = stats.usage();

        let mut content = split_reasoning(&done.text);
        // An answer that is only tool calls has no text part; an empty text
        // part would suggest the model said something and said nothing.
        if !done.tool_calls.is_empty() {
            content.retain(|p| !matches!(p, OutputPart::Text(t) if t.trim().is_empty()));
        }
        for (i, call) in done.tool_calls.iter().enumerate() {
            content.push(OutputPart::ToolCall(ToolCall {
                id: ToolCallId(call.id.clone().unwrap_or_else(|| format!("call_{i}"))),
                name: call.name.clone(),
                arguments: serde_json::from_str(&call.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(call.arguments.clone())),
            }));
        }

        let finish_reason = if !done.tool_calls.is_empty() {
            FinishReason::ToolCall
        } else {
            match done.finish {
                Finish::Eos => {
                    let at_budget = stats.reported()
                        && max_tokens.is_some_and(|max| usage.completion_tokens as usize >= max);
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

        Self {
            message: AssistantMessage { content },
            finish_reason,
            usage,
            stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::ToolCall as StreamToolCall;
    use crate::types::ExecutionStats;

    fn completion(text: &str) -> Completion {
        Completion {
            text: text.into(),
            ..Default::default()
        }
    }

    #[test]
    fn a_finished_reply_is_stop_with_its_text() {
        let r = Response::from_completion(completion("hello"), Some(64));
        assert_eq!(r.text(), "hello");
        assert_eq!(r.reasoning(), None);
        assert!(r.tool_calls().is_empty());
        assert_eq!(*r.finish_reason(), FinishReason::Stop);
        assert!(!r.stats().reported());
    }

    #[test]
    fn hitting_the_budget_is_length_not_stop() {
        let mut done = completion("and then");
        done.stats = Some(ExecutionStats {
            prompt_tokens: 10,
            decode_tokens: 8,
            ..Default::default()
        });
        let r = Response::from_completion(done.clone(), Some(8));
        assert_eq!(*r.finish_reason(), FinishReason::Length);
        assert_eq!(r.usage().total_tokens(), 18);

        // Same stream under a larger budget: the model chose to stop.
        let r = Response::from_completion(done, Some(64));
        assert_eq!(*r.finish_reason(), FinishReason::Stop);
    }

    #[test]
    fn without_stats_the_budget_cannot_be_claimed_reached() {
        // No token count means no grounds to say the budget was hit; `Stop`
        // is the honest default rather than a guess from text length.
        let r = Response::from_completion(completion("x"), Some(1));
        assert_eq!(*r.finish_reason(), FinishReason::Stop);
    }

    #[test]
    fn a_stop_request_is_cancelled_and_keeps_the_partial_text() {
        let mut done = completion("partial");
        done.finish = Finish::Stopped;
        let r = Response::from_completion(done, None);
        assert_eq!(*r.finish_reason(), FinishReason::Cancelled);
        assert_eq!(r.text(), "partial");
    }

    #[test]
    fn tool_calls_are_structured_and_the_reason_says_so() {
        let mut done = completion("");
        done.tool_calls = vec![
            StreamToolCall {
                id: None,
                name: "get_weather".into(),
                arguments: r#"{"city":"Paris"}"#.into(),
            },
            StreamToolCall {
                id: Some("abc".into()),
                name: "broken".into(),
                arguments: "not json".into(),
            },
        ];
        let r = Response::from_completion(done, None);
        assert_eq!(*r.finish_reason(), FinishReason::ToolCall);
        assert_eq!(
            r.text(),
            "",
            "no text part is invented for a tool-only turn"
        );
        let calls = r.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Paris");
        assert_eq!(calls[0].id.as_str(), "call_0", "a missing id is minted");
        assert_eq!(calls[1].id.as_str(), "abc", "a provider id is kept");
        assert_eq!(
            calls[1].arguments,
            serde_json::Value::String("not json".into()),
            "malformed arguments survive as a string"
        );
    }

    #[test]
    fn reasoning_is_separated_from_the_reply() {
        let r = Response::from_completion(completion("<think>\nweigh it\n</think>\n\nYes."), None);
        assert_eq!(r.text(), "Yes.");
        assert_eq!(r.reasoning().as_deref(), Some("weigh it"));
        assert_eq!(*r.finish_reason(), FinishReason::Stop);
    }
}
