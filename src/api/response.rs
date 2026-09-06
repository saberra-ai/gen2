//! [`Response`] — everything one generation produced.

use super::output::{AssistantMessage, FinishReason, GenerationStats, ToolCall, Usage};
use super::session::MessageId;

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
    /// The id of the assistant message this turn appended to its session.
    ///
    /// `None` for one-shot generation, whose session is discarded, and for a
    /// cancelled turn that had produced nothing worth recording.
    pub message_id: Option<MessageId>,
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
    ///
    /// Each carries the id to push its result under — the §28.5 loop:
    ///
    /// ```no_run
    /// # let model = gen2::load("m.gguf")?;
    /// # let mut session = gen2::Session::new();
    /// # fn execute(_: &gen2::output::ToolCall) -> gen2::Result<String> { Ok(String::new()) }
    /// let response = model.turn(&mut session).user("What files are here?").run()?;
    /// for call in response.tool_calls() {
    ///     let result = execute(call)?;
    ///     session.push_tool_result(call.id(), result);
    /// }
    /// let response = model.turn(&mut session).run()?;
    /// # Ok::<(), gen2::Error>(())
    /// ```
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

    /// The id of the assistant message recorded in the session, when one
    /// was. A harness that does not want a cancelled turn's partial reply in
    /// future context removes it by this id (api_spec.md §16.1).
    pub fn message_id(&self) -> Option<MessageId> {
        self.message_id
    }

    /// Build from what the event assembler produced.
    pub(crate) fn assembled(
        message: AssistantMessage,
        finish_reason: FinishReason,
        stats: GenerationStats,
        message_id: Option<MessageId>,
    ) -> Self {
        Self {
            usage: stats.usage(),
            message,
            finish_reason,
            stats,
            message_id,
        }
    }

    /// The same response with no session id: one-shot generation throws its
    /// session away, so the id would name nothing a caller can reach.
    pub(crate) fn detached(mut self) -> Self {
        self.message_id = None;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::output::OutputPart;

    #[test]
    fn accessors_read_the_parts() {
        let r = Response::assembled(
            AssistantMessage {
                content: vec![
                    OutputPart::Reasoning("weigh it".into()),
                    OutputPart::Text("Yes.".into()),
                ],
            },
            FinishReason::Stop,
            GenerationStats::default(),
            Some(MessageId::for_test(3)),
        );
        assert_eq!(r.text(), "Yes.");
        assert_eq!(r.reasoning().as_deref(), Some("weigh it"));
        assert!(r.tool_calls().is_empty());
        assert_eq!(*r.finish_reason(), FinishReason::Stop);
        assert_eq!(r.message_id(), Some(MessageId::for_test(3)));
        assert!(!r.stats().reported());
        assert_eq!(r.detached().message_id(), None);
    }
}
