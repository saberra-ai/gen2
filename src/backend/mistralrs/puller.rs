//! mistral.rs's response stream, as gen2 [`TokenEvent`]s.
//!
//! The contract this has to keep is the one every backend keeps: exactly one
//! terminal event, nothing after it, and no token lost on the way. The last
//! one has a specific trap — providers routinely put the final token and the
//! finish reason in the same chunk, and a puller that treats the finish reason
//! as "stop now" drops that token. It is owed on the next call instead.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mistralrs::{Response, ToolCallResponse};

use crate::backend::TokenPullerDyn;
use crate::engine::ExecError;
use crate::generation::{Token, TokenEvent, ToolCall};

/// A tool call being assembled from streamed fragments.
#[derive(Default)]
struct PartialCall {
    id: Option<String>,
    name: String,
    arguments: String,
}

/// Generic over the stream rather than tied to `BlockingStream`, whose fields
/// are private. The mapping from responses to [`TokenEvent`]s is where the
/// contract lives — one terminal event, nothing after it, no token lost to a
/// finish reason that arrived beside it — and that is worth testing without a
/// model in the room.
pub(super) struct MistralRsPuller<S: Iterator<Item = Response>> {
    stream: S,
    stopped: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    /// Set once the stream has ended, so nothing is produced afterwards.
    done: bool,
    /// Owed because the chunk that carried the finish reason also carried a
    /// token, which goes out first.
    pending_eos: bool,
    /// Tool calls under construction, keyed by the index the provider gives
    /// them so parallel calls do not merge.
    calls: BTreeMap<usize, PartialCall>,
    /// Calls finished and waiting to be handed over one at a time.
    ready: Vec<ToolCall>,
}

impl<S: Iterator<Item = Response>> MistralRsPuller<S> {
    pub(super) fn new(stream: S, stopped: Arc<AtomicBool>, paused: Arc<AtomicBool>) -> Self {
        Self {
            stream,
            stopped,
            paused,
            done: false,
            pending_eos: false,
            calls: BTreeMap::new(),
            ready: Vec::new(),
        }
    }

    /// End the stream, emitting `event` once and nothing after.
    fn finish(&mut self, event: TokenEvent) -> Option<Result<TokenEvent, ExecError>> {
        self.done = true;
        Some(Ok(event))
    }

    /// Fold a streamed tool-call fragment into whatever is being assembled.
    ///
    /// Providers send a name once and then arguments in pieces, so the parts
    /// are accumulated by index rather than treated as whole calls.
    fn absorb_calls(&mut self, calls: &[ToolCallResponse]) {
        for call in calls {
            let slot = self.calls.entry(call.index).or_default();
            if slot.id.is_none() && !call.id.is_empty() {
                slot.id = Some(call.id.clone());
            }
            if !call.function.name.is_empty() {
                slot.name = call.function.name.clone();
            }
            slot.arguments.push_str(&call.function.arguments);
        }
    }

    /// Everything assembled so far becomes deliverable.
    fn seal_calls(&mut self) {
        for (_, partial) in std::mem::take(&mut self.calls) {
            if partial.name.is_empty() {
                continue;
            }
            self.ready.push(ToolCall {
                id: partial.id,
                name: partial.name,
                arguments: partial.arguments,
            });
        }
    }
}

impl<S: Iterator<Item = Response>> TokenPullerDyn for MistralRsPuller<S> {
    fn next_event(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        if self.done {
            return None;
        }
        // Owed from a chunk that carried both a token and a finish reason.
        // Unconditional: the generation has already ended upstream, and
        // reporting a stop here instead would tell the caller a finished reply
        // was cancelled.
        if self.pending_eos && self.ready.is_empty() {
            return self.finish(TokenEvent::Eos);
        }
        if let Some(call) = (!self.ready.is_empty()).then(|| self.ready.remove(0)) {
            return Some(Ok(TokenEvent::ToolCall(call)));
        }
        if self.stopped.load(Ordering::Acquire) {
            return self.finish(TokenEvent::Stopped);
        }
        if self.paused.load(Ordering::Acquire) {
            return Some(Ok(TokenEvent::Paused));
        }

        loop {
            let Some(response) = self.stream.next() else {
                // The stream ended without saying so. That is still an end,
                // and reporting it as one is what stops a caller waiting
                // forever for a terminal event.
                self.seal_calls();
                if !self.ready.is_empty() {
                    self.pending_eos = true;
                    let call = self.ready.remove(0);
                    return Some(Ok(TokenEvent::ToolCall(call)));
                }
                return self.finish(TokenEvent::Eos);
            };

            match response {
                Response::Chunk(chunk) => {
                    let Some(choice) = chunk.choices.first() else {
                        continue;
                    };
                    if let Some(calls) = &choice.delta.tool_calls {
                        self.absorb_calls(calls);
                    }
                    // Reasoning counts as text here. mistral.rs separates the
                    // channel where llama.cpp leaves `<think>` inline, and gen2
                    // has one text channel — so discarding it would make the
                    // same model produce a different transcript depending on
                    // which backend ran it, which is the divergence the
                    // conformance suite exists to catch. A reasoning model that
                    // thinks for its whole token budget would also return
                    // nothing at all.
                    //
                    // Stripping it stays the caller's job, as it already is on
                    // llama.cpp; see the README's note on reasoning models.
                    let text = match (&choice.delta.content, &choice.delta.reasoning_content) {
                        (Some(c), _) if !c.is_empty() => c.clone(),
                        (_, Some(r)) => r.clone(),
                        _ => String::new(),
                    };
                    let finished = choice.finish_reason.is_some();

                    if finished {
                        self.seal_calls();
                        self.pending_eos = true;
                    }
                    if !text.is_empty() {
                        // The token goes out now; the ending is owed. Ending
                        // here would lose it.
                        return Some(Ok(TokenEvent::Token(Token {
                            id: 0,
                            text,
                            logprob: None,
                        })));
                    }
                    if finished {
                        if let Some(call) = (!self.ready.is_empty()).then(|| self.ready.remove(0)) {
                            return Some(Ok(TokenEvent::ToolCall(call)));
                        }
                        return self.finish(TokenEvent::Eos);
                    }
                }
                Response::Done(_) | Response::CompletionDone(_) => {
                    self.seal_calls();
                    if let Some(call) = (!self.ready.is_empty()).then(|| self.ready.remove(0)) {
                        self.pending_eos = true;
                        return Some(Ok(TokenEvent::ToolCall(call)));
                    }
                    return self.finish(TokenEvent::Eos);
                }
                Response::InternalError(e) | Response::ValidationError(e) => {
                    self.done = true;
                    return Some(Err(ExecError::Other(anyhow::anyhow!("mistral.rs: {e}"))));
                }
                Response::ModelError(message, _) | Response::CompletionModelError(message, _) => {
                    self.done = true;
                    return Some(Err(ExecError::Generation(message)));
                }
                // Image, speech and raw responses cannot arise from a chat
                // request, and a backend that started returning them would be
                // doing something this layer has no mapping for.
                _ => continue,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs::{ChatCompletionChunkResponse, ChunkChoice, Delta};

    /// A chunk carrying `text`, and optionally the finish reason that ends the
    /// stream.
    fn chunk(text: &str, finish: Option<&str>) -> Response {
        chunk_with(text, finish, None)
    }

    fn chunk_with(
        text: &str,
        finish: Option<&str>,
        calls: Option<Vec<ToolCallResponse>>,
    ) -> Response {
        Response::Chunk(ChatCompletionChunkResponse {
            id: "test".into(),
            choices: vec![ChunkChoice {
                finish_reason: finish.map(str::to_string),
                index: 0,
                delta: Delta {
                    content: (!text.is_empty()).then(|| text.to_string()),
                    role: "assistant".into(),
                    tool_calls: calls,
                    reasoning_content: None,
                },
                logprobs: None,
            }],
            created: 0,
            model: "test".into(),
            system_fingerprint: String::new(),
            object: "chat.completion.chunk".into(),
            usage: None,
        })
    }

    fn call(index: usize, name: &str, arguments: &str) -> ToolCallResponse {
        ToolCallResponse {
            index,
            id: format!("call-{name}"),
            tp: mistralrs::ToolCallType::Function,
            function: mistralrs::CalledFunction {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    /// The smallest `ChatCompletionResponse` that satisfies the type; a model
    /// error carries one and nothing here reads it.
    fn empty_completion() -> mistralrs::ChatCompletionResponse {
        mistralrs::ChatCompletionResponse {
            id: String::new(),
            choices: Vec::new(),
            created: 0,
            model: String::new(),
            system_fingerprint: String::new(),
            object: String::new(),
            usage: mistralrs::Usage {
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
        }
    }

    fn drive(responses: Vec<Response>) -> Vec<String> {
        let mut puller = MistralRsPuller::new(
            responses.into_iter(),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        );
        let mut trace = Vec::new();
        // Bounded: a puller that never ends should fail the test rather than
        // hang it.
        for _ in 0..64 {
            match puller.next_event() {
                None => break,
                Some(Ok(TokenEvent::Token(t))) => trace.push(format!("token:{}", t.text)),
                Some(Ok(TokenEvent::ToolCall(c))) => {
                    trace.push(format!("call:{}({})", c.name, c.arguments))
                }
                Some(Ok(TokenEvent::Eos)) => trace.push("eos".into()),
                Some(Ok(TokenEvent::Stopped)) => trace.push("stopped".into()),
                Some(Ok(TokenEvent::Paused)) => trace.push("paused".into()),
                Some(Ok(other)) => trace.push(format!("{other:?}")),
                Some(Err(e)) => {
                    trace.push(format!("err:{e:?}"));
                    break;
                }
            }
        }
        trace
    }

    #[test]
    fn content_becomes_tokens_and_the_stream_ends_once() {
        assert_eq!(
            drive(vec![chunk("hel", None), chunk("lo", Some("stop"))]),
            vec!["token:hel", "token:lo", "eos"]
        );
    }

    #[test]
    fn a_token_arriving_with_its_finish_reason_is_not_lost() {
        // The trap every streaming provider sets: the last token and the
        // finish reason share a chunk. Ending on the finish reason drops the
        // token, and the caller never sees the end of the reply.
        assert_eq!(
            drive(vec![chunk("only", Some("stop"))]),
            vec!["token:only", "eos"]
        );
    }

    #[test]
    fn a_finish_reason_alone_just_ends_the_stream() {
        assert_eq!(
            drive(vec![chunk("hi", None), chunk("", Some("length"))]),
            vec!["token:hi", "eos"]
        );
    }

    #[test]
    fn a_stream_that_stops_without_saying_so_still_ends() {
        // No finish reason, no error, the responses simply run out. A caller
        // waiting for a terminal event must not wait forever.
        assert_eq!(
            drive(vec![chunk("truncated", None)]),
            vec!["token:truncated", "eos"]
        );
    }

    #[test]
    fn nothing_follows_the_end() {
        let trace = drive(vec![chunk("a", Some("stop"))]);
        let ended = trace.iter().position(|e| e == "eos").expect("should end");
        assert_eq!(
            ended,
            trace.len() - 1,
            "{} events arrived after the end: {trace:?}",
            trace.len() - 1 - ended
        );
    }

    #[test]
    fn a_tool_call_is_reassembled_from_its_fragments() {
        // Providers send a name once and then arguments in pieces.
        let trace = drive(vec![
            chunk_with("", None, Some(vec![call(0, "get_weather", "{\"city\":")])),
            chunk_with("", None, Some(vec![call(0, "", "\"Paris\"}")])),
            chunk("", Some("tool_calls")),
        ]);
        assert_eq!(
            trace,
            vec!["call:get_weather({\"city\":\"Paris\"})", "eos"],
            "the fragments did not reassemble: {trace:?}"
        );
    }

    #[test]
    fn parallel_tool_calls_stay_separate() {
        let trace = drive(vec![
            chunk_with(
                "",
                None,
                Some(vec![call(0, "first", "{}"), call(1, "second", "{}")]),
            ),
            chunk("", Some("tool_calls")),
        ]);
        assert_eq!(trace, vec!["call:first({})", "call:second({})", "eos"]);
    }

    #[test]
    fn a_model_error_ends_the_stream_as_an_error() {
        let trace = drive(vec![
            chunk("partial", None),
            Response::ModelError("it fell over".into(), empty_completion()),
        ]);
        assert_eq!(trace.first().map(String::as_str), Some("token:partial"));
        assert!(
            trace.last().is_some_and(|t| t.starts_with("err")),
            "an upstream failure must surface as an error: {trace:?}"
        );
        assert!(!trace.iter().any(|t| t == "eos"));
    }

    #[test]
    fn stopping_ends_the_stream_once() {
        let stopped = Arc::new(AtomicBool::new(true));
        let mut puller = MistralRsPuller::new(
            vec![chunk("never seen", None)].into_iter(),
            Arc::clone(&stopped),
            Arc::new(AtomicBool::new(false)),
        );
        assert!(matches!(puller.next_event(), Some(Ok(TokenEvent::Stopped))));
        assert!(
            puller.next_event().is_none(),
            "a stopped stream must produce nothing further"
        );
    }

    #[test]
    fn pausing_reports_paused_without_ending() {
        let paused = Arc::new(AtomicBool::new(true));
        let mut puller = MistralRsPuller::new(
            vec![chunk("later", None)].into_iter(),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&paused),
        );
        assert!(matches!(puller.next_event(), Some(Ok(TokenEvent::Paused))));
        paused.store(false, Ordering::Release);
        assert!(matches!(
            puller.next_event(),
            Some(Ok(TokenEvent::Token(_)))
        ));
    }
}
