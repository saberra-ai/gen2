//! Generic context truncation driver.
//!
//! Lifts cold-start + warm-start truncation out of the llama backend so every
//! `LocalBackend` with a `SessionTokenizer` gets it uniformly.
//!
//! Kept semantics byte-identical to the original llama inline code (see
//! [`crate::backend::llama`] pre-refactor history):
//! - **Cold-start**: batch estimate via avg tokens/msg, then iterative per-msg
//!   drops until fits or only 2 messages remain. System message preserved.
//! - **Warm-start**: algorithmic compaction loop via
//!   [`crate::app::chat::compaction::compact_algorithmic`], then fallback
//!   oldest-non-system drops. KV cache reset is the caller's responsibility.

use std::sync::Arc;

use crate::backend::traits::SessionTokenizer;
use crate::engine::{ExecError, Settings};
use crate::journal::round_len;
use crate::session_rt::prompt::generation_reserve;
use crate::types::message::Message;

/// Cold-start (new session) truncation.
pub struct ColdStart;

/// Warm-start (existing session, new turn) truncation.
pub struct WarmStart;

/// Result of applying truncation. `messages` is the trimmed list; `dropped` is
/// the count of messages removed.
#[derive(Debug, Clone)]
pub struct TruncationOutcome {
    pub messages: Vec<Message>,
    pub dropped: usize,
}

impl ColdStart {
    /// Apply cold-start truncation. Lifted from llama/session.rs:411-461.
    ///
    /// Inputs:
    /// - `tokenizer`: session's tokenizer (chat template baked in).
    /// - `settings`: used for `stopping.max_tokens` → generation reserve.
    /// - `ctx_size`: context window size in tokens (from backend `n_ctx` or
    ///   settings override).
    /// - `messages`: the initial message list (may include system message).
    ///
    /// Semantics:
    /// - Computes `ctx_limit = ctx_size - generation_reserve(...)`.
    /// - If current tokens fit or fewer than 3 messages, returns unchanged.
    /// - Otherwise: batch estimate `est_remove`, drop from oldest non-system,
    ///   then re-tokenize + iterate per-msg until fit.
    pub fn apply(
        tokenizer: Arc<dyn SessionTokenizer>,
        settings: &Settings,
        ctx_size: usize,
        mut messages: Vec<Message>,
    ) -> Result<TruncationOutcome, ExecError> {
        let original = messages.len();
        let gen_reserve = generation_reserve(ctx_size, settings.stopping.max_tokens);
        let ctx_limit = ctx_size.saturating_sub(gen_reserve);

        let mut token_count = tokenizer.count_tokens(&messages)?;

        if token_count > ctx_limit && messages.len() > 2 {
            tracing::warn!(
                "initial context overflow: {} tokens > {} limit, truncating conversation",
                token_count,
                ctx_limit
            );

            // Estimate how many messages to batch-remove based on avg tokens/msg.
            let first_is_system = messages
                .first()
                .map(|m| m.role == "system")
                .unwrap_or(false);
            let removable = if first_is_system {
                messages.len() - 2
            } else {
                messages.len() - 1
            };
            let avg_per_msg = token_count / messages.len().max(1);
            let excess = token_count.saturating_sub(ctx_limit);
            let est_remove = (excess / avg_per_msg.max(1)).min(removable);
            let remove_idx = if first_is_system { 1 } else { 0 };

            let mut removed = 0usize;
            while removed < est_remove && messages.len() > 2 {
                removed += drop_oldest_round(&mut messages, remove_idx);
            }

            token_count = tokenizer.count_tokens(&messages)?;
            while token_count > ctx_limit && messages.len() > 2 {
                if drop_oldest_round(&mut messages, remove_idx) == 0 {
                    break;
                }
                token_count = tokenizer.count_tokens(&messages)?;
            }
        }

        Ok(TruncationOutcome {
            dropped: original.saturating_sub(messages.len()),
            messages,
        })
    }
}

impl WarmStart {
    /// Apply warm-start truncation. Lifted from llama/session.rs:741-777
    /// (Phase 1 algorithmic compaction + Phase 2 fallback drop; Phase 3
    /// validation is folded in). Caller is responsible for clearing KV cache
    /// and re-prefilling after this returns.
    pub fn apply(
        tokenizer: Arc<dyn SessionTokenizer>,
        settings: &Settings,
        ctx_size: usize,
        messages: Vec<Message>,
    ) -> Result<TruncationOutcome, ExecError> {
        let gen_reserve = generation_reserve(ctx_size, settings.stopping.max_tokens);
        let ctx_limit = ctx_size.saturating_sub(gen_reserve);
        let keep_recent = super::compaction::DEFAULT_KEEP_RECENT;

        let mut working = messages;
        let mut dropped = 0_usize;

        // Phase 1: algorithmic compaction until under limit or no further compaction.
        loop {
            let n = tokenizer.count_tokens(&working)?;
            if n <= ctx_limit {
                break;
            }
            let cr = super::compaction::compact_algorithmic(working, keep_recent);
            working = cr.messages;
            if cr.compacted_count == 0 {
                break;
            }
            tracing::info!(
                compacted = cr.compacted_count,
                "warm-session context overflow: applied algorithmic compaction"
            );
            dropped = dropped.saturating_add(cr.compacted_count);
        }

        // Phase 2: drop oldest non-system messages until conversation fits (fallback).
        while working.len() > 2 {
            let n = tokenizer.count_tokens(&working)?;
            if n <= ctx_limit {
                break;
            }
            let first_is_system = working.first().map(|m| m.role == "system").unwrap_or(false);
            let remove_idx = if first_is_system { 1 } else { 0 };
            let removed = drop_oldest_round(&mut working, remove_idx);
            if removed == 0 {
                break;
            }
            dropped += removed;
        }

        // Phase 3: validate it actually fits.
        let final_count = tokenizer.count_tokens(&working)?;
        if final_count > ctx_limit {
            return Err(ExecError::Other(anyhow::anyhow!(
                "conversation too long even after compaction (ctx_size={})",
                ctx_size
            )));
        }

        Ok(TruncationOutcome {
            messages: working,
            dropped,
        })
    }
}

/// Remove the oldest droppable *round* at `at`, returning how many messages went.
///
/// The unit of removal is a round rather than a message, which is the whole
/// difference between this and what it replaces. Dropping one message at a
/// time separates a tool call from its result — the model is then shown an
/// answer to a question the conversation no longer contains, or a call that
/// never resolves. See [`crate::journal::round_len`].
///
/// Refuses to leave fewer than two messages, matching the existing floor:
/// truncating a conversation down to nothing is not a smaller answer, it is a
/// different and useless one.
fn drop_oldest_round(messages: &mut Vec<Message>, at: usize) -> usize {
    if at >= messages.len() || messages.len() <= 2 {
        return 0;
    }
    let span = round_len(messages, at);
    // All or nothing. Clamping the span to fit the floor would drop *part* of
    // a round, which is the exact failure this function exists to prevent —
    // and a truncation that then fails loudly is a far better outcome than one
    // that succeeds having quietly corrupted the transcript.
    if span == 0 || messages.len().saturating_sub(span) < 2 {
        return 0;
    }
    messages.drain(at..at + span);
    span
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::{MessageBody, MessageContent};
    use std::sync::Arc;

    // Fake tokenizer: N tokens per message, uniform.
    struct UniformTokenizer {
        per_message: usize,
    }

    impl SessionTokenizer for UniformTokenizer {
        fn count_tokens(&self, messages: &[Message]) -> Result<usize, ExecError> {
            Ok(messages.len() * self.per_message)
        }
    }

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.into()),
            },
            name: None,
            tool_call_id: None,
        }
    }

    fn user(text: &str) -> Message {
        msg("user", text)
    }

    fn tool_call(id: &str) -> Message {
        Message::assistant_tool_calls(vec![crate::types::message::ToolCall {
            id: id.into(),
            r#type: "function".into(),
            function: crate::types::message::FunctionDefinition {
                description: None,
                name: "search".into(),
                arguments: serde_json::json!({}),
            },
        }])
    }

    fn tool_result(id: &str) -> Message {
        Message::tool_result_for(id, "found it")
    }

    /// Truncation must not leave half a tool round behind.
    ///
    /// Dropping the assistant turn that asked for a tool while keeping its
    /// result shows the model an answer to a question the conversation no
    /// longer contains — and dropping the result while keeping the call leaves
    /// one that never resolves. Both are states no chat template can render
    /// honestly.
    fn assert_no_half_rounds(messages: &[Message], context: &str) {
        let mut called: Vec<&str> = Vec::new();
        let mut answered: Vec<&str> = Vec::new();
        for message in messages {
            match &message.body {
                MessageBody::Tool { tool_calls } => {
                    called.extend(tool_calls.iter().map(|c| c.id.as_str()));
                }
                _ => {
                    if let Some(id) = message.tool_call_id.as_deref() {
                        answered.push(id);
                    }
                }
            }
        }
        for id in &answered {
            assert!(
                called.contains(id),
                "{context}: the transcript answers call {id} but no longer \
                 contains it; called={called:?} answered={answered:?}"
            );
        }
        for id in &called {
            assert!(
                answered.contains(id),
                "{context}: call {id} was left dangling with no result; \
                 called={called:?} answered={answered:?}"
            );
        }
    }

    /// A cold start that has to trim must trim whole rounds.
    #[test]
    fn cold_start_truncation_never_splits_a_tool_round() {
        let messages = vec![
            system("be helpful"),
            user("one"),
            tool_call("c1"),
            tool_result("c1"),
            user("two"),
            tool_call("c2"),
            tool_result("c2"),
            user("three"),
        ];
        // Tight enough that several messages have to go.
        let outcome = ColdStart::apply(
            Arc::new(UniformTokenizer { per_message: 10 }),
            &Settings::default(),
            60,
            messages,
        )
        .expect("truncation should succeed");
        assert_no_half_rounds(&outcome.messages, "cold start");
    }

    /// And so must a warm one.
    #[test]
    fn warm_start_truncation_never_splits_a_tool_round() {
        let messages = vec![
            system("be helpful"),
            user("one"),
            tool_call("c1"),
            tool_result("c1"),
            user("two"),
            tool_call("c2"),
            tool_result("c2"),
            user("three"),
        ];
        let outcome = WarmStart::apply(
            Arc::new(UniformTokenizer { per_message: 10 }),
            &Settings::default(),
            60,
            messages,
        )
        .expect("truncation should succeed");
        assert_no_half_rounds(&outcome.messages, "warm start");
    }

    /// A round that cannot be dropped whole is not dropped at all.
    ///
    /// Clamping it to the floor would take half a round, which is the failure
    /// being prevented. Failing to fit is the honest outcome.
    #[test]
    fn a_round_that_will_not_fit_the_floor_is_left_alone() {
        // Three messages, and the oldest round is two of them: dropping it
        // whole would leave one, below the floor.
        let mut messages = vec![tool_call("c1"), tool_result("c1"), user("now")];
        let before = messages.clone();
        assert_eq!(drop_oldest_round(&mut messages, 0), 0);
        assert_eq!(messages, before, "nothing should have been taken");

        // With one more message there is room, and it goes whole.
        let mut messages = vec![system("s"), tool_call("c1"), tool_result("c1"), user("now")];
        assert_eq!(drop_oldest_round(&mut messages, 1), 2, "the whole round");
        assert_eq!(messages.len(), 2);
    }

    /// Parallel calls are one round and go together.
    #[test]
    fn parallel_calls_are_dropped_as_one_round() {
        let mut calls = Message::assistant_tool_calls(vec![
            crate::types::message::ToolCall {
                id: "a".into(),
                r#type: "function".into(),
                function: crate::types::message::FunctionDefinition {
                    description: None,
                    name: "x".into(),
                    arguments: serde_json::json!({}),
                },
            },
            crate::types::message::ToolCall {
                id: "b".into(),
                r#type: "function".into(),
                function: crate::types::message::FunctionDefinition {
                    description: None,
                    name: "y".into(),
                    arguments: serde_json::json!({}),
                },
            },
        ]);
        calls.name = None;
        let messages = vec![
            system("s"),
            user("one"),
            calls,
            tool_result("a"),
            tool_result("b"),
            user("two"),
        ];
        for ctx in [30, 40, 50, 60, 70] {
            let outcome = ColdStart::apply(
                Arc::new(UniformTokenizer { per_message: 10 }),
                &Settings::default(),
                ctx,
                messages.clone(),
            )
            .expect("truncation should succeed");
            assert_no_half_rounds(&outcome.messages, &format!("ctx={ctx}"));
        }
    }
    fn system(text: &str) -> Message {
        msg("system", text)
    }

    fn settings_with_max(max: Option<usize>) -> Settings {
        let mut s = Settings::default();
        s.stopping.max_tokens = max;
        s
    }

    #[test]
    fn cold_start_no_truncate_when_under_limit() {
        let tok = Arc::new(UniformTokenizer { per_message: 10 }) as Arc<dyn SessionTokenizer>;
        let msgs = vec![system("sys"), user("hi"), user("again")];
        let out = ColdStart::apply(tok, &settings_with_max(Some(16)), 1024, msgs.clone()).unwrap();
        assert_eq!(out.dropped, 0);
        assert_eq!(out.messages.len(), 3);
    }

    #[test]
    fn cold_start_bails_at_two_messages() {
        let tok = Arc::new(UniformTokenizer {
            per_message: 100_000,
        }) as Arc<dyn SessionTokenizer>;
        let msgs = vec![system("sys"), user("hi")];
        let out = ColdStart::apply(tok, &settings_with_max(Some(16)), 128, msgs.clone()).unwrap();
        // len==2: loop condition `messages.len() > 2` is false, no drops.
        assert_eq!(out.dropped, 0);
        assert_eq!(out.messages.len(), 2);
    }

    #[test]
    fn cold_start_drops_oldest_non_system() {
        // 10 messages @ 100 tokens each = 1000. ctx_size=500, max_tokens=16
        // → gen_reserve = 125 (ctx_size/4), ctx_limit=375. Needs to fit in 3 messages (~300).
        let tok = Arc::new(UniformTokenizer { per_message: 100 }) as Arc<dyn SessionTokenizer>;
        let mut msgs = vec![system("sys")];
        for i in 0..10 {
            msgs.push(user(&format!("msg{}", i)));
        }
        let out = ColdStart::apply(tok, &settings_with_max(Some(16)), 500, msgs).unwrap();
        assert!(out.dropped > 0);
        // System message preserved.
        assert_eq!(
            out.messages.first().map(|m| m.role.as_str()),
            Some("system")
        );
        // ctx_limit = 500 - generation_reserve(500, Some(16)) = 500 - 64 = 436
        // Must fit: messages.len() * 100 <= 436  →  len <= 4
        assert!(out.messages.len() <= 4);
    }

    #[test]
    fn warm_start_drops_when_over_limit() {
        let tok = Arc::new(UniformTokenizer { per_message: 100 }) as Arc<dyn SessionTokenizer>;
        let mut msgs = vec![system("sys")];
        for i in 0..10 {
            msgs.push(user(&format!("msg{}", i)));
        }
        let out = WarmStart::apply(tok, &settings_with_max(Some(16)), 500, msgs).unwrap();
        assert!(out.dropped > 0);
        // ctx_limit = 500 - 64 = 436, 100 tokens/msg → len <= 4
        assert!(out.messages.len() <= 4);
    }

    #[test]
    fn warm_start_errors_when_cannot_fit() {
        // Even 2 messages blow the budget.
        let tok = Arc::new(UniformTokenizer {
            per_message: 100_000,
        }) as Arc<dyn SessionTokenizer>;
        let msgs = vec![system("sys"), user("hi")];
        let err = WarmStart::apply(tok, &settings_with_max(Some(16)), 128, msgs).unwrap_err();
        assert!(format!("{:?}", err).contains("too long"));
    }
}
