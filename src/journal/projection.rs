//! Turning a journal into what the model actually sees.
//!
//! The journal is everything that happened. The transcript is a *view* of it,
//! chosen to fit a context window. Keeping those separate is what lets an
//! agent accumulate ten thousand heartbeats without showing the model ten
//! thousand messages — and what lets the view be recomputed differently later
//! without rewriting history.
//!
//! Every projection here selects [`Turn`]s, never records, so none of them can
//! separate a tool call from its result. See [`super::turn`] for why that is a
//! type-level property rather than a rule each projection remembers.

use super::entry::Record;
use super::turn::Turn;
use crate::types::message::Message;

/// Chooses what the model sees.
pub trait Projection {
    /// Which of these turns to keep, oldest first.
    ///
    /// Takes turns rather than records deliberately — see the module docs.
    /// Implementations return the turns they want, in order; dropping is
    /// simply not returning one.
    fn keep<'t>(&self, turns: &'t [Turn]) -> Vec<&'t Turn>;

    /// The transcript this projection produces from a journal.
    ///
    /// Provided, and not worth overriding: grouping and rendering are the same
    /// for everyone, and the only decision is which turns survive.
    fn project(&self, records: &[Record]) -> Vec<Message> {
        let turns = Turn::group(records);
        self.keep(&turns)
            .into_iter()
            .flat_map(Turn::to_messages)
            .collect()
    }
}

/// Keep the most recent turns that fit a token budget.
///
/// The default, and the one an agent runs with unless it is told otherwise.
/// Recency is the right default because it is the only one that is correct
/// without understanding the content: a summary needs a model, and a relevance
/// ranking needs an embedder, and neither should be required to show a model
/// its own last three messages.
#[derive(Debug, Clone, Copy)]
pub struct RecentTurns {
    /// Roughly how many tokens of transcript to produce.
    pub budget: usize,
}

impl RecentTurns {
    pub fn new(budget: usize) -> Self {
        Self { budget }
    }
}

impl Projection for RecentTurns {
    fn keep<'t>(&self, turns: &'t [Turn]) -> Vec<&'t Turn> {
        let mut kept: Vec<&Turn> = Vec::new();
        let mut spent = 0usize;

        // Backwards, because the budget is spent on the newest first — then
        // reversed, because a transcript reads forwards.
        for turn in turns.iter().rev() {
            if !turn.is_conversational() {
                continue;
            }
            let cost = turn.estimated_tokens();
            // A single turn larger than the whole budget is kept anyway when
            // nothing else has been: returning an empty transcript because one
            // tool dumped a large file would leave the model with no idea what
            // it was doing, which is worse than being over budget.
            if spent + cost > self.budget && !kept.is_empty() {
                break;
            }
            spent += cost;
            kept.push(turn);
        }
        kept.reverse();
        kept
    }
}

/// Keep everything.
///
/// For short-lived agents and for tests that want to assert on the whole
/// history without a budget getting in the way.
#[derive(Debug, Clone, Copy, Default)]
pub struct Everything;

impl Projection for Everything {
    fn keep<'t>(&self, turns: &'t [Turn]) -> Vec<&'t Turn> {
        turns.iter().filter(|t| t.is_conversational()).collect()
    }
}

/// Put a fixed instruction in front of whatever `inner` produces.
///
/// The system prompt is not journal history — it is configuration, and an
/// agent that re-derived it from its own log would lose it the first time the
/// budget bit. This is also where a wake's self-prompt belongs: ephemeral,
/// prepended for one inference, never written down as a conversation turn.
pub struct WithPreamble<P> {
    pub preamble: Vec<Message>,
    pub inner: P,
}

impl<P: Projection> Projection for WithPreamble<P> {
    fn keep<'t>(&self, turns: &'t [Turn]) -> Vec<&'t Turn> {
        self.inner.keep(turns)
    }

    fn project(&self, records: &[Record]) -> Vec<Message> {
        let mut out = self.preamble.clone();
        out.extend(self.inner.project(records));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::entry::{EntryId, InputSource, JournalEntry, WakeReason};
    use crate::types::message::MessageBody;

    fn record(id: u64, entry: JournalEntry) -> Record {
        Record {
            id: EntryId(id),
            at: 0,
            entry,
        }
    }

    fn user(id: u64, text: &str) -> Record {
        record(
            id,
            JournalEntry::Input {
                source: InputSource::User,
                content: Message::user(text),
            },
        )
    }

    fn call(id: u64, call_id: &str) -> Record {
        record(
            id,
            JournalEntry::ToolCall {
                id: call_id.into(),
                tool: "t".into(),
                args: serde_json::json!({}),
            },
        )
    }

    fn result(id: u64, call_id: &str, output: &str) -> Record {
        record(
            id,
            JournalEntry::ToolResult {
                id: call_id.into(),
                tool: "t".into(),
                output: output.into(),
                ok: true,
            },
        )
    }

    fn wake(id: u64) -> Record {
        record(
            id,
            JournalEntry::Wake {
                reason: WakeReason::Heartbeat,
            },
        )
    }

    /// Which call ids a transcript asks about, and which it answers.
    fn calls_and_answers(messages: &[Message]) -> (Vec<String>, Vec<String>) {
        let mut called = Vec::new();
        let mut answered = Vec::new();
        for message in messages {
            match &message.body {
                MessageBody::Tool { tool_calls } => {
                    called.extend(tool_calls.iter().map(|c| c.id.clone()));
                }
                _ => {
                    if let Some(id) = &message.tool_call_id {
                        answered.push(id.clone());
                    }
                }
            }
        }
        called.sort();
        answered.sort();
        (called, answered)
    }

    /// A projection may drop a round, but never half of one.
    ///
    /// Stated against the source journal rather than in the abstract, because
    /// "every call has a result" is not true of a *journal*: a process that
    /// died mid-tool leaves a call with no answer, and that is a real history
    /// the projection must be able to represent. What it may not do is invent
    /// a mismatch that was not already there — an answer to a question the
    /// transcript no longer contains, or the loss of an answer that existed.
    fn assert_pairs_intact(source: &[Record], messages: &[Message]) {
        let (called, answered) = calls_and_answers(messages);
        let (_, answered_in_source) = calls_and_answers(&Everything.project(source));

        for id in &answered {
            assert!(
                called.contains(id),
                "the transcript answers call {id} but no longer contains it; \
                 called={called:?} answered={answered:?}"
            );
        }
        for id in &called {
            if answered_in_source.contains(id) {
                assert!(
                    answered.contains(id),
                    "call {id} was answered in the journal but its result was \
                     dropped, leaving it dangling; called={called:?} answered={answered:?}"
                );
            }
        }
    }

    #[test]
    fn heartbeats_never_reach_the_model() {
        // The reason a journal is not a transcript. Ten thousand wakes are ten
        // thousand facts and zero messages.
        let mut records = vec![user(0, "hello")];
        records.extend((1..500).map(wake));
        let messages = Everything.project(&records);
        assert_eq!(messages.len(), 1, "only the actual conversation is shown");
    }

    #[test]
    fn scratch_writes_are_recorded_but_not_shown() {
        let records = vec![
            user(1, "remember this"),
            record(
                2,
                JournalEntry::ScratchSet {
                    key: "plan".into(),
                    value: "step one".into(),
                },
            ),
        ];
        assert_eq!(Everything.project(&records).len(), 1);
    }

    /// A budget that would cut mid-round takes the whole round instead.
    #[test]
    fn a_tight_budget_never_separates_a_call_from_its_result() {
        let records = vec![
            user(1, "a very long earlier message ".repeat(40).trim()),
            call(2, "x"),
            result(3, "x", "some output"),
        ];
        // Enough for the tool round and nothing like enough for the preamble.
        let messages = RecentTurns::new(40).project(&records);
        assert_pairs_intact(&records, &messages);
        assert!(
            messages.len() >= 2,
            "the tool round should survive whole: {messages:?}"
        );
    }

    #[test]
    fn parallel_calls_survive_a_budget_whole_or_not_at_all() {
        let records = vec![
            user(1, "x".repeat(400).as_str()),
            call(2, "a"),
            call(3, "b"),
            result(4, "a", "first"),
            result(5, "b", "second"),
        ];
        for budget in [1, 10, 50, 200, 10_000] {
            let messages = RecentTurns::new(budget).project(&records);
            assert_pairs_intact(&records, &messages);
        }
    }

    /// One oversized turn is kept rather than producing nothing.
    #[test]
    fn a_turn_bigger_than_the_whole_budget_is_still_shown() {
        let records = vec![call(1, "a"), result(2, "a", &"y".repeat(10_000))];
        let messages = RecentTurns::new(10).project(&records);
        assert!(
            !messages.is_empty(),
            "an empty transcript leaves the model with no idea what it was doing"
        );
        assert_pairs_intact(&records, &messages);
    }

    #[test]
    fn the_newest_turns_are_the_ones_kept() {
        let records = vec![user(1, "oldest"), user(2, "middle"), user(3, "newest")];
        let messages = RecentTurns::new(12).project(&records);
        let texts: Vec<String> = messages.iter().map(|m| m.text()).collect();
        assert!(texts.contains(&"newest".to_string()), "got {texts:?}");
        assert!(!texts.contains(&"oldest".to_string()), "got {texts:?}");
    }

    #[test]
    fn a_preamble_is_prepended_and_survives_any_budget() {
        let records: Vec<Record> = (1..40).map(|i| user(i, "chatter")).collect();
        let projection = WithPreamble {
            preamble: vec![Message::system("you are a careful assistant")],
            inner: RecentTurns::new(8),
        };
        let messages = projection.project(&records);
        assert_eq!(messages[0].role, "system");
        assert!(
            messages.len() < records.len(),
            "the budget should still have bitten"
        );
    }

    #[test]
    fn an_empty_journal_projects_to_nothing() {
        assert!(RecentTurns::new(100).project(&[]).is_empty());
        assert!(Everything.project(&[]).is_empty());
    }

    /// Whatever the journal, whatever the budget, the pairs hold.
    ///
    /// The projections are hand-written and a hand-written test only covers
    /// the shapes its author thought of. This covers the ones they did not.
    #[test]
    fn no_journal_and_no_budget_can_produce_a_split_pair() {
        use proptest::prelude::*;

        proptest!(|(ops in prop::collection::vec(0u8..5, 0..60), budget in 0usize..300)| {
            let mut records = Vec::new();
            let mut open: Vec<String> = Vec::new();
            let mut next = 0u64;

            for op in ops {
                next += 1;
                match op {
                    0 => records.push(user(next, "text")),
                    1 => {
                        let id = format!("c{next}");
                        open.push(id.clone());
                        records.push(call(next, &id));
                    }
                    // Answer an outstanding call, oldest or newest, so both
                    // in-order and out-of-order completions are generated.
                    2 if !open.is_empty() => {
                        let id = open.remove(0);
                        records.push(result(next, &id, "out"));
                    }
                    3 if !open.is_empty() => {
                        let id = open.pop().expect("non-empty");
                        records.push(result(next, &id, "out"));
                    }
                    _ => records.push(wake(next)),
                }
            }

            assert_pairs_intact(&records, &RecentTurns::new(budget).project(&records));
            assert_pairs_intact(&records, &Everything.project(&records));
        });
    }
}
