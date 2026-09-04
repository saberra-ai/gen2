//! The unit a projection is allowed to choose.
//!
//! # Why this type exists
//!
//! A transcript can be cut in places that make it incoherent. The one that
//! actually happens: dropping a tool call and keeping its result, so the model
//! is shown an answer to a question nothing asked — or the reverse, a call
//! that never resolves. gen2's own truncation does this today, dropping
//! oldest-first one message at a time with no idea what it is splitting.
//!
//! The obvious fix is to check for it in each projection. That fix lasts until
//! the second projection is written. So the check is not in the projections:
//! **a projection cannot address an individual record at all.** It selects
//! [`Turn`]s, and a `Turn` is constructed only by [`Turn::group`], which puts a
//! call and its results in the same one. Splitting a pair is not a bug this
//! module avoids; it is a thing the type system gives no way to express.

use super::entry::{JournalEntry, Record};
use crate::types::message::Message;

/// An indivisible run of records.
///
/// Whole or not at all. There is deliberately no constructor that takes an
/// arbitrary set of records, and no accessor that hands one back mutably —
/// everything a projection can do is keep this or drop it.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    records: Vec<Record>,
}

impl Turn {
    /// Group records into turns, keeping every tool call with its results.
    ///
    /// The rule is one sentence: a `ToolCall` opens a turn that stays open
    /// until every call it contains has been answered. Anything else is a turn
    /// of its own.
    ///
    /// A call that is never answered — the process died mid-tool — keeps its
    /// turn open to the end of the journal rather than swallowing the rest of
    /// the conversation into it. That would be the wrong trade: one unanswered
    /// call should not make the entire remaining history indivisible.
    pub fn group(records: &[Record]) -> Vec<Turn> {
        let mut turns: Vec<Turn> = Vec::new();
        let mut open: Option<(Vec<Record>, Vec<String>)> = None;

        for record in records {
            match (&mut open, &record.entry) {
                // A result closes out the call it answers.
                (Some((buffered, awaiting)), JournalEntry::ToolResult { id, .. }) => {
                    buffered.push(record.clone());
                    awaiting.retain(|a| a != id);
                    if awaiting.is_empty() {
                        let (records, _) = open.take().expect("just matched");
                        turns.push(Turn { records });
                    }
                }
                // Another call in the same round joins the open turn.
                (Some((buffered, awaiting)), JournalEntry::ToolCall { id, .. }) => {
                    buffered.push(record.clone());
                    awaiting.push(id.clone());
                }
                // Anything else while calls are outstanding belongs to the
                // same round: the model's own commentary between asking and
                // being answered is not separable from either.
                (Some((buffered, _)), _) => buffered.push(record.clone()),

                // A call opens a turn.
                (None, JournalEntry::ToolCall { id, .. }) => {
                    open = Some((vec![record.clone()], vec![id.clone()]));
                }
                // A result with no open call. Its call is not in this journal
                // — truncated, or the journal starts mid-round. It stands
                // alone rather than being dropped: the fact that a tool ran is
                // still true.
                (None, _) => turns.push(Turn {
                    records: vec![record.clone()],
                }),
            }
        }

        // Whatever was still open at the end — an interrupted round.
        if let Some((records, _)) = open {
            turns.push(Turn { records });
        }
        turns
    }

    /// The records in this turn.
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// The messages this turn contributes, in order.
    pub fn to_messages(&self) -> Vec<Message> {
        self.records.iter().flat_map(Record::to_messages).collect()
    }

    /// Roughly how much context this turn costs.
    ///
    /// Characters over four. Deliberately not a tokenizer call: a projection
    /// runs on every turn of every wake, the model's own tokenizer is behind a
    /// session that may not exist yet, and being wrong by a few percent moves
    /// a boundary by one turn. Being *slow* would move it into the request
    /// path.
    pub fn estimated_tokens(&self) -> usize {
        self.to_messages()
            .iter()
            .map(|m| m.text().len() / 4 + 4)
            .sum()
    }

    /// Whether anything here reaches the model.
    ///
    /// A turn of pure bookkeeping — a wake, a scratch write — contributes
    /// nothing to the transcript and should not consume a projection's budget.
    pub fn is_conversational(&self) -> bool {
        self.records.iter().any(|r| r.entry.is_conversational())
    }
}

/// How many messages from `at` form one indivisible tool round.
///
/// The same invariant as [`Turn::group`], for code that holds a plain
/// `Vec<Message>` rather than a journal — chiefly
/// `session_rt::truncate`, which drops messages to fit a context
/// window and had no idea what it was splitting.
///
/// Returns `1` for an ordinary message, and for an assistant turn whose calls
/// are *never* answered. That second case looks wrong and is not: an
/// unanswered call is already a broken pair, and dropping it alone removes the
/// dangling half rather than creating one. The alternative — extending the
/// round to the end of the conversation — would let one interrupted tool call
/// make the entire history undroppable, and truncation would fail instead of
/// truncating.
pub fn round_len(messages: &[Message], at: usize) -> usize {
    use crate::types::message::MessageBody;

    let Some(first) = messages.get(at) else {
        return 0;
    };
    let MessageBody::Tool { tool_calls } = &first.body else {
        return 1;
    };

    let mut awaiting: Vec<&str> = tool_calls.iter().map(|c| c.id.as_str()).collect();
    for (offset, message) in messages[at + 1..].iter().enumerate() {
        match &message.body {
            // A further call in the same round joins it.
            MessageBody::Tool { tool_calls } => {
                awaiting.extend(tool_calls.iter().map(|c| c.id.as_str()));
            }
            _ => {
                if let Some(id) = message.tool_call_id.as_deref() {
                    awaiting.retain(|a| *a != id);
                    if awaiting.is_empty() {
                        return offset + 2;
                    }
                }
            }
        }
    }
    // Never closed.
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::entry::{EntryId, InputSource, JournalEntry, WakeReason};

    fn record(id: u64, entry: JournalEntry) -> Record {
        Record {
            id: EntryId(id),
            at: 0,
            entry,
        }
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

    fn result(id: u64, call_id: &str) -> Record {
        record(
            id,
            JournalEntry::ToolResult {
                id: call_id.into(),
                tool: "t".into(),
                output: "done".into(),
                ok: true,
            },
        )
    }

    fn user(id: u64, text: &str) -> Record {
        record(
            id,
            JournalEntry::Input {
                source: InputSource::User,
                content: crate::types::message::Message::user(text),
            },
        )
    }

    #[test]
    fn a_call_and_its_result_land_in_the_same_turn() {
        let turns = Turn::group(&[user(1, "go"), call(2, "a"), result(3, "a"), user(4, "next")]);
        assert_eq!(turns.len(), 3, "user / call+result / user");
        assert_eq!(turns[1].records().len(), 2);
    }

    /// The case the whole design is for.
    ///
    /// Two calls in one round and two results. If grouping split them, a
    /// projection could keep one call and both results.
    #[test]
    fn parallel_calls_and_all_their_results_are_one_turn() {
        let turns = Turn::group(&[
            call(1, "a"),
            call(2, "b"),
            result(3, "a"),
            result(4, "b"),
            user(5, "thanks"),
        ]);
        assert_eq!(turns.len(), 2);
        assert_eq!(
            turns[0].records().len(),
            4,
            "both calls and both results are indivisible: {:?}",
            turns[0]
        );
    }

    /// Results arriving in a different order than the calls were made.
    #[test]
    fn results_out_of_order_still_close_the_turn() {
        let turns = Turn::group(&[call(1, "a"), call(2, "b"), result(3, "b"), result(4, "a")]);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].records().len(), 4);
    }

    /// An unanswered call must not swallow the rest of history.
    ///
    /// It stays open to the end, which is correct — but if a later round were
    /// folded into it, one crash would make the whole remaining journal a
    /// single indivisible turn and the projection could never trim anything.
    #[test]
    fn an_unanswered_call_keeps_its_turn_open_but_the_journal_still_ends() {
        let turns = Turn::group(&[user(1, "go"), call(2, "a")]);
        assert_eq!(turns.len(), 2);
        assert_eq!(
            turns[1].records().len(),
            1,
            "the call, alone and unanswered"
        );
    }

    /// A result whose call is not present stands alone rather than vanishing.
    #[test]
    fn an_orphaned_result_is_kept_as_its_own_turn() {
        let turns = Turn::group(&[result(1, "gone"), user(2, "hm")]);
        assert_eq!(turns.len(), 2);
    }

    #[test]
    fn bookkeeping_turns_are_not_conversational() {
        let turns = Turn::group(&[record(
            1,
            JournalEntry::Wake {
                reason: WakeReason::Heartbeat,
            },
        )]);
        assert!(!turns[0].is_conversational());
        assert!(turns[0].to_messages().is_empty());
    }

    #[test]
    fn grouping_preserves_every_record_and_their_order() {
        let input = vec![
            user(1, "a"),
            call(2, "x"),
            result(3, "x"),
            user(4, "b"),
            call(5, "y"),
        ];
        let regrouped: Vec<Record> = Turn::group(&input)
            .iter()
            .flat_map(|t| t.records().to_vec())
            .collect();
        assert_eq!(
            regrouped, input,
            "grouping must not lose, duplicate or reorder anything"
        );
    }
}
