//! Session invariants under any sequence of operations.
//!
//! The example-based tests next to [`Session`] cover the sequences someone
//! thought to write down. The bugs that survive are in the ones nobody did:
//! fork after edit after a model swap, deserialize then note the same tools
//! again, clear while mid-conversation. There are too many orderings to
//! enumerate, so these generate them instead and assert what must hold
//! regardless.
//!
//! The invariants are all about one thing — whether the engine's cached
//! prefill still describes this conversation. Getting that wrong in the
//! optimistic direction means a turn runs against a cache built from
//! different messages, different tools, or different weights, which produces
//! plausible nonsense rather than an error.

use proptest::prelude::*;

use super::session::Session;
use crate::types::message::Message;

/// One thing a caller can do to a session.
///
/// Only operations reachable from outside plus the two the engine performs on
/// the caller's behalf (`note_model`, `note_tools`), since those are what
/// actually drive invalidation.
#[derive(Debug, Clone)]
enum Op {
    Push(String),
    Open,
    Edit(usize),
    Clear,
    NoteModel(u64),
    NoteTools(u64),
    Shed(usize),
    RoundTrip,
    Fork,
}

fn any_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        "[a-z]{1,8}".prop_map(Op::Push),
        Just(Op::Open),
        (0usize..6).prop_map(Op::Edit),
        Just(Op::Clear),
        (0u64..3).prop_map(Op::NoteModel),
        (0u64..3).prop_map(Op::NoteTools),
        (0usize..3).prop_map(Op::Shed),
        Just(Op::RoundTrip),
        Just(Op::Fork),
    ]
}

/// Apply one operation, returning the session it produced.
///
/// `Open` stands in for the engine having prefilled this conversation, which
/// is the only way `opened` becomes true outside the engine itself.
fn apply(mut session: Session, op: &Op) -> Session {
    match op {
        Op::Push(text) => session.push_user(text.clone()),
        Op::Open => session.opened = true,
        Op::Edit(n) => session.edit(|m| m.truncate(*n)),
        Op::Clear => session.clear(),
        Op::NoteModel(generation) => {
            session.note_model(*generation);
        }
        Op::NoteTools(fingerprint) => {
            session.note_tools(*fingerprint);
        }
        Op::Shed(n) => session.note_shed(*n),
        Op::RoundTrip => {
            let json = serde_json::to_string(&session).expect("a session must serialize");
            session = serde_json::from_str(&json).expect("and deserialize");
        }
        Op::Fork => session = session.fork(),
    }
    session
}

proptest! {
    /// Serialized state is transcript only: the engine-side bookkeeping is
    /// process-local, so a session read back from disk cannot claim the engine
    /// has a prefill for it.
    #[test]
    fn a_deserialized_session_is_never_already_open(ops in prop::collection::vec(any_op(), 0..24)) {
        let mut session = Session::new();
        for op in &ops {
            session = apply(session, op);
        }

        let json = serde_json::to_string(&session).unwrap();
        let restored: Session = serde_json::from_str(&json).unwrap();

        prop_assert!(
            !restored.opened,
            "a session restored in another process must not believe the engine has its prefill",
        );
        prop_assert_eq!(restored.shed(), 0, "shed counts this process's turns");
        prop_assert_eq!(
            restored.messages().len(),
            session.messages().len(),
            "the transcript is the part that survives the round trip",
        );
    }

    /// A fork is an independent conversation, so it must not inherit the
    /// parent's engine identity or its cached prefill.
    #[test]
    fn a_fork_shares_the_transcript_and_nothing_else(ops in prop::collection::vec(any_op(), 0..24)) {
        let mut session = Session::new();
        for op in &ops {
            session = apply(session, op);
        }

        let fork = session.fork();

        prop_assert_ne!(
            fork.id(),
            session.id(),
            "two conversations sharing one engine identity would overwrite each other",
        );
        prop_assert!(!fork.opened, "the engine has nothing cached for a fork yet");
        prop_assert_eq!(fork.messages(), session.messages());
    }

    /// Anything that changes what the cached prefill describes must close the
    /// conversation. This is the invariant the whole design rests on.
    #[test]
    fn editing_or_clearing_always_closes_the_conversation(
        ops in prop::collection::vec(any_op(), 0..24),
        truncate_to in 0usize..6,
    ) {
        let mut session = Session::new();
        for op in &ops {
            session = apply(session, op);
        }
        session.opened = true;

        let mut edited = session.fork();
        edited.opened = true;
        edited.edit(|m| m.truncate(truncate_to));
        prop_assert!(
            !edited.opened,
            "an edited transcript no longer matches the cached prefill",
        );

        let mut cleared = session.fork();
        cleared.opened = true;
        cleared.clear();
        prop_assert!(!cleared.opened, "a cleared session has nothing to resume");
        prop_assert_eq!(cleared.shed(), 0, "and nothing shed");
        prop_assert!(cleared.messages().is_empty());
    }

    /// Re-declaring the same model or tool set is not a change, so it must not
    /// cost a re-prefill. A ranker or fingerprint that is merely unstable
    /// would show up here as spurious invalidation on every turn.
    #[test]
    fn declaring_the_same_model_and_tools_again_changes_nothing(
        generation in 0u64..5,
        fingerprint in 0u64..5,
        ops in prop::collection::vec(any_op(), 0..16),
    ) {
        let mut session = Session::new();
        for op in &ops {
            session = apply(session, op);
        }

        session.note_model(generation);
        session.note_tools(fingerprint);
        session.opened = true;

        prop_assert!(
            !session.note_model(generation),
            "the same model generation must not reopen the conversation",
        );
        prop_assert!(
            !session.note_tools(fingerprint),
            "the same tool set must not reopen the conversation",
        );
        prop_assert!(session.opened, "and neither may close it");
    }

    /// The other direction, which is the one that corrupts output if it is
    /// wrong: different weights or different tool definitions mean the cache
    /// describes something else.
    #[test]
    fn a_different_model_or_tool_set_always_reopens(
        first in 0u64..5,
        second in 0u64..5,
    ) {
        prop_assume!(first != second);

        let mut by_model = Session::new();
        by_model.note_model(first);
        by_model.opened = true;
        prop_assert!(by_model.note_model(second), "new weights invalidate the prefill");
        prop_assert!(!by_model.opened);

        let mut by_tools = Session::new();
        by_tools.note_tools(first);
        by_tools.opened = true;
        prop_assert!(by_tools.note_tools(second), "new tool definitions invalidate the prefix");
        prop_assert!(!by_tools.opened);
    }

    /// A transcript only shrinks where the caller asked it to. Nothing the
    /// engine does on its own may drop a message the caller still holds.
    #[test]
    fn only_the_caller_ever_loses_a_message(ops in prop::collection::vec(any_op(), 0..32)) {
        let mut session = Session::new();

        for op in &ops {
            let before = session.messages().len();
            session = apply(session, op);
            let after = session.messages().len();

            let expected = match op {
                Op::Push(_) => before + 1,
                Op::Edit(n) => before.min(*n),
                Op::Clear => 0,
                // Everything else is bookkeeping and must leave the
                // transcript exactly as it was.
                _ => before,
            };
            prop_assert_eq!(
                after,
                expected,
                "{:?} changed the transcript from {} to {}",
                op,
                before,
                after,
            );
        }
    }

    /// What the engine still needs to be sent is always a suffix of the
    /// transcript, never a reordering or a gap. A wrong answer here sends the
    /// model a conversation that never happened.
    #[test]
    fn pending_messages_are_always_a_suffix_of_the_transcript(
        ops in prop::collection::vec(any_op(), 0..24),
        sent_through in 0usize..8,
    ) {
        let mut session = Session::new();
        for op in &ops {
            session = apply(session, op);
        }

        let pending = session.pending(sent_through);
        let all = session.messages();

        prop_assert!(
            pending.len() <= all.len(),
            "more is pending than exists: {} of {}",
            pending.len(),
            all.len(),
        );
        let start = all.len() - pending.len();
        prop_assert_eq!(
            &pending[..],
            &all[start..],
            "pending must be the tail of the transcript, in order",
        );
    }
}

/// An unopened conversation owes the engine everything, however much the
/// caller claims was already sent.
#[test]
fn an_unopened_session_always_sends_its_whole_transcript() {
    let mut session = Session::new();
    session.push(Message::user("one"));
    session.push(Message::user("two"));

    for claimed in 0..5 {
        assert_eq!(
            session.pending(claimed).len(),
            2,
            "the engine has no prefill, so a claim of {claimed} sent messages means nothing",
        );
    }
}
