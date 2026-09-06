//! Session invariants under any sequence of operations.
//!
//! The example-based tests next to [`Session`] cover the sequences someone
//! thought to write down. The bugs that survive are in the ones nobody did:
//! fork after edit after a model swap, restore a projection that was
//! compacted twice, remove a tool result after a round trip. There are too
//! many orderings to enumerate, so these generate them instead and assert
//! what must hold regardless.
//!
//! Two families of invariant. The older one is about whether the engine's
//! cached prefill still describes this conversation — getting that wrong in
//! the optimistic direction means a turn runs against a cache built from
//! different messages, tools, or weights, and produces plausible nonsense.
//! The newer one is api_spec.md §29: the event log only grows and never
//! changes, the records are a superset of the projection, the revision moves
//! exactly when context does, and no projection ever shows half a tool round.

use proptest::prelude::*;

use super::session::{MessageId, Session, SessionEvent};
use super::tool_defs::ToolDefinition;
use crate::types::message::{FunctionDefinition, Message, MessageBody, ToolCall};

/// One thing a caller can do to a session.
///
/// Every public mutation, plus the two the engine performs on the caller's
/// behalf (`note_model`, `note_tools`) since those drive invalidation.
/// Indices are taken modulo whatever they index, so a generated op is always
/// applicable to *something*; refusals come from the invariant, not from
/// out-of-range ids.
#[derive(Debug, Clone)]
enum Op {
    Push(String),
    /// An assistant turn making `n` calls, followed by `answered` of the
    /// results — so an in-progress round can exist in the projection.
    PushRound {
        calls: u8,
        answered: u8,
    },
    Open,
    Edit(usize),
    Clear,
    NoteModel(u64),
    NoteTools(u64),
    Shed(usize),
    RoundTrip,
    Fork,
    SetSystem(String),
    AppendSystem(String),
    ClearSystem,
    AddTool(String),
    RemoveTool(String),
    ReplaceMessage(usize, String),
    RemoveMessage(usize),
    ReplaceMessages(Vec<String>),
    /// Indices into `all_messages()`, in the order to restore.
    RestoreContext(Vec<usize>),
    ForkAt(usize),
    Rollback(usize),
}

fn any_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        "[a-z]{1,8}".prop_map(Op::Push),
        (1u8..3, 0u8..3).prop_map(|(calls, answered)| Op::PushRound {
            calls,
            answered: answered.min(calls),
        }),
        Just(Op::Open),
        (0usize..6).prop_map(Op::Edit),
        Just(Op::Clear),
        (0u64..3).prop_map(Op::NoteModel),
        (0u64..3).prop_map(Op::NoteTools),
        (0usize..3).prop_map(Op::Shed),
        Just(Op::RoundTrip),
        Just(Op::Fork),
        "[a-z]{1,6}".prop_map(Op::SetSystem),
        "[a-z]{1,6}".prop_map(Op::AppendSystem),
        Just(Op::ClearSystem),
        "[a-c]".prop_map(Op::AddTool),
        "[a-c]".prop_map(Op::RemoveTool),
        (0usize..8, "[a-z]{1,4}").prop_map(|(i, t)| Op::ReplaceMessage(i, t)),
        (0usize..8).prop_map(Op::RemoveMessage),
        prop::collection::vec("[a-z]{1,4}", 0..3).prop_map(Op::ReplaceMessages),
        prop::collection::vec(0usize..12, 0..5).prop_map(Op::RestoreContext),
        (0usize..8).prop_map(Op::ForkAt),
        (0usize..6).prop_map(Op::Rollback),
    ]
}

fn call_message(ids: &[String]) -> Message {
    Message::assistant_tool_calls(
        ids.iter()
            .map(|id| ToolCall {
                id: id.clone(),
                r#type: "function".into(),
                function: FunctionDefinition {
                    description: None,
                    name: "t".into(),
                    arguments: serde_json::json!({}),
                },
            })
            .collect(),
    )
}

/// Pick an active id by index, wrapping. `None` on an empty projection.
fn active_at(session: &Session, i: usize) -> Option<MessageId> {
    let ids = session.active_ids();
    (!ids.is_empty()).then(|| ids[i % ids.len()])
}

/// Apply one operation, returning the session it produced.
///
/// `Open` stands in for the engine having prefilled this conversation, which
/// is the only way `opened` becomes true outside the engine itself. Refused
/// operations are allowed to be refused; what they must not do is change
/// anything, which the property below checks.
fn apply(mut session: Session, op: &Op, counter: &mut u64) -> Session {
    match op {
        Op::Push(text) => {
            session.push_user(text.clone());
        }
        Op::PushRound { calls, answered } => {
            let ids: Vec<String> = (0..*calls)
                .map(|_| {
                    *counter += 1;
                    format!("c{counter}")
                })
                .collect();
            session.push(call_message(&ids));
            for id in ids.iter().take(usize::from(*answered)) {
                session.push_tool_result(id, "done");
            }
        }
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
        Op::SetSystem(text) => session.set_system(text.clone()),
        Op::AppendSystem(text) => session.append_system(text),
        Op::ClearSystem => session.clear_system(),
        Op::AddTool(name) => session.add_tool(ToolDefinition::new(name.clone())),
        Op::RemoveTool(name) => {
            session.remove_tool(name);
        }
        Op::ReplaceMessage(i, text) => {
            if let Some(id) = active_at(&session, *i) {
                let _ = session.replace_message(id, Message::user(text.clone()));
            }
        }
        Op::RemoveMessage(i) => {
            if let Some(id) = active_at(&session, *i) {
                let _ = session.remove_message(id);
            }
        }
        Op::ReplaceMessages(texts) => {
            let _ = session.replace_messages(texts.iter().map(Message::user));
        }
        Op::RestoreContext(indices) => {
            let all: Vec<MessageId> = session.all_messages().iter().map(|r| r.id).collect();
            if !all.is_empty() {
                let ids: Vec<MessageId> = indices.iter().map(|i| all[i % all.len()]).collect();
                let _ = session.restore_context(ids);
            }
        }
        Op::ForkAt(i) => {
            if let Some(id) = active_at(&session, *i)
                && let Ok(fork) = session.fork_at(id)
            {
                session = fork;
            }
        }
        Op::Rollback(n) => session.rollback_to(*n),
    }
    session
}

/// The context a runtime renders from: what the revision must track.
fn context_of(s: &Session) -> (Option<String>, Vec<String>, Vec<MessageId>) {
    (
        s.system().map(str::to_string),
        s.tools().names().iter().map(|n| n.to_string()).collect(),
        s.active_ids().to_vec(),
    )
}

/// The oracle for the tool-round invariant, written independently of the
/// session's own check: a result must follow a call that made it, and a call
/// whose result exists anywhere in the records must show that result.
fn round_violation(s: &Session) -> Option<String> {
    let calls_of = |m: &Message| -> Vec<String> {
        match &m.body {
            MessageBody::Tool { tool_calls } => tool_calls.iter().map(|c| c.id.clone()).collect(),
            _ => Vec::new(),
        }
    };
    let mut declared: Vec<String> = Vec::new();
    let mut answered: Vec<String> = Vec::new();
    for m in s.messages() {
        let calls = calls_of(m);
        if !calls.is_empty() {
            declared.extend(calls);
        } else if m.role == "tool" {
            match m.tool_call_id.as_deref() {
                Some(id) if declared.iter().any(|d| d == id) => answered.push(id.to_string()),
                Some(id) => return Some(format!("result for `{id}` with no call shown")),
                None if declared.is_empty() => return Some("id-less result with no call".into()),
                None => {}
            }
        }
    }
    let answered_in_records: Vec<String> = s
        .all_messages()
        .iter()
        .filter(|r| r.message.role == "tool")
        .filter_map(|r| r.message.tool_call_id.clone())
        .collect();
    declared
        .iter()
        .find(|c| !answered.contains(c) && answered_in_records.contains(c))
        .map(|c| format!("call `{c}` shown without its result"))
}

proptest! {
    /// api_spec.md §29, invariants 5 and 6, plus §7.10, over any sequence:
    /// the log only grows and earlier events never change; every active
    /// message is a record and records are never lost; the revision moves
    /// exactly when the rendered context does; a refused mutation changes
    /// nothing; and no projection ever shows half a tool round.
    #[test]
    fn the_log_only_grows_the_records_only_accrue_and_the_revision_tracks_context(
        ops in prop::collection::vec(any_op(), 0..32),
    ) {
        let mut session = Session::new();
        let mut counter = 0;
        for op in &ops {
            let events_before: Vec<SessionEvent> = session.events().to_vec();
            let records_before = session.all_messages().len();
            let revision_before = session.revision();
            let context_before = context_of(&session);
            let id_before = session.id().clone();

            session = apply(session, op, &mut counter);

            let same_session = session.id() == &id_before;
            // Invariant 5: earlier events are immutable and the log only grows.
            prop_assert!(
                session.events().starts_with(&events_before),
                "{op:?} rewrote history: {:?} -> {:?}",
                events_before,
                session.events(),
            );
            // Invariant 6: records accrue; the projection is drawn from them.
            prop_assert!(
                session.all_messages().len() >= records_before,
                "{op:?} lost a record",
            );
            for m in session.messages() {
                prop_assert!(
                    session.all_messages().iter().any(|r| r.active && r.message == m),
                    "{op:?} left an active message that is not an active record",
                );
            }
            prop_assert_eq!(
                session.all_messages().iter().filter(|r| r.active).count(),
                session.len(),
                "{:?}: active flags disagree with the projection", op,
            );
            // §7.10: the revision is the event count, and it moves exactly
            // when the context does — strictly up on a change, not at all
            // on a no-op or a refusal.
            let context_events = session
                .events()
                .iter()
                .filter(|e| !matches!(e, SessionEvent::MessageRecorded { .. }))
                .count() as u64;
            prop_assert_eq!(session.revision().as_u64(), context_events);
            if same_session {
                let changed = context_of(&session) != context_before;
                let minted = session.all_messages().len() > records_before;
                if changed || minted {
                    prop_assert!(
                        session.revision() > revision_before,
                        "{op:?} changed the context without bumping the revision",
                    );
                } else {
                    prop_assert_eq!(
                        session.revision(), revision_before,
                        "{:?} bumped the revision without changing the context", op,
                    );
                }
            } else {
                // A fork: the parent's log plus the fork event.
                prop_assert!(session.events().len() == events_before.len() + 1);
                let forked_from_parent = matches!(
                    session.events().last(),
                    Some(SessionEvent::Forked { parent, .. }) if parent == &id_before
                );
                prop_assert!(forked_from_parent, "a fork must record its parent");
            }
            // The tool-round invariant, on every projection ever produced.
            prop_assert_eq!(round_violation(&session), None, "after {:?}", op);
        }
    }

    /// Reads never move the revision.
    #[test]
    fn reads_do_not_change_the_revision(ops in prop::collection::vec(any_op(), 0..16)) {
        let mut session = Session::new();
        let mut counter = 0;
        for op in &ops {
            session = apply(session, op, &mut counter);
        }
        let before = session.revision();
        let events = session.events().to_vec();
        let _ = (
            session.messages().len(),
            session.all_messages().len(),
            session.events().len(),
            session.system().map(str::len),
            session.tools().len(),
            session.latest().is_some(),
            session.latest_text(),
            session.active_ids().len(),
            session.pending(3).len(),
            session.len(),
            session.is_empty(),
            session.shed(),
        );
        let _ = session.fork();
        prop_assert_eq!(session.revision(), before);
        prop_assert_eq!(session.events(), &events[..]);
    }

    /// Serialized state is the event log: the engine-side bookkeeping is
    /// process-local, so a session read back from disk cannot claim the engine
    /// has a prefill for it — and everything derived comes back identical.
    #[test]
    fn a_deserialized_session_is_never_already_open_and_replays_exactly(
        ops in prop::collection::vec(any_op(), 0..24),
    ) {
        let mut session = Session::new();
        let mut counter = 0;
        for op in &ops {
            session = apply(session, op, &mut counter);
        }

        let json = serde_json::to_string(&session).unwrap();
        let restored: Session = serde_json::from_str(&json).unwrap();

        prop_assert!(
            !restored.opened,
            "a session restored in another process must not believe the engine has its prefill",
        );
        prop_assert_eq!(restored.shed(), 0, "shed counts this process's turns");
        prop_assert_eq!(restored.id(), session.id());
        prop_assert_eq!(restored.events(), session.events());
        prop_assert_eq!(restored.revision(), session.revision());
        prop_assert_eq!(restored.system(), session.system());
        prop_assert_eq!(restored.tools(), session.tools());
        prop_assert_eq!(restored.active_ids(), session.active_ids());
        prop_assert_eq!(restored.messages(), session.messages());
        prop_assert_eq!(restored.all_messages(), session.all_messages());
    }

    /// A fork is an independent conversation, so it must not inherit the
    /// parent's engine identity or its cached prefill — but it does inherit
    /// everything else, provenance included.
    #[test]
    fn a_fork_shares_the_state_and_provenance_and_nothing_engine_side(
        ops in prop::collection::vec(any_op(), 0..24),
    ) {
        let mut session = Session::new();
        let mut counter = 0;
        for op in &ops {
            session = apply(session, op, &mut counter);
        }

        let fork = session.fork();

        prop_assert_ne!(
            fork.id(),
            session.id(),
            "two conversations sharing one engine identity would overwrite each other",
        );
        prop_assert!(!fork.opened, "the engine has nothing cached for a fork yet");
        prop_assert_eq!(fork.messages(), session.messages());
        prop_assert_eq!(fork.active_ids(), session.active_ids());
        prop_assert_eq!(fork.system(), session.system());
        prop_assert_eq!(fork.tools(), session.tools());
        prop_assert!(fork.events().starts_with(session.events()));
        prop_assert_eq!(fork.all_messages().len(), session.all_messages().len());
    }

    /// Anything that changes what the cached prefill describes must close the
    /// conversation. This is the invariant the whole design rests on.
    #[test]
    fn editing_or_clearing_always_closes_the_conversation(
        ops in prop::collection::vec(any_op(), 0..24),
        truncate_to in 0usize..6,
    ) {
        let mut session = Session::new();
        let mut counter = 0;
        for op in &ops {
            session = apply(session, op, &mut counter);
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

        let mut re_instructed = session.fork();
        re_instructed.opened = true;
        re_instructed.set_system("something else entirely");
        prop_assert!(!re_instructed.opened, "a new system prompt is a new prefix");

        let mut re_tooled = session.fork();
        re_tooled.opened = true;
        re_tooled.add_tool(ToolDefinition::new("zzz_never_generated"));
        prop_assert!(!re_tooled.opened, "a new tool set is a new prefix");
    }

    /// Re-declaring the same model or tool set is not a change, so it must not
    /// cost a re-prefill.
    #[test]
    fn declaring_the_same_model_and_tools_again_changes_nothing(
        generation in 0u64..5,
        fingerprint in 0u64..5,
        ops in prop::collection::vec(any_op(), 0..16),
    ) {
        let mut session = Session::new();
        let mut counter = 0;
        for op in &ops {
            session = apply(session, op, &mut counter);
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

        // The same goes for first-order state: setting what is already set
        // is not a change.
        let revision = session.revision();
        if let Some(system) = session.system().map(str::to_string) {
            session.set_system(system);
        }
        let tools = session.tools().clone();
        session.set_tools(tools);
        prop_assert_eq!(session.revision(), revision);
        prop_assert!(session.opened);
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

    /// The projection only shrinks where the caller asked it to. Nothing the
    /// engine does on its own may drop a message the caller still holds.
    #[test]
    fn only_the_caller_ever_loses_a_message(ops in prop::collection::vec(any_op(), 0..32)) {
        let mut session = Session::new();
        let mut counter = 0;

        for op in &ops {
            let before = session.messages().len();
            session = apply(session, op, &mut counter);
            let after = session.messages().len();

            let ok = match op {
                Op::Push(_) => after == before + 1,
                Op::PushRound { answered, .. } => after == before + 1 + usize::from(*answered),
                // Trimming may take a half-round with it, never more than asked.
                Op::Edit(n) | Op::Rollback(n) => after <= before.min(*n),
                Op::Clear => after == 0,
                Op::ReplaceMessage(..) => after == before,
                Op::RemoveMessage(_) => after == before || after + 1 == before,
                Op::ReplaceMessages(texts) => after == before || after == texts.len(),
                Op::RestoreContext(ids) => after == before || after == ids.len(),
                Op::ForkAt(_) => after <= before,
                // Everything else is bookkeeping or instruction state and
                // must leave the projection exactly as it was.
                _ => after == before,
            };
            prop_assert!(ok, "{:?} changed the projection from {} to {}", op, before, after);
        }
    }

    /// What the engine still needs to be sent is always a suffix of the
    /// rendered transcript — system prompt first — never a reordering or a
    /// gap. A wrong answer here sends the model a conversation that never
    /// happened.
    #[test]
    fn pending_messages_are_always_a_suffix_of_the_transcript(
        ops in prop::collection::vec(any_op(), 0..24),
        sent_through in 0usize..8,
    ) {
        let mut session = Session::new();
        let mut counter = 0;
        for op in &ops {
            session = apply(session, op, &mut counter);
        }

        let pending = session.pending(sent_through);
        let all = session.transcript();

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
        if let Some(system) = session.system() {
            prop_assert_eq!(all[0].role.as_str(), "system");
            prop_assert_eq!(all[0].text(), system);
            prop_assert!(session.messages().iter().all(|m| m.role != "system"));
        }
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
