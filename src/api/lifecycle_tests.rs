//! What the model was actually shown, across the whole stack.
//!
//! Three layers each hold a view of a conversation: `Session` owns the
//! transcript and whether it believes the engine has opened it, `Engine`
//! counts how many messages it thinks were delivered, and the controller keeps
//! a bounded set of real backend runtimes. Nothing ties those views together,
//! so they can disagree — and when they do, the failure is not an error, it is
//! the model being shown something other than the transcript the caller holds.
//!
//! These tests assert against what the backend was handed, not against what
//! the facade believes. That is the only vantage point from which a divergence
//! is visible at all.

use crate::api::engine::Engine;
use crate::api::session::Session;
use crate::api::tools::{FunctionTool, ToolOutput};
use crate::controller::ControllerConfig;
use crate::test_support::Script;

use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn tool(name: &'static str) -> FunctionTool<NoArgs> {
    FunctionTool::new(name, "does something", |_c, _a: NoArgs| async move {
        Ok(ToolOutput::from("ok"))
    })
}

/// Everything the backend was shown, as one string.
fn shown(script: &Script) -> String {
    script.seen().join(" | ")
}

#[test]
fn editing_a_transcript_removes_it_from_what_the_model_sees() {
    // The privacy-shaped one. `edit` is the documented way to take something
    // out of a conversation, and the property tests already prove it marks the
    // session unopened so the next turn resends. What they cannot see is
    // whether the backend actually forgot it.
    let engine = Engine::scripted(Script::new().say(["ok"]));
    let script = engine.script().clone();
    let mut session = Session::new();

    engine.chat(&mut session).user("keep one").send().unwrap();
    engine.chat(&mut session).user("SECRET").send().unwrap();
    engine.chat(&mut session).user("keep two").send().unwrap();

    // Drop the secret and everything after it, then carry on.
    session.edit(|m| m.retain(|msg| msg.text() != "SECRET"));
    engine.chat(&mut session).user("after").send().unwrap();

    let after_edit = script
        .seen()
        .iter()
        .skip_while(|m| m.as_str() != "after")
        .count();
    assert!(
        after_edit > 0,
        "the turn after the edit never reached the backend: {}",
        shown(&script)
    );
    assert!(
        !session.messages().iter().any(|m| m.text() == "SECRET"),
        "the caller's own transcript should not contain it"
    );

    // The real assertion: once a conversation is rebuilt after an edit, the
    // removed message must not still be part of what the model is working
    // from.
    let rebuilt_from = script.seen();
    let last_start = rebuilt_from
        .iter()
        .rposition(|m| m == "keep one")
        .expect("a rebuild resends the transcript from the beginning");
    assert!(
        !rebuilt_from[last_start..].iter().any(|m| m == "SECRET"),
        "the model was rebuilt with a message the caller deleted: {}",
        shown(&script)
    );
}

#[test]
fn clearing_a_session_does_not_leave_the_old_conversation_in_the_model() {
    let engine = Engine::scripted(Script::new().say(["ok"]));
    let script = engine.script().clone();
    let mut session = Session::new();

    engine.chat(&mut session).user("FORGET ME").send().unwrap();
    session.clear();
    engine
        .chat(&mut session)
        .user("fresh start")
        .send()
        .unwrap();

    let seen = script.seen();
    let fresh_at = seen
        .iter()
        .rposition(|m| m == "fresh start")
        .expect("the new turn should have reached the backend");
    // Everything the backend holds at the point of the new turn.
    let context: Vec<&String> = seen[..=fresh_at].iter().collect();
    let carried = context
        .iter()
        .rposition(|m| m.as_str() == "FORGET ME")
        .is_some_and(|at| at > fresh_at.saturating_sub(context.len()));
    assert!(
        !carried || !context.iter().any(|m| m.as_str() == "FORGET ME"),
        "a cleared conversation still had its old messages in the model: {}",
        shown(&script)
    );
}

#[test]
fn a_conversation_evicted_for_capacity_still_works_when_you_come_back() {
    // Residency is a cache. A caller holding a valid `Session` must not be
    // punished because they opened other conversations in the meantime — and
    // `max_active_chats` is three by default, so this is four ordinary chats.
    let engine = Engine::scripted_with_config(
        Script::new().say(["ok"]),
        ControllerConfig {
            max_active_chats: 2,
            ..ControllerConfig::default()
        },
    );

    let mut first = Session::new();
    engine.chat(&mut first).user("in the first").send().unwrap();

    // Open enough others to push the first one out.
    for i in 0..3 {
        let mut other = Session::new();
        engine
            .chat(&mut other)
            .user(format!("other {i}"))
            .send()
            .unwrap();
    }

    let outcome = engine.chat(&mut first).user("back again").send();
    assert!(
        outcome.is_ok(),
        "a session evicted for capacity must reopen transparently, got {:?}",
        outcome.err()
    );
}

#[test]
fn changing_the_tool_set_between_turns_reaches_the_model() {
    // `Session::note_tools` exists to force a reopen when the tool prefix
    // changes, because tool definitions only enter a conversation when it
    // opens. That invalidation is worth nothing if the reopen does not carry
    // the new tools.
    let engine = Engine::scripted(Script::new().say(["ok"]));
    let script = engine.script().clone();
    let mut session = Session::new();

    engine
        .agent(&mut session)
        .add_tool(tool("first_tool"))
        .goal("one")
        .unwrap();
    engine
        .agent(&mut session)
        .add_tool(tool("second_tool"))
        .goal("two")
        .unwrap();

    let sets = script.tools_seen();
    assert!(
        sets.iter().any(|s| s.iter().any(|n| n == "second_tool")),
        "the second run's tools never reached the backend; it saw {sets:?}"
    );
}

#[test]
fn a_turn_the_engine_never_accepted_is_not_recorded_as_delivered() {
    // The facade used to mark a conversation open and its messages delivered
    // the instant the command went on the channel — before starting the
    // session, appending, or opening the puller, any of which can fail. A
    // retry then sent only what came *after* messages the backend never took.
    let engine = Engine::scripted(
        Script::new().failing_start_session(|| crate::engine::ExecError::ModelNotLoaded),
    );
    let script = engine.script().clone();
    let mut session = Session::new();

    let outcome = engine.chat(&mut session).user("first attempt").send();
    assert!(outcome.is_err(), "the turn should have failed");
    assert!(
        !session.opened,
        "a conversation the engine never accepted must not be marked open"
    );

    // The retry has to carry the message the first attempt never delivered.
    let seen_before = script.seen().len();
    let _ = engine.chat(&mut session).user("second attempt").send();
    let delivered: Vec<String> = script.seen().into_iter().skip(seen_before).collect();
    assert!(
        delivered.iter().any(|m| m == "first attempt"),
        "the retry skipped a message the backend never accepted; it sent {delivered:?}"
    );
}

#[test]
fn an_accepted_turn_is_recorded_so_the_next_one_sends_only_what_is_new() {
    // The other half: acknowledgement has to actually record delivery, or
    // every turn would resend the whole conversation.
    let engine = Engine::scripted(Script::new().say(["ok"]));
    let script = engine.script().clone();
    let mut session = Session::new();

    engine.chat(&mut session).user("first").send().unwrap();
    assert!(session.opened, "an accepted turn opens the conversation");

    let seen_before = script.seen().len();
    engine.chat(&mut session).user("second").send().unwrap();
    let delivered: Vec<String> = script.seen().into_iter().skip(seen_before).collect();

    assert!(
        !delivered.iter().any(|m| m == "first"),
        "a warm conversation resent history the engine already had: {delivered:?}"
    );
    assert!(
        delivered.iter().any(|m| m == "second"),
        "the new message never reached the backend: {delivered:?}"
    );
}
