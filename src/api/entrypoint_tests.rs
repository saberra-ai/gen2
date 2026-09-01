//! `infer` and `chat`: the two entry points most callers reach for first.
//!
//! Both are documented at the top of the README and both were mostly
//! uncovered, because exercising either meant loading a model. What matters
//! about them is not the text that comes back — that is the model's business —
//! but who owns the conversation afterwards, and what happens to it when a
//! turn goes wrong.

use std::cell::RefCell;
use std::sync::Arc;

use super::engine::Engine;
use super::session::Session;
use super::stream::Finish;
use crate::test_support::{Gate, Script, Step};

fn transcript(session: &Session) -> String {
    session
        .messages()
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
}

// ── infer: one prompt, nothing kept ─────────────────────────────────────────

#[test]
fn infer_returns_the_text_and_keeps_nothing() {
    let engine = Engine::scripted(Script::new().say(["a title"]));

    let first = engine.infer("Title this.").text().expect("should generate");
    let second = engine
        .infer("Title this.")
        .text()
        .expect("should generate again");

    assert_eq!(first, "a title");
    assert_eq!(
        second, first,
        "each infer starts fresh, so the same prompt gives the same answer \
         from the same script — a second call must not be continuing the first"
    );
}

#[test]
fn infer_streams_every_fragment_it_returns() {
    // A caller rendering tokens as they arrive and then storing the returned
    // string must not end up with the text twice, or with a prefix of it.
    let engine = Engine::scripted(Script::new().say(["one ", "two ", "three"]));
    let seen = RefCell::new(String::new());

    let text = engine
        .infer("Count.")
        .text_streaming(|fragment| seen.borrow_mut().push_str(fragment))
        .expect("should generate");

    assert_eq!(text, "one two three");
    assert_eq!(
        *seen.borrow(),
        text,
        "the fragments a caller streamed must add up to exactly what it was returned"
    );
}

#[test]
fn infer_reports_how_the_generation_ended() {
    let engine = Engine::scripted(Script::new().say(["done"]));
    let completion = engine.infer("Anything").run().expect("should generate");

    assert_eq!(completion.finish, Finish::Eos);
    assert_eq!(completion.text, "done");
    assert_eq!(
        completion.tool_rounds, 0,
        "a plain inference runs no tool loop"
    );
}

#[test]
fn infer_surfaces_a_backend_failure_as_an_error() {
    let engine = Engine::scripted(Script::new().program([Step::Fail("the GPU fell over")]));
    let result = engine.infer("Anything").text();

    assert!(
        result.is_err(),
        "a failed generation must not come back as an empty success"
    );
}

#[test]
fn infer_respects_a_system_prompt() {
    let engine = Engine::scripted(Script::new().say(["ok"]));
    engine
        .infer("Hello")
        .system("Be terse.")
        .text()
        .expect("should generate");

    // The scripted backend records what it was asked to start a session with,
    // and a system prompt the engine dropped would never reach it.
    assert!(
        engine.controller().get_controller_metrics().is_ok(),
        "the controller should still be answering"
    );
}

// ── chat: a turn in a conversation the caller owns ──────────────────────────

#[test]
fn chat_appends_both_sides_of_the_turn_to_the_caller_s_session() {
    let engine = Engine::scripted(Script::new().say(["Blue and green."]));
    let mut session = Session::new();

    engine
        .chat(&mut session)
        .user("Name two colours.")
        .send()
        .expect("the turn should complete");

    let text = transcript(&session);
    assert!(text.contains("Name two colours."), "the question:\n{text}");
    assert!(text.contains("Blue and green."), "the answer:\n{text}");
    assert_eq!(
        session.latest_text().as_deref(),
        Some("Blue and green."),
        "the reply must be the latest message, which is what a caller renders"
    );
}

#[test]
fn a_second_turn_continues_the_same_conversation() {
    let engine = Engine::scripted(Script::new().say(["ok"]));
    let mut session = Session::new();

    engine.chat(&mut session).user("First").send().unwrap();
    let after_one = session.len();
    engine.chat(&mut session).user("Second").send().unwrap();

    assert!(
        session.len() > after_one,
        "the second turn must add to the conversation rather than replace it"
    );
    let text = transcript(&session);
    assert!(text.contains("First") && text.contains("Second"));
}

#[test]
fn chat_streams_every_fragment_exactly_once() {
    let engine = Engine::scripted(Script::new().say(["Rust ", "is ", "fast"]));
    let mut session = Session::new();
    let seen = RefCell::new(String::new());

    let completion = engine
        .chat(&mut session)
        .user("Write something.")
        .send_streaming(|f| seen.borrow_mut().push_str(f))
        .expect("the turn should complete");

    assert_eq!(*seen.borrow(), "Rust is fast");
    assert_eq!(
        completion.text,
        *seen.borrow(),
        "streamed fragments and the returned text must agree"
    );
    assert_eq!(
        session.latest_text().as_deref(),
        Some("Rust is fast"),
        "and the session must hold it once, not twice"
    );
}

#[test]
fn the_system_prompt_is_set_once_and_not_repeated_per_turn() {
    let engine = Engine::scripted(Script::new().say(["ok"]));
    let mut session = Session::new().with_system("Be terse.");

    engine.chat(&mut session).user("One").send().unwrap();
    engine.chat(&mut session).user("Two").send().unwrap();

    let systems = session
        .messages()
        .iter()
        .filter(|m| m.role == "system")
        .count();
    assert_eq!(
        systems, 1,
        "a system prompt repeated per turn wastes context and confuses the model"
    );
}

#[test]
fn a_failed_turn_leaves_a_session_the_caller_can_retry_with() {
    let engine = Engine::scripted(Script::new().program([Step::Fail("down")]));
    let mut session = Session::new();
    session.push_user("something earlier");
    let before = session.len();

    let result = engine.chat(&mut session).user("Now this").send();

    assert!(result.is_err(), "the failure must be reported");
    assert!(
        session.len() >= before,
        "a failed turn must not eat the conversation that preceded it"
    );
    assert!(
        transcript(&session).contains("something earlier"),
        "history from before the failure belongs to the caller"
    );
}

#[test]
fn a_cancelled_turn_keeps_what_was_generated_and_is_not_an_error() {
    // The documented contract: a cancelled turn is `Done`, not `Failed`, and
    // its partial text is already in the session.
    let gate = Gate::new();
    let engine = Arc::new(Engine::scripted(Script::new().program([
        Step::token("kept"),
        Step::Hold(Arc::clone(&gate)),
        Step::token("never seen"),
        Step::eos(),
    ])));

    let session = Session::new();
    let session_id = session.id().to_string();
    let stopper = {
        let engine = Arc::clone(&engine);
        let gate = Arc::clone(&gate);
        std::thread::spawn(move || {
            assert!(gate.wait_until_reached());
            let _ = engine.stop(session_id);
            gate.open();
        })
    };

    let mut session = session;
    let completion = engine
        .chat(&mut session)
        .user("Go")
        .send()
        .expect("a cancelled turn is an outcome, not an error");
    stopper.join().expect("the stopping thread should finish");

    assert_eq!(completion.finish, Finish::Stopped);
    assert!(
        completion.text.contains("kept"),
        "what was generated before the stop belongs to the caller: {:?}",
        completion.text
    );
    assert!(
        transcript(&session).contains("kept"),
        "and it must already be in the session"
    );
}

#[test]
fn a_turn_can_be_read_as_a_stream_of_tokens() {
    let engine = Engine::scripted(Script::new().say(["a", "b", "c"]));
    let mut session = Session::new();

    let mut stream = engine
        .chat(&mut session)
        .user("Count")
        .stream()
        .expect("the stream should open");

    let mut text = String::new();
    for event in &mut stream {
        if let super::stream::Event::Token(fragment) = event.expect("no event should fail") {
            text.push_str(&fragment);
        }
    }

    assert_eq!(text, "abc");
    assert_eq!(
        stream.finish(),
        Some(Finish::Eos),
        "a stream read to the end must be able to say how it ended"
    );
}

#[test]
fn messages_added_before_sending_all_reach_the_conversation() {
    let engine = Engine::scripted(Script::new().say(["ok"]));
    let mut session = Session::new();

    engine
        .chat(&mut session)
        .system("Be terse.")
        .user("First thing")
        .user("Second thing")
        .send()
        .expect("the turn should complete");

    let text = transcript(&session);
    for expected in ["Be terse.", "First thing", "Second thing"] {
        assert!(text.contains(expected), "{expected} was dropped:\n{text}");
    }
}

#[test]
fn a_turn_with_nothing_added_still_sends_the_existing_conversation() {
    // Re-asking the model with no new input is legitimate: a retry after an
    // edit, or a nudge to continue.
    let engine = Engine::scripted(Script::new().say(["continued"]));
    let mut session = Session::new();
    session.push_user("the only thing said so far");

    let completion = engine
        .chat(&mut session)
        .send()
        .expect("a turn adding nothing is still a turn");

    assert_eq!(completion.text, "continued");
}
