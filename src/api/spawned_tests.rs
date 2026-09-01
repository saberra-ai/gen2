//! The off-thread API: what a caller consuming a stream of [`Update`]s gets.
//!
//! This is the surface the README shows for anything that streams — a turn or
//! an agent run on a worker thread, tokens arriving as they decode, the
//! session handed back at the end — and it had no tests at all. Every path
//! through it needed a running model to observe, so none of it was observed.
//!
//! Against a script it is ordinary. What matters here is not the text: it is
//! that exactly one terminal update arrives, that the session comes back on
//! it, and that a caller who cancels or abandons the stream is left with
//! something coherent rather than a lost conversation.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use schemars::JsonSchema;
use serde::Deserialize;

use super::engine::Engine;
use super::session::Session;
use super::spawned::Update;
use super::stream::Finish;
use super::tools::{FunctionTool, ToolOutput};
use crate::test_support::{Gate, Script, Step};

#[derive(Deserialize, JsonSchema)]
struct City {
    /// City to look up.
    city: String,
}

/// Text from every `Delta`, joined.
fn deltas(updates: &[Update]) -> String {
    updates
        .iter()
        .filter_map(|u| match u {
            Update::Delta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

/// Assert the stream contract: exactly one terminal update, and it is last.
///
/// The same shape [`assert_valid_trace`](crate::test_support::assert_valid_trace)
/// enforces at the controller. Restated here because this is a different
/// vocabulary — a consumer of `Update` never sees a `ControllerEvent` — and a
/// caller matching on `Done` to hand the session onward is broken by either
/// zero of them or two.
#[track_caller]
fn assert_one_ending(updates: &[Update]) {
    let endings: Vec<usize> = updates
        .iter()
        .enumerate()
        .filter(|(_, u)| matches!(u, Update::Done { .. } | Update::Failed { .. }))
        .map(|(i, _)| i)
        .collect();

    assert_eq!(
        endings.len(),
        1,
        "a run must end exactly once, got {} endings in {} updates",
        endings.len(),
        updates.len(),
    );
    assert_eq!(
        endings[0],
        updates.len() - 1,
        "the ending must be the last update; {} arrived after it",
        updates.len() - 1 - endings[0],
    );
}

// ── A spawned turn ──────────────────────────────────────────────────────────

#[test]
fn a_spawned_turn_streams_its_tokens_and_hands_the_session_back() {
    let engine = Arc::new(Engine::scripted(Script::new().say(["hel", "lo"])));
    let turn = engine.chat_owned(Session::new()).user("Hi").spawn();

    let updates: Vec<Update> = turn.collect();

    assert_one_ending(&updates);
    assert_eq!(deltas(&updates), "hello");

    let Some(Update::Done {
        completion,
        session,
    }) = updates.last()
    else {
        panic!("expected Done, got {:?}", updates.last());
    };
    assert_eq!(completion.text, "hello");
    assert_eq!(completion.finish, Finish::Eos);
    assert_eq!(
        session.latest_text().as_deref(),
        Some("hello"),
        "the reply must already be in the session the caller gets back"
    );
}

#[test]
fn the_session_that_comes_back_carries_what_the_caller_put_in() {
    let engine = Arc::new(Engine::scripted(Script::new().say(["ok"])));
    let mut session = Session::new().with_system("Be terse.");
    session.push_user("earlier turn");

    let turn = engine.chat_owned(session).user("Now this.").spawn();
    let updates: Vec<Update> = turn.collect();

    let Some(Update::Done { session, .. }) = updates.last() else {
        panic!("expected Done");
    };
    let text: Vec<String> = session
        .messages()
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect();
    let joined = text.join("\n");
    assert!(
        joined.contains("Be terse."),
        "system prompt lost:\n{joined}"
    );
    assert!(joined.contains("earlier turn"), "history lost:\n{joined}");
    assert!(joined.contains("Now this."), "this turn lost:\n{joined}");
}

#[test]
fn a_spawned_turn_that_generates_nothing_still_ends() {
    let engine = Arc::new(Engine::scripted(Script::new().say([])));
    let updates: Vec<Update> = engine
        .chat_owned(Session::new())
        .user("Hi")
        .spawn()
        .collect();

    assert_one_ending(&updates);
    assert_eq!(deltas(&updates), "");
}

#[test]
fn a_backend_failure_reaches_the_caller_with_the_session_intact() {
    // The reason `Failed` carries a session at all: a caller who loses the
    // conversation on every transient error cannot retry.
    let engine = Arc::new(Engine::scripted(
        Script::new().program([Step::Fail("the GPU fell over")]),
    ));
    let updates: Vec<Update> = engine
        .chat_owned(Session::new())
        .user("Hi")
        .spawn()
        .collect();

    assert_one_ending(&updates);
    match updates.last() {
        Some(Update::Failed { session, .. }) => {
            assert!(
                !session.messages().is_empty(),
                "a failed turn must still hand back the conversation to retry with"
            );
        }
        // A backend error that the controller reports as a completed-but-empty
        // turn is also acceptable; what is not acceptable is losing the
        // session or never ending.
        Some(Update::Done { session, .. }) => {
            assert!(!session.messages().is_empty());
        }
        other => panic!("expected an ending, got {other:?}"),
    }
}

#[test]
fn cancelling_a_spawned_turn_ends_it_and_keeps_what_was_generated() {
    // A cancelled reply is still a reply — the documented contract, and the
    // one a UI depends on when a user hits stop.
    let gate = Gate::new();
    let engine = Arc::new(Engine::scripted(Script::new().program([
        Step::token("kept"),
        Step::Hold(Arc::clone(&gate)),
        Step::token("never seen"),
        Step::eos(),
    ])));

    let mut turn = engine.chat_owned(Session::new()).user("Hi").spawn();
    let first = turn.next().expect("a first update");
    assert!(
        matches!(first, Update::Delta(ref t) if t == "kept"),
        "{first:?}"
    );
    assert!(gate.wait_until_reached(), "the turn should be mid-flight");

    turn.cancel().expect("cancel should reach the controller");
    gate.open();

    let mut updates = vec![first];
    updates.extend(turn);
    assert_one_ending(&updates);
    assert!(
        deltas(&updates).contains("kept"),
        "what was generated before the stop belongs to the caller: {:?}",
        deltas(&updates)
    );
}

#[test]
fn a_canceller_works_from_another_thread() {
    // The whole reason `Canceller` exists: the handle that stops a turn has to
    // be movable somewhere else, because the thread consuming the stream is
    // busy consuming the stream.
    let gate = Gate::new();
    let engine = Arc::new(Engine::scripted(Script::new().program([
        Step::token("kept"),
        Step::Hold(Arc::clone(&gate)),
        Step::token("never seen"),
        Step::eos(),
    ])));

    let turn = engine.chat_owned(Session::new()).user("Hi").spawn();
    let canceller = turn.canceller();

    let stopper = std::thread::spawn({
        let gate = Arc::clone(&gate);
        move || {
            assert!(gate.wait_until_reached());
            let _ = canceller.cancel();
            gate.open();
        }
    });

    let updates: Vec<Update> = turn.collect();
    stopper.join().expect("the stopping thread should finish");

    assert_one_ending(&updates);
}

#[test]
fn dropping_the_stream_mid_turn_does_not_wedge_the_engine() {
    // A caller that stops reading is ordinary: a closed tab, an abandoned
    // request. The engine must still serve the next one.
    let engine = Arc::new(Engine::scripted(Script::new().say(["a", "b", "c"])));

    for _ in 0..20 {
        drop(engine.chat_owned(Session::new()).user("Hi").spawn());
    }

    let updates: Vec<Update> = engine
        .chat_owned(Session::new())
        .user("Still there?")
        .spawn()
        .collect();
    assert_one_ending(&updates);
}

#[test]
fn a_spawned_turn_reports_the_session_it_belongs_to() {
    let engine = Arc::new(Engine::scripted(Script::new().say(["ok"])));
    let session = Session::new();
    let expected = session.id().to_string();

    let turn = engine.chat_owned(session).user("Hi").spawn();
    assert_eq!(
        turn.session_id(),
        expected,
        "a caller holding several turns identifies them by session"
    );
    let _: Vec<Update> = turn.collect();
}

// ── A spawned agent ─────────────────────────────────────────────────────────

fn weather() -> (Arc<AtomicUsize>, FunctionTool<City>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let tool = FunctionTool::new(
        "get_weather",
        "Current weather for a city",
        move |_ctx, a: City| {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutput::from(format!("sunny in {}", a.city)))
            }
        },
    );
    (calls, tool)
}

#[test]
fn a_spawned_agent_reports_each_tool_call_and_its_result() {
    // The updates a UI renders as "running get_weather…" and then its outcome.
    // Nothing else tells a consumer a tool ran.
    let (calls, tool) = weather();
    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("It is sunny."), Step::eos()],
    ])));

    let updates: Vec<Update> = engine
        .agent_owned(Session::new())
        .add_tool(tool)
        .goal("Weather in Paris?")
        .spawn()
        .collect();

    assert_one_ending(&updates);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let announced: Vec<&str> = updates
        .iter()
        .filter_map(|u| match u {
            Update::ToolCall { tool, .. } => Some(tool.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        announced,
        ["get_weather"],
        "a consumer learns a tool is running only from ToolCall"
    );

    let results: Vec<(&str, bool)> = updates
        .iter()
        .filter_map(|u| match u {
            Update::ToolResult { tool, result, .. } => Some((tool.as_str(), result.is_ok())),
            _ => None,
        })
        .collect();
    assert_eq!(
        results,
        [("get_weather", true)],
        "and learns it finished only from ToolResult"
    );
}

#[test]
fn a_tool_call_is_announced_before_its_result() {
    // Ordering a UI depends on: a spinner cannot be replaced by an outcome it
    // was never told about.
    let (_, tool) = weather();
    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("done"), Step::eos()],
    ])));

    let updates: Vec<Update> = engine
        .agent_owned(Session::new())
        .add_tool(tool)
        .goal("Weather?")
        .spawn()
        .collect();

    let call_at = updates
        .iter()
        .position(|u| matches!(u, Update::ToolCall { .. }))
        .expect("a call should be announced");
    let result_at = updates
        .iter()
        .position(|u| matches!(u, Update::ToolResult { .. }))
        .expect("a result should be announced");
    assert!(
        call_at < result_at,
        "the result at {result_at} preceded the call at {call_at}"
    );
}

#[test]
fn a_failing_tool_reports_its_failure_through_the_stream() {
    let failing = FunctionTool::new(
        "get_weather",
        "Current weather for a city",
        |_ctx, _a: City| async move { Err(super::tools::ToolError::Failed("service is down".into())) },
    );
    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("I could not check."), Step::eos()],
    ])));

    let updates: Vec<Update> = engine
        .agent_owned(Session::new())
        .add_tool(failing)
        .goal("Weather?")
        .spawn()
        .collect();

    assert_one_ending(&updates);
    let failed = updates
        .iter()
        .any(|u| matches!(u, Update::ToolResult { result, .. } if result.is_err()));
    assert!(
        failed,
        "a consumer must be able to see that a tool failed, not just that it ran"
    );
}

#[test]
fn a_spawned_agent_hands_its_session_back_with_the_whole_exchange() {
    let (_, tool) = weather();
    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("It is sunny in Paris."), Step::eos()],
    ])));

    let updates: Vec<Update> = engine
        .agent_owned(Session::new())
        .add_tool(tool)
        .goal("Weather in Paris?")
        .spawn()
        .collect();

    let Some(Update::Done { session, .. }) = updates.last() else {
        panic!("expected Done, got {:?}", updates.last());
    };
    let transcript: String = session
        .messages()
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript.contains("sunny in Paris"),
        "the tool result the answer rests on must survive into the caller's \
         session:\n{transcript}"
    );
}

#[test]
fn a_spawned_agent_reports_the_session_it_belongs_to() {
    let engine = Arc::new(Engine::scripted(Script::new().say(["done"])));
    let session = Session::new();
    let expected = session.id().to_string();

    let run = engine.agent_owned(session).goal("Anything").spawn();
    assert_eq!(run.session_id(), expected);
    let _: Vec<Update> = run.collect();
}

#[test]
fn dropping_an_agent_run_mid_flight_does_not_wedge_the_engine() {
    let (_, tool) = weather();
    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("done"), Step::eos()],
    ])));

    for _ in 0..10 {
        let (_, t) = weather();
        drop(
            engine
                .agent_owned(Session::new())
                .add_tool(t)
                .goal("Weather?")
                .spawn(),
        );
    }

    let updates: Vec<Update> = engine
        .agent_owned(Session::new())
        .add_tool(tool)
        .goal("Weather?")
        .spawn()
        .collect();
    assert_one_ending(&updates);
}

// ── Steering ────────────────────────────────────────────────────────────────

#[test]
fn a_follow_up_reaches_the_model_at_the_next_step() {
    // The difference from an interrupt: nothing is cut short, the message just
    // joins the conversation before the model writes again.
    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![Step::token("first answer"), Step::eos()],
        vec![Step::token("second answer"), Step::eos()],
    ])));

    let run = engine.agent_owned(Session::new()).goal("Start").spawn();
    let steering = run.steering();
    steering.follow_up("also check the tests");
    assert!(
        steering.is_pending(),
        "a queued message is pending until the loop takes it"
    );

    let updates: Vec<Update> = run.collect();
    assert_one_ending(&updates);

    let Some(Update::Done { session, .. }) = updates.last() else {
        panic!("expected Done, got {:?}", updates.last());
    };
    let transcript: String = session
        .messages()
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript.contains("also check the tests"),
        "a follow-up must reach the conversation, or the caller's steer is \
         silently dropped:\n{transcript}"
    );
}

#[test]
fn a_spawned_run_can_interrupt_its_own_generation_but_a_borrowed_one_cannot() {
    // Not a limitation to work around: a borrowed agent has no engine handle
    // to ask, so its steer lands at the next step boundary instead. The
    // difference is worth stating because it changes what a caller can promise
    // a user who pressed stop.
    let engine = Arc::new(Engine::scripted(Script::new().say(["done"])));

    let run = engine.agent_owned(Session::new()).goal("Go").spawn();
    assert!(
        run.steering().can_interrupt_generation(),
        "a spawned run owns an engine handle and can cut a generation short"
    );
    let _: Vec<Update> = run.collect();

    let mut session = Session::new();
    let borrowed = engine.agent(&mut session);
    assert!(
        !borrowed.steering().can_interrupt_generation(),
        "a borrowed agent has no handle to stop with, and must say so rather \
         than appearing to offer it"
    );
}

#[test]
fn queued_steers_keep_their_order() {
    let engine = Arc::new(Engine::scripted(Script::new().say(["done"])));
    let run = engine.agent_owned(Session::new()).goal("Go").spawn();
    let steering = run.steering();

    steering.follow_up("first");
    steering.follow_up("second");
    steering.follow_up("third");
    assert!(steering.is_pending());

    let updates: Vec<Update> = run.collect();
    let Some(Update::Done { session, .. }) = updates.last() else {
        panic!("expected Done");
    };
    let transcript: String = session
        .messages()
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");

    let positions: Vec<Option<usize>> = ["first", "second", "third"]
        .iter()
        .map(|needle| transcript.find(needle))
        .collect();
    if positions.iter().all(|p| p.is_some()) {
        let p: Vec<usize> = positions.into_iter().flatten().collect();
        assert!(
            p[0] < p[1] && p[1] < p[2],
            "steers arrived out of order: {p:?}"
        );
    }
}

#[test]
fn steering_a_finished_run_is_harmless() {
    // A UI's stop button outlives the run it belongs to. Pressing it after the
    // answer arrived must not panic or resurrect anything.
    let engine = Arc::new(Engine::scripted(Script::new().say(["done"])));
    let run = engine.agent_owned(Session::new()).goal("Go").spawn();
    let steering = run.steering();
    let _: Vec<Update> = run.collect();

    steering.follow_up("too late");
    steering.interrupt("also too late");
    assert!(
        steering.is_pending(),
        "the queue still accepts, and nobody reads it"
    );
}
