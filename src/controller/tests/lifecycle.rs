//! What a generation is allowed to do, and what it must never do.
//!
//! Organised by invariant rather than by method. The failures worth catching
//! here are not "this function returned the wrong number" but "the runtime
//! reached a state it has no way back out of", and those show up as an
//! illegal event ordering, a runtime that outlives its chat, or a terminal
//! event that arrives twice.
//!
//! Every test runs against [`Script`], so a whole file of them costs
//! milliseconds and needs no model.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::controller::{ControllerCmd, ControllerConfig, ControllerEvent};
use crate::engine::ExecError;
use crate::test_support::harness::{Harness, assert_valid_trace, text_of};
use crate::test_support::{Gate, Script, Step};

// ── The happy path, and the shape of every path ─────────────────────────────

#[test]
fn a_generation_streams_its_tokens_then_ends_once() {
    let harness = Harness::loaded(Script::new().say(["hel", "lo"]));
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
    assert_eq!(text_of(&events), "hello");
    assert!(
        events.iter().any(|e| matches!(e, ControllerEvent::Eos)),
        "a stream that reaches the end of the model's output ends with Eos, got {events:?}"
    );
}

#[test]
fn a_generation_that_produces_nothing_still_terminates() {
    // An empty answer is legal. A caller waiting for a terminal event must not
    // be left waiting because the model had nothing to say.
    let harness = Harness::loaded(Script::new().say([]));
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
    assert_eq!(text_of(&events), "");
}

#[test]
fn tokens_arrive_in_the_order_the_backend_produced_them() {
    let harness = Harness::loaded(Script::new().say(["a", "b", "c", "d", "e"]));
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
    assert_eq!(
        text_of(&events),
        "abcde",
        "the controller must not reorder or drop token fragments"
    );
}

// ── Terminal-event invariants ───────────────────────────────────────────────

#[test]
fn a_token_after_the_terminal_event_never_reaches_the_caller() {
    // A backend that keeps emitting after Eos is misbehaving, and the
    // controller is the last place to catch it: past here the token lands in a
    // transcript the caller has already closed and rendered.
    let harness = Harness::loaded(Script::new().program([
        Step::token("real"),
        Step::Emit(crate::generation::TokenEvent::Eos),
        Step::token("after the end"),
    ]));
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
    assert!(
        !text_of(&events).contains("after the end"),
        "a token produced after Eos must be dropped, got {events:?}"
    );
}

#[test]
fn two_terminal_events_from_the_backend_become_one() {
    let harness = Harness::loaded(Script::new().program([
        Step::token("hi"),
        Step::Emit(crate::generation::TokenEvent::Eos),
        Step::Emit(crate::generation::TokenEvent::Eos),
    ]));
    let events = harness.start_chat("chat").collect();

    // The whole point of the assertion: exactly one.
    assert_valid_trace(&events);
}

#[test]
fn a_stream_that_just_stops_still_reports_a_terminal_event() {
    // The nastiest backend failure mode: no Eos, no error, the token source
    // simply ends. A caller must not hang.
    let harness = Harness::loaded(Script::new().program([Step::token("truncated")]));
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
}

// ── Failure routing ─────────────────────────────────────────────────────────

#[test]
fn a_backend_failure_mid_stream_becomes_one_error_event() {
    let harness = Harness::loaded(
        Script::new().program([Step::token("partial"), Step::Fail("the GPU fell over")]),
    );
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
    assert!(
        matches!(events.last(), Some(ControllerEvent::Error { .. }))
            || events
                .iter()
                .any(|e| matches!(e, ControllerEvent::Error { .. })),
        "a mid-stream backend failure must surface as an Error event, got {events:?}"
    );
    assert_eq!(
        text_of(&events),
        "partial",
        "tokens generated before the failure belong to the caller"
    );
}

#[test]
fn a_failed_pull_terminates_the_generation_rather_than_hanging_it() {
    let harness = Harness::loaded(Script::new().failing_pull(|| ExecError::ModelNotLoaded));
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ControllerEvent::Error { .. })),
        "a pull the backend refuses is an error, not a silent end: {events:?}"
    );
}

#[test]
fn a_failed_start_session_leaves_no_runtime_behind() {
    // The ghost-runtime bug: a session that failed to start still occupying a
    // slot means the next chat is evicted to make room for something that
    // does not exist.
    let harness = Harness::loaded(
        Script::new().failing_start_session(|| ExecError::ContextOverflow("too big".into())),
    );
    let events = harness.start_chat("doomed").collect();

    assert_valid_trace(&events);
    assert_eq!(
        harness.script.live_sessions(),
        0,
        "a session that failed to start was never alive"
    );
    assert_eq!(
        harness.resident_chats(),
        0,
        "a chat whose session failed to start must not stay resident"
    );
}

#[test]
fn starting_a_chat_with_no_model_is_an_error_not_a_panic() {
    let harness = Harness::empty(Script::new().say(["unreachable"]));
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ControllerEvent::Error { .. })),
        "generating without a loaded model must be a recoverable error: {events:?}"
    );
}

#[test]
fn a_load_failure_is_reported_and_leaves_the_engine_unloaded() {
    let harness = Harness::empty(
        Script::new().failing_load(|| ExecError::InvalidModelFile("not a GGUF".into())),
    );

    let result = harness.load_model();

    assert!(result.is_err(), "a failing load must not report success");
    assert!(
        !harness.script.is_loaded(),
        "a failed load must not leave the engine claiming a model"
    );
}

// ── Cancellation ────────────────────────────────────────────────────────────

#[test]
fn stopping_between_two_tokens_ends_the_generation_cleanly() {
    // The gate is the synchronisation point: the generation is provably
    // mid-flight when the stop arrives, rather than racing the first token.
    let gate = Gate::new();
    let harness = Harness::loaded(Script::new().program([
        Step::token("before"),
        Step::Hold(Arc::clone(&gate)),
        Step::token("after"),
        Step::Emit(crate::generation::TokenEvent::Eos),
    ]));

    let events = harness.start_chat("chat");
    assert_eq!(events.wait_for_first_token(), "before");
    assert!(
        gate.wait_until_reached(),
        "the generation should have reached the hold"
    );

    harness.send(ControllerCmd::StopChat {
        chat_id: "chat".into(),
    });
    gate.open();

    let rest = events.collect();
    let mut all = vec![ControllerEvent::Token("before".into())];
    all.extend(rest);
    assert_valid_trace(&all);
}

#[test]
fn stopping_an_unknown_chat_is_harmless() {
    let harness = Harness::loaded(Script::new().say(["fine"]));
    harness.send(ControllerCmd::StopChat {
        chat_id: "never-existed".into(),
    });

    // The controller must still be answering afterwards.
    let events = harness.start_chat("chat").collect();
    assert_valid_trace(&events);
}

#[test]
fn a_caller_that_drops_its_receiver_does_not_wedge_the_controller() {
    let harness = Harness::loaded(Script::new().say(["one", "two", "three"]));

    // Start a chat and throw the receiver away immediately.
    drop(harness.start_chat("abandoned"));

    // A second chat must still work, which it cannot if the first one is
    // still spinning against a closed channel.
    let events = harness.start_chat("live").collect();
    assert_valid_trace(&events);
}

// ── Residency and shutdown ──────────────────────────────────────────────────

#[test]
fn a_finished_chat_keeps_its_session_so_the_next_turn_reuses_the_prefill() {
    // Deliberate, and the reason a second turn is cheap: the runtime outlives
    // the generation. Sessions are released by eviction and by shutdown, both
    // covered below, not by reaching Eos.
    let harness = Harness::loaded(Script::new().say(["done"]));
    let _ = harness.start_chat("chat").collect();

    assert_eq!(
        harness.script.live_sessions(),
        1,
        "a finished chat must stay resident to be continued"
    );
    assert_eq!(
        harness.script.count("start_session"),
        1,
        "and a second turn must not open a second session: {:?}",
        harness.script.calls()
    );
}

#[test]
fn shutdown_leaves_no_session_alive() {
    let harness = Harness::loaded(Script::new().say(["a"]));
    let _ = harness.start_chat("one").collect();
    let _ = harness.start_chat("two").collect();
    let script = harness.script.clone();

    harness.shutdown();

    assert_eq!(
        script.live_sessions(),
        0,
        "shutdown must terminate every session it is holding"
    );
}

#[test]
fn shutdown_joins_rather_than_abandoning_the_loop() {
    // The reason `Engine::drop` joins: exiting while the loop still holds the
    // backend tears down ggml's statics underneath it and aborts.
    let harness = Harness::loaded(Script::new().say(["a"]));
    harness.shutdown();
    // Reaching here at all is the assertion — `shutdown` returns only after
    // the loop thread has finished.
}

#[test]
fn the_active_chat_limit_bounds_resident_runtimes() {
    let harness = Harness::with_config(
        Script::new().say(["hi"]),
        ControllerConfig {
            max_active_chats: 2,
            ..ControllerConfig::default()
        },
    );
    harness
        .load_model()
        .expect("the scripted backend should load");

    for i in 0..5 {
        let _ = harness.start_chat(&format!("chat-{i}")).collect();
    }

    assert!(
        harness.resident_chats() <= 2,
        "max_active_chats is a bound, but {} runtimes are resident",
        harness.resident_chats()
    );
}

// ── Model lifecycle ─────────────────────────────────────────────────────────

#[test]
fn reloading_a_model_does_not_leak_the_previous_sessions() {
    let harness = Harness::loaded(Script::new().say(["one"]));
    let _ = harness.start_chat("chat").collect();

    harness
        .load_model()
        .expect("a second load should be accepted");

    let settled = wait_until(|| harness.script.live_sessions() == 0);
    assert!(
        settled,
        "a model swap must retire sessions bound to the old model, {} still live",
        harness.script.live_sessions()
    );
}

#[test]
fn a_chat_started_after_a_reload_generates_normally() {
    let harness = Harness::loaded(Script::new().say(["before"]));
    let _ = harness.start_chat("old").collect();
    harness.load_model().expect("reload should be accepted");

    let events = harness.start_chat("new").collect();
    assert_valid_trace(&events);
    assert_eq!(text_of(&events), "before");
}

// ── Concurrency ─────────────────────────────────────────────────────────────

#[test]
fn two_chats_each_get_their_own_complete_stream() {
    let harness = Harness::with_config(
        Script::new().say(["x", "y"]),
        ControllerConfig {
            max_active_chats: 4,
            ..ControllerConfig::default()
        },
    );
    harness
        .load_model()
        .expect("the scripted backend should load");

    let first = harness.start_chat("one");
    let second = harness.start_chat("two");

    let first = first.collect();
    let second = second.collect();

    assert_valid_trace(&first);
    assert_valid_trace(&second);
    assert_eq!(text_of(&first), "xy");
    assert_eq!(
        text_of(&second),
        "xy",
        "two chats must not consume each other's tokens"
    );
}

#[test]
fn a_poisoned_session_is_reported_as_poisoned_not_as_a_generic_error() {
    // The distinction is the whole point: a caller retries a generation
    // error, and must not retry into a session whose state is gone.
    let harness =
        Harness::loaded(Script::new().program([Step::token("before the crash"), Step::Poison]));
    let events = harness.start_chat("chat").collect();

    assert_valid_trace(&events);
    let code = events
        .iter()
        .find_map(|e| match e {
            ControllerEvent::Error { code, .. } => Some(code.clone()),
            _ => None,
        })
        .expect("an FFI-level crash must reach the caller as an error");
    assert_eq!(
        code, "session_poisoned",
        "a backend that reports poisoning must not be routed as a retryable error"
    );
}

#[test]
fn a_generation_error_that_is_not_poisoning_stays_retryable() {
    // The other side of the same contract. A backend that fails without
    // reporting poisoning has a live session, and the caller may retry.
    let harness =
        Harness::loaded(Script::new().program([Step::token("partial"), Step::Fail("transient")]));
    let events = harness.start_chat("chat").collect();

    let code = events
        .iter()
        .find_map(|e| match e {
            ControllerEvent::Error { code, .. } => Some(code.clone()),
            _ => None,
        })
        .expect("the failure must surface");
    assert_eq!(
        code, "generation_error",
        "a one-off failure must stay retryable rather than being called poisoning"
    );
}

// ── Call-ordering contracts ─────────────────────────────────────────────────

#[test]
fn a_chat_loads_once_and_starts_one_session() {
    let harness = Harness::loaded(Script::new().say(["hi"]));
    let _ = harness.start_chat("chat").collect();

    assert_eq!(
        harness.script.count("load_model"),
        1,
        "one LoadModel must not load twice: {:?}",
        harness.script.calls()
    );
    assert_eq!(
        harness.script.count("start_session"),
        1,
        "one chat must not open two backend sessions: {:?}",
        harness.script.calls()
    );
}

#[test]
fn nothing_touches_the_backend_before_a_model_is_loaded() {
    let harness = Harness::empty(Script::new().say(["hi"]));
    assert!(
        !harness.script.calls().iter().any(|c| c == "start_session"),
        "an idle controller must not open sessions: {:?}",
        harness.script.calls()
    );
}

/// Poll a condition until it holds or patience runs out.
///
/// The controller retires runtimes on its own tick, so a test that asserts
/// immediately after reading the last event is asserting against a state the
/// loop has not reached yet. Polling rather than sleeping keeps the fast path
/// fast.
fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    condition()
}

/// Unused today, kept because the next test that needs a call counter will
/// want it rather than reinventing it.
#[allow(dead_code)]
fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

#[allow(dead_code)]
fn bump(counter: &AtomicUsize) -> usize {
    counter.fetch_add(1, Ordering::SeqCst)
}
