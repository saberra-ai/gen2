//! Churn, and what has to come back to zero afterwards.
//!
//! Inference runtimes fail from lifecycle bugs more often than from algorithm
//! bugs. A leaked thread, a session that outlives its chat, a receiver dropped
//! at the wrong moment — none of these fail a single-shot test, and all of
//! them take down a process that has been running for a day.
//!
//! The pattern here is one shape: establish a baseline, churn hard, assert the
//! baseline came back. Against [`Script`] the churn is free, so these run with
//! the ordinary suite rather than being nightly-only.
//!
//! Numbers are deliberately modest. A leak shows up at a hundred iterations as
//! readily as at a hundred thousand, and a suite nobody runs catches nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::controller::{ControllerCmd, ControllerConfig, ControllerEvent};
use crate::test_support::harness::{Harness, assert_valid_trace};
use crate::test_support::{Gate, Script, Step};

/// Threads this process is running right now.
///
/// The leak check that matters: the controller owns a thread, and an engine
/// that fails to join it on drop leaks one per engine. Counted through the
/// platform rather than tracked in-process, so a thread the crate forgot about
/// is still counted.
#[cfg(target_os = "macos")]
fn thread_count() -> usize {
    // `ps -M` lists one line per thread of a process, plus a header.
    let out = std::process::Command::new("ps")
        .args(["-M", &std::process::id().to_string()])
        .output()
        .expect("ps should run");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .count()
        .saturating_sub(1)
}

#[cfg(target_os = "linux")]
fn thread_count() -> usize {
    std::fs::read_dir("/proc/self/task")
        .map(|d| d.count())
        .unwrap_or(0)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn thread_count() -> usize {
    0
}

/// Wait for a condition, or give up. Threads retire asynchronously, so an
/// assertion made the instant after a drop is asserting against a state the OS
/// has not reached.
fn settles(mut condition: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    condition()
}

#[test]
fn a_hundred_controllers_started_and_stopped_leak_no_threads() {
    // The invariant `Engine::drop` exists to hold. Each controller owns a
    // thread holding the backend's native context; one that is stopped but not
    // joined leaves it behind, and a long-lived host accumulates them until it
    // runs out.
    let baseline = thread_count();

    for _ in 0..100 {
        let harness = Harness::loaded(Script::new().say(["hi"]));
        harness.shutdown();
    }

    let settled = settles(|| thread_count() <= baseline + 2);
    assert!(
        settled,
        "started from {baseline} threads and ended at {} after 100 controllers",
        thread_count(),
    );
}

#[test]
fn a_thousand_chats_through_one_controller_leave_nothing_resident() {
    // Session churn on a single controller: the runtime map must not grow
    // without bound, and every backend session must eventually be released.
    let harness = Harness::with_config(
        Script::new().say(["ok"]),
        ControllerConfig {
            max_active_chats: 4,
            ..ControllerConfig::default()
        },
    );
    harness
        .load_model()
        .expect("the scripted backend should load");

    for i in 0..1_000 {
        let events = harness.start_chat(&format!("chat-{i}")).collect_to_end();
        assert_valid_trace(&events);
    }

    assert!(
        harness.resident_chats() <= 4,
        "the runtime map grew past max_active_chats to {}",
        harness.resident_chats(),
    );
    assert!(
        harness.script.live_sessions() <= 4,
        "backend sessions outlived their runtimes: {} still open",
        harness.script.live_sessions(),
    );
}

#[test]
fn shutting_down_after_heavy_churn_still_releases_every_session() {
    let harness = Harness::with_config(
        Script::new().say(["ok"]),
        ControllerConfig {
            max_active_chats: 8,
            ..ControllerConfig::default()
        },
    );
    harness
        .load_model()
        .expect("the scripted backend should load");
    for i in 0..200 {
        let _ = harness.start_chat(&format!("chat-{i}")).collect_to_end();
    }

    let script = harness.script.clone();
    harness.shutdown();

    assert_eq!(
        script.live_sessions(),
        0,
        "shutdown after churn must still terminate every session"
    );
}

#[test]
fn abandoning_every_receiver_does_not_wedge_the_controller() {
    // A caller dropping the receiver mid-generation is ordinary — a cancelled
    // request, a closed tab. Doing it repeatedly must not accumulate runtimes
    // spinning against closed channels.
    let harness = Harness::loaded(Script::new().say(["a", "b", "c", "d"]));

    for i in 0..200 {
        drop(harness.start_chat(&format!("abandoned-{i}")));
    }

    // The controller must still be answering, which it cannot be if the
    // abandoned runs are still holding it.
    let events = harness.start_chat("live").collect();
    assert_valid_trace(&events);
}

#[test]
fn repeated_model_reloads_do_not_accumulate_sessions() {
    // A host swapping models under a live engine: every reload invalidates the
    // sessions bound to the previous weights, and none of them may survive it.
    let harness = Harness::loaded(Script::new().say(["ok"]));

    for i in 0..50 {
        let _ = harness.start_chat(&format!("chat-{i}")).collect_to_end();
        harness.load_model().expect("a reload should be accepted");
    }

    let settled = settles(|| harness.script.live_sessions() == 0);
    assert!(
        settled,
        "sessions survived the model swaps that invalidated them: {} still open",
        harness.script.live_sessions(),
    );
}

#[test]
fn a_storm_of_stops_for_chats_that_do_not_exist_is_absorbed() {
    let harness = Harness::loaded(Script::new().say(["ok"]));

    for i in 0..2_000 {
        harness.send(ControllerCmd::StopChat {
            chat_id: format!("never-existed-{i}"),
        });
    }

    let events = harness.start_chat("live").collect();
    assert_valid_trace(&events);
}

#[test]
fn stopping_a_generation_mid_flight_repeatedly_leaves_the_loop_healthy() {
    // The interesting cancellation race, run enough times that a
    // once-in-a-while ordering problem has a chance to appear.
    for round in 0..40 {
        let gate = Gate::new();
        let harness = Harness::loaded(Script::new().program([
            Step::token("before"),
            Step::Hold(Arc::clone(&gate)),
            Step::token("after"),
            Step::eos(),
        ]));

        let events = harness.start_chat("chat");
        assert_eq!(events.wait_for_first_token(), "before");
        assert!(
            gate.wait_until_reached(),
            "round {round}: the generation should have reached the hold"
        );
        harness.send(ControllerCmd::StopChat {
            chat_id: "chat".into(),
        });
        gate.open();

        let mut all = vec![ControllerEvent::Token("before".into())];
        all.extend(events.collect());
        assert_valid_trace(&all);
    }
}

#[test]
fn a_backend_that_fails_every_generation_does_not_degrade_the_controller() {
    // Failure is not a special case that happens once. A provider that is down
    // fails every call, and the loop has to keep answering rather than
    // accumulating whatever it allocates per failure.
    let failures = Arc::new(AtomicUsize::new(0));
    let harness = Harness::loaded(Script::new().program([Step::Fail("down")]));

    for i in 0..300 {
        let events = harness.start_chat(&format!("chat-{i}")).collect_to_end();
        assert_valid_trace(&events);
        if events
            .iter()
            .any(|e| matches!(e, ControllerEvent::Error { .. }))
        {
            failures.fetch_add(1, Ordering::SeqCst);
        }
    }

    assert_eq!(
        failures.load(Ordering::SeqCst),
        300,
        "every run should have reported its failure"
    );
    assert!(
        harness.resident_chats() <= ControllerConfig::default().max_active_chats,
        "failed runs accumulated runtimes: {} resident",
        harness.resident_chats(),
    );
}
