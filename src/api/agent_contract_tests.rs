//! What the agent loop does, on model behaviour tests choose.
//!
//! The live agent tests are useful and cannot be the primary proof: real model
//! behaviour is probabilistic, so a test that needs two tool calls in one turn
//! can only hope for them, and one that needs a budget exceeded has to coax
//! the model into being verbose. The concurrency test in `live_inference.rs`
//! openly accepts the model deciding to make a single call.
//!
//! Here the model is a script. The loop's own responsibilities — dispatch,
//! argument validation, failure routing, budgets, approval, progress
//! detection — become ordinary assertions.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use schemars::JsonSchema;
use serde::Deserialize;

use super::agent::ApprovalMode;
use super::engine::Engine;
use super::session::Session;
use super::stream::{Budget, Finish};
use super::tools::{Decision, FunctionTool, ToolOutput};
use crate::test_support::{Script, Step};

#[derive(Deserialize, JsonSchema)]
struct City {
    /// City to look up.
    city: String,
}

/// Everything in the transcript, tool payloads included.
///
/// Serialized rather than read field by field: a tool result and an assistant
/// reply live in different shapes of `MessageBody`, and what these tests care
/// about is whether some text reached the conversation at all.
fn transcript_of(session: &Session) -> String {
    session
        .messages()
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A tool that records every call it received.
fn recording(name: &'static str) -> (Arc<AtomicUsize>, FunctionTool<City>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let tool = FunctionTool::new(name, "Current weather for a city", move |_ctx, a: City| {
        let counter = Arc::clone(&counter);
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput::from(format!("sunny in {}", a.city)))
        }
    });
    (calls, tool)
}

// ── Dispatch ────────────────────────────────────────────────────────────────

#[test]
fn a_tool_call_reaches_the_handler_registered_under_that_name() {
    let (calls, weather) = recording("get_weather");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("It is sunny."), Step::eos()],
    ]));
    let mut session = Session::new();

    let done = engine
        .agent(&mut session)
        .add_tool(weather)
        .goal("What is the weather in Paris?")
        .expect("the run should complete");

    assert_eq!(calls.load(Ordering::SeqCst), 1, "the handler must have run");
    assert_eq!(done.tool_rounds, 1);
    assert_eq!(done.text, "It is sunny.");
}

#[test]
fn a_call_to_an_unregistered_tool_is_reported_rather_than_ignored() {
    // Silently dropping it leaves the model waiting for a result that never
    // comes, and it will call again.
    let (calls, weather) = recording("get_weather");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_wether", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("I cannot."), Step::eos()],
    ]));
    let mut session = Session::new();

    let done = engine
        .agent(&mut session)
        .add_tool(weather)
        .goal("What is the weather?")
        .expect("a typo in a tool name is the model's problem to recover from, not a crash");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "nothing may run for a name that was never registered"
    );
    let transcript = transcript_of(&session);
    assert!(
        transcript.contains("get_wether") || transcript.to_lowercase().contains("unknown"),
        "the model must be told the call failed so it can correct itself:\n{transcript}"
    );
    assert_eq!(done.finish, Finish::Eos);
}

#[test]
fn arguments_that_do_not_match_the_schema_never_reach_the_handler() {
    // The handler declares its own argument type, so it is entitled to assume
    // the shape. Validation is the loop's job.
    let (calls, weather) = recording("get_weather");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"town":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("Sorry."), Step::eos()],
    ]));
    let mut session = Session::new();

    engine
        .agent(&mut session)
        .add_tool(weather)
        .goal("What is the weather?")
        .expect("bad arguments are recoverable");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "a handler must not be called with arguments its type cannot hold"
    );
}

#[test]
fn malformed_json_arguments_are_recoverable() {
    let (calls, weather) = recording("get_weather");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", "{not json at all"),
            Step::eos(),
        ],
        vec![Step::token("Sorry."), Step::eos()],
    ]));
    let mut session = Session::new();

    let done = engine
        .agent(&mut session)
        .add_tool(weather)
        .goal("What is the weather?")
        .expect("a model emitting broken JSON must not fail the run");

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(done.finish, Finish::Eos);
}

#[test]
fn a_tool_that_fails_reports_the_failure_to_the_model() {
    // A failing tool is information, not an outage: the model may try another
    // approach, and only it knows what that is.
    let failing = FunctionTool::new(
        "get_weather",
        "Current weather for a city",
        |_ctx, _a: City| async move {
            Err(super::tools::ToolError::Failed(
                "the weather service is down".into(),
            ))
        },
    );
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("I could not check."), Step::eos()],
    ]));
    let mut session = Session::new();

    let done = engine
        .agent(&mut session)
        .add_tool(failing)
        .goal("What is the weather?")
        .expect("a failing tool must not fail the run");

    let transcript = transcript_of(&session);
    assert!(
        transcript.contains("weather service is down"),
        "the model needs the reason to choose what to do next:\n{transcript}"
    );
    assert_eq!(done.finish, Finish::Eos);
}

// ── Budgets ─────────────────────────────────────────────────────────────────

#[test]
fn a_step_budget_stops_a_model_that_keeps_calling_tools() {
    // The runaway-agent guard. Without it a model that always asks for one
    // more tool never returns. Each call names a different city, so this is
    // the budget stopping it rather than the repeat detector below.
    let (calls, weather) = recording("get_weather");
    let wandering = (0..50).map(|i| {
        vec![
            Step::tool_call("get_weather", &format!(r#"{{"city":"city-{i}"}}"#)),
            Step::eos(),
        ]
    });
    let engine = Engine::scripted(Script::new().turns(wandering));
    let mut session = Session::new();

    let done = engine
        .agent(&mut session)
        .add_tool(weather)
        .max_steps(3)
        .goal("Keep checking the weather")
        .expect("hitting a budget is an outcome, not an error");

    assert_eq!(done.finish, Finish::OutOfBudget(Budget::Steps));
    assert!(
        calls.load(Ordering::SeqCst) <= 3,
        "the budget is a bound: {} calls ran under a limit of 3",
        calls.load(Ordering::SeqCst),
    );
}

#[test]
fn a_model_repeating_one_call_is_stopped_before_its_budget_runs_out() {
    // Progress detection, and it fires earlier than the step budget on
    // purpose: a model asking the identical question a third time is stuck,
    // and spending the rest of the budget confirming that helps nobody.
    let (calls, weather) = recording("get_weather");
    let stuck = std::iter::repeat_n(
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        50,
    );
    let engine = Engine::scripted(Script::new().turns(stuck));
    let mut session = Session::new();

    let done = engine
        .agent(&mut session)
        .add_tool(weather)
        .max_steps(40)
        .goal("Keep checking the weather")
        .expect("giving up is an outcome, not an error");

    assert!(
        matches!(done.finish, Finish::GaveUp(_)),
        "an identical call repeated should end the run as stuck, got {:?}",
        done.finish,
    );
    assert!(
        calls.load(Ordering::SeqCst) < 40,
        "giving up early is the point; {} calls ran",
        calls.load(Ordering::SeqCst),
    );
}

#[test]
fn a_run_that_answers_immediately_uses_no_steps() {
    let (calls, weather) = recording("get_weather");
    let engine =
        Engine::scripted(Script::new().turns([vec![Step::token("Paris is sunny."), Step::eos()]]));
    let mut session = Session::new();

    let done = engine
        .agent(&mut session)
        .add_tool(weather)
        .goal("What is the weather?")
        .expect("answering without tools is the simplest path, not a failure");

    assert_eq!(done.tool_rounds, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(done.finish, Finish::Eos);
    assert_eq!(done.text, "Paris is sunny.");
}

// ── Approval ────────────────────────────────────────────────────────────────

/// A tool that declares itself risky, and counts the times it actually ran.
fn risky(name: &'static str) -> (Arc<AtomicUsize>, FunctionTool<City>) {
    let (calls, tool) = recording(name);
    (calls, tool.risky())
}

#[test]
fn a_denied_tool_never_runs() {
    // The entire point of approval. If a denial could still execute, the
    // callback would be theatre.
    let (calls, delete) = risky("delete_everything");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("delete_everything", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("Understood."), Step::eos()],
    ]));
    let mut session = Session::new();

    let outcome = engine
        .agent(&mut session)
        .add_tool(delete)
        .approval(ApprovalMode::AskOnRisky)
        .on_approval(|_name, _args, _spec| Decision::Deny("not today".into()))
        .goal("Clean up");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "a denied tool must not execute"
    );
    // Denial ends the run rather than handing the refusal back for the model
    // to work around, which is what `Decision::Deny` documents: a denied
    // action is not something to retry into. The reason travels with it.
    let error = outcome.expect_err("a denial ends the run");
    assert!(
        error.to_string().contains("not today"),
        "the caller's own reason should survive to the error: {error}"
    );
}

#[test]
fn an_allowed_risky_tool_runs() {
    let (calls, delete) = risky("delete_everything");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("delete_everything", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("Done."), Step::eos()],
    ]));
    let mut session = Session::new();

    engine
        .agent(&mut session)
        .add_tool(delete)
        .approval(ApprovalMode::AskOnRisky)
        .on_approval(|_name, _args, _spec| Decision::Allow)
        .goal("Clean up")
        .expect("an allowed call runs");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn a_safe_tool_is_not_put_through_approval() {
    // Asking about every safe call trains the user to approve without reading,
    // which is worse than not asking.
    let asked = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&asked);
    let (calls, weather) = recording("get_weather");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("Sunny."), Step::eos()],
    ]));
    let mut session = Session::new();

    engine
        .agent(&mut session)
        .add_tool(weather)
        .approval(ApprovalMode::AskOnRisky)
        .on_approval(move |_name, _args, _spec| {
            seen.fetch_add(1, Ordering::SeqCst);
            Decision::Allow
        })
        .goal("What is the weather?")
        .expect("the run should complete");

    assert_eq!(
        asked.load(Ordering::SeqCst),
        0,
        "AskOnRisky must not ask about a tool that declared itself safe"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn approval_is_off_unless_asked_for() {
    // The default is unattended: a library that stopped for approval nobody
    // configured would deadlock every headless caller.
    let (calls, delete) = risky("delete_everything");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("delete_everything", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("Done."), Step::eos()],
    ]));
    let mut session = Session::new();

    engine
        .agent(&mut session)
        .add_tool(delete)
        .goal("Clean up")
        .expect("the run should complete");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "without an approval mode, a risky tool runs like any other"
    );
}

// ── The transcript ──────────────────────────────────────────────────────────

#[test]
fn the_session_carries_the_whole_exchange_including_the_tool_result() {
    // The caller owns the transcript, so everything the model saw has to be in
    // it — otherwise a resumed conversation is missing the evidence the
    // answer was based on.
    let (_, weather) = recording("get_weather");
    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("It is sunny in Paris."), Step::eos()],
    ]));
    let mut session = Session::new();

    engine
        .agent(&mut session)
        .add_tool(weather)
        .goal("What is the weather in Paris?")
        .expect("the run should complete");

    let transcript = transcript_of(&session);
    assert!(
        transcript.contains("What is the weather in Paris?"),
        "the goal is part of the conversation:\n{transcript}"
    );
    assert!(
        transcript.contains("sunny in Paris"),
        "the tool result the answer rests on must be in the transcript:\n{transcript}"
    );
}
