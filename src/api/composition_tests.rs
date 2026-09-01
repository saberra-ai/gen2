//! Composing tools: bundling them, reusing a configuration, and delegating to
//! a sub-agent.
//!
//! Three README features that had almost no coverage between them. All three
//! exist to be *reused*, so the failures worth catching are about what
//! survives reuse: a bundle registered twice, a configuration that quietly
//! drops half its settings on the second run, a sub-agent that reaches tools
//! its parent never gave it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use schemars::JsonSchema;
use serde::Deserialize;

use super::agent::ApprovalMode;
use super::agent_config::AgentConfig;
use super::engine::Engine;
use super::session::Session;
use super::stream::{Budget, Finish};
use super::tools::{AgentTool, FunctionTool, ToolOutput, ToolSearch, ToolSet};
use crate::test_support::{Script, Step};

#[derive(Deserialize, JsonSchema)]
struct City {
    /// City to look up.
    city: String,
}

fn counting(name: &'static str) -> (Arc<AtomicUsize>, FunctionTool<City>) {
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

fn plain(name: &'static str) -> FunctionTool<City> {
    counting(name).1
}

// ── ToolSet ─────────────────────────────────────────────────────────────────

#[test]
fn a_tool_set_keeps_what_was_put_in_it_and_in_order() {
    let set = ToolSet::new()
        .add(plain("read_file"))
        .add(plain("write_file"))
        .add(plain("list_dir"));

    assert_eq!(set.len(), 3);
    assert!(!set.is_empty());
    assert_eq!(set.names(), ["read_file", "write_file", "list_dir"]);
}

#[test]
fn an_empty_tool_set_is_legal() {
    // A bundle assembled from a filtered list can legitimately come out empty,
    // and that is the caller's business rather than an error.
    let set = ToolSet::new();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
    assert!(set.names().is_empty());
}

#[test]
fn extend_adds_a_whole_iterator_at_once() {
    let set = ToolSet::new()
        .add(plain("first"))
        .extend([plain("second"), plain("third")]);
    assert_eq!(set.names(), ["first", "second", "third"]);
}

#[test]
fn a_tool_set_can_be_registered_with_more_than_one_agent() {
    // The reason it is `Clone`: the same bundle is resident in one agent and
    // deferred in another, and neither may consume it.
    let filesystem = ToolSet::new()
        .add(plain("read_file"))
        .add(plain("write_file"));

    let engine = Engine::scripted(Script::new().say(["done"]));
    let mut first = Session::new();
    let mut second = Session::new();

    engine
        .agent(&mut first)
        .add_tools(filesystem.clone())
        .goal("One")
        .expect("the first agent should run");
    engine
        .agent(&mut second)
        .add_tools(filesystem)
        .goal("Two")
        .expect("the bundle must survive being registered once already");
}

#[test]
fn a_bundled_tool_is_dispatched_the_same_as_one_added_directly() {
    let (calls, weather) = counting("get_weather");
    let set = ToolSet::new().add(weather);
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
        .add_tools(set)
        .goal("Weather?")
        .expect("the run should complete");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "bundling a tool must not change whether it can be called"
    );
}

// ── AgentConfig ─────────────────────────────────────────────────────────────

#[test]
fn a_config_reports_what_it_holds() {
    let config = AgentConfig::new()
        .add_tool(plain("a"))
        .add_tool(plain("b"))
        .max_steps(4);
    assert_eq!(config.len(), 2);
    assert!(!config.is_empty());
    assert!(AgentConfig::new().is_empty());
}

#[test]
fn a_config_builds_the_same_agent_every_time() {
    // The point of a reusable configuration: the second run must be
    // configured like the first, not like a default.
    let (calls, weather) = counting("get_weather");
    let config = AgentConfig::new().add_tool(weather);

    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("One."), Step::eos()],
        vec![
            Step::tool_call("get_weather", r#"{"city":"Berlin"}"#),
            Step::eos(),
        ],
        vec![Step::token("Two."), Step::eos()],
    ]));
    let mut session = Session::new();

    config
        .agent(&engine, &mut session)
        .goal("Paris?")
        .expect("the first run should complete");
    config
        .agent(&engine, &mut session)
        .goal("Berlin?")
        .expect("the second run should complete");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the tool must be registered on the second run as well as the first"
    );
}

#[test]
fn a_configs_budget_applies_to_every_run_it_builds() {
    // A budget that only bound the first run would let the second one loop.
    let (_, weather) = counting("get_weather");
    let config = AgentConfig::new().add_tool(weather).max_steps(2);

    let wandering = (0..40).map(|i| {
        vec![
            Step::tool_call("get_weather", &format!(r#"{{"city":"city-{i}"}}"#)),
            Step::eos(),
        ]
    });
    let engine = Engine::scripted(Script::new().turns(wandering));
    let mut session = Session::new();

    for run in 0..2 {
        let done = config
            .agent(&engine, &mut session)
            .goal("Keep going")
            .expect("hitting a budget is an outcome");
        assert_eq!(
            done.finish,
            Finish::OutOfBudget(Budget::Steps),
            "run {run} was not bounded by the configuration's budget"
        );
    }
}

#[test]
fn a_config_carries_its_approval_mode_into_every_run() {
    let (calls, delete) = counting("delete_everything");
    let config = AgentConfig::new()
        .add_tool(delete.risky())
        .approval(ApprovalMode::AskOnRisky);

    let engine = Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("delete_everything", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("Understood."), Step::eos()],
    ]));
    let mut session = Session::new();

    let outcome = config
        .agent(&engine, &mut session)
        .on_approval(|_n, _a, _s| super::tools::Decision::Deny("no".into()))
        .goal("Clean up");

    assert!(outcome.is_err(), "a denial ends the run");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "a configuration that carried AskOnRisky must still gate the call"
    );
}

#[test]
fn a_config_can_build_an_off_thread_run_too() {
    let (calls, weather) = counting("get_weather");
    let config = AgentConfig::new().add_tool(weather);
    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("Sunny."), Step::eos()],
    ])));

    let updates: Vec<super::spawned::Update> = config
        .agent_owned(&engine, Session::new())
        .goal("Weather?")
        .spawn()
        .collect();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the same configuration must apply whether the run is on this thread \
         or another"
    );
    assert!(
        updates
            .iter()
            .any(|u| matches!(u, super::spawned::Update::Done { .. })),
        "the spawned run should have ended cleanly"
    );
}

#[test]
fn deferred_tools_in_a_config_survive_into_the_agent() {
    let config = AgentConfig::new()
        .add_tool(plain("resident"))
        .defer_tool(plain("deferred"))
        .tool_search(ToolSearch::Bm25);
    assert_eq!(config.len(), 2, "both loadings count toward what is held");

    let engine = Engine::scripted(Script::new().say(["done"]));
    let mut session = Session::new();
    config
        .agent(&engine, &mut session)
        .goal("Anything")
        .expect("a config mixing resident and deferred tools must build");
}

// ── AgentTool: an agent as somebody else's tool ─────────────────────────────

#[test]
fn a_sub_agent_presents_itself_as_an_ordinary_tool() {
    let engine = Arc::new(Engine::scripted(Script::new().say(["done"])));
    let researcher = AgentTool::new("researcher", "Investigates a question", Arc::clone(&engine));

    let spec = super::tools::Tool::spec(&researcher);
    assert_eq!(spec.name, "researcher");
    assert!(
        !spec.description.is_empty(),
        "the parent model chooses to delegate by reading this"
    );
    assert!(
        spec.input_schema.is_object(),
        "a tool the model calls needs an argument schema"
    );
}

#[test]
fn a_sub_agent_runs_its_own_loop_and_returns_its_answer() {
    // Two turns for the parent's delegation, two for the sub-agent's own run:
    // the script is shared, which is what makes the nesting observable.
    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("researcher", r#"{"task":"look into rust"}"#),
            Step::eos(),
        ],
        vec![Step::token("the delegate said so"), Step::eos()],
        vec![Step::token("parent answer"), Step::eos()],
    ])));

    let researcher =
        AgentTool::new("researcher", "Investigates a question", Arc::clone(&engine)).max_steps(2);

    let mut session = Session::new();
    let done = engine
        .agent(&mut session)
        .add_tool(researcher)
        .max_steps(3)
        .goal("Find out about rust")
        .expect("delegating must not fail the parent run");

    assert!(
        done.tool_rounds >= 1,
        "the parent should have delegated at least once"
    );
}

#[test]
fn a_sub_agent_gets_only_the_tools_it_was_given() {
    // The main reason to delegate: narrowing what a delegate can reach. If the
    // parent's tools leaked in, the boundary would be decorative.
    let (parent_calls, parent_only) = counting("parent_only");
    let (child_calls, child_only) = counting("child_only");

    let engine = Arc::new(Engine::scripted(Script::new().turns([
        vec![
            Step::tool_call("researcher", r#"{"task":"go"}"#),
            Step::eos(),
        ],
        // The sub-agent's own turn: it reaches for the parent's tool, which it
        // was never given.
        vec![
            Step::tool_call("parent_only", r#"{"city":"Paris"}"#),
            Step::eos(),
        ],
        vec![Step::token("could not"), Step::eos()],
        vec![Step::token("parent answer"), Step::eos()],
    ])));

    let researcher = AgentTool::new("researcher", "Investigates", Arc::clone(&engine))
        .tools([child_only])
        .max_steps(3);

    let mut session = Session::new();
    engine
        .agent(&mut session)
        .add_tool(parent_only)
        .add_tool(researcher)
        .max_steps(4)
        .goal("Delegate this")
        .expect("the run should complete");

    assert_eq!(
        parent_calls.load(Ordering::SeqCst),
        0,
        "the sub-agent reached a tool that belongs to its parent — the whole \
         point of giving a delegate its own set is that it cannot"
    );
    let _ = child_calls;
}

#[test]
fn a_sub_agents_step_budget_is_its_own() {
    // A delegate that ran away would spend the parent's budget as well as its
    // own, and the parent could not tell why it never came back.
    let (calls, child_tool) = counting("child_tool");
    let looping = std::iter::once(vec![
        Step::tool_call("researcher", r#"{"task":"go"}"#),
        Step::eos(),
    ])
    .chain((0..40).map(|i| {
        vec![
            Step::tool_call("child_tool", &format!(r#"{{"city":"c-{i}"}}"#)),
            Step::eos(),
        ]
    }));

    let engine = Arc::new(Engine::scripted(Script::new().turns(looping)));
    let researcher = AgentTool::new("researcher", "Investigates", Arc::clone(&engine))
        .tools([child_tool])
        .max_steps(2);

    let mut session = Session::new();
    let done = engine
        .agent(&mut session)
        .add_tool(researcher)
        .max_steps(3)
        .goal("Delegate")
        .expect("a delegate hitting its budget must not fail the parent");

    assert!(
        calls.load(Ordering::SeqCst) <= 2,
        "the delegate ran {} tool calls against its own limit of 2",
        calls.load(Ordering::SeqCst),
    );
    let _ = done;
}
