//! Model switching, cache identity, residency, and concurrency, on scripted
//! models (api_spec.md §4.2, §4.5, §17, §23).
//!
//! Every assertion is against what the backend was handed — how many
//! runtimes it started, what each one saw, when weights were unloaded and
//! reloaded — not against what the facade believes. A switch that "works"
//! by re-prefilling every turn is correct but not what §17.2 promises; a
//! reuse that shows the model an edited transcript is worse than either.

use std::sync::Arc;

use crate::api::runtime::Runtime;
use crate::api::session::Session;
use crate::api::tool_defs::{ToolDefinition, ToolSet};
use crate::generation::ThinkingMode;
use crate::test_support::Script;
use crate::types::message::Message;

fn two_models(
    runtime: &Runtime,
) -> (
    crate::api::model::Model,
    Script,
    crate::api::model::Model,
    Script,
) {
    let a = Script::new().say(["from a"]);
    let b = Script::new().say(["from b"]);
    let qwen = runtime.scripted_in(a.clone(), 0);
    let llama = runtime.scripted_in(b.clone(), 0);
    (qwen, a, llama, b)
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(name).description("does something")
}

// ── §17 / §28.4: switching is a normal turn ─────────────────────────────

#[test]
fn the_same_session_on_two_models_gets_a_runtime_on_each() {
    let runtime = Runtime::new().unwrap();
    let (qwen, a, llama, b) = two_models(&runtime);
    let mut session = Session::new().with_system("Be concise.");

    let first = qwen
        .turn(&mut session)
        .user("My name is Bob")
        .run()
        .unwrap();
    assert_eq!(first.text(), "from a");
    let second = llama
        .turn(&mut session)
        .user("What is my name?")
        .run()
        .unwrap();
    assert_eq!(second.text(), "from b");

    // One runtime per model, and the second model was shown the whole
    // conversation the first one built — system prompt included.
    assert_eq!(a.count("start_session"), 1);
    assert_eq!(b.count("start_session"), 1);
    assert_eq!(
        b.seen(),
        vec![
            "Be concise.",
            "My name is Bob",
            "from a",
            "What is my name?"
        ],
        "the second model must see the transcript, not just the new message"
    );
    assert_eq!(session.bound_engines().len(), 2);
    // The session itself is model-agnostic: nothing was converted.
    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "user", "assistant"]);
}

#[test]
fn switching_back_rebuilds_so_the_first_model_sees_the_seconds_reply() {
    // §17.2: qwen → llama → qwen. Qwen's runtime is still there, but the
    // tail now holds llama's reply — an assistant message qwen did not
    // generate, which the backends' append path would skip. Correctness
    // first: qwen is rebuilt, and shown llama's reply.
    let runtime = Runtime::new().unwrap();
    let (qwen, a, llama, _b) = two_models(&runtime);
    let mut session = Session::new();

    qwen.turn(&mut session).user("one").run().unwrap();
    llama.turn(&mut session).user("two").run().unwrap();
    qwen.turn(&mut session).user("three").run().unwrap();

    assert_eq!(
        a.count("start_session"),
        2,
        "a foreign reply forces a rebuild"
    );
    assert_eq!(a.count("append_messages"), 0);
    let seen = a.seen();
    assert_eq!(
        &seen[1..],
        ["one", "from a", "two", "from b", "three"],
        "the rebuilt runtime saw llama's reply in its place"
    );
}

#[test]
fn switching_back_appends_when_the_tail_holds_no_foreign_reply() {
    // §17.2, the reuse case: the parked qwen binding is still valid when
    // the session comes back with nothing but its own reply and new user
    // messages after the prefix qwen holds — here because llama's reply
    // was removed before switching back.
    let runtime = Runtime::new().unwrap();
    let (qwen, a, llama, _b) = two_models(&runtime);
    let mut session = Session::new();

    qwen.turn(&mut session).user("one").run().unwrap();
    let reply = llama.turn(&mut session).user("two").run().unwrap();
    session.remove_message(reply.message_id().unwrap()).unwrap();
    qwen.turn(&mut session).user("three").run().unwrap();

    assert_eq!(a.count("start_session"), 1, "qwen's runtime was reused");
    assert_eq!(a.count("append_messages"), 1, "and appended to once");
    let seen = a.seen();
    assert_eq!(seen[0], "one");
    assert_eq!(&seen[1..], ["from a", "two", "three"]);
}

#[test]
fn an_untouched_session_appends_on_the_same_model() {
    let runtime = Runtime::new().unwrap();
    let (qwen, a, _llama, _b) = two_models(&runtime);
    let mut session = Session::new();
    qwen.turn(&mut session).user("one").run().unwrap();
    qwen.turn(&mut session).user("two").run().unwrap();
    assert_eq!(a.count("start_session"), 1);
    assert_eq!(a.count("append_messages"), 1);
}

// ── §17.1: cache identity ───────────────────────────────────────────────

#[test]
fn editing_a_message_invalidates_the_parked_runtime() {
    let runtime = Runtime::new().unwrap();
    let (qwen, a, llama, _b) = two_models(&runtime);
    let mut session = Session::new();

    let first = session.push_user("one");
    qwen.turn(&mut session).run().unwrap();
    llama.turn(&mut session).user("two").run().unwrap();
    // Rewrite history under qwen's parked runtime.
    session
        .replace_message(first, Message::user("ONE"))
        .unwrap();
    qwen.turn(&mut session).user("three").run().unwrap();

    assert_eq!(
        a.count("start_session"),
        2,
        "a changed prefix must rebuild qwen's runtime"
    );
    let seen = a.seen();
    assert!(
        seen[seen.len() - 5..].starts_with(&["ONE".to_string()]),
        "the rebuilt runtime saw the edited message: {seen:?}"
    );
    assert!(
        !seen[seen.len() - 5..].iter().any(|m| m == "one"),
        "and not the old one: {seen:?}"
    );
}

#[test]
fn changing_the_system_prompt_invalidates_every_runtime() {
    let runtime = Runtime::new().unwrap();
    let (qwen, a, llama, b) = two_models(&runtime);
    let mut session = Session::new().with_system("A");
    qwen.turn(&mut session).user("one").run().unwrap();
    llama.turn(&mut session).user("two").run().unwrap();
    session.set_system("B");
    qwen.turn(&mut session).user("three").run().unwrap();
    llama.turn(&mut session).user("four").run().unwrap();
    assert_eq!(a.count("start_session"), 2);
    assert_eq!(b.count("start_session"), 2);
    assert_eq!(a.seen().iter().filter(|m| m.as_str() == "B").count(), 1);
    assert_eq!(b.seen().iter().filter(|m| m.as_str() == "B").count(), 1);
}

#[test]
fn changing_the_tools_invalidates_and_the_rebuild_carries_them() {
    let runtime = Runtime::new().unwrap();
    let (qwen, a, llama, _b) = two_models(&runtime);
    let mut session = Session::new().with_tools(ToolSet::new().with(tool("read")));
    qwen.turn(&mut session).user("one").run().unwrap();
    llama.turn(&mut session).user("two").run().unwrap();
    session.add_tool(tool("write"));
    qwen.turn(&mut session).user("three").run().unwrap();
    assert_eq!(a.count("start_session"), 2);
    assert_eq!(
        a.tools_seen().last().unwrap(),
        &["read".to_string(), "write".to_string()]
    );
}

#[test]
fn a_different_reasoning_policy_reopens_only_that_model() {
    let runtime = Runtime::new().unwrap();
    let (qwen, a, llama, b) = two_models(&runtime);
    let mut session = Session::new();
    qwen.turn(&mut session)
        .user("one")
        .reasoning(ThinkingMode::Off)
        .run()
        .unwrap();
    llama.turn(&mut session).user("two").run().unwrap();
    qwen.turn(&mut session)
        .user("three")
        .reasoning(ThinkingMode::On)
        .run()
        .unwrap();
    assert_eq!(a.count("start_session"), 2, "the policy is pinned at start");
    assert_eq!(a.thinking_seen(), vec![ThinkingMode::Off, ThinkingMode::On]);
    assert_eq!(b.count("start_session"), 1);
}

#[test]
fn the_fingerprint_moves_with_the_revision_and_only_then() {
    let mut s = Session::new();
    let f0 = s.fingerprint();
    assert_eq!(s.fingerprint(), f0, "reading is free and stable");

    s.set_system("A");
    let f1 = s.fingerprint();
    assert_ne!(f1, f0, "the system prompt is context");
    s.set_system("A");
    assert_eq!(
        s.fingerprint(),
        f1,
        "setting the same prompt is not a change"
    );

    let id = s.push_user("hi");
    let f2 = s.fingerprint();
    assert_ne!(f2, f1);

    s.set_tools(ToolSet::new().with(tool("read")));
    let f3 = s.fingerprint();
    assert_ne!(f3, f2, "tools are context");
    s.set_tools(ToolSet::new().with(tool("read")).with(tool("write")));
    let f4 = s.fingerprint();
    assert_ne!(f4, f3, "tool order and membership are context");

    let replaced = s.replace_message(id, Message::user("HI")).unwrap();
    let f5 = s.fingerprint();
    assert_ne!(f5, f4, "content is context");
    // Replacing with identical text still mints a new record: identity is
    // part of the fingerprint too.
    s.replace_message(replaced, Message::user("HI")).unwrap();
    assert_ne!(s.fingerprint(), f5, "message identity is context");

    // Same content built the same way → same fingerprint: it is a digest of
    // the context, not of the session's random id.
    let mut x = Session::new().with_system("S");
    x.push_user("a");
    let mut y = Session::new().with_system("S");
    y.push_user("a");
    assert_eq!(x.fingerprint(), y.fingerprint());
    assert_eq!(x.revision(), y.revision());
}

// ── §4.2 / §4.5: residency ──────────────────────────────────────────────

#[test]
fn an_evicted_model_is_restored_by_its_next_turn() {
    let runtime = Runtime::builder()
        .resident_budget_mb(10_000)
        .build()
        .unwrap();
    let (qwen, a, llama, _b) = two_models(&runtime);
    assert!(runtime.residency().is_resident(qwen.id()));
    assert!(runtime.residency().is_resident(llama.id()));

    runtime.evict(&qwen).unwrap();
    let after = runtime.residency();
    assert!(!after.is_resident(qwen.id()), "{after:?}");
    assert!(after.is_resident(llama.id()));
    assert_eq!(a.count("unload_model"), 1);
    assert!(!a.is_loaded());

    // The handle is still good: the turn restores the weights first.
    let text = qwen.generate("hi").text().unwrap();
    assert_eq!(text, "from a");
    assert_eq!(a.count("reload_model"), 1);
    assert!(runtime.residency().is_resident(qwen.id()));

    // Evicting again is idempotent, and so is evicting what is not there.
    runtime.evict(&qwen).unwrap();
    runtime.evict(&qwen).unwrap();
    assert_eq!(a.count("unload_model"), 2);
    runtime.preload(&qwen).unwrap();
    assert_eq!(a.count("reload_model"), 2);
    assert!(runtime.residency().is_resident(qwen.id()));
    assert_eq!(
        runtime.stats().evictions,
        0,
        "explicit evictions are not counted"
    );
}

#[test]
fn a_session_survives_its_model_being_evicted_between_turns() {
    let runtime = Runtime::new().unwrap();
    let (qwen, a, _llama, _b) = two_models(&runtime);
    let mut session = Session::new().with_tools(ToolSet::new().with(tool("read")));
    qwen.turn(&mut session).user("one").run().unwrap();
    runtime.evict(&qwen).unwrap();
    qwen.turn(&mut session).user("two").run().unwrap();
    assert_eq!(
        a.count("start_session"),
        2,
        "the runtime went with the weights"
    );
    assert_eq!(a.tools_seen().last().unwrap(), &["read".to_string()]);
    let seen = a.seen();
    assert_eq!(&seen[seen.len() - 3..], ["one", "from a", "two"]);
}

#[test]
fn loading_past_the_budget_evicts_the_least_recently_used() {
    let runtime = Runtime::builder().resident_budget_mb(100).build().unwrap();
    let a = Script::new().say(["a"]);
    let b = Script::new().say(["b"]);
    let c = Script::new().say(["c"]);
    let ma = runtime.scripted_in(a.clone(), 60);
    let mb = runtime.scripted_in(b.clone(), 60);
    // 60 + 60 > 100: loading b evicted a, the only other resident model.
    let r = runtime.residency();
    assert!(!r.is_resident(ma.id()), "{r:?}");
    assert!(r.is_resident(mb.id()));
    assert_eq!(a.count("unload_model"), 1);
    assert_eq!(runtime.stats().evictions, 1);

    // Using a restores it, at b's expense.
    ma.generate("x").text().unwrap();
    let r = runtime.residency();
    assert!(r.is_resident(ma.id()));
    assert!(!r.is_resident(mb.id()), "{r:?}");
    assert_eq!(a.count("reload_model"), 1);

    // A third model that fits alongside a evicts nothing.
    let mc = runtime.scripted_in(c.clone(), 30);
    let r = runtime.residency();
    assert!(r.is_resident(ma.id()) && r.is_resident(mc.id()));
    assert_eq!(runtime.stats().evictions, 2, "{:?}", runtime.stats());

    // Now b comes back (60): a is the least recently used of {a, c}.
    mb.generate("y").text().unwrap();
    let r = runtime.residency();
    assert!(r.is_resident(mb.id()));
    assert!(r.is_resident(mc.id()), "c was used more recently than a");
    assert!(!r.is_resident(ma.id()));
    assert_eq!(r.resident_mb(), 90);
    let stats = runtime.stats();
    assert_eq!(stats.models, 3);
    assert_eq!(stats.resident_models, 2);
    assert_eq!(stats.estimated_resident_mb, 90);
}

#[test]
fn a_model_that_does_not_fit_at_all_is_still_loaded() {
    // Nothing to evict cannot mean nothing to load: the engine's own
    // admission decides, and a scripted engine admits.
    let runtime = Runtime::builder().resident_budget_mb(10).build().unwrap();
    let a = Script::new().say(["a"]);
    let ma = runtime.scripted_in(a.clone(), 60);
    assert!(runtime.residency().is_resident(ma.id()));
    assert_eq!(ma.generate("x").text().unwrap(), "a");
}

#[test]
fn residency_controls_refuse_a_model_from_another_runtime() {
    let one = Runtime::new().unwrap();
    let two = Runtime::new().unwrap();
    let foreign = two.scripted_in(Script::new(), 0);
    assert!(one.evict(&foreign).is_err());
    assert!(one.preload(&foreign).is_err());
    assert!(one.residency().models.is_empty());
}

#[test]
fn hardware_and_residency_are_reachable_under_advanced() {
    let runtime = Runtime::new().unwrap();
    let hw: crate::advanced::runtime::HardwareProfile = runtime.hardware();
    assert!(hw.cpu_cores > 0);
    let snapshot: crate::advanced::runtime::ResidencySnapshot = runtime.residency();
    assert!(snapshot.models.is_empty());
    let stats: crate::advanced::runtime::RuntimeStats = runtime.stats();
    assert_eq!(stats.models, 0);
}

// ── §23: concurrency ────────────────────────────────────────────────────

#[test]
fn runtime_and_model_cross_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Runtime>();
    assert_send_sync::<crate::api::model::Model>();
    assert_send_sync::<Session>();
}

#[test]
fn two_sessions_on_one_model_from_two_threads_both_complete() {
    let runtime = Runtime::new().unwrap();
    let script = Script::new().say(["ok"]);
    let model = Arc::new(runtime.scripted_in(script.clone(), 0));
    let handles: Vec<_> = (0..2)
        .map(|i| {
            let model = Arc::clone(&model);
            std::thread::spawn(move || {
                let mut session = Session::new();
                let text = model
                    .turn(&mut session)
                    .user(format!("thread {i}"))
                    .run()
                    .map(|r| r.text().to_string());
                (text, session)
            })
        })
        .collect();
    for h in handles {
        let (text, session) = h.join().expect("no panic");
        assert_eq!(text.unwrap(), "ok");
        assert_eq!(session.messages().len(), 2);
    }
    assert_eq!(script.count("start_session"), 2);
    assert_eq!(script.live_sessions(), 2, "both conversations are held");
}

// ── S2.3 ⬜: the rebuild after a controller-side eviction ────────────────

#[test]
fn a_conversation_rebuilt_after_capacity_eviction_keeps_its_tools_and_thinking() {
    // One resident conversation at a time, so the second session evicts the
    // first inside the controller — without the facade noticing.
    let runtime = Runtime::builder().max_active_sessions(1).build().unwrap();
    let script = Script::new().say(["ok"]);
    let model = runtime.scripted_in(script.clone(), 0);
    let mut first = Session::new().with_tools(ToolSet::new().with(tool("read")));
    model
        .turn(&mut first)
        .user("one")
        .reasoning(ThinkingMode::Off)
        .run()
        .unwrap();
    let mut other = Session::new();
    model.turn(&mut other).user("other").run().unwrap();

    model
        .turn(&mut first)
        .user("two")
        .reasoning(ThinkingMode::Off)
        .run()
        .unwrap();

    // Three starts: first, other, and first again — rebuilt from the
    // transcript the facade carried, with what it was opened with.
    assert_eq!(script.count("start_session"), 3);
    assert_eq!(script.tools_seen()[2], vec!["read".to_string()]);
    assert_eq!(script.thinking_seen()[2], ThinkingMode::Off);
    // And shown the conversation exactly once — not the tail twice.
    let seen = script.seen();
    assert_eq!(&seen[seen.len() - 3..], ["one", "ok", "two"], "{seen:?}");
}
