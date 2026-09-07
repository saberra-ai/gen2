//! End-to-end proof that the crate actually runs inference.
//!
//! Unit tests cover the pieces; this drives the real thing — a real GGUF
//! through the real llama.cpp backend, asserting on real decoded tokens. It is
//! the test that would have caught a broken extraction that still compiled.
//!
//! It goes through the public API, so as an external test target it reaches
//! exactly what any other consumer reaches. That makes it proof of two things
//! at once: the engine generates, and the API is sufficient to make it.
//!
//! Point `PIO_TEST_MODEL` at a small instruct GGUF and run:
//!
//! ```sh
//! PIO_TEST_MODEL=/path/SmolLM2-360M-Instruct-Q4_K_M.gguf \
//!   cargo test --test live_inference --no-default-features --features metal -- --nocapture
//! ```
//!
//! Without `PIO_TEST_MODEL` the tests skip. Skipping is not passing: if the
//! variable IS set, every failure below is a hard failure — no silent fallback,
//! no "backend unavailable" escape hatch.

#![cfg(feature = "backend-llamacpp")]
// The first half of this file drives the previous facade, which stays until
// pio-app has switched (roadmap S5); its types are deprecated aliases now.
#![allow(deprecated)]

use std::path::PathBuf;

use gen2::Session;
use gen2::legacy::{Engine, Event, Finish};

fn test_model() -> Option<PathBuf> {
    let raw = std::env::var("PIO_TEST_MODEL").ok()?;
    let path = PathBuf::from(raw);
    assert!(
        path.exists(),
        "PIO_TEST_MODEL points at a file that does not exist: {}",
        path.display()
    );
    Some(path)
}

/// Load → generate → text, in three lines.
#[test]
fn generates_real_tokens_from_a_real_model() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = Engine::load(model).expect("real GGUF should load");
    let text = engine
        .infer("Reply with exactly one word: hello")
        .max_tokens(24)
        .greedy()
        .text()
        .expect("generation should succeed");

    eprintln!("--- generated: {text:?}");

    assert!(
        !text.trim().is_empty(),
        "decoded an empty string — the model loaded but produced no tokens"
    );
    assert!(
        text.chars().any(|c| c.is_alphabetic()),
        "output has no letters, so this is not decoded text: {text:?}"
    );
}

/// The stream reports how the generation ended, and ends on `Eos` rather than
/// running out of budget for a prompt this small.
#[test]
fn stream_reports_a_clean_finish() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = Engine::load(model).expect("real GGUF should load");
    let mut session = Session::new();
    let mut stream = engine
        .chat(&mut session)
        .user("Reply with exactly one word: hello")
        .max_tokens(24)
        .greedy()
        .stream()
        .expect("stream should start");

    let mut text = String::new();
    for event in &mut stream {
        // An error arrives as `Err` from the iterator, so it cannot be read
        // past — that is what stops a truncated reply looking complete.
        if let Event::Token(t) = event.expect("no event should be an error") {
            text.push_str(&t);
        }
    }

    assert!(!text.trim().is_empty(), "stream produced no text");
    assert_eq!(
        stream.finish(),
        Some(Finish::Eos),
        "expected the model to stop on its own, not be cut off"
    );
}

/// `.greedy()` is reproducible. Catches a sampler or KV cache that survived the
/// extraction but wired itself to the wrong state — a fresh chat must not
/// inherit the previous one's.
#[test]
fn greedy_decoding_is_reproducible() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = Engine::load(model).expect("real GGUF should load");
    let prompt = "Count: one two three";

    let first = engine.infer(prompt).max_tokens(16).greedy().text().unwrap();
    let second = engine.infer(prompt).max_tokens(16).greedy().text().unwrap();

    assert!(!first.trim().is_empty(), "first generation was empty");
    assert_eq!(
        first, second,
        "same prompt gave different text across two turns — \
         sampler or session state is not being reset"
    );
}

/// A seed pins sampling *without* pinning it to the single greedy answer.
///
/// `greedy()` forces temperature zero, so it proves nothing about the seed —
/// argmax is deterministic whatever the RNG does. This is the test that
/// actually exercises `.seed()`, and it failed before the seed reached the
/// sampler at all: five runs at one seed gave five different answers, because
/// `Settings::with_gen_spec_overrides` merged every sampling field except that
/// one.
#[test]
fn a_seed_makes_sampled_output_reproducible() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = Engine::load(model).expect("real GGUF should load");
    let prompt = "Invent a two-word name for a coffee shop.";
    let sample = |seed: u64| {
        engine
            .infer(prompt)
            .max_tokens(12)
            .temperature(0.9)
            .seed(seed)
            .text()
            .expect("generation should succeed")
    };

    let first = sample(42);
    assert!(!first.trim().is_empty(), "seeded generation was empty");
    for run in 1..4 {
        assert_eq!(
            sample(42),
            first,
            "run {run} at seed 42 differed from the first — a seed that does \
             not pin sampling is a reproducibility knob that does nothing"
        );
    }

    // And it must still be a seed rather than a constant: different seeds have
    // to be able to produce different text, or the fix would be indistinguishable
    // from having silently forced greedy decoding.
    let others: std::collections::HashSet<String> = (1..6u64).map(|s| sample(s * 7717)).collect();
    assert!(
        others.len() > 1,
        "every seed produced the same text, so sampling is not seeded but fixed"
    );
}

/// `max_tokens` is honoured, so a caller can bound a generation. A budget that
/// is ignored is how a runaway decode loop reaches production.
#[test]
fn respects_the_max_tokens_budget() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    const BUDGET: usize = 8;
    let engine = Engine::load(model).expect("real GGUF should load");

    let mut tokens = 0_usize;
    let mut session = Session::new();
    let stream = engine
        .chat(&mut session)
        .user("Write a long story about a robot.")
        .max_tokens(BUDGET)
        .greedy()
        .stream()
        .expect("stream should start");

    for event in stream {
        if let Event::Token(_) = event.expect("no event should be an error") {
            tokens += 1;
        }
        assert!(
            tokens <= BUDGET,
            "generated {tokens} tokens against a budget of {BUDGET}"
        );
    }

    assert!(tokens > 0, "budget-limited generation produced no tokens");
}

/// A second turn on the same chat id continues that conversation — the model
/// can answer a follow-up that only makes sense with the first turn in context.
#[test]
fn a_named_chat_continues_across_turns() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = Engine::load(model).expect("real GGUF should load");
    let mut session = Session::new();

    engine
        .chat(&mut session)
        .user("My favourite colour is blue. Reply with just: ok")
        .max_tokens(16)
        .greedy()
        .send()
        .expect("first turn should succeed");
    assert_eq!(session.len(), 2, "user + assistant");

    // Carries no colour of its own — only answerable from the first turn.
    engine
        .chat(&mut session)
        .user("What is my favourite colour? Answer in one word.")
        .max_tokens(16)
        .greedy()
        .send()
        .expect("second turn should succeed");

    let reply = session.latest_text().unwrap_or_default();
    eprintln!(
        "--- transcript: {} messages, latest: {reply:?}",
        session.len()
    );
    assert_eq!(session.len(), 4, "the session holds the whole conversation");
    assert!(!reply.trim().is_empty(), "second turn was empty");
}

/// The session owns the transcript: it can be read, edited, and rebuilt.
#[test]
fn the_caller_owns_the_transcript() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = Engine::load(model).expect("real GGUF should load");
    let mut session = Session::new().with_system("Answer in one word.");

    engine
        .chat(&mut session)
        .user("Name a colour.")
        .max_tokens(16)
        .greedy()
        .send()
        .unwrap();
    // The system prompt is session state, not a message (api_spec.md §7.3).
    assert_eq!(session.len(), 2, "user + assistant");
    assert_eq!(session.system(), Some("Answer in one word."));
    assert_eq!(session.messages()[0].role, "user");
    assert_eq!(session.latest().unwrap().role, "assistant");

    // Editing invalidates the engine's cached prefill, so the next turn is
    // answered from the edited history rather than the original.
    session.edit(|m| m.clear());
    assert_eq!(session.len(), 0);
    assert_eq!(
        session.system(),
        Some("Answer in one word."),
        "instructions survive an edit"
    );

    engine
        .chat(&mut session)
        .user("Name a fruit.")
        .max_tokens(16)
        .greedy()
        .send()
        .expect("a turn after an edit should succeed");
    assert_eq!(session.len(), 2, "new user + new assistant");
    // Nothing was lost: the first exchange is still on record.
    assert_eq!(
        session.all_messages().len(),
        4,
        "two exchanges recorded, one active"
    );
    assert!(session.all_messages().iter().filter(|r| !r.active).count() == 2);

    // A transcript can be rebuilt from stored messages after a restart.
    let restored = Session::from_messages(session.messages().to_vec());
    assert_eq!(restored.len(), session.len());
    assert_ne!(restored.id(), session.id(), "a fresh conversation id");

    // And the whole session — history included — round-trips as JSON.
    let json = serde_json::to_string(&session).unwrap();
    let back: Session = serde_json::from_str(&json).unwrap();
    assert_eq!(back.id(), session.id());
    assert_eq!(back.messages(), session.messages());
    assert_eq!(back.all_messages().len(), 4);
}

/// Dropping the engine shuts the controller down and joins its thread.
///
/// This is the regression guard for a real failure: without it the loop is
/// still holding the llama.cpp context when the process exits, and teardown
/// aborts inside ggml's static destructors — every test passing, then SIGABRT.
/// The whole file exercises it, since none of these tests shut down by hand.
#[test]
fn engine_shuts_down_cleanly_on_drop() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    {
        let engine = Engine::load(model.clone()).expect("real GGUF should load");
        assert!(engine.is_model_loaded(), "model should be loaded");
    } // drop: stops the loop and waits for the backend to be released

    // Loading again proves the previous engine really let go of the backend.
    let engine = Engine::load(model).expect("a second engine should load after the first dropped");
    engine.shutdown().expect("explicit shutdown should succeed");
}

/// Embedding is a separate capability: an engine can hold only an embedder,
/// and the vectors it returns have to be semantically meaningful, not just
/// well-shaped.
///
/// Set `PIO_TEST_EMBEDDER` to a GGUF embedding model.
#[test]
fn embeds_text_into_comparable_vectors() {
    let Ok(raw) = std::env::var("PIO_TEST_EMBEDDER") else {
        eprintln!("SKIP: set PIO_TEST_EMBEDDER to run embedding");
        return;
    };
    let path = PathBuf::from(raw);
    assert!(path.exists(), "PIO_TEST_EMBEDDER does not exist");

    // No chat model — embedding must not depend on one.
    let engine = Engine::builder()
        .embedder(&path)
        .build()
        .expect("an embedder-only engine should build");
    assert!(engine.is_embedder_loaded());

    let corpus = [
        "The cat sat on the mat.".to_string(),
        "A feline rested on the rug.".to_string(),
        "Rust has a borrow checker.".to_string(),
    ];
    let vectors = engine.embed(&corpus).expect("embedding should succeed");

    assert_eq!(vectors.len(), 3, "one vector per input, in order");
    let dims = vectors[0].len();
    assert!(dims > 0, "vectors must not be empty");
    assert!(
        vectors.iter().all(|v| v.len() == dims),
        "every vector shares the model's dimensionality"
    );
    assert!(
        vectors.iter().flatten().any(|x| *x != 0.0),
        "an all-zero embedding means the model never ran"
    );

    // The real assertion: paraphrases must be closer than unrelated text. A
    // well-shaped vector that encodes nothing would pass every check above.
    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    };
    let paraphrase = cos(&vectors[0], &vectors[1]);
    let unrelated = cos(&vectors[0], &vectors[2]);
    eprintln!("--- paraphrase {paraphrase:.3} vs unrelated {unrelated:.3}");
    assert!(
        paraphrase > unrelated,
        "paraphrase ({paraphrase:.3}) should score above unrelated ({unrelated:.3})"
    );

    // Single-input convenience returns one vector of the same width.
    let one = engine.embed_one("Where did the cat sit?").unwrap();
    assert_eq!(one.len(), dims);
}

/// The tool loop: the model asks, the handler answers, generation resumes.
///
/// Needs a model whose template supports native tool calling — SmolLM2 does
/// not. Set `PIO_TEST_TOOL_MODEL` to e.g. a Qwen3 GGUF.
#[test]
fn runs_a_tool_loop_and_answers_from_the_result() {
    let Some(model) = tool_model() else {
        eprintln!("SKIP: set PIO_TEST_TOOL_MODEL to run the tool loop");
        return;
    };

    let engine = Engine::load(model).expect("tool model should load");
    let mut session = Session::new();
    let mut dispatched = Vec::new();

    let done = engine
        .chat(&mut session)
        .user("What is the weather in Paris? Use the tool.")
        .tools(vec![weather_tool()], "Call a tool when you need data.")
        .on_tool(|call| {
            dispatched.push(call.name.clone());
            r#"{"temp_c":18,"sky":"clear"}"#.to_string()
        })
        .max_tokens(256)
        .send()
        .expect("the tool loop should complete");

    eprintln!("--- rounds {} answer {:?}", done.tool_rounds, done.text);

    assert_eq!(dispatched, ["get_weather"], "the tool should be dispatched");
    assert_eq!(done.tool_rounds, 1, "one round: ask, answer, done");
    assert_eq!(done.finish, Finish::Eos);

    // The answer must come from the tool's result, not the model's guess.
    assert!(
        done.text.contains("18"),
        "final answer should use the tool's data, got: {:?}",
        done.text
    );

    // Both halves are recorded, in the order they happened.
    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
}

/// A depth of zero stops immediately with the model still asking, rather than
/// looping. This is the runaway guard.
#[test]
fn tool_depth_bounds_the_loop() {
    let Some(model) = tool_model() else {
        eprintln!("SKIP: set PIO_TEST_TOOL_MODEL to run the tool loop");
        return;
    };

    let engine = Engine::load(model).expect("tool model should load");
    let mut session = Session::new();
    let mut dispatched = 0;

    let done = engine
        .chat(&mut session)
        .user("What is the weather in Paris? Use the tool.")
        .tools(vec![weather_tool()], "Call a tool when you need data.")
        .on_tool(|_| {
            dispatched += 1;
            "{}".to_string()
        })
        .tool_depth(0)
        .max_tokens(256)
        .send()
        .expect("a depth-limited loop still returns");

    assert_eq!(
        done.finish,
        Finish::ToolDepthReached,
        "should report why it stopped"
    );
    assert_eq!(dispatched, 0, "no dispatch past the limit");
}

fn tool_model() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("PIO_TEST_TOOL_MODEL").ok()?);
    assert!(path.exists(), "PIO_TEST_TOOL_MODEL does not exist");
    Some(path)
}

fn weather_tool() -> gen2::advanced::wire::ToolSpec {
    gen2::advanced::wire::ToolSpec {
        r#type: "function".into(),
        function: gen2::advanced::wire::FunctionDefinition {
            name: "get_weather".into(),
            description: Some("Current weather for a city".into()),
            arguments: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        },
    }
}

/// A conversation that outgrows the context window keeps working, and says so.
///
/// The engine sheds its oldest messages to make room; the session keeps the
/// whole transcript and records how much fell out of the model's view. Nothing
/// errors, and later turns still generate.
#[test]
fn a_session_survives_outgrowing_the_context_window() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    // The smallest window the engine will give, so overflow arrives quickly.
    let engine = Engine::builder()
        .model(model)
        .context(2048)
        .build()
        .expect("model should load at a small context");

    let mut session = Session::new();
    assert!(session.fully_in_context(), "nothing shed before we start");

    let filler = "Discuss at length: ".to_string() + &"lorem ipsum dolor sit amet ".repeat(60);
    let mut shed_reported = 0;

    for turn in 1..=6 {
        let done = engine
            .chat(&mut session)
            .user(format!("{filler} (turn {turn}). Reply in one sentence."))
            .max_tokens(48)
            .greedy()
            .send()
            .unwrap_or_else(|e| panic!("turn {turn} should still generate, got: {e}"));

        shed_reported += done.dropped + done.compacted;
        assert!(
            !done.text.trim().is_empty(),
            "turn {turn} produced no text after overflow"
        );
    }

    eprintln!(
        "--- {} messages held, {} shed from the model's view",
        session.len(),
        session.shed()
    );

    // The transcript is complete: two messages per turn, nothing rewritten.
    assert_eq!(session.len(), 12, "the session keeps every message");

    // And the divergence is visible rather than silent.
    assert!(
        session.shed() > 0,
        "six long turns at 2048 context should have overflowed"
    );
    assert_eq!(
        session.shed(),
        shed_reported,
        "the session's count matches what the turns reported"
    );
    assert!(!session.fully_in_context());
}

/// The agent owns dispatch: it resolves the tool the model named, validates the
/// arguments against that tool's schema, and records both halves of the turn.
#[cfg(feature = "agent")]
#[test]
fn an_agent_dispatches_a_registered_tool_and_answers_from_it() {
    let Some(model) = tool_model() else {
        eprintln!("SKIP: set PIO_TEST_TOOL_MODEL to run the agent");
        return;
    };

    let engine = Engine::load(model).expect("tool model should load");
    let mut session = Session::new();
    let mut calls = Vec::new();

    let done = engine
        .agent(&mut session)
        .add_tool(weather_agent_tool())
        .max_steps(4)
        .run_streaming(
            Some("What is the weather in Paris? Use the tool.".into()),
            |step| {
                if let gen2::agent::AgentStep::Calling { tool, .. } = step {
                    calls.push(tool.to_string());
                }
            },
        )
        .expect("the agent should complete");

    assert_eq!(calls, ["get_weather"], "the registry dispatched the call");
    assert_eq!(done.tool_rounds, 1);
    assert!(
        done.text.contains("18"),
        "answered from the tool: {:?}",
        done.text
    );

    // One assistant turn per round — a duplicate here means the agent and the
    // chat layer both appended.
    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
}

/// A deferred tool is absent from the prompt until the model searches for it,
/// at which point its spec joins the conversation.
#[cfg(feature = "agent")]
#[test]
fn an_agent_hydrates_a_deferred_tool_through_search() {
    let Some(model) = tool_model() else {
        eprintln!("SKIP: set PIO_TEST_TOOL_MODEL to run the agent");
        return;
    };

    let engine = Engine::load(model).expect("tool model should load");
    let mut session = Session::new();
    let mut calls = Vec::new();

    engine
        .agent(&mut session)
        .add_tool(weather_agent_tool())
        .defer_tool(resize_agent_tool())
        .tool_search(gen2::agent::ToolSearch::Bm25)
        .max_steps(4)
        .run_streaming(
            Some(
                "Resize /tmp/a.png to 200 pixels wide. You do not have that tool — \
                 call search_tools first."
                    .into(),
            ),
            |step| {
                if let gen2::agent::AgentStep::Calling { tool, .. } = step {
                    calls.push(tool.to_string());
                }
            },
        )
        .expect("the agent should complete");

    assert!(
        calls.contains(&gen2::agent::SEARCH_TOOL.to_string()),
        "the model should have searched, got {calls:?}"
    );
}

#[cfg(feature = "agent")]
fn weather_agent_tool() -> gen2::agent::FunctionTool<WeatherArgs> {
    gen2::agent::FunctionTool::new(
        "get_weather",
        "Current weather for a city",
        |_ctx, a: WeatherArgs| async move {
            Ok(gen2::agent::ToolOutput::Json(serde_json::json!({
                "city": a.city, "temp_c": 18, "sky": "clear"
            })))
        },
    )
}

#[cfg(feature = "agent")]
fn resize_agent_tool() -> gen2::agent::FunctionTool<ResizeArgs> {
    gen2::agent::FunctionTool::new(
        "resize_image",
        "Shrink a picture to a smaller width",
        |_ctx, a: ResizeArgs| async move {
            Ok(gen2::agent::ToolOutput::from(format!(
                "resized to {}px",
                a.width
            )))
        },
    )
}

#[cfg(feature = "agent")]
#[derive(serde::Deserialize, gen2::schemars::JsonSchema)]
struct WeatherArgs {
    /// City to look up.
    city: String,
}

#[cfg(feature = "agent")]
#[derive(serde::Deserialize, gen2::schemars::JsonSchema)]
struct ResizeArgs {
    /// Target width in pixels.
    width: u32,
}

/// Independent tool calls in one turn run concurrently.
///
/// Measured by peak overlap rather than wall-clock: total elapsed includes
/// generation, which swamps the tool time and would make the assertion
/// meaningless. Each tool raises a counter on entry and lowers it on exit, so a
/// peak above one is direct evidence two ran at the same time — the property
/// `ExecutionPolicy::parallel_safe` is supposed to buy.
#[cfg(feature = "agent")]
#[test]
fn parallel_safe_tools_in_one_turn_run_concurrently() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let Some(model) = tool_model() else {
        eprintln!("SKIP: set PIO_TEST_TOOL_MODEL to run the agent");
        return;
    };

    let engine = Engine::load(model).expect("tool model should load");
    let mut session = Session::new();
    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let total = Arc::new(AtomicUsize::new(0));

    let slow = |name: &'static str,
                live: Arc<AtomicUsize>,
                peak: Arc<AtomicUsize>,
                total: Arc<AtomicUsize>| {
        gen2::agent::FunctionTool::new(
            name,
            format!("Check the {name} system"),
            move |_c, _a: NoArgs| {
                let (live, peak, total) =
                    (Arc::clone(&live), Arc::clone(&peak), Arc::clone(&total));
                async move {
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                    live.fetch_sub(1, Ordering::SeqCst);
                    total.fetch_add(1, Ordering::SeqCst);
                    Ok(gen2::agent::ToolOutput::from("ok"))
                }
            },
        )
    };

    let _ = engine
        .agent(&mut session)
        .add_tool(slow("alpha", live.clone(), peak.clone(), total.clone()))
        .add_tool(slow("beta", live.clone(), peak.clone(), total.clone()))
        .add_tool(slow("gamma", live.clone(), peak.clone(), total.clone()))
        .max_steps(3)
        .goal("Check alpha, beta and gamma. Call all three tools at once.")
        .expect("the agent should complete");

    let (ran, overlap) = (total.load(Ordering::SeqCst), peak.load(Ordering::SeqCst));
    eprintln!("--- {ran} tool runs, peak overlap {overlap}");

    assert!(
        ran > 0,
        "the model called no tools, so nothing was exercised"
    );
    if ran >= 2 {
        assert!(
            overlap > 1,
            "{ran} tools ran but never overlapped — dispatch is sequential"
        );
    } else {
        eprintln!("(model issued one call at a time; concurrency not exercised)");
    }
}

#[derive(serde::Deserialize, gen2::schemars::JsonSchema)]
struct NoArgs {}

/// A spawned agent streams updates and can be steered mid-run.
#[cfg(feature = "agent")]
#[test]
fn a_spawned_agent_interrupt_cuts_the_generation_short() {
    use std::sync::Arc;

    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = Arc::new(Engine::load(model).expect("model should load"));
    let prompt = "Write an extremely long, detailed essay about the history of rust.";

    // Baseline: how much it writes when left alone.
    let mut baseline = 0usize;
    for update in engine
        .agent_owned(Session::new())
        .goal(prompt)
        .max_steps(1)
        .spawn()
    {
        if let gen2::agent::Update::Delta(t) = update {
            baseline += t.len();
        }
    }

    let run = engine
        .agent_owned(Session::new())
        .goal(prompt)
        .max_steps(1)
        .spawn();

    let steering = run.steering();
    assert!(
        steering.can_interrupt_generation(),
        "a spawned run owns an engine, so it must be able to stop one"
    );

    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(250));
        steering.interrupt("stop, just say OK");
    });

    let mut chars = 0usize;
    let mut saw_done = false;
    for update in run {
        match update {
            gen2::agent::Update::Delta(t) => chars += t.len(),
            gen2::agent::Update::Done { .. } => saw_done = true,
            gen2::agent::Update::Failed { error, .. } => panic!("run failed: {error}"),
            _ => {}
        }
    }

    eprintln!("--- baseline {baseline} chars, interrupted {chars} chars");
    assert!(saw_done, "an interrupted run still ends with Done");
    assert!(baseline > 0, "nothing to compare against");
    // The point of an owned run: the generation is cut, not merely queued
    // behind a step boundary that never arrives in a single-step run.
    assert!(
        chars < baseline,
        "interrupt produced {chars} chars vs {baseline} uninterrupted — \
         the generation was not cut short"
    );
}

/// The borrowed agent cannot stop a generation; it says so rather than
/// pretending.
#[cfg(feature = "agent")]
#[test]
fn a_borrowed_agent_cannot_cut_a_generation_short() {
    let mut session = Session::new();
    // No engine needed: this is a property of how the handle was built.
    let engine = match test_model() {
        Some(m) => Engine::load(m).expect("model should load"),
        None => {
            eprintln!("SKIP: set PIO_TEST_MODEL");
            return;
        }
    };
    let agent = engine.agent(&mut session);
    assert!(
        !agent.steering().can_interrupt_generation(),
        "a borrowed agent has no owned engine, and must not claim otherwise"
    );
}

/// Two runs over one session, with different tool sets, both complete and the
/// transcript stays coherent.
///
/// The reopen itself is unit-tested (`Session::note_tools`); what this adds is
/// that the reopened conversation still renders and generates — a full history
/// replayed into a fresh `StartChat` with a different tool list is the case
/// most likely to break the template.
///
/// It deliberately does *not* assert that the model calls the new tool. It can
/// and does, but greedily this model declines given a transcript full of the
/// old one, and that is a fact about the model, not about the reopen.
#[cfg(feature = "agent")]
#[test]
fn a_session_survives_a_tool_set_change_between_runs() {
    let Some(model) = tool_model() else {
        eprintln!("SKIP: set PIO_TEST_TOOL_MODEL to run the agent");
        return;
    };

    let engine = Engine::load(model).expect("tool model should load");
    let mut session = Session::new();

    engine
        .agent(&mut session)
        .add_tool(named_tool("alpha"))
        .max_steps(3)
        .greedy()
        .goal("Call the alpha tool with city=Paris.")
        .expect("first run should complete");
    let after_first = session.len();
    assert!(after_first >= 2, "the first run recorded nothing");

    let done = engine
        .agent(&mut session)
        .add_tool(named_tool("beta"))
        .max_steps(3)
        .greedy()
        .goal("Now call the beta tool with city=Rome.")
        .expect("a reopened conversation should still generate");

    eprintln!(
        "--- {} messages after two runs, second reply {:?}",
        session.len(),
        done.text.chars().take(40).collect::<String>()
    );
    assert!(
        session.len() > after_first,
        "the second run added nothing to the transcript"
    );
    assert!(
        !done.text.trim().is_empty(),
        "the second run produced no text"
    );

    // Roles stay well-formed across the reopen — no duplicate assistant turns,
    // no tool result without a preceding call.
    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    for pair in roles.windows(2) {
        assert_ne!(
            (pair[0], pair[1]),
            ("assistant", "assistant"),
            "duplicate assistant turns: {roles:?}"
        );
    }
}

#[cfg(feature = "agent")]
use gen2::agent::AgentStep as pio_agent_step;

#[cfg(feature = "agent")]
fn named_tool(name: &'static str) -> gen2::agent::FunctionTool<WeatherArgs> {
    gen2::agent::FunctionTool::new(
        name,
        format!("The {name} tool — call it with any city"),
        |_c, a: WeatherArgs| async move {
            Ok(gen2::agent::ToolOutput::from(format!("ok for {}", a.city)))
        },
    )
}

/// A model can be swapped on a live engine, and conversations survive it.
///
/// The engine stays up — sessions, tools and settings persist. What cannot
/// persist is the cached prefill: it was produced by weights that are no longer
/// loaded, so every live session reopens on its next turn. That is the part
/// worth testing, because getting it wrong means answering from another
/// model's state with no symptom until the output is subtly wrong.
///
/// Needs `PIO_TEST_MODEL` and `PIO_TEST_TOOL_MODEL` to be different files.
#[test]
fn a_model_can_be_swapped_on_a_live_engine() {
    let (Some(first), Some(second)) = (test_model(), tool_model()) else {
        eprintln!("SKIP: set PIO_TEST_MODEL and PIO_TEST_TOOL_MODEL");
        return;
    };
    if first == second {
        eprintln!("SKIP: the two models must differ");
        return;
    }

    let engine = Engine::load(&first).expect("first model should load");
    assert_eq!(engine.model_generation(), 0);

    let mut session = Session::new();
    engine
        .chat(&mut session)
        .user("Say hello.")
        .max_tokens(16)
        .greedy()
        .send()
        .expect("first model should generate");
    let before = session.len();

    // Swap. The engine stays up.
    engine
        .load_model(&second)
        .expect("second model should load");
    assert_eq!(engine.model_generation(), 1, "the swap is observable");
    assert!(engine.is_model_loaded());

    // The same session keeps generating — its prefill is rebuilt against the
    // new weights rather than reused from the old ones.
    let done = engine
        .chat(&mut session)
        .user("Say goodbye.")
        .max_tokens(16)
        .greedy()
        .send()
        .expect("the swapped model should generate on the same session");

    eprintln!(
        "--- gen {} · {} msgs · {:?}",
        engine.model_generation(),
        session.len(),
        done.text.chars().take(40).collect::<String>()
    );
    assert!(!done.text.trim().is_empty(), "no text after the swap");
    assert!(session.len() > before, "the transcript kept growing");

    // The whole conversation survived — nothing was rewritten to fit the swap.
    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "user", "assistant"]);
}

/// A swap refused up front leaves the loaded model alone.
///
/// The controller tears the old model down before reading the new one, so a
/// load that fails part-way leaves nothing loaded — a typo in a path would cost
/// you a working model. The path is checked before any of that happens, which
/// is what makes the common mistake survivable.
#[test]
fn a_swap_refused_up_front_does_not_disturb_the_loaded_model() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let engine = Engine::load(model).expect("model should load");
    let before = engine.model_generation();

    let err = engine
        .load_model("/nonexistent/model.gguf")
        .expect_err("loading a missing file must fail");
    eprintln!("--- refused: {err}");

    // Nothing moved: bumping the generation here would invalidate every live
    // session's prefill for a model that is still perfectly loaded.
    assert_eq!(
        engine.model_generation(),
        before,
        "a failed swap is not a swap"
    );
    assert!(engine.is_model_loaded(), "the old model is still there");

    let mut session = Session::new();
    assert!(
        !engine
            .chat(&mut session)
            .user("Still working?")
            .max_tokens(16)
            .greedy()
            .send()
            .expect("the surviving model should still generate")
            .text
            .trim()
            .is_empty()
    );
}

/// An agent can be asked for a structured final answer.
///
/// The grammar applies to one extra turn after the work is done, not to the
/// whole run — constraining every turn would forbid the tool-call syntax the
/// model needs to get anywhere. So this checks both halves: the tool still ran,
/// and the answer still parses.
#[cfg(feature = "agent")]
#[test]
fn an_agent_can_return_a_structured_answer() {
    let Some(model) = tool_model() else {
        eprintln!("SKIP: set PIO_TEST_TOOL_MODEL to run the agent");
        return;
    };

    let engine = Engine::load(model).expect("tool model should load");
    let mut session = Session::new();
    let mut called = Vec::new();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" },
            "temperature_c": { "type": "number" }
        },
        "required": ["city", "temperature_c"]
    });

    let done = engine
        .agent(&mut session)
        .add_tool(weather_agent_tool())
        .max_steps(4)
        .greedy()
        .answer_as(
            gen2::advanced::generation::GrammarSpec::JsonSchema(schema),
            "Give the final answer as JSON with keys city and temperature_c.",
        )
        .run_streaming(
            Some("What is the weather in Paris? Use the tool.".into()),
            |s| {
                if let pio_agent_step::Calling { tool, .. } = s {
                    called.push(tool.to_string());
                }
            },
        )
        .expect("the agent should complete");

    eprintln!("--- called {called:?} · answer {:?}", done.text);
    assert_eq!(called, ["get_weather"], "the tool still ran unconstrained");

    // The grammar made this sound rather than hopeful.
    let parsed: serde_json::Value =
        serde_json::from_str(&done.text).expect("the shaped answer must parse as JSON");
    assert!(parsed.get("city").is_some(), "got {parsed}");
    assert!(parsed.get("temperature_c").is_some(), "got {parsed}");
}

/// Sending images to a text-only model fails before anything is generated.
#[test]
fn images_on_a_text_only_model_are_refused_not_attempted() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let engine = Engine::load(model).expect("model should load");
    // SmolLM2 is text-only; this is the precondition the guard exists for.
    if engine.supports_images() {
        eprintln!("SKIP: PIO_TEST_MODEL is multimodal");
        return;
    }

    let mut session = Session::new();
    let err = engine
        .chat(&mut session)
        .user_with_images("What is this?", ["/tmp/a.png"])
        .send()
        .expect_err("a text-only model must refuse images");

    eprintln!("--- refused: {err}");
    assert_eq!(
        err.code(),
        Some("unsupported"),
        "routable, not a backend crash"
    );

    // Recoverable: nothing was generated, so dropping the image and retrying
    // works on the same session.
    assert!(
        !engine
            .chat(&mut session)
            .user("Never mind — just say hello.")
            .max_tokens(16)
            .greedy()
            .send()
            .expect("the same session should still work")
            .text
            .trim()
            .is_empty()
    );
}

// ── The inference-first facade (api_spec.md §4–§6) ───────────────────────────

/// Hello world, as the spec writes it: load, generate, text.
#[test]
fn facade_load_and_generate_prints_a_token() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let model = gen2::load(model).expect("gen2::load should load a real GGUF");
    let text = model
        .generate("Reply with exactly one word: hello")
        .max_tokens(8)
        .text()
        .expect("generation should succeed");

    eprintln!("--- facade generated: {text:?}");
    assert!(
        !text.trim().is_empty(),
        "the reply is empty — the model loaded but the facade returned no text"
    );
}

/// A `Runtime` hands out cloneable, shareable models, and the handle answers
/// what it is and what it can do.
#[test]
fn facade_runtime_model_is_shareable_and_describes_itself() {
    use gen2::Runtime;
    use gen2::model::ModelSourceKind;
    use gen2::output::FinishReason;

    let Some(path) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let runtime = Runtime::new().expect("a runtime builds");
    let model = runtime
        .load(&path)
        .expect("the runtime should load a real GGUF");
    let a = model.clone();
    let b = model.clone();

    // Two threads, sequentially: the second starts after the first finishes,
    // which is the sharing an app does when it moves inference off the UI.
    let first = std::thread::spawn(move || {
        a.generate("Reply with exactly one word: hello")
            .max_tokens(8)
            .greedy()
            .run()
    })
    .join()
    .expect("no panic")
    .expect("first generation should succeed");
    let second = std::thread::spawn(move || {
        b.generate("Reply with exactly one word: goodbye")
            .max_tokens(8)
            .greedy()
            .run()
    })
    .join()
    .expect("no panic")
    .expect("second generation should succeed");
    eprintln!(
        "--- shared clones: {:?} / {:?} ({} / {})",
        first.text(),
        second.text(),
        first.finish_reason(),
        second.finish_reason()
    );
    for r in [&first, &second] {
        assert!(
            !r.text().is_empty() || r.reasoning().is_some(),
            "a clone must generate: {r:?}"
        );
        assert!(
            matches!(r.finish_reason(), FinishReason::Stop | FinishReason::Length),
            "a plain reply ends by stopping or by budget, got {}",
            r.finish_reason()
        );
        assert!(r.stats().reported(), "llama.cpp reports stats: {r:?}");
        assert!(r.usage().completion_tokens > 0);
    }

    let info = model.info();
    eprintln!("--- info: {info:?}");
    assert!(
        info.context_window.is_some(),
        "the loaded model's context window must be reported"
    );
    assert_eq!(info.source, ModelSourceKind::LocalFile);
    assert!(info.local);
    assert!(info.architecture.is_some(), "a GGUF names its architecture");
    assert_eq!(
        info.name.as_deref(),
        path.file_stem().and_then(|s| s.to_str())
    );

    let caps = model.capabilities();
    eprintln!("--- capabilities: {caps:?}");
    assert!(caps.text);
    assert!(
        caps.structured_output,
        "llama.cpp constrains output by grammar"
    );
    assert_eq!(runtime.models(), vec![model.id()]);
}

// ── The turn facade: tools, streaming, structure, cancellation (§10–§16, §18) ──

fn facade_model() -> Option<gen2::Model> {
    let path = test_model()?;
    Some(gen2::load(path).expect("gen2::load should load a real GGUF"))
}

fn weather_definition() -> gen2::tool_defs::ToolDefinition {
    gen2::tool_defs::ToolDefinition::new("get_weather")
        .description("Current weather for a city")
        .schema(serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }))
}

/// §28.5 verbatim: the model declares a call, the harness answers it, the
/// model runs again with no new user message and answers from the result.
#[test]
fn facade_tool_loop_declares_a_call_and_answers_from_the_result() {
    use gen2::Session;
    use gen2::model::ThinkingMode;
    use gen2::output::FinishReason;
    use gen2::tool_defs::ToolSet;

    let Some(model) = facade_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };
    assert!(model.capabilities().tools, "Qwen3's template renders tools");

    let mut session = Session::new()
        .with_system("You have a weather tool. Use it for weather questions.")
        .with_tools(ToolSet::new().with(weather_definition()));

    let response = model
        .turn(&mut session)
        .user("What is the weather in Paris? Use the get_weather tool.")
        .reasoning(ThinkingMode::Off)
        .max_tokens(256)
        .greedy()
        .run()
        .expect("the first turn should run");
    eprintln!(
        "--- first turn: {} · text {:?} · calls {:?}",
        response.finish_reason(),
        response.text(),
        response.tool_calls()
    );
    assert_eq!(*response.finish_reason(), FinishReason::ToolCall);
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1, "one call: {calls:?}");
    let call = calls[0];
    assert_eq!(call.name(), "get_weather");
    assert!(
        call.arguments()["city"]
            .as_str()
            .is_some_and(|c| c.to_lowercase().contains("paris")),
        "the call names the city: {:?}",
        call.arguments()
    );

    for call in response.tool_calls() {
        session.push_tool_result(call.id(), r#"{"temp_c":18,"sky":"clear"}"#);
    }

    let response = model
        .turn(&mut session)
        .reasoning(ThinkingMode::Off)
        .max_tokens(128)
        .greedy()
        .run()
        .expect("the continuation should run with no new user message");
    eprintln!("--- answer: {:?}", response.text());
    assert!(
        response.tool_calls().is_empty(),
        "the model should answer, not call again: {response:?}"
    );
    assert!(
        response.text().contains("18"),
        "the answer should use the tool's data, got {:?}",
        response.text()
    );
    assert!(
        response.stats().reported() && response.usage().completion_tokens > 0,
        "a continued turn reports its stats too: {:?}",
        response.stats()
    );

    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
    assert_eq!(
        session.messages()[2].tool_call_id.as_deref(),
        Some(call.id().as_str()),
        "the result answers the call by id"
    );
}

/// §28.10: a stream yields text deltas and ends with `Finished`; with the
/// reasoning channel on, thinking arrives as `ReasoningDelta`, not as text.
#[test]
fn facade_streaming_yields_semantic_events() {
    use gen2::Session;
    use gen2::event::Event;
    use gen2::model::ThinkingMode;
    use gen2::output::FinishReason;

    let Some(model) = facade_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let mut session = Session::new();
    let mut stream = model
        .turn(&mut session)
        .user("Reply with one short sentence about the sea.")
        .reasoning(ThinkingMode::Off)
        .max_tokens(32)
        .greedy()
        .stream()
        .expect("a stream starts");
    let mut deltas = String::new();
    let mut text_deltas = 0;
    let mut last = None;
    for event in stream.by_ref() {
        let event = event.expect("no event fails");
        match &event {
            Event::TextDelta(t) => {
                text_deltas += 1;
                deltas.push_str(t);
            }
            Event::Finished(_) => last = Some(event.clone()),
            _ => {}
        }
    }
    eprintln!("--- {text_deltas} text deltas: {deltas:?} · last {last:?}");
    assert!(text_deltas >= 1, "at least one TextDelta");
    assert!(
        matches!(
            last,
            Some(Event::Finished(FinishReason::Stop | FinishReason::Length))
        ),
        "the last event is Finished: {last:?}"
    );
    let response = stream.finish().expect("the outcome");
    assert_eq!(
        response.text(),
        deltas,
        "finish() returns what was streamed"
    );
    assert!(response.stats().reported());
    assert_eq!(response.message_id(), session.active_ids().last().copied());

    // With thinking on, Qwen3 opens with `<think>`: that is a reasoning
    // delta, and never leaks into the text.
    let mut session = Session::new();
    let mut stream = model
        .turn(&mut session)
        .user("Is 17 prime? Think it through, then answer.")
        .reasoning(ThinkingMode::On)
        .max_tokens(96)
        .greedy()
        .stream()
        .expect("a stream starts");
    let mut reasoning = 0;
    let mut text = String::new();
    for event in stream.by_ref() {
        match event.expect("no event fails") {
            Event::ReasoningDelta(_) => reasoning += 1,
            Event::TextDelta(t) => text.push_str(&t),
            _ => {}
        }
    }
    let response = stream.finish().expect("the outcome");
    eprintln!(
        "--- {reasoning} reasoning deltas · reasoning {:?} · text {:?}",
        response.reasoning(),
        response.text()
    );
    assert!(
        reasoning >= 1,
        "the reasoning channel streams as ReasoningDelta"
    );
    assert!(response.reasoning().is_some());
    assert!(
        !text.contains("<think>"),
        "no scaffold in the text: {text:?}"
    );
    assert!(
        !response.text().contains("</think>"),
        "no scaffold in the reply: {:?}",
        response.text()
    );
}

/// §28.11: a typed value straight out of a generation, enforced by grammar.
#[test]
fn facade_structured_output_is_typed_and_grammar_enforced() {
    #[derive(Debug, serde::Deserialize, gen2::schemars::JsonSchema)]
    #[serde(rename_all = "lowercase")]
    enum Sky {
        Clear,
        Cloudy,
        Rain,
    }
    #[derive(Debug, serde::Deserialize, gen2::schemars::JsonSchema)]
    struct Weather {
        city: String,
        sky: Sky,
        temp_c: i32,
    }

    let Some(model) = facade_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };
    assert!(model.capabilities().structured_output);

    let weather: Weather = model
        .generate("Paris was clear and 18 degrees today.")
        .system("Extract the weather report.")
        .greedy()
        .structured()
        .expect("the grammar makes the reply decode");
    eprintln!("--- structured: {weather:?}");
    assert!(
        weather.city.to_lowercase().contains("paris"),
        "city: {weather:?}"
    );
    assert!(matches!(weather.sky, Sky::Clear), "sky: {weather:?}");
    assert_eq!(weather.temp_c, 18, "{weather:?}");

    // The same on a session turn: the JSON reply becomes part of the chat.
    let mut session = gen2::Session::new().with_system("Extract the weather report.");
    let weather: Weather = model
        .turn(&mut session)
        .user("Berlin: cloudy, 9 degrees.")
        .greedy()
        .structured()
        .expect("decodes");
    eprintln!("--- turn structured: {weather:?}");
    assert!(matches!(weather.sky, Sky::Cloudy), "{weather:?}");
    assert_eq!(session.len(), 2);
}

/// §16 and §16.1: cancel from another thread after the first delta; the
/// stream ends `Cancelled`, the partial reply is recorded under an id, and a
/// harness can drop it from the active context without losing the record.
#[test]
fn facade_cancellation_keeps_the_partial_reply_and_lets_the_harness_remove_it() {
    use gen2::Session;
    use gen2::event::Event;
    use gen2::model::ThinkingMode;
    use gen2::output::FinishReason;

    let Some(model) = facade_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let mut session = Session::new();
    let mut stream = model
        .turn(&mut session)
        .user("Count from 1 to 500, one number per line.")
        .reasoning(ThinkingMode::Off)
        .max_tokens(400)
        .greedy()
        .stream()
        .expect("a stream starts");
    let cancel = stream.canceller();

    let first = stream.next().expect("a first event").expect("not an error");
    eprintln!("--- first event: {first:?}");
    assert!(matches!(first, Event::TextDelta(_)), "{first:?}");
    let stopper = std::thread::spawn(move || cancel.cancel());

    let mut events = 0;
    let mut last = None;
    for event in stream.by_ref() {
        events += 1;
        last = Some(event.expect("not an error"));
    }
    stopper.join().expect("the cancelling thread finishes");
    eprintln!("--- {events} more events · last {last:?}");
    assert_eq!(last, Some(Event::Finished(FinishReason::Cancelled)));

    let response = stream.finish().expect("the outcome");
    assert_eq!(*response.finish_reason(), FinishReason::Cancelled);
    assert!(!response.text().is_empty(), "the partial text is kept");
    assert!(
        response.text().lines().count() < 300,
        "the cancel cut the count short: {} lines",
        response.text().lines().count()
    );
    let id = response
        .message_id()
        .expect("the partial reply was recorded");
    assert_eq!(session.latest_text(), Some(response.text()));
    session
        .remove_message(id)
        .expect("the harness can remove it");
    assert_eq!(session.len(), 1, "gone from the active context");
    assert_eq!(session.all_messages().len(), 2, "still on record");

    // And the session still works afterwards.
    let response = model
        .turn(&mut session)
        .user("Reply with exactly one word: hello")
        .reasoning(ThinkingMode::Off)
        .max_tokens(8)
        .greedy()
        .run()
        .expect("the next turn runs");
    eprintln!("--- after cancel: {:?}", response.text());
    assert!(!response.text().trim().is_empty());
}

/// §18.1: a system prompt set between turns changes the next turn, and the
/// change is only a revision bump — no message moved.
#[test]
fn facade_dynamic_system_prompt_changes_the_next_turn() {
    use gen2::Session;
    use gen2::model::ThinkingMode;

    let Some(model) = facade_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let mut session = Session::new().with_system("You are a helpful assistant.");
    let before = model
        .turn(&mut session)
        .user("Say hello.")
        .reasoning(ThinkingMode::Off)
        .max_tokens(16)
        .greedy()
        .run()
        .expect("runs");
    eprintln!("--- before: {:?}", before.text());
    assert!(
        !before.text().to_uppercase().contains("BANANA"),
        "nothing about bananas yet: {:?}",
        before.text()
    );

    let revision = session.revision();
    let len = session.len();
    session.set_system("Reply with exactly the single word BANANA, in capitals, and nothing else.");
    assert_eq!(session.revision().as_u64(), revision.as_u64() + 1);
    assert_eq!(session.len(), len, "no message moved");

    let after = model
        .turn(&mut session)
        .user("Say hello.")
        .reasoning(ThinkingMode::Off)
        .max_tokens(16)
        .greedy()
        .run()
        .expect("runs");
    eprintln!("--- after: {:?}", after.text());
    assert!(
        after.text().to_uppercase().contains("BANANA"),
        "the new system prompt governs the next turn: {:?}",
        after.text()
    );
}

/// §22: the same turn, awaited. One-shot text, a session turn, and a stream
/// over the async surface, each producing real tokens.
#[cfg(feature = "tokio")]
#[tokio::test]
async fn facade_run_async_produces_text() {
    use gen2::Session;
    use gen2::model::ThinkingMode;
    use gen2::output::FinishReason;

    let Some(model) = facade_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let text = model
        .generate("Reply with the single word: hello")
        .reasoning(ThinkingMode::Off)
        .max_tokens(8)
        .greedy()
        .text_async()
        .await
        .expect("one-shot async text");
    eprintln!("--- text_async: {text:?}");
    assert!(!text.trim().is_empty(), "a real token came back");

    let mut session = Session::new().with_system("Answer in one short sentence.");
    let response = model
        .turn(&mut session)
        .user("What colour is the sky on a clear day?")
        .reasoning(ThinkingMode::Off)
        .max_tokens(32)
        .greedy()
        .run_async()
        .await
        .expect("an async turn");
    eprintln!(
        "--- run_async: {:?} · {:?}",
        response.text(),
        response.finish_reason()
    );
    assert!(!response.text().trim().is_empty());
    assert!(matches!(
        response.finish_reason(),
        FinishReason::Stop | FinishReason::Length
    ));
    assert!(response.stats().reported());
    assert_eq!(
        response.message_id(),
        session.active_ids().last().copied(),
        "the reply was appended to the caller's session"
    );
    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant"]);

    // The same session, blocking: the async turn left it continuable.
    let follow = model
        .turn(&mut session)
        .user("And at night?")
        .reasoning(ThinkingMode::Off)
        .max_tokens(32)
        .greedy()
        .run()
        .expect("a blocking follow-up");
    eprintln!("--- follow-up: {:?}", follow.text());
    assert!(!follow.text().trim().is_empty());
    assert_eq!(session.len(), 4);
}

/// §22 streaming: the async stream yields the §15 events, `finish()` returns
/// what was streamed, and a cancel from another task ends it as cancelled.
#[cfg(feature = "tokio")]
#[tokio::test]
async fn facade_stream_async_yields_events_and_cancels_across_tasks() {
    use futures::StreamExt;
    use gen2::Session;
    use gen2::event::Event;
    use gen2::model::ThinkingMode;
    use gen2::output::FinishReason;

    let Some(model) = facade_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };

    let mut session = Session::new();
    let mut stream = model
        .turn(&mut session)
        .user("Reply with one short sentence about the sea.")
        .reasoning(ThinkingMode::Off)
        .max_tokens(32)
        .greedy()
        .stream_async()
        .await
        .expect("a stream starts");
    let mut deltas = String::new();
    let mut text_deltas = 0;
    let mut last = None;
    while let Some(event) = stream.next().await {
        let event = event.expect("no event fails");
        match &event {
            Event::TextDelta(t) => {
                text_deltas += 1;
                deltas.push_str(t);
            }
            Event::Finished(_) => last = Some(event.clone()),
            _ => {}
        }
    }
    eprintln!("--- {text_deltas} text deltas: {deltas:?} · last {last:?}");
    assert!(text_deltas >= 1, "at least one TextDelta");
    assert!(
        matches!(
            last,
            Some(Event::Finished(FinishReason::Stop | FinishReason::Length))
        ),
        "the last event is Finished: {last:?}"
    );
    let response = stream.finish().await.expect("the outcome");
    assert_eq!(
        response.text(),
        deltas,
        "finish() returns what was streamed"
    );
    assert!(response.stats().reported());
    assert_eq!(response.message_id(), session.active_ids().last().copied());

    // Cancel from a spawned task after the first delta (§16, across tasks).
    let mut session = Session::new();
    let mut stream = model
        .turn(&mut session)
        .user("Write a long story about a lighthouse keeper.")
        .reasoning(ThinkingMode::Off)
        .max_tokens(400)
        .greedy()
        .stream_async()
        .await
        .expect("a stream starts");
    let cancel = stream.canceller();
    let first = stream.next().await.expect("a first event").expect("ok");
    eprintln!("--- first event before cancel: {first:?}");
    tokio::spawn(async move { cancel.cancel() })
        .await
        .expect("the cancelling task");
    let mut events = 0;
    let mut last = None;
    while let Some(event) = stream.next().await {
        events += 1;
        last = Some(event.expect("no event fails"));
    }
    eprintln!("--- {events} events after cancel · last {last:?}");
    assert_eq!(last, Some(Event::Finished(FinishReason::Cancelled)));
    assert!(
        events < 300,
        "the cancel cut the generation short: {events} events"
    );
    let response = stream.finish().await.expect("the outcome");
    assert_eq!(*response.finish_reason(), FinishReason::Cancelled);
    if let Some(id) = response.message_id() {
        session
            .remove_message(id)
            .expect("the partial reply can be removed");
    }
    // The session runs again after a cancel.
    let again = model
        .turn(&mut session)
        .user("Say hi.")
        .reasoning(ThinkingMode::Off)
        .max_tokens(8)
        .greedy()
        .run_async()
        .await
        .expect("a turn after a cancel");
    eprintln!("--- after cancel: {:?}", again.text());
    assert!(!again.text().trim().is_empty());
}

// ── Model switching, residency, concurrency (§4.2, §4.5, §17, §23, §28.4) ──

/// A second, different model for the switching tests. `PIO_TEST_SECOND_MODEL`
/// — Llama-3.2-3B-Instruct against Qwen3-0.6B in the reference run.
fn second_model() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("PIO_TEST_SECOND_MODEL").ok()?);
    assert!(path.exists(), "PIO_TEST_SECOND_MODEL does not exist");
    Some(path)
}

fn two_model_paths() -> Option<(PathBuf, PathBuf)> {
    let (Some(first), Some(second)) = (test_model(), second_model()) else {
        eprintln!("SKIP: set PIO_TEST_MODEL and PIO_TEST_SECOND_MODEL");
        return None;
    };
    if first == second {
        eprintln!("SKIP: the two models must differ");
        return None;
    }
    Some((first, second))
}

/// §28.4: two models from one runtime, one session switched between them
/// mid-chat. The second model answers from what the first was told.
#[test]
fn facade_switches_models_mid_chat_and_the_second_remembers_bob() {
    use gen2::model::ThinkingMode;
    use gen2::{Runtime, Session};

    let Some((first, second)) = two_model_paths() else {
        return;
    };
    let runtime = Runtime::new().expect("a runtime builds");
    let qwen = runtime.load(&first).expect("first model loads");
    let llama = runtime.load(&second).expect("second model loads");
    assert_eq!(runtime.models(), vec![qwen.id(), llama.id()]);

    let mut session = Session::new().with_system("Be concise.");
    let a = qwen
        .turn(&mut session)
        .user("My name is Bob.")
        .max_tokens(32)
        .greedy()
        .reasoning(ThinkingMode::Off)
        .run()
        .expect("the first model answers");
    let b = llama
        .turn(&mut session)
        .user("What is my name? Reply with just the name.")
        .max_tokens(16)
        .greedy()
        .reasoning(ThinkingMode::Off)
        .run()
        .expect("the second model answers on the same session");
    eprintln!("--- qwen: {:?}\n--- llama: {:?}", a.text(), b.text());
    assert!(
        b.text().contains("Bob"),
        "the second model must answer from the transcript the first built: {:?}",
        b.text()
    );
    // The session was never converted or migrated (invariant 2).
    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "user", "assistant"]);
    assert_eq!(session.system(), Some("Be concise."));
}

/// §17.2: qwen → llama → qwen. Correctness is what this proves live: the
/// session comes back to the first model and answers from what the second
/// model said in between — llama's reply is an assistant message qwen
/// never generated, so qwen's runtime is rebuilt with it rather than
/// appended to (the backends append only their own replies).
///
/// Reuse — appending when the tail holds no foreign reply — is proved on
/// the scripted engine in `api::switching_tests`, by counting
/// `append_messages` against `start_session`. It is not provable here:
/// llama.cpp reports `prompt_tokens` as the KV position at pull time — the
/// whole context, appended or rebuilt — so the numbers below are printed
/// with the time to first token for the record, and only their cumulative
/// shape is asserted.
#[test]
fn facade_switching_away_and_back_answers_from_the_other_models_reply() {
    use gen2::model::ThinkingMode;
    use gen2::{Runtime, Session};

    let Some((first, second)) = two_model_paths() else {
        return;
    };
    let runtime = Runtime::new().expect("a runtime builds");
    let qwen = runtime.load(&first).expect("first model loads");
    let llama = runtime.load(&second).expect("second model loads");

    let system = format!(
        "{}Answer with one word.",
        "You are a terse assistant. ".repeat(60)
    );
    let mut session = Session::new().with_system(system);
    let t1 = qwen
        .turn(&mut session)
        .user("Say hi.")
        .max_tokens(8)
        .greedy()
        .reasoning(ThinkingMode::Off)
        .run()
        .expect("turn 1");
    let t2 = llama
        .turn(&mut session)
        .user("The secret word is PINEAPPLE. Repeat the secret word.")
        .max_tokens(8)
        .greedy()
        .reasoning(ThinkingMode::Off)
        .run()
        .expect("turn 2");
    let t3 = qwen
        .turn(&mut session)
        .user("What did you just say the secret word was?")
        .max_tokens(16)
        .greedy()
        .reasoning(ThinkingMode::Off)
        .run()
        .expect("turn 3");
    let (p1, p2, p3) = (
        t1.usage().prompt_tokens,
        t2.usage().prompt_tokens,
        t3.usage().prompt_tokens,
    );
    eprintln!(
        "--- qwen {:?} · llama {:?} · qwen again {:?}",
        t1.text(),
        t2.text(),
        t3.text()
    );
    eprintln!(
        "--- prompt tokens (cumulative): qwen {p1} · llama {p2} · qwen again {p3} · ttft us {} / {} / {}",
        t1.stats()
            .time_to_first_token()
            .map(|d| d.as_micros())
            .unwrap_or(0),
        t2.stats()
            .time_to_first_token()
            .map(|d| d.as_micros())
            .unwrap_or(0),
        t3.stats()
            .time_to_first_token()
            .map(|d| d.as_micros())
            .unwrap_or(0),
    );
    for (t, r) in [(1, &t1), (2, &t2), (3, &t3)] {
        assert!(r.stats().reported(), "turn {t} must report stats: {r:?}");
        assert!(!r.text().trim().is_empty(), "turn {t} produced no text");
    }
    assert!(
        t2.text().to_uppercase().contains("PINEAPPLE"),
        "llama should repeat the word: {:?}",
        t2.text()
    );
    assert!(
        t3.text().to_uppercase().contains("PINEAPPLE"),
        "qwen must see llama's reply after switching back: {:?}",
        t3.text()
    );
    assert!(p1 > 200, "the system prompt should be long: {p1} tokens");
    assert!(
        p3 > p1,
        "qwen's context grew across the switch: {p3} vs {p1}"
    );
    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(
        roles,
        [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant"
        ]
    );
}

/// §4.2/§4.5: a handle outlives its weights. Evict, watch residency, use it
/// again, and the weights come back.
#[test]
fn facade_evicted_model_is_restored_by_its_next_generation() {
    use gen2::Runtime;
    use gen2::model::ThinkingMode;

    let Some((first, second)) = two_model_paths() else {
        return;
    };
    let runtime = Runtime::new().expect("a runtime builds");
    let qwen = runtime.load(&first).expect("first model loads");
    let llama = runtime.load(&second).expect("second model loads");

    let before = runtime.residency();
    eprintln!(
        "--- before: {before:?}\n--- hardware: {:?}",
        runtime.hardware()
    );
    assert!(before.is_resident(qwen.id()) && before.is_resident(llama.id()));
    assert!(before.resident_mb() > 0);

    runtime.evict(&qwen).expect("evict");
    let evicted = runtime.residency();
    eprintln!("--- evicted: {evicted:?}");
    assert!(!evicted.is_resident(qwen.id()), "{evicted:?}");
    assert!(evicted.is_resident(llama.id()));
    assert!(evicted.resident_mb() < before.resident_mb());

    let text = qwen
        .generate("Reply with exactly one word: hi")
        .max_tokens(8)
        .greedy()
        .reasoning(ThinkingMode::Off)
        .text()
        .expect("an evicted model still generates — restored on use");
    eprintln!("--- restored: {text:?}");
    assert!(!text.trim().is_empty());
    let after = runtime.residency();
    eprintln!("--- after: {after:?}\n--- stats: {:?}", runtime.stats());
    assert!(after.is_resident(qwen.id()), "{after:?}");
    assert!(after.is_resident(llama.id()));
    assert_eq!(runtime.stats().resident_models, 2);

    // Preload is the eager form of the same restore.
    runtime.evict(&llama).expect("evict");
    assert!(!runtime.residency().is_resident(llama.id()));
    runtime.preload(&llama).expect("preload");
    assert!(runtime.residency().is_resident(llama.id()));
}

/// §23: two sessions on one model from two threads at once. The controller
/// serialises them; both complete.
#[test]
fn facade_two_sessions_on_one_model_run_from_two_threads() {
    use gen2::Session;
    use gen2::model::ThinkingMode;

    let Some(model) = facade_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL");
        return;
    };
    let handles: Vec<_> = [
        "Reply with exactly one word: red",
        "Reply with exactly one word: blue",
    ]
    .into_iter()
    .map(|prompt| {
        let model = model.clone();
        std::thread::spawn(move || {
            let mut session = Session::new();
            let r = model
                .turn(&mut session)
                .user(prompt)
                .max_tokens(8)
                .greedy()
                .reasoning(ThinkingMode::Off)
                .run()
                .map(|r| r.text().to_string());
            (r, session)
        })
    })
    .collect();
    for h in handles {
        let (text, session) = h.join().expect("no panic");
        let text = text.expect("each thread's turn completes");
        eprintln!("--- thread: {text:?}");
        assert!(!text.trim().is_empty());
        assert_eq!(session.messages().len(), 2);
    }
}
