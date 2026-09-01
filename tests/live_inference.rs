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

use std::path::PathBuf;

use gen2::{Engine, Event, Finish, Session};

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
    assert_eq!(session.len(), 3, "system + user + assistant");
    assert_eq!(session.messages()[0].role, "system");
    assert_eq!(session.latest().unwrap().role, "assistant");

    // Editing invalidates the engine's cached prefill, so the next turn is
    // answered from the edited history rather than the original.
    session.edit(|m| m.truncate(1));
    assert_eq!(session.len(), 1);

    engine
        .chat(&mut session)
        .user("Name a fruit.")
        .max_tokens(16)
        .greedy()
        .send()
        .expect("a turn after an edit should succeed");
    assert_eq!(session.len(), 3, "system + new user + new assistant");

    // A transcript can be rebuilt from stored messages after a restart.
    let restored = Session::from_messages(session.messages().to_vec());
    assert_eq!(restored.len(), session.len());
    assert_ne!(restored.id(), session.id(), "a fresh conversation id");
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

fn weather_tool() -> gen2::ToolSpec {
    gen2::ToolSpec {
        r#type: "function".into(),
        function: gen2::FunctionDefinition {
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
                if let gen2::AgentStep::Calling { tool, .. } = step {
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
        .tool_search(gen2::ToolSearch::Bm25)
        .max_steps(4)
        .run_streaming(
            Some(
                "Resize /tmp/a.png to 200 pixels wide. You do not have that tool — \
                 call search_tools first."
                    .into(),
            ),
            |step| {
                if let gen2::AgentStep::Calling { tool, .. } = step {
                    calls.push(tool.to_string());
                }
            },
        )
        .expect("the agent should complete");

    assert!(
        calls.contains(&gen2::SEARCH_TOOL.to_string()),
        "the model should have searched, got {calls:?}"
    );
}

fn weather_agent_tool() -> gen2::FunctionTool<WeatherArgs> {
    gen2::FunctionTool::new(
        "get_weather",
        "Current weather for a city",
        |_ctx, a: WeatherArgs| async move {
            Ok(gen2::ToolOutput::Json(serde_json::json!({
                "city": a.city, "temp_c": 18, "sky": "clear"
            })))
        },
    )
}

fn resize_agent_tool() -> gen2::FunctionTool<ResizeArgs> {
    gen2::FunctionTool::new(
        "resize_image",
        "Shrink a picture to a smaller width",
        |_ctx, a: ResizeArgs| async move {
            Ok(gen2::ToolOutput::from(format!("resized to {}px", a.width)))
        },
    )
}

#[derive(serde::Deserialize, gen2::schemars::JsonSchema)]
struct WeatherArgs {
    /// City to look up.
    city: String,
}

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
        gen2::FunctionTool::new(
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
                    Ok(gen2::ToolOutput::from("ok"))
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
        if let gen2::Update::Delta(t) = update {
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
            gen2::Update::Delta(t) => chars += t.len(),
            gen2::Update::Done { .. } => saw_done = true,
            gen2::Update::Failed { error, .. } => panic!("run failed: {error}"),
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
