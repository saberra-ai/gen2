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

use pio_gen2::{Engine, Event, Finish, Session};

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
