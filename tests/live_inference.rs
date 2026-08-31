//! End-to-end proof that the extracted crate actually runs inference.
//!
//! Unit tests cover the pieces; this drives the real thing — a real GGUF
//! through the real llama.cpp backend, asserting on real decoded tokens. It is
//! the test that would have caught a broken extraction that still compiled.
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

use pio_gen2::engine::{Engine, LoadRequest};
use pio_gen2::generation::{GenSpec, TokenEvent};
use pio_gen2::session_rt::SessionSpec;
use pio_gen2::{Message, MessageBody, MessageContent};

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

fn user_message(text: &str) -> Message {
    Message {
        name: None,
        role: "user".into(),
        body: MessageBody::Content {
            content: MessageContent::SingleText(text.into()),
        },
    }
}

fn loaded_engine(model: PathBuf) -> Engine {
    let mut engine = Engine::new();
    engine
        .load_model(LoadRequest {
            model_path: model,
            ..Default::default()
        })
        .expect("real GGUF should load through the llama.cpp backend");
    engine
}

/// A sampler pinned to be reproducible: temperature 0 and a fixed seed.
///
/// `GenSpec::default()` leaves both `None`, which means backend-default
/// sampling with a random seed — correct for production, useless for asserting
/// on exact text.
fn deterministic_spec(max_tokens: usize) -> GenSpec {
    GenSpec {
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        seed: Some(42),
        ..Default::default()
    }
}

/// Drain a session's token stream into decoded text plus the terminal event.
fn generate_with(engine: &Engine, prompt: &str, gen_spec: GenSpec) -> (String, Option<TokenEvent>) {
    let session = engine
        .start_session(SessionSpec {
            messages: vec![user_message(prompt)],
            ..Default::default()
        })
        .expect("session should start on a loaded model");

    let mut puller = session
        .pull(gen_spec)
        .expect("pull should start a generation");

    let mut text = String::new();
    let mut terminal = None;
    for event in puller.by_ref() {
        // A decode error is a failure, never something to skip past — swallowing
        // it is how an empty generation gets reported as a successful one.
        match event.expect("decode step returned an error") {
            TokenEvent::Token(t) => text.push_str(&t.text),
            ev @ (TokenEvent::Eos | TokenEvent::Stopped) => {
                terminal = Some(ev);
                break;
            }
            _ => {}
        }
    }
    (text, terminal)
}

/// The load → session → decode path produces real text from a real model.
#[test]
fn generates_real_tokens_from_a_real_model() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = loaded_engine(model);
    let (text, terminal) = generate_with(
        &engine,
        "Reply with exactly one word: hello",
        deterministic_spec(24),
    );

    eprintln!("--- generated: {text:?} (terminal: {terminal:?})");

    assert!(
        !text.trim().is_empty(),
        "decoded an empty string — the model loaded but produced no tokens"
    );
    assert!(
        text.chars().any(|c| c.is_alphabetic()),
        "output has no letters, so this is not decoded text: {text:?}"
    );
    assert!(
        terminal.is_some(),
        "stream ended without Eos or Stopped — the generation did not terminate cleanly"
    );
}

/// At temperature 0 with a fixed seed, the same prompt twice gives the same
/// text. Catches a sampler or KV cache that survived the move but wired itself
/// to the wrong state — a fresh session must not inherit the previous one's.
#[test]
fn greedy_decoding_is_reproducible() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let engine = loaded_engine(model);
    let prompt = "Count: one two three";

    let (first, _) = generate_with(&engine, prompt, deterministic_spec(16));
    let (second, _) = generate_with(&engine, prompt, deterministic_spec(16));

    assert!(!first.trim().is_empty(), "first generation was empty");
    assert_eq!(
        first, second,
        "same prompt gave different text across two sessions — \
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

    let engine = loaded_engine(model);
    let session = engine
        .start_session(SessionSpec {
            messages: vec![user_message("Write a long story about a robot.")],
            ..Default::default()
        })
        .expect("session should start");

    const BUDGET: usize = 8;
    let mut puller = session
        .pull(deterministic_spec(BUDGET))
        .expect("pull should start");

    let mut tokens = 0_usize;
    for event in puller.by_ref() {
        match event.expect("decode step returned an error") {
            TokenEvent::Token(_) => tokens += 1,
            TokenEvent::Eos | TokenEvent::Stopped => break,
            _ => {}
        }
        assert!(
            tokens <= BUDGET,
            "generated {tokens} tokens against a budget of {BUDGET}"
        );
    }

    assert!(tokens > 0, "budget-limited generation produced no tokens");
}
