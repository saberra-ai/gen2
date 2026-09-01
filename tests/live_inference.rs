//! End-to-end proof that the extracted crate actually runs inference.
//!
//! Unit tests cover the pieces; this drives the real thing — a real GGUF
//! through the real llama.cpp backend, asserting on real decoded tokens. It is
//! the test that would have caught a broken extraction that still compiled.
//!
//! It goes through the controller because that is the crate's public API. As an
//! external test target it can reach exactly what any other consumer can, so it
//! doubles as proof that the narrowed surface is actually sufficient to load a
//! model and generate from it.
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
use std::sync::mpsc::{channel, sync_channel};

use pio_gen2::controller::start_controller;
use pio_gen2::{ControllerCmd, ControllerEvent, ControllerHandle, GenSpec, Message, Settings};

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

/// A controller with the model loaded and confirmed ready.
fn loaded_controller(model: PathBuf) -> ControllerHandle {
    let handle = start_controller();
    let (resp, resp_rx) = channel();

    handle
        .send(ControllerCmd::LoadModel {
            model_path: model,
            mmproj_path: None,
            settings: Settings::default(),
            api_key: None,
            api_format: None,
            resp,
        })
        .expect("controller should accept a LoadModel command");

    resp_rx
        .recv()
        .expect("controller should answer LoadModel")
        .expect("real GGUF should load through the llama.cpp backend");

    handle
}

/// Stop the controller and let it release the backend.
///
/// Not optional hygiene: the loop runs on its own thread holding the llama.cpp
/// context, and a test binary that reaches `exit()` while it is still alive
/// aborts inside ggml's static destructors — a passing test with a SIGABRT
/// after it. Shutdown is fire-and-forget (no ack on the command), so this waits
/// for the loop to actually wind down.
fn shutdown(handle: ControllerHandle) {
    let _ = handle.send(ControllerCmd::Shutdown);
    drop(handle);
    std::thread::sleep(std::time::Duration::from_millis(250));
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

/// Run one chat turn to completion, returning the decoded text and the
/// terminal event.
fn generate(
    handle: &ControllerHandle,
    chat_id: &str,
    prompt: &str,
    gen_spec: GenSpec,
) -> (String, Option<ControllerEvent>) {
    let (tx, rx) = sync_channel(handle.config().event_channel_capacity);

    handle
        .send(ControllerCmd::StartChat {
            chat_id: chat_id.into(),
            messages: vec![Message::user(prompt)],
            gen_spec,
            thinking: Default::default(),
            model_id: None,
            model_size_bytes: None,
            tools: None,
            tx,
        })
        .expect("controller should accept a StartChat command");

    let mut text = String::new();
    let mut terminal = None;
    for event in rx {
        match event {
            ControllerEvent::Token(t) => text.push_str(&t),
            // A generation error is a failure, never something to read past —
            // swallowing it is how an empty result gets reported as success.
            ControllerEvent::Error { code, message } => {
                panic!("generation failed [{code}]: {message}")
            }
            ev @ (ControllerEvent::Eos | ControllerEvent::Stopped) => {
                terminal = Some(ev);
                break;
            }
            _ => {}
        }
    }
    (text, terminal)
}

/// The load → chat → decode path produces real text from a real model.
#[test]
fn generates_real_tokens_from_a_real_model() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let handle = loaded_controller(model);
    let (text, terminal) = generate(
        &handle,
        "live-1",
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
        matches!(terminal, Some(ControllerEvent::Eos)),
        "expected the stream to end on Eos, got {terminal:?}"
    );

    shutdown(handle);
}

/// At temperature 0 with a fixed seed, the same prompt twice gives the same
/// text. Catches a sampler or KV cache that survived the move but wired itself
/// to the wrong state — a fresh chat must not inherit the previous one's.
#[test]
fn greedy_decoding_is_reproducible() {
    let Some(model) = test_model() else {
        eprintln!("SKIP: set PIO_TEST_MODEL to run live inference");
        return;
    };

    let handle = loaded_controller(model);
    let prompt = "Count: one two three";

    let (first, _) = generate(&handle, "repro-a", prompt, deterministic_spec(16));
    let (second, _) = generate(&handle, "repro-b", prompt, deterministic_spec(16));

    assert!(!first.trim().is_empty(), "first generation was empty");
    assert_eq!(
        first, second,
        "same prompt gave different text across two chats — \
         sampler or session state is not being reset"
    );

    shutdown(handle);
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
    let handle = loaded_controller(model);
    let (tx, rx) = sync_channel(handle.config().event_channel_capacity);

    handle
        .send(ControllerCmd::StartChat {
            chat_id: "budget-1".into(),
            messages: vec![Message::user("Write a long story about a robot.")],
            gen_spec: deterministic_spec(BUDGET),
            thinking: Default::default(),
            model_id: None,
            model_size_bytes: None,
            tools: None,
            tx,
        })
        .expect("controller should accept a StartChat command");

    let mut tokens = 0_usize;
    for event in rx {
        match event {
            ControllerEvent::Token(_) => tokens += 1,
            ControllerEvent::Error { code, message } => {
                panic!("generation failed [{code}]: {message}")
            }
            ControllerEvent::Eos | ControllerEvent::Stopped => break,
            _ => {}
        }
        assert!(
            tokens <= BUDGET,
            "generated {tokens} tokens against a budget of {BUDGET}"
        );
    }

    assert!(tokens > 0, "budget-limited generation produced no tokens");

    shutdown(handle);
}
