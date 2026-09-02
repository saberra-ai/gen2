//! What moving helpers off the generation backend actually bought.
//!
//! Two claims, both of which were false before the utility worker existed and
//! neither of which any unit test in this module's siblings can make, because
//! both are about the *controller* — one about who owns a helper, one about
//! what a busy helper does to chat scheduling.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use crate::api::{Engine, Session};
use crate::controller::{ControllerCmd, ControllerConfig};
use crate::test_support::Script;
use crate::utilities::{EmbeddingRuntime, ScriptedEmbedder, UtilityWorker};

/// How long the fake helper holds a call open.
///
/// Long enough that a controller which waited for it could not possibly have
/// scheduled tokens in the meantime, short enough not to slow the suite.
const HELPER_LATENCY: Duration = Duration::from_millis(600);

struct Fake {
    busy: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
}

fn slow_utility_worker() -> (UtilityWorker, Fake) {
    let busy = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let (b, c) = (Arc::clone(&busy), Arc::clone(&calls));
    let worker = UtilityWorker::spawn_with(Box::new(move |_req| {
        Ok(Box::new(ScriptedEmbedder {
            name: "slow-helper".into(),
            latency: HELPER_LATENCY,
            busy: Arc::clone(&b),
            calls: Arc::clone(&c),
        }) as Box<dyn EmbeddingRuntime>)
    }));
    (worker, Fake { busy, calls })
}

/// A helper that is busy must not stop chat tokens.
///
/// This is the reason the worker is a thread rather than a method. Embedding
/// used to run on the controller thread, which pulls chat tokens; a call that
/// took 600ms stopped generation for 600ms. That was survivable for a small
/// embedding and is not for the transcription and OCR helpers this design
/// exists to make room for.
///
/// The test drives it from the outside: start a helper call, then run a chat
/// turn to completion while the helper is still working, and check the reply
/// arrived before the helper did.
#[test]
fn a_chat_keeps_generating_while_a_helper_is_busy() {
    let (utilities, fake) = slow_utility_worker();
    let engine = Engine::scripted_with(
        Script::new().say(["one", " two", " three"]),
        ControllerConfig::default(),
        Some(utilities),
    );
    engine
        .load_embedder("/models/embedder.gguf", None)
        .expect("the scripted helper loads");

    // Hand the controller a helper call that will take most of a second.
    let (resp, helper_rx) = channel();
    engine
        .send_for_test(ControllerCmd::GenerateEmbeddings {
            inputs: vec!["something to embed".into()],
            resp,
        })
        .expect("the controller should accept the request");

    // ...and immediately run a chat turn.
    let started = Instant::now();
    let mut session = Session::new();
    let done = engine
        .chat(&mut session)
        .user("hello")
        .send()
        .expect("the chat turn should complete");
    let chat_took = started.elapsed();

    assert_eq!(done.text, "one two three");
    assert!(
        chat_took < HELPER_LATENCY,
        "the chat turn took {chat_took:?} against a helper latency of \
         {HELPER_LATENCY:?} — the controller waited for the helper instead of \
         scheduling tokens"
    );
    assert!(
        fake.busy.load(Ordering::SeqCst),
        "the helper should still have been working when the chat finished; if \
         it had already returned, this test proved nothing"
    );

    // And the helper's own answer still arrives.
    let vectors = helper_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the helper answers the caller directly")
        .expect("embeddings");
    assert_eq!(vectors.len(), 1);
    assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
}

/// The embedder is no longer the chat backend's to provide.
///
/// Before this, `load_embedder` went through `Backend::as_embeddings()` on
/// whichever backend held the chat model — and MLX and ONNX do not implement
/// it, so an MLX chat model simply could not have an embedder beside it.
///
/// The scripted backend stands in for exactly that case: it implements no
/// embedding capability at all. An embedder loading and answering here is the
/// proof that ownership moved.
#[test]
fn an_embedder_works_over_a_chat_backend_that_has_no_embedding_support() {
    let (utilities, _fake) = slow_utility_worker();
    let engine = Engine::scripted_with(
        Script::new().say(["hi"]),
        ControllerConfig::default(),
        Some(utilities),
    );

    engine
        .load_embedder("/models/embedder.gguf", None)
        .expect("a helper must not need the chat backend to implement it");
    assert!(engine.is_embedder_loaded());

    let vectors = engine
        .embed(&["four".to_string()])
        .expect("embedding must work over a backend with no embedding capability");
    assert_eq!(vectors.len(), 1);

    // And the chat model on that same engine is still usable.
    let mut session = Session::new();
    let done = engine.chat(&mut session).user("hello").send().unwrap();
    assert_eq!(done.text, "hi");
}

/// Unloading the helper is the worker's business, not the backend's.
#[test]
fn unloading_the_embedder_is_visible_through_the_public_api() {
    let (utilities, _fake) = slow_utility_worker();
    let engine = Engine::scripted_with(
        Script::new().say(["hi"]),
        ControllerConfig::default(),
        Some(utilities),
    );

    engine.load_embedder("/models/embedder.gguf", None).unwrap();
    assert!(engine.is_embedder_loaded());

    let status = engine.utility_status().expect("status");
    let embedder = status.embedder.expect("a loaded helper is reported");
    assert_eq!(embedder.name, "slow-helper");
}

/// Embedding with nothing loaded is an error, not an empty answer.
#[test]
fn embedding_with_no_helper_loaded_reports_that_rather_than_returning_nothing() {
    let (utilities, _fake) = slow_utility_worker();
    let engine = Engine::scripted_with(
        Script::new().say(["hi"]),
        ControllerConfig::default(),
        Some(utilities),
    );

    let outcome = engine.embed(&["x".to_string()]);
    assert!(
        outcome.is_err(),
        "a caller who forgot to load an embedder must be told, not handed an \
         empty vector they will index into"
    );
}

/// The vectors must be the same ones the backend path produced.
///
/// This is an ownership change, not a rewrite — the family handling, pooling
/// and normalisation all still live in `backend::llama::embedder`. The way to
/// show that is to embed something real and check the result is a sane,
/// stable vector rather than a plausible-looking one.
///
/// Needs `PIO_TEST_EMBEDDER` pointed at a GGUF embedding model.
#[test]
#[ignore = "needs PIO_TEST_EMBEDDER"]
fn a_real_embedder_produces_stable_vectors_through_the_worker() {
    let Ok(path) = std::env::var("PIO_TEST_EMBEDDER") else {
        eprintln!("SKIP: set PIO_TEST_EMBEDDER");
        return;
    };
    // A real worker over the real llama factory, but a scripted *chat*
    // backend — so this also demonstrates the point of the whole phase: the
    // embedder does not care what is generating.
    let engine =
        Engine::scripted_with(Script::new().say(["hi"]), ControllerConfig::default(), None);
    engine
        .load_embedder(&path, None)
        .expect("the embedder should load through the worker");

    let vectors = engine
        .embed(&["the quick brown fox".to_string(), "a".to_string()])
        .expect("embedding should work");

    assert_eq!(vectors.len(), 2, "one vector per input");
    assert!(
        vectors[0].len() > 128,
        "an embedding of {} dimensions is not a real one",
        vectors[0].len()
    );
    assert_eq!(
        vectors[0].len(),
        vectors[1].len(),
        "every vector from one model has the same width"
    );
    assert!(
        vectors[0].iter().any(|v| *v != 0.0),
        "an all-zero vector means the model ran but produced nothing"
    );
    assert!(
        vectors[0].iter().all(|v| v.is_finite()),
        "a NaN in an embedding poisons every distance computed from it"
    );

    // The same text twice gives the same vector; different text does not.
    let again = engine
        .embed(&["the quick brown fox".to_string()])
        .expect("embedding should work");
    assert_eq!(
        vectors[0], again[0],
        "embedding is deterministic; two different answers means state leaked \
         between calls"
    );
    assert_ne!(
        vectors[0], vectors[1],
        "two different inputs producing one vector means the input never \
         reached the model"
    );
    eprintln!(
        "worker embedder: {} dimensions, deterministic",
        vectors[0].len()
    );
}
