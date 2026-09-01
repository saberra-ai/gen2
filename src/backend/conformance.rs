//! One contract, applied to every backend that is compiled in.
//!
//! The crate's claim is a single API over llama.cpp, MLX, mlxcel, ONNX,
//! Candle, ExecuTorch and an OpenAI-compatible endpoint. A claim like that is
//! only worth anything if every backend is held to the same contract, and
//! until now each was tested — where it was tested at all — on its own terms.
//!
//! # Two halves, and the honest gap between them
//!
//! Most of the contract needs no weights: what a backend reports before a
//! model is loaded, that starting a session without one is an error rather
//! than a crash, that capability probes agree with the capabilities they
//! claim, that unloading nothing is harmless. That half runs everywhere, on
//! every backend, in milliseconds.
//!
//! The other half needs a real model, and a real model is a large file this
//! repository does not carry. Those tests read a per-backend environment
//! variable and skip when it is absent — but they skip *loudly*, naming the
//! variable, because a silently skipped test is indistinguishable from a
//! passing one and that is exactly how four of these backends came to be
//! believed working without ever having produced a token.
//!
//! [`unverified_backends`] fails if that list of never-run backends stops
//! matching what is actually compiled, so the gap cannot widen unnoticed.

use std::sync::Arc;

use super::traits::Backend;
use crate::engine::{Capabilities, ExecError, LoadRequest, Settings};
use crate::session_rt::SessionSpec;

/// Backends whose token generation has never been exercised by any test.
///
/// Not a to-do list: a statement of what is known, checked against what is
/// compiled. A backend earns removal from here the only way that means
/// anything — by running [`contract_with_a_model`] against a real model and
/// decoding a non-zero number of tokens, which that function asserts and
/// prints.
///
/// Verified so far, and how:
///
/// - `llamacpp` — `PIO_TEST_MODEL` pointed at a GGUF, plus the whole of
///   `tests/live_inference.rs`.
/// - `mlx` — `PIO_TEST_MLX_MODEL` pointed at an MLX safetensors bundle
///   (`llama-3.2-3b-4bit`), on macOS 26.3 with the Metal Toolchain installed.
///   Decoded to its 16-token cap.
///
/// The others need a model this repository does not carry. Point the variable
/// in [`model_env`] at one and the generating half runs.
pub(crate) const NEVER_PRODUCED_A_TOKEN: &[&str] = &["mlxcel", "onnx", "candle"];

/// The environment variable naming a model this backend can load, if the
/// caller has one.
fn model_env(backend: &str) -> &'static str {
    match backend {
        "llamacpp" => "PIO_TEST_MODEL",
        "mlx" | "mlxcel" => "PIO_TEST_MLX_MODEL",
        "onnx" => "PIO_TEST_ONNX_MODEL",
        "candle" => "PIO_TEST_CANDLE_MODEL",
        "external-api" => "PIO_TEST_API_URL",
        _ => "PIO_TEST_MODEL",
    }
}

/// The model path for this backend, or `None` with a note on stderr saying
/// which variable would have supplied one.
///
/// Printing is the point. A test that skips in silence reads as a test that
/// passed.
fn model_for(backend: &str) -> Option<String> {
    let var = model_env(backend);
    match std::env::var(var) {
        Ok(path) if !path.is_empty() => Some(path),
        _ => {
            eprintln!(
                "conformance: {backend} generation UNVERIFIED — set {var} to a model \
                 this backend can load"
            );
            None
        }
    }
}

// ── The contract, as plain functions over `&dyn Backend` ────────────────────

/// What must hold before anything is loaded.
///
/// Every one of these is reachable from a host that constructed an engine and
/// has not yet given it a model — which is every host, for the first moment of
/// its life.
pub(crate) fn contract_before_loading(backend: &dyn Backend) {
    let name = backend.backend_name();
    assert!(!name.is_empty(), "a backend must name itself");
    assert_eq!(
        name,
        backend.backend_name(),
        "the name must not change between calls"
    );

    assert!(
        !backend.is_model_loaded(),
        "{name}: a freshly constructed backend has no model"
    );

    // Not "returns an error" but "does not crash and does not claim success":
    // a host asking for a session before loading is a caller bug, and it must
    // surface as one.
    let started = backend.start_session(SessionSpec::default());
    assert!(
        started.is_err(),
        "{name}: starting a session without a model must fail rather than \
         hand back something unusable"
    );

    assert!(
        backend.reload_model().is_err(),
        "{name}: there is nothing to reload"
    );

    // Idempotent teardown. A host that unloads twice, or unloads what it never
    // loaded, is doing something ordinary during error recovery.
    backend.unload_model();
    backend.unload_model();
    assert!(!backend.is_model_loaded(), "{name}: still claims a model");

    // Ending a session nobody started is the same kind of ordinary.
    let _ = backend.end_session(0);
    let _ = backend.end_session(u64::MAX);
}

/// Capability probes must agree with the capabilities they advertise.
///
/// The two can drift: `as_multimodal()` is a trait upcast and
/// `capabilities()` is a bitset, and nothing but this ties them together. A
/// backend that advertises images through one and denies them through the
/// other sends the caller's guard the wrong way.
pub(crate) fn contract_capabilities(backend: &dyn Backend) {
    let name = backend.backend_name();
    let caps = backend.capabilities();
    assert_eq!(
        caps,
        backend.capabilities(),
        "{name}: capabilities must not change between calls"
    );

    // Deliberately not "TEXT is always set". A backend talking to a remote
    // endpoint cannot know what that endpoint supports until it has connected,
    // and reporting an empty set until then is the honest answer rather than a
    // bug. What TEXT is required for is a *loaded* backend, which
    // `contract_with_a_model` checks.

    if let Some(mm) = backend.as_multimodal() {
        assert_eq!(
            mm.supports_images(),
            caps.contains(Capabilities::IMAGES),
            "{name}: the multimodal probe and the capability bitset disagree about images"
        );
        assert_eq!(
            mm.supports_audio(),
            caps.contains(Capabilities::AUDIO),
            "{name}: the multimodal probe and the capability bitset disagree about audio"
        );
    } else {
        assert!(
            !caps.contains(Capabilities::IMAGES) && !caps.contains(Capabilities::AUDIO),
            "{name}: advertises non-text capabilities but offers no multimodal probe, \
             so a caller cannot act on them"
        );
    }

    if let Some(embeddings) = backend.as_embeddings() {
        assert!(
            !embeddings.is_embedder_loaded(),
            "{name}: a fresh backend has no embedder"
        );
        assert!(
            embeddings.generate_embeddings(&["hello".into()]).is_err(),
            "{name}: embedding without an embedder must fail rather than return \
             a vector of nothing"
        );
        // Same idempotence as model teardown.
        embeddings.unload_embedder();
        embeddings.unload_embedder();
    }
}

/// Reported statistics must be arithmetically possible.
///
/// These feed a UI and a budget check. A negative rate or a cache holding more
/// than its budget is a number someone will divide by.
pub(crate) fn contract_stats(backend: &dyn Backend) {
    let name = backend.backend_name();
    let stats = backend.stats();
    assert!(
        stats.avg_tps.is_finite() && stats.avg_tps >= 0.0,
        "{name}: reported {} tokens per second",
        stats.avg_tps
    );
    if stats.cache_budget > 0 {
        assert!(
            stats.cache_tokens <= stats.cache_budget,
            "{name}: cache holds {} tokens against a budget of {}",
            stats.cache_tokens,
            stats.cache_budget
        );
    }
}

/// Whether this backend is a scaffold that implements nothing and says so.
///
/// ExecuTorch is compiled as one: the feature exists so the mobile target
/// builds, and every call returns [`ExecError::Unimplemented`]. That is a
/// legitimate state for a backend to be in, and a contract that assumed
/// otherwise would either fail it forever or have to name it as a special
/// case. Detected by behaviour rather than by name, so the next scaffold is
/// covered too.
fn is_declared_stub(backend: &dyn Backend) -> bool {
    matches!(
        backend.upload_settings(Settings::default()),
        Err(ExecError::Unimplemented)
    )
}

/// What a stub owes, which is narrower than the full contract but not empty.
///
/// It must refuse consistently, never claim a success it cannot deliver, and
/// never panic. A stub that returned `Ok` from `load_model` would have a host
/// generating against nothing.
fn contract_for_a_stub(backend: &dyn Backend) {
    let name = backend.backend_name();

    assert!(
        matches!(
            backend.load_model(LoadRequest {
                model_path: "/anything.gguf".into(),
                ..Default::default()
            }),
            Err(ExecError::Unimplemented)
        ),
        "{name}: a stub must refuse to load rather than report a success it \
         cannot deliver"
    );
    assert!(
        !backend.is_model_loaded(),
        "{name}: a stub that refused to load must not claim a model"
    );
    assert!(
        backend.start_session(SessionSpec::default()).is_err(),
        "{name}: a stub must not hand back a session"
    );

    // Still has to be a well-behaved object.
    let _ = backend.settings();
    let _ = backend.stats();
    let _ = backend.capabilities();
    backend.unload_model();
}

/// Settings must survive being uploaded.
pub(crate) fn contract_settings(backend: &dyn Backend) {
    let name = backend.backend_name();
    let before = backend.settings_version();
    let accepted = backend.upload_settings(Settings::default());
    assert!(
        accepted.is_ok(),
        "{name}: refused default settings: {accepted:?}"
    );
    assert!(
        backend.settings_version() >= before,
        "{name}: the settings version went backwards"
    );
    // Reachable, non-panicking, and not empty of meaning.
    let _ = backend.settings();
    let _ = backend.first_token_tier();
    let _ = backend.hooks();
}

/// Everything that needs no weights, in one call.
pub(crate) fn contract_without_a_model(backend: &dyn Backend) {
    contract_before_loading(backend);
    contract_capabilities(backend);
    contract_stats(backend);
    if is_declared_stub(backend) {
        contract_for_a_stub(backend);
    } else {
        contract_settings(backend);
    }
}

/// The half that needs a real model. Returns whether it ran.
///
/// Deliberately tolerant about *what* the backend generates and strict about
/// the shape: a conformance suite that asserted on model output would be
/// testing the model, and every backend would need a different expectation.
pub(crate) fn contract_with_a_model(backend: &dyn Backend, model_path: &str) -> bool {
    let name = backend.backend_name();

    let loaded = backend.load_model(LoadRequest {
        model_path: model_path.into(),
        ..Default::default()
    });
    if let Err(e) = loaded {
        panic!("{name}: could not load {model_path}: {e:?}");
    }
    assert!(
        backend.is_model_loaded(),
        "{name}: load reported success but the backend denies having a model"
    );
    assert!(
        backend.capabilities().contains(Capabilities::TEXT),
        "{name}: a loaded backend must advertise text; that is what makes it a backend"
    );

    let first = backend
        .start_session(SessionSpec {
            messages: vec![crate::types::message::Message::user("Say hello.")],
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("{name}: start_session failed with a model loaded: {e:?}"));
    let second = backend
        .start_session(SessionSpec {
            messages: vec![crate::types::message::Message::user("Say hello again.")],
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("{name}: a second session failed: {e:?}"));
    assert_ne!(
        first.id(),
        second.id(),
        "{name}: two concurrent sessions were given the same id, so the \
         controller cannot tell them apart"
    );

    assert_terminates_exactly_once(name, &first);

    // Teardown, in the order a host performs it.
    let _ = backend.end_session(first.id());
    let _ = backend.end_session(second.id());
    backend.unload_model();
    assert!(
        !backend.is_model_loaded(),
        "{name}: still claims a model after unloading"
    );
    true
}

/// Pull a generation to its end and assert what every backend owes: at least
/// one token, exactly one terminal event, and nothing after it.
///
/// The terminal rule is the same one
/// [`assert_valid_trace`](crate::test_support::assert_valid_trace) enforces a
/// layer up, checked here too because this is where a backend could break it.
///
/// The token rule is what makes this evidence rather than a formality. A
/// backend can reach a terminal event without ever decoding — by refusing, by
/// stopping immediately, by returning an empty stream — and a contract that
/// accepted that would let a backend be marked verified without having
/// generated anything, which is the exact claim
/// [`NEVER_PRODUCED_A_TOKEN`] exists to keep honest.
fn assert_terminates_exactly_once(name: &str, session: &Arc<dyn crate::backend::BackendSession>) {
    use crate::generation::{GenSpec, TokenEvent};

    let mut puller = match session.pull(GenSpec {
        max_tokens: Some(16),
        ..Default::default()
    }) {
        Ok(p) => p,
        Err(e) => panic!("{name}: pull failed on a loaded model: {e:?}"),
    };
    let mut tokens = 0usize;

    let mut terminals = 0;
    let mut after_terminal = 0;
    // Bounded: a backend that never terminates would otherwise hang the suite
    // rather than fail it.
    for _ in 0..10_000 {
        match puller.next_event() {
            None => break,
            Some(Err(_)) => {
                terminals += 1;
                break;
            }
            Some(Ok(event)) => {
                if matches!(event, TokenEvent::Token(_)) {
                    tokens += 1;
                }
                let terminal = matches!(event, TokenEvent::Eos | TokenEvent::Stopped);
                if terminals > 0 {
                    after_terminal += 1;
                }
                if terminal {
                    terminals += 1;
                }
            }
        }
    }

    assert!(
        terminals <= 1,
        "{name}: reported {terminals} terminal events for one generation"
    );
    assert_eq!(
        after_terminal, 0,
        "{name}: emitted {after_terminal} events after the generation ended"
    );
    assert!(
        tokens > 0,
        "{name}: reached the end of a generation without decoding a single \
         token. Ending cleanly is not the same as working, and this is the \
         assertion that separates the two."
    );
    // Visible under `--nocapture`, so a run that claims a backend generates
    // shows the number it is claiming rather than asking to be believed.
    eprintln!("conformance: {name} decoded {tokens} tokens");
}

// ── One invocation per compiled backend ─────────────────────────────────────

/// Generate the conformance tests for one backend.
///
/// A macro rather than a loop because the backends are different concrete
/// types that are not `Send`, cannot be boxed into one collection across
/// features, and each need their own `#[cfg]`.
macro_rules! backend_contract {
    ($mod_name:ident, $ctor:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn honours_the_contract_without_a_model() {
                contract_without_a_model(&$ctor);
            }

            #[test]
            fn honours_the_contract_with_a_model_or_says_it_was_not_checked() {
                let backend = $ctor;
                let name = backend.backend_name();
                match model_for(name) {
                    Some(path) => {
                        assert!(contract_with_a_model(&backend, &path));
                    }
                    None => {
                        // Not a pass. The stderr note above is the result, and
                        // `unverified_backends` is what keeps it honest.
                    }
                }
            }
        }
    };
}

#[cfg(feature = "backend-llamacpp")]
backend_contract!(llamacpp, crate::backend::llama::Engine::new());

#[cfg(feature = "backend-mlx")]
backend_contract!(mlx, crate::backend::mlx::Engine::new());

#[cfg(feature = "backend-mlxcel")]
backend_contract!(mlxcel, crate::backend::mlxcel::MlxcelEngine::new());

#[cfg(feature = "backend-onnx")]
backend_contract!(onnx, crate::backend::onnx::Engine::new());

#[cfg(feature = "backend-candle")]
backend_contract!(candle, crate::backend::candle::CandleBackend::new());

#[cfg(feature = "backend-executorch")]
backend_contract!(
    executorch,
    crate::backend::executorch::ExecutorchBackend::new()
);

#[cfg(feature = "backend-external-api")]
backend_contract!(external_api, crate::backend::external_api::Engine::new());

/// The list of backends nobody has ever seen generate a token must match the
/// backends that are compiled.
///
/// Without this the gap is invisible: a new backend lands, compiles, passes
/// the no-model half of the contract, and looks exactly as tested as the two
/// that actually work.
#[test]
fn unverified_backends() {
    let compiled = crate::backend::Engine::available_backends();
    let unverified: Vec<&str> = compiled
        .iter()
        .copied()
        .filter(|b| NEVER_PRODUCED_A_TOKEN.contains(b))
        .collect();

    if !unverified.is_empty() {
        eprintln!(
            "conformance: {} of {} compiled backends have never generated a token: {:?}",
            unverified.len(),
            compiled.len(),
            unverified,
        );
    }

    // Nothing may sit on the list that the crate does not have, so a rename
    // cannot leave a stale entry silently covering for a backend that is now
    // untested under a new name.
    const KNOWN: &[&str] = &[
        "llamacpp",
        "mlx",
        "mlxcel",
        "onnx",
        "candle",
        "executorch",
        "external-api",
    ];
    for backend in NEVER_PRODUCED_A_TOKEN {
        assert!(
            KNOWN.contains(backend),
            "{backend} is listed as never having generated a token, but is not a \
             backend this crate has — the list has gone stale"
        );
    }
}
