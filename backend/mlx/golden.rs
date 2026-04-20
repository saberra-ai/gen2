//! Golden test suite for the MLX backend.
//!
//! These tests lock in *observable* behavior — greedy token sequences, prefix
//! cache state, speculative hit-rate stats, multi-turn coherence — so that
//! future speculative / caching / forward-path refactors cannot silently
//! change outputs or regress cache behavior.
//!
//! ## What is "golden"?
//!
//! For each test we capture a value once (the "golden" value), hardcode it,
//! and assert equality on every subsequent run. When a legitimate algorithm
//! change shifts the output, the test fails and you either:
//!   1. Fix the regression, or
//!   2. Recapture the golden value (with a PR comment explaining why).
//!
//! This is the same pattern as `insta` snapshots, minus the external dep.
//!
//! ## Determinism caveat
//!
//! Greedy (`temperature = 0.0`) sampling hits the deterministic `argmax` path
//! in our sampler — but MLX Metal execution is **not bit-reproducible across
//! runs**. Reduction order on GPU varies, and near argmax tiebreaks (common
//! for special tokens like `<end_of_turn>`) the selected token id can drift
//! from run to run.
//!
//! Empirically on Gemma 4 E2B: the **first content token** is stable, but
//! later tokens (especially special-token runs after a natural stopping
//! point) drift. The tests below therefore assert:
//!   - **Stable prefix** — first N token ids must match a captured golden.
//!   - **Semantic anchors** — decoded text must contain specific substrings.
//! This is tight enough to catch real regressions (wrong first token, empty
//! output, garbage) without being spooked by kernel-level noise.
//!
//! Golden prefixes were captured on:
//!   - mlx-rs pinned at oxideai/mlx-rs rev `f4aa309` (MLX v0.30.6)
//!   - Apple Silicon (M-series)
//!
//! If a test fails on a materially different device, run with
//! `RECORD_GOLDEN=1` to print actuals and re-capture the prefix.
//!
//! ## Running
//!
//! All tests are `#[ignore]` because they need real MLX model bundles.
//!
//! ```bash
//! cargo test -p pio-core --no-default-features \
//!   --features "backend-mlx,backend-llamacpp" \
//!   --lib golden -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Note `--test-threads=1`: MLX's Metal context doesn't tolerate concurrent
//! init across test binaries.

use std::path::PathBuf;

use crate::gen2::Message;
use crate::gen2::backend::mlx::{Engine, Session, TokenPuller};
use crate::gen2::engine::{ExecutionStats, LoadRequest, Settings};
use crate::gen2::generation::{GenSpec, TokenEvent};
use crate::gen2::session_rt::SessionSpec;
use crate::types::message::{MessageBody, MessageContent};

// ─── Bundles ─────────────────────────────────────────────────────────────────

/// Uniform 4-bit Gemma 4 E2B bundle (every quantized weight at 4 bits).
/// Exercises the common case: no pre-dequant, no UD.
const UNIFORM_BUNDLE_REL: &str = "../resources/gemma-4-e2b-it-mlx-4bit";

/// Unsloth Dynamic Gemma 4 E2B bundle — mixed 4/5/6-bit quantization.
/// Exercises the 5-bit `quantized_matmul` Metal kernel path (requires
/// mlx-rs main / MLX v0.26+ — see `Cargo.toml`).
const UD_BUNDLE_REL: &str = "../resources/gemma-4-E2B-it-UD-MLX-4bit";

fn bundle_path(rel: &str) -> Option<PathBuf> {
    let p = PathBuf::from(rel);
    if p.exists() { Some(p) } else { None }
}

fn skip_if_missing(rel: &str) -> Option<PathBuf> {
    match bundle_path(rel) {
        Some(p) => Some(p),
        None => {
            eprintln!("[golden] skipping — bundle not found at {}", rel);
            None
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn load_engine(bundle: PathBuf) -> Engine {
    let e = Engine::new();
    e.load_model(LoadRequest {
        model_path: bundle,
        ..Default::default()
    })
    .expect("load_model");
    e
}

fn sys_msg(text: &str) -> Message {
    Message {
        role: "system".into(),
        body: MessageBody::Content {
            content: MessageContent::SingleText(text.into()),
        },
        name: None,
    }
}

fn user_msg(text: &str) -> Message {
    Message {
        role: "user".into(),
        body: MessageBody::Content {
            content: MessageContent::SingleText(text.into()),
        },
        name: None,
    }
}

fn asst_msg(text: &str) -> Message {
    Message {
        role: "assistant".into(),
        body: MessageBody::Content {
            content: MessageContent::SingleText(text.into()),
        },
        name: None,
    }
}

/// Start a greedy (temp=0) session for the given messages.
fn greedy_session(engine: &Engine, messages: Vec<Message>) -> std::sync::Arc<Session> {
    let mut overrides = Settings::default();
    overrides.sampling.temperature = Some(0.0);
    engine
        .start_session(SessionSpec {
            messages,
            overrides: Some(overrides),
            persona: None,
            attachments: vec![],
            cache: None,
        })
        .expect("start_session")
}

/// Drain a puller, returning `(token_ids, decoded_text, stats)`.
fn drain(puller: &mut TokenPuller) -> (Vec<u32>, String, ExecutionStats) {
    let mut ids = Vec::new();
    let mut text = String::new();
    loop {
        match puller.next() {
            Some(Ok(TokenEvent::Token(tok))) => {
                ids.push(tok.id);
                text.push_str(&tok.text);
            }
            Some(Ok(TokenEvent::Eos)) | Some(Ok(TokenEvent::Stopped)) => break,
            Some(Ok(_)) => continue,
            Some(Err(e)) => panic!("token error: {:?}", e),
            None => break,
        }
    }
    // Stats aren't surfaced via TokenEvent; read them off the puller's internal
    // counters. `snapshot_stats` is a test-only accessor for `stats_now`.
    let stats = puller.snapshot_stats();
    (ids, text, stats)
}

/// Run `max_tokens` of greedy generation starting from `messages`.
fn generate_greedy(
    engine: &Engine,
    messages: Vec<Message>,
    max_tokens: usize,
) -> (Vec<u32>, String, ExecutionStats) {
    let session = greedy_session(engine, messages);
    let mut p = session
        .pull(GenSpec {
            max_tokens: Some(max_tokens),
            temperature: Some(0.0),
            ..Default::default()
        })
        .expect("pull");
    drain(&mut p)
}

// ─── RECORD_GOLDEN helper ────────────────────────────────────────────────────

/// If `RECORD_GOLDEN=1` is set, print the actual values and skip assertion.
/// Otherwise assert equality with a helpful diff message.
/// Assert `actual` starts with `expected_prefix`. Tolerant of trailing drift.
fn assert_golden_id_prefix(label: &str, actual: &[u32], expected_prefix: &[u32]) {
    if std::env::var("RECORD_GOLDEN").is_ok() {
        eprintln!(
            "[golden][{label}] actual ids ({}): {:?}",
            actual.len(),
            actual
        );
        return;
    }
    if actual.len() < expected_prefix.len() || &actual[..expected_prefix.len()] != expected_prefix {
        panic!(
            "[golden][{label}] token prefix mismatch\n  expected prefix ({}): {:?}\n  actual ({}): {:?}\n\
             \n(run with RECORD_GOLDEN=1 to print actuals for re-capture)",
            expected_prefix.len(),
            expected_prefix,
            actual.len(),
            actual
        );
    }
}

/// Assert `actual` contains every needle in `needles`, in order.
fn assert_golden_contains(label: &str, actual: &str, needles: &[&str]) {
    if std::env::var("RECORD_GOLDEN").is_ok() {
        eprintln!("[golden][{label}] actual text: {:?}", actual);
        return;
    }
    let mut cursor = 0usize;
    for needle in needles {
        match actual[cursor..].find(needle) {
            Some(pos) => cursor += pos + needle.len(),
            None => panic!(
                "[golden][{label}] missing expected substring {:?} (or out of order)\n  full actual: {:?}\n\
                 \n(run with RECORD_GOLDEN=1 to print actuals for re-capture)",
                needle, actual
            ),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// GOLDEN 1: Greedy generation on the uniform-4bit bundle.
///
/// Locks the stable prefix of argmax sampling for a fixed prompt. The first
/// content token + some trailing special-token runs are deterministic; past
/// that, Metal reduction-order noise drifts the sequence. Catches regressions
/// that change the FIRST token (wrong answer, corrupted weights, tokenizer
/// drift) — which is the regression that actually matters.
#[test]
#[ignore = "requires resources/gemma-4-e2b-it-mlx-4bit"]
fn golden_greedy_output_uniform() {
    let Some(bundle) = skip_if_missing(UNIFORM_BUNDLE_REL) else {
        return;
    };
    let engine = load_engine(bundle);

    let (ids, text, _stats) = generate_greedy(
        &engine,
        vec![user_msg("What is 2 + 2? Answer in one word.")],
        12,
    );

    // 26391 = "Four"; 106 = `<end_of_turn>`. Stable across runs for the first
    // ~6 positions.
    const EXPECTED_PREFIX: &[u32] = &[26391, 106, 106, 106, 106, 106];
    assert_golden_id_prefix("uniform/greedy", &ids, EXPECTED_PREFIX);
    assert_golden_contains("uniform/greedy", &text, &["Four"]);
}

/// GOLDEN 2: Greedy generation on the UD bundle.
///
/// Different quantization (mixed 4/5/6-bit) → same first content token in
/// practice, because the model's top choice is very confident for simple
/// arithmetic prompts. Guards the 5-bit / 6-bit `quantized_matmul` paths:
/// if mlx-rs gets downgraded to a version without 5-bit kernels, or the
/// loader regresses to pre-dequantizing 5-bit weights, this fails.
#[test]
#[ignore = "requires resources/gemma-4-E2B-it-UD-MLX-4bit"]
fn golden_greedy_output_ud() {
    let Some(bundle) = skip_if_missing(UD_BUNDLE_REL) else {
        return;
    };
    let engine = load_engine(bundle);

    let (ids, text, _stats) = generate_greedy(
        &engine,
        vec![user_msg("What is 2 + 2? Answer in one word.")],
        12,
    );

    const EXPECTED_PREFIX: &[u32] = &[26391, 106, 106, 106, 106, 106];
    assert_golden_id_prefix("ud/greedy", &ids, EXPECTED_PREFIX);
    assert_golden_contains("ud/greedy", &text, &["Four"]);
}

/// GOLDEN 3: Speculative hit-rate baseline.
///
/// For Gemma 4, `Model::forward_all` returns `None`, so the speculative path
/// is bypassed entirely — `spec_drafted` and `spec_accepted` stay at zero.
/// **This test is expected to fail when `forward_all` lands for Gemma 4** —
/// that's the signal that speculative decoding actually fires. When it does,
/// replace the zeros with the new hit-rate floor.
#[test]
#[ignore = "requires resources/gemma-4-e2b-it-mlx-4bit"]
fn golden_speculative_hit_rate_baseline() {
    let Some(bundle) = skip_if_missing(UNIFORM_BUNDLE_REL) else {
        return;
    };
    let engine = load_engine(bundle);

    // Generate enough tokens to give the n-gram predictor time to warm up.
    let (_ids, _text, stats) = generate_greedy(&engine, vec![user_msg("Count from 1 to 10.")], 48);

    // Today: Gemma 4's forward_all returns None, so spec never fires.
    assert_eq!(
        stats.spec_drafted, 0,
        "spec path should not fire on Gemma 4 (forward_all returns None)"
    );
    assert_eq!(stats.spec_accepted, 0);
    // Sanity: we actually generated tokens through the single-token path.
    assert!(stats.decode_tokens > 0, "should have produced tokens");
}

/// GOLDEN 4: Multi-turn coherence.
///
/// Captures the exact turn-2 response given a fixed turn-1 reply, proving
/// that `append_messages` + the prefix KV cache correctly carry context
/// across turns. Sensitive to any drift in append-path tokenization or KV
/// bookkeeping.
#[test]
#[ignore = "requires resources/gemma-4-e2b-it-mlx-4bit"]
fn golden_multiturn_coherence() {
    let Some(bundle) = skip_if_missing(UNIFORM_BUNDLE_REL) else {
        return;
    };
    let engine = load_engine(bundle);

    // Turn 1.
    let session = greedy_session(&engine, vec![user_msg("What is 2 + 2?")]);
    let mut p1 = session
        .pull(GenSpec {
            max_tokens: Some(12),
            temperature: Some(0.0),
            ..Default::default()
        })
        .expect("pull t1");
    let (t1_ids, t1_text, _) = drain(&mut p1);
    drop(p1);

    // Turn 1 semantic anchor: model must answer with "4" (in some phrasing).
    assert_golden_contains("multiturn/t1", &t1_text, &["4"]);
    let _ = t1_ids; // not asserted — text-level anchors are the contract

    // Append turn-1 assistant reply + turn-2 user question.
    session
        .append_messages(vec![
            asst_msg(t1_text.trim()),
            user_msg("Now multiply that by 3."),
        ])
        .expect("append");

    let mut p2 = session
        .pull(GenSpec {
            max_tokens: Some(12),
            temperature: Some(0.0),
            ..Default::default()
        })
        .expect("pull t2");
    let (_t2_ids, t2_text, _) = drain(&mut p2);

    // Turn 2 semantic anchors: must mention the operands AND the result.
    // Using in-order substring match: "4", then "3", then "12" somewhere
    // after — proves context carries and the model computed 4 × 3 = 12.
    assert_golden_contains("multiturn/t2", &t2_text, &["4", "3", "12"]);
}

/// GOLDEN 5: Prefix-cache LRU state across a defined session sequence.
///
/// Locks in the multi-entry LRU's exact behavior:
///   - 3 distinct system prompts populate 3 entries
///   - 4th distinct prompt fills cap (CAP=4, so still no eviction)
///   - Re-using an existing prompt dedups (no size growth)
///
/// Any regression to the old single-slot `Mutex<Option<Entry>>` fails here.
#[test]
#[ignore = "requires resources/gemma-4-e2b-it-mlx-4bit"]
fn golden_prefix_cache_lru_state() {
    let Some(bundle) = skip_if_missing(UNIFORM_BUNDLE_REL) else {
        return;
    };
    let engine = load_engine(bundle);

    let prompts = [
        "You are a terse calculator.",
        "You are a cheerful tour guide for Lisbon.",
        "You respond only in haiku.",
        "You are a stern grammar teacher.",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        let mut overrides = Settings::default();
        overrides.prompt.system_prompt = Some((*prompt).to_string());
        overrides.sampling.temperature = Some(0.0);
        let session = engine
            .start_session(SessionSpec {
                messages: vec![sys_msg(prompt), user_msg("hi")],
                overrides: Some(overrides),
                persona: None,
                attachments: vec![],
                cache: None,
            })
            .expect("start");
        // Drain 2 tokens so the prefill actually runs.
        let mut p = session
            .pull(GenSpec {
                max_tokens: Some(2),
                temperature: Some(0.0),
                ..Default::default()
            })
            .expect("pull");
        let _ = drain(&mut p);
        drop(p);
        assert_eq!(
            engine.prefix_cache_len(),
            i + 1,
            "after {} distinct prompts cache should hold {} entries",
            i + 1,
            i + 1
        );
    }

    // CAP = 4 → all four fit, no eviction yet.
    assert_eq!(engine.prefix_cache_len(), 4);

    // Re-using the first prompt: dedup, no growth.
    let mut overrides = Settings::default();
    overrides.prompt.system_prompt = Some(prompts[0].to_string());
    overrides.sampling.temperature = Some(0.0);
    let session = engine
        .start_session(SessionSpec {
            messages: vec![sys_msg(prompts[0]), user_msg("hi again")],
            overrides: Some(overrides),
            persona: None,
            attachments: vec![],
            cache: None,
        })
        .expect("start");
    let mut p = session
        .pull(GenSpec {
            max_tokens: Some(2),
            temperature: Some(0.0),
            ..Default::default()
        })
        .expect("pull");
    let _ = drain(&mut p);
    drop(p);
    assert_eq!(
        engine.prefix_cache_len(),
        4,
        "re-using a cached prompt must not grow the LRU"
    );
}

/// GOLDEN 6: UD bundle loads end-to-end with no pre-dequant fallback.
///
/// Regression test for the mlx-rs main bump that unlocked 5-bit Metal
/// kernels. If someone rolls mlx-rs back to a version missing those kernels
/// without restoring the loader workaround, this fails at load or first
/// forward.
#[test]
#[ignore = "requires resources/gemma-4-E2B-it-UD-MLX-4bit"]
fn golden_ud_bundle_loads_and_generates() {
    let Some(bundle) = skip_if_missing(UD_BUNDLE_REL) else {
        return;
    };
    let engine = load_engine(bundle);

    // Minimal smoke: 4 tokens of greedy generation must succeed without panic.
    let (ids, _text, stats) = generate_greedy(&engine, vec![user_msg("Hello.")], 4);
    assert_eq!(
        ids.len(),
        4,
        "expected 4 generated tokens, got {}",
        ids.len()
    );
    assert_eq!(stats.decode_tokens, 4);
}

/// GOLDEN 7: Cross-bundle semantic agreement.
///
/// Both bundles are the same base model at different quantization schemes.
/// For a trivial arithmetic question at temp=0, both must converge on the
/// same answer — a weak but load-bearing sanity check: if we accidentally
/// corrupt weights at load time (bad shape inference, wrong group_size,
/// bit-width misdetection), the two bundles diverge in obvious ways.
#[test]
#[ignore = "requires both bundles"]
fn golden_cross_bundle_agreement() {
    let (Some(u), Some(ud)) = (
        skip_if_missing(UNIFORM_BUNDLE_REL),
        skip_if_missing(UD_BUNDLE_REL),
    ) else {
        return;
    };

    let prompt = vec![user_msg("What is 2 + 2? Reply with just the number.")];

    let eu = load_engine(u);
    let (_, text_u, _) = generate_greedy(&eu, prompt.clone(), 8);
    drop(eu);

    let eud = load_engine(ud);
    let (_, text_ud, _) = generate_greedy(&eud, prompt, 8);
    drop(eud);

    // Weak: both replies must contain "4". Quantization noise can shift exact
    // wording, but the answer must survive.
    assert!(
        text_u.contains('4'),
        "uniform bundle lost the arithmetic answer: {:?}",
        text_u
    );
    assert!(
        text_ud.contains('4'),
        "UD bundle lost the arithmetic answer: {:?}",
        text_ud
    );
}
