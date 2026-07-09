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
//!
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
            ..Default::default()
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
                ..Default::default()
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
            ..Default::default()
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

// ─── DiffusionGemma (slice 1: load + forward) ─────────────────────────────────

/// 15GB MLX 4-bit DiffusionGemma 26B-A4B checkpoint (block-diffusion Gemma 4).
/// Absolute path — not a repo-relative bundle like the Gemma 4 E2B fixtures.
const DIFFUSION_GEMMA_DIR: &str = "/Users/victor/models/diffusiongemma-26B-A4B-it-4bit";

/// ~16GB MLX 4-bit **autoregressive** Gemma 4 26B-A4B checkpoint — same
/// backbone / quant / weight-class as `DIFFUSION_GEMMA_DIR` and ollama's
/// `gemma4:26b-mlx`. Used to benchmark gen2's AR (KV-cached) chat path so we
/// can isolate "diffusion is slow" from "gen2 is slow".
const GEMMA4_AR_DIR: &str = "/Users/victor/models/gemma-4-26b-a4b-4bit";

/// SLICE 1: DiffusionGemma loads and forwards without the embedding-shape panic.
///
/// Loads the checkpoint, runs the encoder on a short prompt, then ONE decoder
/// forward over a random 256-token canvas, and asserts the logits shape is
/// `(1, canvas_length, vocab)`. Numerical parity is NOT checked here — this
/// locks in structural correctness (it loads, maps weights, and forwards).
#[test]
#[ignore = "requires the 15GB DiffusionGemma checkpoint"]
fn diffusion_gemma_loads_and_forwards() {
    use std::path::Path;

    use mlx_rs::ops::indexing::IndexOp;

    use super::loader::build_diffusion_gemma_model;

    let dir = Path::new(DIFFUSION_GEMMA_DIR);
    if !dir.exists() {
        eprintln!("skipping: {DIFFUSION_GEMMA_DIR} not present");
        return;
    }

    let (model, _config) = build_diffusion_gemma_model(dir).expect("build diffusion_gemma model");

    let vocab = model.config.vocab_size;
    let canvas_len = model.config.canvas_length;

    // Short prompt (arbitrary token ids in range).
    let prompt: Vec<u32> = vec![2, 100, 200, 300, 400, 500, 600, 1];
    let encoder_cache = model.encode(&prompt);
    eprintln!(
        "encoder ran: {} layers cached, first KV shape {:?}",
        encoder_cache.len(),
        encoder_cache[0].0.shape()
    );

    // Random canvas of length `canvas_length` (deterministic LCG, no deps).
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let canvas: Vec<u32> = (0..canvas_len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % (vocab as u32)
        })
        .collect();

    let logits = model.decode(&canvas, &encoder_cache, None);
    let shape = logits.shape();
    eprintln!("decoder logits shape: {:?}", shape);

    assert_eq!(
        shape,
        &[1, canvas_len as i32, vocab as i32],
        "expected logits (1, {canvas_len}, {vocab}), got {shape:?}"
    );

    // Print the first few logits as a smoke signal (forces evaluation).
    let head = logits.index((0..1, 0..1, 0..8));
    let head_vals: Vec<f32> = head.as_slice::<f32>().to_vec();
    eprintln!("first 8 logits[0,0]: {:?}", head_vals);

    // Softcap bound sanity: all finite and within [-cap, cap] if softcap set.
    if let Some(cap) = model.config.final_logit_softcapping {
        for v in &head_vals {
            assert!(v.is_finite(), "non-finite logit: {v}");
            assert!(v.abs() <= cap + 1e-2, "logit {v} exceeds softcap {cap}");
        }
    }
}

/// SLICE 2: DiffusionGemma generates coherent text via the entropy-bound
/// denoising loop.
///
/// Loads the checkpoint, builds a chat-formatted prompt, runs the entropy-bound
/// denoising generation loop (random canvas → 48 denoising steps → argmax
/// canvas, EOS-trimmed), decodes the result, and prints it. The win condition
/// is **coherent generated text** — if the output is garbage/repetition that
/// signals the slice-1 forward is not numerically faithful.
#[test]
#[ignore = "requires the 15GB DiffusionGemma checkpoint"]
fn diffusion_gemma_generates_text() {
    use std::path::Path;

    use super::loader::build_diffusion_gemma_model;
    use super::model::DiffusionGenParams;
    use crate::gen2::backend::mlx::tokenizer::HfTokenizer;

    let dir = Path::new(DIFFUSION_GEMMA_DIR);
    if !dir.exists() {
        eprintln!("skipping: {DIFFUSION_GEMMA_DIR} not present");
        return;
    }

    let (model, _config) = build_diffusion_gemma_model(dir).expect("build diffusion_gemma model");
    let tok = HfTokenizer::from_dir(dir).expect("load tokenizer");

    // Generation parameters from the checkpoint generation_config.
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let params = DiffusionGenParams::from_json(&raw);
    eprintln!("gen params: {params:?}");

    // Build the chat prompt manually following the model's chat_template.jinja:
    //   <bos><|turn>user\n{content}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
    // (single user turn, add_generation_prompt, thinking disabled).
    let user = "What is the capital of France? Answer in one sentence.";
    let prompt =
        format!("<bos><|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    // add_special = false: the literal special tokens are already in the string.
    let prompt_ids = tok.encode(&prompt, false).expect("tokenize prompt");
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let out_ids = model.diffusion_generate(&prompt_ids, &params);
    eprintln!("generated {} token ids", out_ids.len());

    let text = tok.decode(&out_ids).expect("decode output");
    eprintln!("\n=== GENERATED TEXT ===\n{text}\n=== END ===\n");

    assert!(!out_ids.is_empty(), "generation produced no tokens");
    assert!(
        !text.trim().is_empty(),
        "generation decoded to empty text (ids: {out_ids:?})"
    );
}

/// SLICE 3: DiffusionGemma chats through the SAME Engine/Session/TokenPuller
/// path autoregressive models use.
///
/// `Engine::load_model` → `start_session([user(..)])` → `pull(GenSpec)` →
/// drain `TokenEvent`s with the standard `drain()` helper. This proves the
/// denoising loop is wired behind the normal model interface: callers pass
/// `messages` (not raw ids), the prompt is built by Session's chat template,
/// and the result streams out as `TokenEvent::Token` then `Eos` — exactly as
/// for AR models. Non-streaming compute, sequential emit (`PrecomputedPuller`).
#[test]
#[ignore = "requires the 15GB DiffusionGemma checkpoint"]
fn diffusion_gemma_chat_via_engine() {
    use std::path::Path;

    let dir = Path::new(DIFFUSION_GEMMA_DIR);
    if !dir.exists() {
        eprintln!("skipping: {DIFFUSION_GEMMA_DIR} not present");
        return;
    }

    // Standard load path — same as every AR golden test.
    let engine = load_engine(PathBuf::from(DIFFUSION_GEMMA_DIR));

    // Standard session start: caller passes messages, NOT raw token ids.
    let session = engine
        .start_session(SessionSpec {
            messages: vec![user_msg(
                "What is the capital of France? Answer in one sentence.",
            )],
            ..Default::default()
        })
        .expect("start_session");

    // Standard pull → drain. `max_tokens` caps the emitted canvas length.
    let mut puller = session
        .pull(GenSpec {
            max_tokens: Some(64),
            ..Default::default()
        })
        .expect("pull");
    let (ids, text, stats) = drain(&mut puller);

    eprintln!("\n=== ENGINE-PATH GENERATED TEXT ===\n{text}\n=== END ===\n");
    eprintln!(
        "ids ({}): {:?}\nstats: prompt={} decode={}",
        ids.len(),
        ids,
        stats.prompt_tokens,
        stats.decode_tokens
    );

    assert!(!ids.is_empty(), "engine path produced no tokens");
    assert!(
        text.contains("Paris"),
        "expected the answer to mention Paris, got: {text:?}"
    );
}

/// VALIDATION: 20-turn DiffusionGemma conversation through the native engine.
///
/// This is a stress / sanity pass, NOT a feature test — it produces a clear
/// transcript and lets us read whether the generation settings (48 denoising
/// steps, entropy_bound 0.1, t_min/t_max 0.4–0.8, canvas_length 256,
/// max_tokens=128) make sense in practice. It exercises context-dependent
/// turns (5/6/11/16), instruction-following (7/9/15/17), and an empty-prompt
/// edge case (19).
///
/// One Session, re-rendered each turn via `append_messages` (the same
/// multi-turn path as `golden_multiturn_coherence`). Runs ~20 min total.
#[test]
#[ignore = "requires the 15GB DiffusionGemma checkpoint; ~20 min"]
fn diffusion_gemma_twenty_turn_pass() {
    use std::path::Path;
    use std::time::Instant;

    use crate::gen2::generation::ThinkingMode;

    let dir = Path::new(DIFFUSION_GEMMA_DIR);
    if !dir.exists() {
        eprintln!("skipping: {DIFFUSION_GEMMA_DIR} not present");
        return;
    }

    // Load model + tokenizer ONCE.
    let engine = load_engine(PathBuf::from(DIFFUSION_GEMMA_DIR));

    // The curated 20-turn prompt list (per the validation spec).
    let prompts: [&str; 20] = [
        "Hi! My name is Victor and I'm building an AI app in Rust.",
        "What's the capital of Japan?",
        "What is 17 times 23?",
        "List three primary colors.",
        "What's my name?",                 // context
        "What language am I building in?", // context
        "Write a two-line haiku about the ocean.",
        "Translate 'good morning' into French.",
        "Is 91 a prime number? Answer yes or no and why.",
        "Give me a one-sentence definition of recursion.",
        "What did I say I'm building?", // context
        "Suggest a name for my app.",
        "Continue this story in one sentence: The robot opened the door and…",
        "What's heavier, a kilogram of steel or a kilogram of feathers?",
        "Reply with exactly the word: PONG",           // format
        "Summarize our conversation in one sentence.", // context
        "Count from 1 to 5.",
        "What's the opposite of 'hot'?",
        "", // edge: empty
        "Goodbye — say something friendly.",
    ];

    // IMPORTANT: `ThinkingMode::default()` is `Auto`, which maps to
    // `enable_thinking = Some(true)` (see generation/thinking.rs). With
    // thinking ON, the DiffusionGemma chat template emits a
    // `<|channel>thought ... <channel|>` scaffold, and the denoised canvas
    // decodes to a bare `thought` token instead of a clean answer (observed:
    // every turn returned just "thought"). Force thinking OFF so replies are
    // clean prose — this is the chat default the validation spec calls for.
    let session = engine
        .start_session(SessionSpec {
            messages: vec![user_msg(prompts[0])],
            thinking: ThinkingMode::Off,
            ..Default::default()
        })
        .expect("start_session");

    let mut per_turn_secs: Vec<f64> = Vec::with_capacity(20);
    let mut transcript: Vec<(usize, String, String)> = Vec::with_capacity(20);
    let run_start = Instant::now();

    for (i, &prompt) in prompts.iter().enumerate() {
        let turn = i + 1;

        // Turn 1's user message is already in the session; subsequent turns
        // append the prior assistant reply + the new user message.
        if i > 0 {
            let prev_reply = transcript[i - 1].2.trim().to_string();
            session
                .append_messages(vec![asst_msg(&prev_reply), user_msg(prompt)])
                .expect("append_messages");
        }

        let t0 = Instant::now();
        let mut puller = session
            .pull(GenSpec {
                max_tokens: Some(128),
                ..Default::default()
            })
            .expect("pull");
        let (_ids, text, _stats) = drain(&mut puller);
        drop(puller);
        let secs = t0.elapsed().as_secs_f64();
        per_turn_secs.push(secs);

        eprintln!("\n--- TURN {turn} ({secs:.1}s) ---");
        eprintln!("PROMPT: {prompt:?}");
        eprintln!("REPLY : {text}");

        transcript.push((turn, prompt.to_string(), text));
    }

    let total = run_start.elapsed().as_secs_f64();
    let avg = total / 20.0;

    // Latency drift: compare the average of the first 5 turns vs the last 5.
    let first5: f64 = per_turn_secs[..5].iter().sum::<f64>() / 5.0;
    let last5: f64 = per_turn_secs[15..].iter().sum::<f64>() / 5.0;

    eprintln!("\n==================== SUMMARY ====================");
    eprintln!("total wall time : {total:.1}s ({:.1} min)", total / 60.0);
    eprintln!("avg per turn    : {avg:.1}s");
    eprintln!("first-5 avg     : {first5:.1}s");
    eprintln!("last-5 avg      : {last5:.1}s");
    eprintln!(
        "latency drift   : {:+.1}s ({:+.0}%) across context growth",
        last5 - first5,
        if first5 > 0.0 {
            (last5 - first5) / first5 * 100.0
        } else {
            0.0
        }
    );
    eprintln!("per-turn secs   : {per_turn_secs:?}");
    eprintln!("=================================================\n");

    // This is a validation pass: the deliverable is the printed transcript +
    // timings above, read with --nocapture. The only hard assertion is that
    // the run actually produced 20 turns and didn't silently die.
    assert_eq!(transcript.len(), 20, "expected 20 completed turns");
}

/// A/B VARIANT: the same 20-turn DiffusionGemma validation pass as
/// `diffusion_gemma_twenty_turn_pass`, but run at **24 denoising steps**
/// instead of the checkpoint's default 48, to test whether we can halve the
/// denoising work (~2x chat speedup) without losing answer quality.
///
/// Identical in every other respect: same model, same 20 prompts, same
/// `ThinkingMode::Off`, same `max_tokens=128`, same multi-turn session path.
/// The step-count override is applied test-only via the
/// `PIO_MLX_DENOISING_STEPS` env var (read in `session.rs` where the denoising
/// params are assembled). It does NOT change any production default — when the
/// var is unset, generation still runs at 48 steps.
#[test]
#[ignore = "requires the 15GB DiffusionGemma checkpoint; ~10-15 min"]
fn diffusion_gemma_twenty_turn_pass_24steps() {
    use std::path::Path;
    use std::time::Instant;

    use crate::gen2::generation::ThinkingMode;

    let dir = Path::new(DIFFUSION_GEMMA_DIR);
    if !dir.exists() {
        eprintln!("skipping: {DIFFUSION_GEMMA_DIR} not present");
        return;
    }

    // Test-only override: run the denoising loop at 24 steps for THIS run.
    // `session.rs` reads this when building `DiffusionGenParams`; unset = 48.
    // SAFETY: single-threaded test (`--test-threads=1`); no other thread reads
    // the environment concurrently.
    unsafe {
        std::env::set_var("PIO_MLX_DENOISING_STEPS", "24");
    }

    // Load model + tokenizer ONCE.
    let engine = load_engine(PathBuf::from(DIFFUSION_GEMMA_DIR));

    // The curated 20-turn prompt list (identical to the 48-step pass).
    let prompts: [&str; 20] = [
        "Hi! My name is Victor and I'm building an AI app in Rust.",
        "What's the capital of Japan?",
        "What is 17 times 23?",
        "List three primary colors.",
        "What's my name?",                 // context
        "What language am I building in?", // context
        "Write a two-line haiku about the ocean.",
        "Translate 'good morning' into French.",
        "Is 91 a prime number? Answer yes or no and why.",
        "Give me a one-sentence definition of recursion.",
        "What did I say I'm building?", // context
        "Suggest a name for my app.",
        "Continue this story in one sentence: The robot opened the door and…",
        "What's heavier, a kilogram of steel or a kilogram of feathers?",
        "Reply with exactly the word: PONG",           // format
        "Summarize our conversation in one sentence.", // context
        "Count from 1 to 5.",
        "What's the opposite of 'hot'?",
        "", // edge: empty
        "Goodbye — say something friendly.",
    ];

    // Thinking OFF — same rationale as the 48-step pass (clean prose replies,
    // no `thought` scaffold). This is the chat default the validation calls for.
    let session = engine
        .start_session(SessionSpec {
            messages: vec![user_msg(prompts[0])],
            thinking: ThinkingMode::Off,
            ..Default::default()
        })
        .expect("start_session");

    let mut per_turn_secs: Vec<f64> = Vec::with_capacity(20);
    let mut transcript: Vec<(usize, String, String)> = Vec::with_capacity(20);
    let run_start = Instant::now();

    for (i, &prompt) in prompts.iter().enumerate() {
        let turn = i + 1;

        if i > 0 {
            let prev_reply = transcript[i - 1].2.trim().to_string();
            session
                .append_messages(vec![asst_msg(&prev_reply), user_msg(prompt)])
                .expect("append_messages");
        }

        let t0 = Instant::now();
        let mut puller = session
            .pull(GenSpec {
                max_tokens: Some(128),
                ..Default::default()
            })
            .expect("pull");
        let (_ids, text, _stats) = drain(&mut puller);
        drop(puller);
        let secs = t0.elapsed().as_secs_f64();
        per_turn_secs.push(secs);

        eprintln!("\n--- TURN {turn} ({secs:.1}s) ---");
        eprintln!("PROMPT: {prompt:?}");
        eprintln!("REPLY : {text}");

        transcript.push((turn, prompt.to_string(), text));
    }

    let total = run_start.elapsed().as_secs_f64();
    let avg = total / 20.0;

    let first5: f64 = per_turn_secs[..5].iter().sum::<f64>() / 5.0;
    let last5: f64 = per_turn_secs[15..].iter().sum::<f64>() / 5.0;

    eprintln!("\n================ SUMMARY (24 denoising steps) ================");
    eprintln!("total wall time : {total:.1}s ({:.1} min)", total / 60.0);
    eprintln!("avg per turn    : {avg:.1}s");
    eprintln!("first-5 avg     : {first5:.1}s");
    eprintln!("last-5 avg      : {last5:.1}s");
    eprintln!(
        "latency drift   : {:+.1}s ({:+.0}%) across context growth",
        last5 - first5,
        if first5 > 0.0 {
            (last5 - first5) / first5 * 100.0
        } else {
            0.0
        }
    );
    eprintln!("per-turn secs   : {per_turn_secs:?}");
    eprintln!("(48-step baseline: 64.9s/turn avg)");
    eprintln!("==============================================================\n");

    // Clean up the override so it cannot leak into any other test in-process.
    unsafe {
        std::env::remove_var("PIO_MLX_DENOISING_STEPS");
    }

    assert_eq!(transcript.len(), 20, "expected 20 completed turns");
}

/// BENCHMARK: 20-turn **autoregressive** Gemma 4 26B-A4B conversation through
/// gen2's KV-cached multi-turn chat path.
///
/// Mirrors `diffusion_gemma_twenty_turn_pass` (same 20 prompts, same
/// `ThinkingMode::Off`, same `max_tokens=128`, one Session carried across turns
/// via `append_messages` + the prefix KV cache — gen2's REAL fast path), but
/// runs the AR sibling of the DiffusionGemma weights. Goal: measure whether
/// gen2's *engine* is competitive with ollama (`gemma4:26b-mlx`, 0.78s/turn)
/// and quantify any gen2-AR gap, isolating it from the diffusion cost
/// (gen2-Diffusion-24step: 33.4s/turn, 667s total).
///
/// Deliverable is the printed transcript + per-turn latency table + 3-way
/// comparison, read with --nocapture. Test-only; changes no production code.
#[test]
#[ignore = "requires the ~16GB autoregressive Gemma-4 26B-A4B checkpoint"]
fn gemma4_ar_twenty_turn_pass() {
    use std::path::Path;
    use std::time::Instant;

    use crate::gen2::generation::ThinkingMode;

    let dir = Path::new(GEMMA4_AR_DIR);
    if !dir.exists() {
        eprintln!("skipping: {GEMMA4_AR_DIR} not present");
        return;
    }

    // Load model + tokenizer ONCE.
    let engine = load_engine(PathBuf::from(GEMMA4_AR_DIR));

    // The curated 20-turn prompt list — IDENTICAL to the diffusion passes.
    let prompts: [&str; 20] = [
        "Hi! My name is Victor and I'm building an AI app in Rust.",
        "What's the capital of Japan?",
        "What is 17 times 23?",
        "List three primary colors.",
        "What's my name?",                 // context
        "What language am I building in?", // context
        "Write a two-line haiku about the ocean.",
        "Translate 'good morning' into French.",
        "Is 91 a prime number? Answer yes or no and why.",
        "Give me a one-sentence definition of recursion.",
        "What did I say I'm building?", // context
        "Suggest a name for my app.",
        "Continue this story in one sentence: The robot opened the door and…",
        "What's heavier, a kilogram of steel or a kilogram of feathers?",
        "Reply with exactly the word: PONG",           // format
        "Summarize our conversation in one sentence.", // context
        "Count from 1 to 5.",
        "What's the opposite of 'hot'?",
        "", // edge: empty
        "Goodbye — say something friendly.",
    ];

    // Greedy (temp=0) for reproducibility; thinking OFF for clean prose replies
    // (mirrors the diffusion passes' chat default).
    let mut overrides = Settings::default();
    overrides.sampling.temperature = Some(0.0);
    let session = engine
        .start_session(SessionSpec {
            messages: vec![user_msg(prompts[0])],
            overrides: Some(overrides),
            thinking: ThinkingMode::Off,
            ..Default::default()
        })
        .expect("start_session");

    // Defensive thought-scaffold stripper for display. With ThinkingMode::Off
    // replies are already clean prose, but if a `<think>…</think>` or Gemma-4
    // `<|channel>thought…<channel|>` scaffold ever leaks through, drop it so the
    // transcript reads as the actual answer.
    fn strip_scaffold(raw: &str) -> String {
        let mut s = raw.to_string();
        for (open, close) in [("<think>", "</think>"), ("<|channel>thought", "<channel|>")] {
            if let Some(start) = s.find(open) {
                if let Some(end_rel) = s[start..].find(close) {
                    let end = start + end_rel + close.len();
                    s.replace_range(start..end, "");
                } else {
                    // Unclosed opener: drop from the opener onward.
                    s.truncate(start);
                }
            }
        }
        s.trim().to_string()
    }

    let mut per_turn_secs: Vec<f64> = Vec::with_capacity(20);
    let mut per_turn_toks: Vec<usize> = Vec::with_capacity(20);
    let mut transcript: Vec<(usize, String, String)> = Vec::with_capacity(20);
    let run_start = Instant::now();

    for (i, &prompt) in prompts.iter().enumerate() {
        let turn = i + 1;

        // Turn 1's user message is already in the session; subsequent turns
        // append the prior assistant reply + the new user message (KV-cached
        // context carry — the same multi-turn path as golden_multiturn_coherence).
        if i > 0 {
            let prev_reply = transcript[i - 1].2.trim().to_string();
            session
                .append_messages(vec![asst_msg(&prev_reply), user_msg(prompt)])
                .expect("append_messages");
        }

        let t0 = Instant::now();
        let mut puller = session
            .pull(GenSpec {
                max_tokens: Some(128),
                temperature: Some(0.0),
                ..Default::default()
            })
            .expect("pull");
        let (ids, text, _stats) = drain(&mut puller);
        drop(puller);
        let secs = t0.elapsed().as_secs_f64();
        per_turn_secs.push(secs);
        per_turn_toks.push(ids.len());

        // Strip any thought scaffold for display, then truncate to ~70 chars.
        let clean = strip_scaffold(text.trim());
        let display: String = clean.chars().take(70).collect();
        let ellipsis = if clean.chars().count() > 70 {
            "…"
        } else {
            ""
        };
        eprintln!(
            "TURN {turn:>2}  {secs:>6.2}s  {:>4} tok  | {display}{ellipsis}",
            ids.len()
        );

        transcript.push((turn, prompt.to_string(), clean));
    }

    let total = run_start.elapsed().as_secs_f64();
    let avg = total / 20.0;
    let warmup = per_turn_secs[0];
    let steady_avg: f64 = per_turn_secs[1..].iter().sum::<f64>() / 19.0;
    let total_toks: usize = per_turn_toks.iter().sum();
    let gen_secs: f64 = per_turn_secs.iter().sum();
    let tok_per_s = if gen_secs > 0.0 {
        total_toks as f64 / gen_secs
    } else {
        0.0
    };

    eprintln!("\n=============== FULL TRANSCRIPT (verbatim) ===============");
    for (turn, prompt, reply) in &transcript {
        eprintln!("\n--- TURN {turn} ---");
        eprintln!("PROMPT: {prompt:?}");
        eprintln!("REPLY : {reply}");
    }

    eprintln!("\n=============== SUMMARY (gen2 AR Gemma-4 26B-A4B) ===============");
    eprintln!("total wall time : {total:.2}s ({:.2} min)", total / 60.0);
    eprintln!("avg per turn    : {avg:.2}s  (all 20 turns)");
    eprintln!("turn-1 (warmup) : {warmup:.2}s");
    eprintln!("steady avg      : {steady_avg:.2}s  (turns 2-20)");
    eprintln!("total tokens    : {total_toks}");
    eprintln!("overall tok/s   : {tok_per_s:.1}");
    eprintln!("per-turn secs   : {per_turn_secs:?}");
    eprintln!("per-turn toks   : {per_turn_toks:?}");
    eprintln!("\n--------------- 3-WAY COMPARISON (avg/turn) ---------------");
    eprintln!("gen2-AR (this)         : {avg:.2}s/turn ({total:.1}s total)");
    eprintln!("ollama gemma4:26b-mlx  : 0.78s/turn (15.6s total)");
    eprintln!("gen2-Diffusion 24-step : 33.40s/turn (667s total)");
    eprintln!("================================================================\n");

    // Benchmark pass: the deliverable is the printed transcript + timings above.
    // The only hard assertion is that all 20 turns completed.
    assert_eq!(transcript.len(), 20, "expected 20 completed turns");
}

/// FAST PATH (`PIO_MLX_FAST=1`) — Stage A: bf16 activations + fused SDPA +
/// step-buffer KV cache, SERIAL decode. Runs the SAME 20-turn conversation as
/// `gemma4_ar_twenty_turn_pass` with the fast path enabled, asserts the answers
/// stay coherent + correct (391, Tokyo, Bonjour, 91 not prime, context on turns
/// 5/6/11/16, PONG), prints the transcript + tok/s + speedup, and checks
/// determinism (two fast-path runs are byte-identical).
///
/// The flag is read once at model construction (`Gemma4Model::fast`); the
/// default path is untouched when it's unset.
#[test]
#[ignore = "requires the ~16GB autoregressive Gemma-4 26B-A4B checkpoint (PIO_MLX_FAST)"]
fn gemma4_fast_twenty_turn_pass() {
    use std::path::Path;
    use std::time::Instant;

    use crate::gen2::generation::ThinkingMode;

    let dir = Path::new(GEMMA4_AR_DIR);
    if !dir.exists() {
        eprintln!("skipping: {GEMMA4_AR_DIR} not present");
        return;
    }

    // Enable the fast path BEFORE loading (flag read at model construction).
    // SAFETY: test is single-threaded (--test-threads=1).
    unsafe {
        std::env::set_var("PIO_MLX_FAST", "1");
    }

    let prompts: [&str; 20] = [
        "Hi! My name is Victor and I'm building an AI app in Rust.",
        "What's the capital of Japan?",
        "What is 17 times 23?",
        "List three primary colors.",
        "What's my name?",                 // context
        "What language am I building in?", // context
        "Write a two-line haiku about the ocean.",
        "Translate 'good morning' into French.",
        "Is 91 a prime number? Answer yes or no and why.",
        "Give me a one-sentence definition of recursion.",
        "What did I say I'm building?", // context
        "Suggest a name for my app.",
        "Continue this story in one sentence: The robot opened the door and…",
        "What's heavier, a kilogram of steel or a kilogram of feathers?",
        "Reply with exactly the word: PONG",           // format
        "Summarize our conversation in one sentence.", // context
        "Count from 1 to 5.",
        "What's the opposite of 'hot'?",
        "", // edge: empty
        "Goodbye — say something friendly.",
    ];

    // Inner closure: run the full 20-turn conversation once, return the
    // (per-turn ids, per-turn clean text, per-turn secs, per-turn toks).
    #[allow(clippy::type_complexity)] // one-off test harness return bundle
    fn run_once(prompts: &[&str; 20]) -> (Vec<Vec<u32>>, Vec<String>, Vec<f64>, Vec<usize>, f64) {
        let engine = load_engine(PathBuf::from(GEMMA4_AR_DIR));
        let mut overrides = Settings::default();
        overrides.sampling.temperature = Some(0.0);
        let session = engine
            .start_session(SessionSpec {
                messages: vec![user_msg(prompts[0])],
                overrides: Some(overrides),
                thinking: ThinkingMode::Off,
                ..Default::default()
            })
            .expect("start_session");

        fn strip_scaffold(raw: &str) -> String {
            let mut s = raw.to_string();
            for (open, close) in [("<think>", "</think>"), ("<|channel>thought", "<channel|>")] {
                if let Some(start) = s.find(open) {
                    if let Some(end_rel) = s[start..].find(close) {
                        let end = start + end_rel + close.len();
                        s.replace_range(start..end, "");
                    } else {
                        s.truncate(start);
                    }
                }
            }
            s.trim().to_string()
        }

        let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(20);
        let mut all_text: Vec<String> = Vec::with_capacity(20);
        let mut secs: Vec<f64> = Vec::with_capacity(20);
        let mut toks: Vec<usize> = Vec::with_capacity(20);
        let run_start = Instant::now();

        for (i, &prompt) in prompts.iter().enumerate() {
            if i > 0 {
                let prev_reply = all_text[i - 1].trim().to_string();
                session
                    .append_messages(vec![asst_msg(&prev_reply), user_msg(prompt)])
                    .expect("append_messages");
            }
            let t0 = Instant::now();
            let mut puller = session
                .pull(GenSpec {
                    max_tokens: Some(128),
                    temperature: Some(0.0),
                    ..Default::default()
                })
                .expect("pull");
            let (ids, text, _stats) = drain(&mut puller);
            drop(puller);
            secs.push(t0.elapsed().as_secs_f64());
            toks.push(ids.len());
            all_ids.push(ids);
            all_text.push(strip_scaffold(text.trim()));
        }
        let total = run_start.elapsed().as_secs_f64();
        (all_ids, all_text, secs, toks, total)
    }

    // ── Run 1 (timed) ────────────────────────────────────────────────────────
    let (ids1, text1, per_turn_secs, per_turn_toks, total) = run_once(&prompts);

    for (i, (secs, toks)) in per_turn_secs.iter().zip(per_turn_toks.iter()).enumerate() {
        let display: String = text1[i].chars().take(70).collect();
        let ellipsis = if text1[i].chars().count() > 70 {
            "…"
        } else {
            ""
        };
        eprintln!(
            "TURN {:>2}  {secs:>6.2}s  {toks:>4} tok  | {display}{ellipsis}",
            i + 1
        );
    }

    let avg = total / 20.0;
    let warmup = per_turn_secs[0];
    let steady_avg: f64 = per_turn_secs[1..].iter().sum::<f64>() / 19.0;
    let total_toks: usize = per_turn_toks.iter().sum();
    let gen_secs: f64 = per_turn_secs.iter().sum();
    let tok_per_s = if gen_secs > 0.0 {
        total_toks as f64 / gen_secs
    } else {
        0.0
    };

    eprintln!("\n=============== FULL TRANSCRIPT (FAST PATH, verbatim) ===============");
    for (i, prompt) in prompts.iter().enumerate() {
        eprintln!("\n--- TURN {} ---", i + 1);
        eprintln!("PROMPT: {prompt:?}");
        eprintln!("REPLY : {}", text1[i]);
    }

    eprintln!("\n=============== SUMMARY (gen2 AR FAST Gemma-4 26B-A4B) ===============");
    eprintln!("total wall time : {total:.2}s ({:.2} min)", total / 60.0);
    eprintln!("avg per turn    : {avg:.2}s  (all 20 turns)");
    eprintln!("turn-1 (warmup) : {warmup:.2}s");
    eprintln!("steady avg      : {steady_avg:.2}s  (turns 2-20)");
    eprintln!("total tokens    : {total_toks}");
    eprintln!("overall tok/s   : {tok_per_s:.1}");
    eprintln!("per-turn secs   : {per_turn_secs:?}");
    eprintln!("per-turn toks   : {per_turn_toks:?}");
    eprintln!("\n--------------- SPEEDUP (avg/turn) ---------------");
    eprintln!("gen2-AR FAST (this)    : {avg:.2}s/turn ({total:.1}s total, {tok_per_s:.1} tok/s)");
    eprintln!("gen2-AR DEFAULT (base) : 2.15s/turn (43.0s total, 14.9 tok/s)  [same machine]");
    eprintln!("=====================================================================\n");

    // ── Coherence / correctness assertions ──────────────────────────────────
    let lc: Vec<String> = text1.iter().map(|t| t.to_lowercase()).collect();
    assert!(
        lc[1].contains("tokyo"),
        "turn2 should say Tokyo: {:?}",
        text1[1]
    );
    assert!(
        text1[2].contains("391"),
        "turn3 should compute 391: {:?}",
        text1[2]
    );
    assert!(
        lc[4].contains("victor"),
        "turn5 should recall name Victor: {:?}",
        text1[4]
    );
    assert!(
        lc[5].contains("rust"),
        "turn6 should recall Rust: {:?}",
        text1[5]
    );
    assert!(
        lc[7].contains("bonjour"),
        "turn8 should translate to Bonjour: {:?}",
        text1[7]
    );
    assert!(
        lc[8].contains("not")
            && (lc[8].contains("7") || lc[8].contains("13") || lc[8].contains("prime")),
        "turn9 should say 91 is not prime (7×13): {:?}",
        text1[8]
    );
    assert!(
        lc[10].contains("rust") || lc[10].contains("app"),
        "turn11 should recall building app/Rust: {:?}",
        text1[10]
    );
    assert!(
        text1[14].to_uppercase().contains("PONG"),
        "turn15 should reply PONG: {:?}",
        text1[14]
    );
    assert!(
        lc[15].contains("tokyo")
            || lc[15].contains("rust")
            || lc[15].contains("victor")
            || lc[15].contains("recursion"),
        "turn16 summary should retain prior context: {:?}",
        text1[15]
    );

    // ── Determinism: a second fast-path run must be byte-identical ──────────
    let (ids2, _text2, _s2, _t2, _tot2) = run_once(&prompts);
    for (i, (a, b)) in ids1.iter().zip(ids2.iter()).enumerate() {
        assert_eq!(
            a,
            b,
            "determinism: fast-path run 2 diverged at turn {} from run 1",
            i + 1
        );
    }
    eprintln!("DETERMINISM: two fast-path runs produced byte-identical token ids ✓");

    unsafe {
        std::env::remove_var("PIO_MLX_FAST");
    }
}

/// DIAGNOSTIC ABLATION (not a pass/fail gate): fixed-workload fast-path decode
/// for clean critical-path timing. Reads `PIO_MLX_ABLATE` (set by the caller)
/// once at model construction; correctness is intentionally irrelevant — we ONLY
/// measure ms/token. Run once per ablation (baseline / moe / attn / lmhead) and
/// diff the ms/token to localize each component's real (overlap-accounted) cost.
///
/// Workload: one warmup turn (excluded), then a single steady-state turn that
/// generates a FIXED `STEADY_TOKENS` tokens via greedy decode. Wall-time of the
/// steady turn / tokens = ms/token. `max_tokens` forces the exact count even when
/// the ablated model emits garbage (no EOS reliance).
#[test]
#[ignore = "diagnostic: requires the ~16GB Gemma-4 26B-A4B checkpoint (PIO_MLX_FAST + PIO_MLX_ABLATE)"]
fn gemma4_fast_ablate_decode() {
    use std::path::Path;
    use std::time::Instant;

    use crate::gen2::generation::ThinkingMode;

    if !Path::new(GEMMA4_AR_DIR).exists() {
        eprintln!("skipping: {GEMMA4_AR_DIR} not present");
        return;
    }

    // Fixed steady-state token budget. Large enough to dominate the per-turn
    // prefill + dispatch overhead, small enough to keep each run a few minutes.
    const STEADY_TOKENS: usize = 200;
    const WARMUP_TOKENS: usize = 32;

    // Fast path is the subject under test. Flag read at model construction.
    // SAFETY: single-threaded (--test-threads=1).
    unsafe {
        std::env::set_var("PIO_MLX_FAST", "1");
        // FIXED workload: decode EXACTLY `STEADY_TOKENS` forwards, ignoring
        // EOS / loop-detector stops so EVERY ablation runs the same workload
        // (ablated models emit garbage that would otherwise trip early stop,
        // making ms/token incomparable). The gate value is the step COUNT (now
        // honoured by the engine — see `ArPuller::fixed_steps`), NOT a boolean.
        std::env::set_var("PIO_MLX_FIXED_STEPS", STEADY_TOKENS.to_string());
        // Single-token decode (no speculative batching) so #emitted tokens ==
        // #fast forwards — clean ms/token attribution to the fast forward.
        std::env::set_var("PIO_MLX_SPEC", "0");
    }
    let ablate = std::env::var("PIO_MLX_ABLATE").unwrap_or_else(|_| "(none/baseline)".into());

    let engine = load_engine(PathBuf::from(GEMMA4_AR_DIR));
    let mut overrides = Settings::default();
    overrides.sampling.temperature = Some(0.0);
    let session = engine
        .start_session(SessionSpec {
            messages: vec![user_msg(
                "Write a long detailed essay about the history of computing.",
            )],
            overrides: Some(overrides),
            thinking: ThinkingMode::Off,
            ..Default::default()
        })
        .expect("start_session");

    // ── Warmup turn (excluded from timing): primes Metal kernels + KV cache ──
    {
        let mut p = session
            .pull(GenSpec {
                max_tokens: Some(WARMUP_TOKENS),
                temperature: Some(0.0),
                ..Default::default()
            })
            .expect("pull warmup");
        let _ = drain(&mut p);
    }

    // ── Steady-state timed turn: FIXED token count, greedy ───────────────────
    session
        .append_messages(vec![
            asst_msg("(warmup)"),
            user_msg("Continue the essay with many more concrete details and examples."),
        ])
        .expect("append");

    let t0 = Instant::now();
    let (ids, _text, stats) = {
        let mut p = session
            .pull(GenSpec {
                max_tokens: Some(STEADY_TOKENS),
                temperature: Some(0.0),
                ..Default::default()
            })
            .expect("pull steady");
        drain(&mut p)
    };
    let secs = t0.elapsed().as_secs_f64();
    // Count decode FORWARDS (stats.decode_tokens), not emitted Token EVENTS:
    // the lmhead ablation argmaxes all-zeros → token 0, whose decoded text may
    // be empty (no Token event), but the forward still ran. `decode_tokens` is
    // bumped once per forward step regardless of emitted text, so it is the
    // true denominator for ms/forward. Under PIO_MLX_FIXED_STEPS it equals
    // STEADY_TOKENS exactly.
    let forwards = stats.decode_tokens as usize;
    let emitted = ids.len();
    let ms_per_tok = if forwards > 0 {
        secs * 1000.0 / forwards as f64
    } else {
        0.0
    };
    let tok_per_s = if secs > 0.0 {
        forwards as f64 / secs
    } else {
        0.0
    };

    eprintln!("\n=============== ABLATION DECODE RESULT ===============");
    eprintln!("PIO_MLX_ABLATE  : {ablate}");
    eprintln!("decode forwards : {forwards} (requested {STEADY_TOKENS}, emitted {emitted})");
    eprintln!("steady wall     : {secs:.3}s");
    eprintln!("ms / token      : {ms_per_tok:.3}");
    eprintln!("tok / s         : {tok_per_s:.2}");
    eprintln!("=====================================================\n");

    unsafe {
        std::env::remove_var("PIO_MLX_FAST");
        std::env::remove_var("PIO_MLX_FIXED_STEPS");
        std::env::remove_var("PIO_MLX_SPEC");
    }

    assert!(forwards > 0, "steady turn produced no decode forwards");
    // GATE TRUSTWORTHINESS (STEP 1): with `PIO_MLX_FIXED_STEPS` now honoured by
    // the engine, the steady turn must run EXACTLY `STEADY_TOKENS` forwards —
    // EOS / loop early-termination is suppressed. A mismatch means the gate is
    // broken and every ms/token number below is untrustworthy (the original
    // ablation-confounding bug).
    assert_eq!(
        forwards, STEADY_TOKENS,
        "PIO_MLX_FIXED_STEPS gate broken: decoded {forwards} forwards, expected exactly {STEADY_TOKENS}"
    );
}

/// DIAGNOSTIC (not a pass/fail gate): run the first 6 turns with the fast path
/// OFF then ON, printing both transcripts so we can see exactly which turns the
/// fast path diverges on vs the (known-good) default path. Helps localise a
/// fast-path numerics/cache bug without re-running the full 20-turn gate.
#[test]
#[ignore = "diagnostic: requires the ~16GB Gemma-4 26B-A4B checkpoint"]
fn gemma4_fast_vs_default_diag() {
    use std::path::Path;

    use crate::gen2::generation::ThinkingMode;

    if !Path::new(GEMMA4_AR_DIR).exists() {
        eprintln!("skipping: {GEMMA4_AR_DIR} not present");
        return;
    }

    let prompts: [&str; 6] = [
        "Hi! My name is Victor and I'm building an AI app in Rust.",
        "What's the capital of Japan?",
        "What is 17 times 23?",
        "List three primary colors.",
        "What's my name?",
        "What language am I building in?",
    ];

    fn run(prompts: &[&str; 6]) -> Vec<String> {
        let engine = load_engine(PathBuf::from(GEMMA4_AR_DIR));
        let mut overrides = Settings::default();
        overrides.sampling.temperature = Some(0.0);
        let session = engine
            .start_session(SessionSpec {
                messages: vec![user_msg(prompts[0])],
                overrides: Some(overrides),
                thinking: ThinkingMode::Off,
                ..Default::default()
            })
            .expect("start_session");
        let mut out: Vec<String> = Vec::new();
        for (i, &prompt) in prompts.iter().enumerate() {
            if i > 0 {
                let prev = out[i - 1].clone();
                session
                    .append_messages(vec![asst_msg(&prev), user_msg(prompt)])
                    .expect("append");
            }
            let mut p = session
                .pull(GenSpec {
                    max_tokens: Some(64),
                    temperature: Some(0.0),
                    ..Default::default()
                })
                .expect("pull");
            let (_ids, text, _s) = drain(&mut p);
            drop(p);
            out.push(text.trim().to_string());
        }
        out
    }

    unsafe {
        std::env::remove_var("PIO_MLX_FAST");
    }
    let def = run(&prompts);
    unsafe {
        std::env::set_var("PIO_MLX_FAST", "1");
    }
    let fast = run(&prompts);
    unsafe {
        std::env::remove_var("PIO_MLX_FAST");
    }

    eprintln!("\n=============== DEFAULT vs FAST (first 6 turns) ===============");
    for i in 0..6 {
        eprintln!("\n--- TURN {} : {:?} ---", i + 1, prompts[i]);
        eprintln!("DEFAULT: {}", def[i].chars().take(120).collect::<String>());
        eprintln!("FAST   : {}", fast[i].chars().take(120).collect::<String>());
    }
    eprintln!("================================================================\n");
}
