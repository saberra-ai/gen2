//! MEASUREMENT-ONLY decode profiler for the AR Gemma-4 26B-A4B MLX model.
//!
//! Two experiments (both `#[ignore]`, both load the real ~16GB checkpoint):
//!
//! - `profile_context_scaling` — prefill to {128,1024,4096,8192} then time 30
//!   pure decode steps at each, reporting ms/token. FLAT ⇒ dispatch/compile
//!   overhead; STRONGLY GROWING ⇒ KV-concat/attention bandwidth.
//! - `profile_component_breakdown` — at context ~1024, run 30 decode steps with
//!   `PIO_MLX_PROFILE` boundaries on and print a ranked ms/token + % breakdown.
//!   Also prints the un-instrumented baseline (one eval/token at the sampler).
//!
//! Run:
//! ```bash
//! CARGO_TARGET_DIR=/tmp/pio-target cargo test -p pio-core \
//!   --no-default-features --features backend-mlx,backend-llamacpp --lib \
//!   profile_ -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Drives the `Model` directly (load → forward in a loop) so we control the
//! exact prefill length and bypass session/sampler scheduling. The only "work"
//! per decode step matches the puller hot path: `model.forward(&[tok], pos,
//! &mut cache, &rope)` then a full-vocab `as_slice` + argmax for the sampler
//! sync component.

#![cfg(test)]

use std::path::Path;
use std::time::Instant;

use mlx_rs::Array;

use super::loader::build_gemma4_model;
use super::model::{KvCache, RotaryEmbedding};

const GEMMA4_AR_DIR: &str = "/Users/victor/models/gemma-4-26b-a4b-4bit";

/// Greedy argmax over a `[1,1,vocab]` logits row, forcing the full-vocab CPU
/// sync that the real sampler pays each token (`as_slice::<f32>`).
fn sampler_argmax(logits: &Array) -> (u32, f64) {
    let t = Instant::now();
    let slice: &[f32] = logits.as_slice::<f32>();
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    (best, t.elapsed().as_secs_f64() * 1e3)
}

/// One decode step. Returns `(next_token, sampler_sync_ms)`.
/// `eval`-free except the sampler's `as_slice`, which forces the single
/// per-token eval (this is the un-instrumented baseline boundary).
fn decode_step(
    model: &super::model::Model,
    rope: &RotaryEmbedding,
    cache: &mut KvCache,
    token: u32,
    pos: usize,
) -> (u32, f64) {
    let logits = model.forward(&[token], pos, cache, rope);
    sampler_argmax(&logits)
}

/// Prefill `n_prompt` tokens (a deterministic ramp) in one forward, returning
/// the cache, the next position, and the first sampled token.
fn prefill(
    model: &super::model::Model,
    rope: &RotaryEmbedding,
    cache_slots: usize,
    n_prompt: usize,
) -> (KvCache, usize, u32) {
    let mut cache: KvCache = vec![None; cache_slots];
    // A benign in-vocab prompt: token 2 (BOS-ish) then a ramp. Content is
    // irrelevant to timing; we only need a real cache of the right length.
    let prompt: Vec<u32> = (0..n_prompt).map(|i| (i % 1000 + 10) as u32).collect();
    let logits = model.forward(&prompt, 0, &mut cache, rope);
    let (tok, _) = sampler_argmax(&logits);
    (cache, n_prompt, tok)
}

fn missing() -> bool {
    if !Path::new(GEMMA4_AR_DIR).exists() {
        eprintln!("skipping: {GEMMA4_AR_DIR} not present");
        true
    } else {
        false
    }
}

// ─── Experiment A: context-scaling curve ─────────────────────────────────────

#[test]
#[ignore = "requires the ~16GB autoregressive Gemma-4 26B-A4B checkpoint"]
fn profile_context_scaling() {
    if missing() {
        return;
    }
    let dir = Path::new(GEMMA4_AR_DIR);
    let (model, _cfg) = build_gemma4_model(dir).expect("build gemma4 model");
    let model = super::model::Model::Gemma4(model);
    // Gemma4 ignores the passed rope (uses its own internal local/global ropes).
    let rope = RotaryEmbedding::new(256, 131072, 10_000.0);
    let cache_slots = model.num_non_shared_layers();

    let contexts = [128usize, 1024, 4096, 8192];
    const STEPS: usize = 30;
    const WARMUP: usize = 3;

    eprintln!("\n=== Experiment A: context-scaling curve (AR Gemma-4 26B-A4B) ===");
    eprintln!(
        "{:>8}  {:>10}  {:>10}  {:>10}",
        "ctx_len", "ms/token", "tok/s", "sampler_ms"
    );

    for &ctx in &contexts {
        let (mut cache, mut pos, mut tok) = prefill(&model, &rope, cache_slots, ctx);

        // Warmup decode steps (don't time — first steps pay kernel JIT / alloc).
        for _ in 0..WARMUP {
            let (next, _) = decode_step(&model, &rope, &mut cache, tok, pos);
            tok = next;
            pos += 1;
        }

        let mut total_ms = 0.0f64;
        let mut sampler_ms = 0.0f64;
        for _ in 0..STEPS {
            let t = Instant::now();
            let (next, s_ms) = decode_step(&model, &rope, &mut cache, tok, pos);
            total_ms += t.elapsed().as_secs_f64() * 1e3;
            sampler_ms += s_ms;
            tok = next;
            pos += 1;
        }
        let per = total_ms / STEPS as f64;
        eprintln!(
            "{:>8}  {:>10.2}  {:>10.2}  {:>10.3}",
            ctx,
            per,
            1000.0 / per,
            sampler_ms / STEPS as f64
        );
    }
    eprintln!(
        "(FLAT across ctx ⇒ dispatch/compile overhead; GROWING ⇒ KV-concat/attention bandwidth)\n"
    );
}

// ─── Experiment B: per-component breakdown ───────────────────────────────────

#[test]
#[ignore = "requires the ~16GB autoregressive Gemma-4 26B-A4B checkpoint"]
fn profile_component_breakdown() {
    if missing() {
        return;
    }
    // Force profiling on for the instrumented window (runtime override, so we
    // can flip it off later in-process — MLX `Array` is `!Send`).
    super::model::profile::set_override(Some(true));
    assert!(
        super::model::profile::enabled(),
        "profiling must be active for the breakdown"
    );

    let dir = Path::new(GEMMA4_AR_DIR);
    let (model, _cfg) = build_gemma4_model(dir).expect("build gemma4 model");
    let model = super::model::Model::Gemma4(model);
    let rope = RotaryEmbedding::new(256, 131072, 10_000.0);
    let cache_slots = model.num_non_shared_layers();

    const CTX: usize = 1024;
    const STEPS: usize = 30;
    const WARMUP: usize = 3;

    let (mut cache, mut pos, mut tok) = prefill(&model, &rope, cache_slots, CTX);

    // Warmup with profiling on (so kernels are hot), then reset accumulators.
    for _ in 0..WARMUP {
        let (next, _) = decode_step(&model, &rope, &mut cache, tok, pos);
        tok = next;
        pos += 1;
    }
    super::model::profile::reset();

    // ── Instrumented window: per-component boundaries on. ──
    let mut instrumented_total_ms = 0.0f64;
    let mut sampler_ms_acc = 0.0f64;
    for _ in 0..STEPS {
        let t = Instant::now();
        let (next, s_ms) = decode_step(&model, &rope, &mut cache, tok, pos);
        instrumented_total_ms += t.elapsed().as_secs_f64() * 1e3;
        sampler_ms_acc += s_ms;
        tok = next;
        pos += 1;
    }

    let snap = super::model::profile::snapshot();
    // Component nanos → ms/token. Per-layer components (attn.*, moe.*, ffn.*)
    // accumulate across ALL layers per token, so dividing by STEPS already gives
    // the full-model per-token cost of that component.
    let mut rows: Vec<(&str, f64)> = snap
        .iter()
        .map(|(label, nanos, _hits)| (*label, (*nanos as f64) / 1e6 / STEPS as f64))
        .collect();
    // Add the sampler sync as a component (measured directly, not via profile).
    rows.push((
        "sampler_sync(as_slice+argmax)",
        sampler_ms_acc / STEPS as f64,
    ));

    let instrumented_per = instrumented_total_ms / STEPS as f64;
    let sum_components: f64 = rows.iter().map(|(_, ms)| ms).sum();

    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\n=== Experiment B: per-component breakdown (ctx≈{CTX}, {STEPS} steps) ===");
    eprintln!(
        "instrumented per-token total (eval barriers inflate this): {:.2} ms ({:.1} tok/s)",
        instrumented_per,
        1000.0 / instrumented_per
    );
    eprintln!("sum of timed components: {sum_components:.2} ms/token");
    eprintln!(
        "\n{:<34}  {:>10}  {:>8}",
        "component", "ms/token", "% of sum"
    );
    eprintln!("{}", "-".repeat(56));
    for (label, ms) in &rows {
        eprintln!(
            "{:<34}  {:>10.3}  {:>7.1}%",
            label,
            ms,
            100.0 * ms / sum_components
        );
    }

    // ── Un-instrumented baseline: profiling OFF, one eval/token at sampler. ──
    // Flip the runtime override off so prof_eval becomes a no-op and the lazy
    // graph pipelines normally; the only forced eval is the sampler's as_slice.
    super::model::profile::set_override(Some(false));
    assert!(
        !super::model::profile::enabled(),
        "profiling must be off for the baseline"
    );
    let baseline = {
        // Warmup (graph shape changes when barriers vanish).
        for _ in 0..WARMUP {
            let (next, _) = decode_step(&model, &rope, &mut cache, tok, pos);
            tok = next;
            pos += 1;
        }
        let t = Instant::now();
        for _ in 0..STEPS {
            let (next, _) = decode_step(&model, &rope, &mut cache, tok, pos);
            tok = next;
            pos += 1;
        }
        (t.elapsed().as_secs_f64() * 1e3) / STEPS as f64
    };

    eprintln!(
        "\nun-instrumented baseline (no eval barriers, one eval/token at sampler): \
         {:.2} ms/token = {:.1} tok/s",
        baseline,
        1000.0 / baseline
    );
    eprintln!(
        "(instrumented total is inflated by {:.2} ms/token of lost pipelining)\n",
        instrumented_per - baseline
    );
}
