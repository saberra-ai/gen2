//! Grammar-constrained decode for the mlxcel backend (roadmap S4).
//!
//! ## Why a separate loop (the fast-path tradeoff, stated plainly)
//! mlxcel's fast path is [`mlxcel::MlxInferenceSession::generate_streaming`],
//! which wraps `CxxGenerator::generate_streaming`. That loop:
//!   1. carries a **fixed** `SamplingConfig` and samples on the GPU with the
//!      fused sampler — it exposes **no per-step logit hook** to a caller; and
//!   2. is **pipelined** — at iteration `n` it pre-samples the token at `n+1`
//!      *before* the `on_token` callback for token `n` fires
//!      (`generate.rs:1287` samples `next_y`; `:1415` calls `on_token`).
//!
//! Grammar masking needs the opposite ordering: know token `n` **before**
//! sampling token `n+1`, so the matcher can be advanced and the next mask
//! computed. mlxcel's server does this only inside its `BatchScheduler`
//! (`compute_mask` → sample → `consume_token`, `scheduler.rs:3885/3934`), which
//! is not reachable through the embeddable single-session API. So a
//! grammar-constrained generation **cannot** ride `generate_streaming` — it
//! takes this manual per-step loop instead. Greedy/text generations keep the
//! fast path (see [`super::worker`]); only `spec.grammar.is_some()` diverts here.
//!
//! ## Mechanism (mirrors mlxcel's own on-device mask)
//! We mirror mlxcel's `apply_structured_mask_to_logits` (server/structured.rs:722):
//! build a `[1, vocab]` bias array that is `0.0` for grammar-allowed token ids
//! and `-inf` for disallowed ones, then `mlxcel_core::add` it onto the logits
//! **on the GPU** — the mask stays on-device, exactly as mlxcel's server does.
//! The allowed-set itself is computed by **pio-core's own** [`GrammarMatcher`]
//! (`backend/common/grammar.rs`, our canonical llguidance engine with the
//! toktrie cache) so all four `GrammarSpec` variants (JSON-schema / JSON-object
//! / regex / Lark) work, not just JSON-schema. Token ids align between
//! pio-core's tokenizer (`tokenizers 0.21`) and mlxcel's model tokenizer
//! (`0.22`) because both are built from the same `tokenizer.json` vocab.
//!
//! ## Decode primitives (the manual loop)
//! Mirrors what `generate_streaming` does internally, minus the pipeline:
//!   - prefill: `model.forward_last_logits(input, &mut caches, None, last_pos)`
//!     (generate.rs:1215/1230);
//!   - per step: `model.forward(reshaped_token, &mut caches, None)`
//!     (generate.rs:1302);
//!   - mask: pio-core `GrammarMatcher::apply_mask` computes the allowed set;
//!     we lift it to an on-device `-inf` bias and `add`;
//!   - sample: greedy argmax over the masked logits (deterministic; grammar
//!     tool-calls want determinism);
//!   - advance: `GrammarMatcher::observe(token)` after each accepted token.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;

use mlxcel::LoadedModel;
use mlxcel::tokenizer::MlxcelTokenizer;
use mlxcel_core::generate::LanguageModel;

use gen2::advanced::plugin::{ExecError, GrammarMatcher, GrammarSpec, HfTokenizer};

use super::worker::DecodedToken;

/// Run one grammar-constrained streaming generation on the loaded model.
///
/// Drives the manual per-step masked loop described in the module docs. Pushes
/// each decoded `(id, text)` onto `tokens_tx`. Returns when: EOS, `max_tokens`
/// reached, the grammar reaches a stopped/accepting terminal state, the stop
/// flag flips, or the consumer hangs up.
///
/// `model` / `tokenizer` are borrowed from the worker's `LoadedState` (both
/// `!Send`, thread-confined). `hf_tok` is pio-core's own tokenizer for the same
/// model dir — it drives the grammar matcher (token ids align with the model).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_grammar_generation(
    model: &LoadedModel,
    tokenizer: &MlxcelTokenizer,
    hf_tok: &HfTokenizer,
    spec: GrammarSpec,
    prompt_ids_i32: &[i32],
    max_tokens: usize,
    eos_token_ids: &[i32],
    stop: &Arc<AtomicBool>,
    tokens_tx: &SyncSender<DecodedToken>,
) -> Result<(), ExecError> {
    // Build the grammar matcher over pio-core's tokenizer (the toktrie is
    // cached per vocab — 932d4ca5 — so back-to-back grammar gens are cheap).
    let mut matcher = GrammarMatcher::new(hf_tok, spec)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("build grammar matcher: {e}")))?;

    // Fresh KV caches for this generation (mirrors generate_streaming's
    // reset_with_model → model.make_caches()).
    let mut caches = model.make_caches();

    // Prefill: one forward over the whole prompt, taking the last position's
    // logits (mirrors generate.rs:1229-1230, the no-align branch — correctness
    // over the tile-alignment perf trick, which is a fast-path-only detail).
    let prompt_len = prompt_ids_i32.len();
    let input = mlxcel_core::from_slice_i32(prompt_ids_i32, &[1, prompt_len as i32]);
    let mut logits =
        model.forward_last_logits(&input, &mut caches, None, prompt_len.saturating_sub(1));

    let vocab = vocab_from_logits(&logits);

    let mut n = 0usize;
    loop {
        if n >= max_tokens {
            break;
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // ---- mask → sample (the grammar critical section) ----
        // 1. Compute the grammar-allowed bias (0.0 allowed, -inf disallowed)
        //    from pio-core's matcher, then apply it ON-DEVICE (mirrors
        //    mlxcel apply_structured_mask_to_logits:760-774).
        let bias = match grammar_bias(&mut matcher, vocab) {
            Ok(b) => b,
            Err(e) => {
                // A stuck matcher (empty mask) is a real, surfaceable error:
                // the schema is unsatisfiable from here. Fail loud rather than
                // emit an off-grammar token.
                return Err(e);
            }
        };
        let bias_arr = mlxcel_core::from_slice_f32(&bias, &[1, vocab as i32]);
        let masked = mlxcel_core::add(&logits, &bias_arr);

        // 2. Greedy argmax over the masked logits on host. Deterministic —
        //    exactly what grammar-constrained tool-calling wants.
        let token_id = argmax_host(&masked, vocab);

        // EOS check *before* emit (matches generate_streaming:1397).
        if eos_token_ids.contains(&token_id) {
            break;
        }

        // 3. Advance the matcher with the accepted token.
        if let Err(e) = matcher.observe(token_id as u32) {
            return Err(ExecError::Other(anyhow::anyhow!(
                "grammar observe token {token_id}: {e}"
            )));
        }

        // Decode id→text and push to the puller (skip_special=true).
        let text = tokenizer
            .decode(&[token_id as u32], true)
            .unwrap_or_default();
        if tokens_tx
            .send(DecodedToken {
                id: token_id as u32,
                text,
            })
            .is_err()
        {
            // Consumer hung up.
            break;
        }

        // Grammar reached a terminal accepting state (JSON object complete):
        // stop cleanly. `compute_mask_or_eos` flips the EOS bit when done, but
        // stopping here avoids one wasted forward when the matcher is stopped.
        if matcher.is_stopped() {
            break;
        }

        // ---- advance one step: forward the just-sampled token ----
        // `[1, 1]` is already the shape `forward` expects for a single decode
        // token (generate_streaming reshapes its GPU-sampled token array to the
        // same shape via `reshape_token_for_forward`; we build it directly).
        let next_input = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        logits = model.forward(&next_input, &mut caches, None);

        n += 1;
    }

    Ok(())
}

/// Vocab size from a `[.., vocab]` logits array (last axis).
fn vocab_from_logits(logits: &mlxcel_core::MlxArray) -> usize {
    let shape = mlxcel_core::array_shape(logits);
    shape.last().copied().unwrap_or(0) as usize
}

/// Compute the grammar-allowed additive bias: `0.0` for allowed token ids,
/// `f32::NEG_INFINITY` for disallowed. Mirrors mlxcel's
/// `apply_structured_mask_to_logits` bias construction (server/structured.rs).
///
/// We reuse pio-core's [`GrammarMatcher::apply_mask`], which sets disallowed
/// entries of a logits slice to `-inf`. We hand it an all-`0.0` scratch buffer,
/// so on return the buffer *is* the additive bias.
fn grammar_bias(matcher: &mut GrammarMatcher, vocab: usize) -> Result<Vec<f32>, ExecError> {
    let mut bias = vec![0.0f32; vocab];
    matcher
        .apply_mask(&mut bias)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("grammar compute mask: {e}")))?;
    // Guard: an all-`-inf` bias means no token can extend the partial output —
    // the matcher is stuck. Surface it (mirrors mlxcel's empty-mask error at
    // structured.rs:741) instead of sampling an arbitrary token.
    if bias.iter().all(|b| b.is_infinite() && b.is_sign_negative()) {
        return Err(ExecError::Other(anyhow::anyhow!(
            "grammar mask allows no token — schema unsatisfiable from current state"
        )));
    }
    Ok(bias)
}

/// Greedy argmax over a `[1, vocab]` (or `[1, 1, vocab]`) masked-logits array,
/// on host. Reads the last-axis slice to `f32` and takes the max index. `-inf`
/// entries (masked-out tokens) never win.
fn argmax_host(logits: &mlxcel_core::MlxArray, vocab: usize) -> i32 {
    let last = mlxcel_core::slice_last_logits(logits);
    let f = mlxcel_core::astype(&last, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    let bytes = mlxcel_core::array_to_raw_bytes(&f);
    let mut best_id: i32 = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, chunk) in bytes.chunks_exact(4).enumerate().take(vocab) {
        let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if v > best_val {
            best_val = v;
            best_id = i as i32;
        }
    }
    best_id
}
