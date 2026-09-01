//! Model-fit contracts, over synthetic inputs only.
//!
//! Every machine here is constructed, never detected, and every model is a
//! GGUF header written into a temp file — so the verdicts assert the crate's
//! sizing arithmetic rather than whatever RAM the runner happens to have.
//!
//! The invariants are monotonicity ones: they say how a verdict must *move*
//! when one input moves, which is what callers actually rely on ("buy more
//! RAM and it will still load", "ask for less context and it can only get
//! better"). Point assertions about a specific byte count would pin the
//! current heuristic instead of the contract.

use std::io::{Seek, Write};
use std::path::{Path, PathBuf};

use gen2::api::{FitVerdict, ModelInfo};
use gen2::{GpuBackend, HardwareProfile};

// ── Synthetic inputs ────────────────────────────────────────────────────────

const GIB: u64 = 1024 * 1024 * 1024;

fn machine(ram_gib: u64) -> HardwareProfile {
    HardwareProfile {
        total_ram_bytes: ram_gib * GIB,
        cpu_cores: 8,
        gpu_backend: GpuBackend::Metal,
        vram_bytes: 0,
    }
}

/// Architecture fields of a 7B-class GGUF, as `(key, value)` u32 pairs.
/// `block_count` × `head_count_kv` × (`embedding_length` / `head_count`) is
/// what the KV-per-token cost is derived from, so these four decide how
/// expensive a token of context is.
fn arch_dims(arch: &str) -> Vec<(String, u32)> {
    vec![
        (format!("{arch}.context_length"), 32_768),
        (format!("{arch}.embedding_length"), 4_096),
        (format!("{arch}.block_count"), 32),
        (format!("{arch}.attention.head_count"), 32),
        (format!("{arch}.attention.head_count_kv"), 8),
    ]
}

/// Write a GGUF file whose header declares `arch` + `dims` and whose length on
/// disk is `file_bytes`.
///
/// The body is a hole: the fit path reads the header and `len()`, never the
/// weights, so a sparse file models a multi-gigabyte model without writing
/// multiple gigabytes.
fn write_gguf(
    dir: &Path,
    name: &str,
    file_bytes: u64,
    arch: Option<&str>,
    dims: &[(String, u32)],
) -> PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create gguf fixture");

    let mut kv: Vec<u8> = Vec::new();
    let mut kv_count: u64 = 0;
    if let Some(arch) = arch {
        push_string_kv(&mut kv, "general.architecture", arch);
        push_u32_kv(&mut kv, "general.file_type", 15); // Q4_K_M
        kv_count += 2;
    } else {
        // A header with no architecture, no file_type and no block_count is
        // the "unusable metadata" case the loader has to refuse.
        push_string_kv(&mut kv, "general.name", "headerless fixture");
        kv_count += 1;
    }
    for (key, value) in dims {
        push_u32_kv(&mut kv, key, *value);
        kv_count += 1;
    }

    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap(); // version
    f.write_all(&0u64.to_le_bytes()).unwrap(); // tensor count
    f.write_all(&kv_count.to_le_bytes()).unwrap();
    f.write_all(&kv).unwrap();

    let header_len = f.stream_position().unwrap();
    assert!(
        file_bytes >= header_len,
        "fixture size {file_bytes} is smaller than its own header"
    );
    f.set_len(file_bytes).expect("extend fixture sparsely");
    drop(f);
    path
}

fn push_string_kv(out: &mut Vec<u8>, key: &str, value: &str) {
    push_len_prefixed(out, key.as_bytes());
    out.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
    push_len_prefixed(out, value.as_bytes());
}

fn push_u32_kv(out: &mut Vec<u8>, key: &str, value: u32) {
    push_len_prefixed(out, key.as_bytes());
    out.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_len_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// A 7B-class model of `size_mib` on disk.
fn model(dir: &Path, size_mib: u64) -> ModelInfo {
    let path = write_gguf(
        dir,
        &format!("model-{size_mib}mib.gguf"),
        size_mib * 1024 * 1024,
        Some("llama"),
        &arch_dims("llama"),
    );
    ModelInfo::read(&path).expect("fixture header is readable")
}

/// Verdicts ranked worst-last, so "never gets better / never gets worse" can
/// be stated as an ordering. `FitVerdict` is `#[non_exhaustive]`, so a new
/// variant lands here as an unranked worst case rather than a compile error.
fn severity(v: FitVerdict) -> u8 {
    match v {
        FitVerdict::Fits => 0,
        FitVerdict::ContextTooLarge => 1,
        FitVerdict::TooLarge => 2,
        _ => u8::MAX,
    }
}

const RAM_LADDER: &[u64] = &[2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128];
const CONTEXT_LADDER: &[u32] = &[
    2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 262_144, 1_000_000,
];
const SIZE_LADDER_MIB: &[u64] = &[64, 256, 512, 1_024, 2_048, 4_096, 8_192];

// ── Monotonicity in memory ──────────────────────────────────────────────────

#[test]
fn more_memory_never_un_fits_a_model_that_already_fit() {
    let dir = tempfile::tempdir().unwrap();
    for &size in SIZE_LADDER_MIB {
        let info = model(dir.path(), size);
        for &context in CONTEXT_LADDER {
            let mut worst_so_far = u8::MAX;
            for &ram in RAM_LADDER {
                let verdict = info.fits(&machine(ram), Some(context)).verdict;
                let rank = severity(verdict);
                assert!(
                    rank <= worst_so_far,
                    "a {size} MiB model at {context} context went from a better verdict to \
                     {verdict:?} when RAM grew to {ram} GiB — adding memory must never take \
                     away a fit, or a user who upgrades their machine loses a model",
                );
                worst_so_far = rank;
            }
        }
    }
}

#[test]
fn more_memory_never_shrinks_the_largest_workable_context() {
    let dir = tempfile::tempdir().unwrap();
    for &size in SIZE_LADDER_MIB {
        let info = model(dir.path(), size);
        let mut previous = 0;
        for &ram in RAM_LADDER {
            let max_context = info.max_context(&machine(ram));
            assert!(
                max_context >= previous,
                "a {size} MiB model offered {max_context} context on {ram} GiB but {previous} on \
                 less — the auto-sized context must not fall as memory rises",
            );
            previous = max_context;
        }
    }
}

#[test]
fn vram_rather_than_ram_bounds_the_fit_when_a_discrete_gpu_reports_some() {
    let dir = tempfile::tempdir().unwrap();
    let info = model(dir.path(), 8_192);
    let mut hw = machine(128);
    hw.gpu_backend = GpuBackend::Cuda;
    hw.vram_bytes = 4 * GIB;
    // A discrete card can only run what its own memory holds; the host's 128
    // GiB is not available to it, and treating it as available would
    // over-commit the GPU at load time.
    assert_eq!(
        info.fits(&hw, Some(2_048)).verdict,
        FitVerdict::TooLarge,
        "an 8 GiB model must not be called a fit for a 4 GiB card just because the host is roomy",
    );
}

// ── Monotonicity in requested context ───────────────────────────────────────

#[test]
fn asking_for_more_context_never_improves_the_verdict() {
    let dir = tempfile::tempdir().unwrap();
    for &size in SIZE_LADDER_MIB {
        let info = model(dir.path(), size);
        for &ram in RAM_LADDER {
            let hw = machine(ram);
            let mut best_so_far = 0;
            for &context in CONTEXT_LADDER {
                let verdict = info.fits(&hw, Some(context)).verdict;
                let rank = severity(verdict);
                assert!(
                    rank >= best_so_far,
                    "a {size} MiB model on {ram} GiB improved to {verdict:?} when the request \
                     grew to {context} context — a bigger ask can never be easier to satisfy",
                );
                best_so_far = rank;
            }
        }
    }
}

#[test]
fn asking_for_more_context_never_lowers_the_memory_estimate() {
    let dir = tempfile::tempdir().unwrap();
    let info = model(dir.path(), 2_048);
    let mut previous = 0;
    for &context in CONTEXT_LADDER {
        let needed = info.memory_needed(context);
        assert!(
            needed >= previous,
            "{context} context was estimated at {needed} bytes, below the {previous} bytes a \
             smaller context needed — KV cache only grows with context",
        );
        previous = needed;
    }
}

/// The context floor `fit_context` will not size below.
const CONTEXT_FLOOR: u32 = 2_048;

#[test]
fn the_context_that_max_context_offers_actually_fits() {
    let dir = tempfile::tempdir().unwrap();
    for &size in SIZE_LADDER_MIB {
        let info = model(dir.path(), size);
        for &ram in RAM_LADDER {
            let hw = machine(ram);
            let fit = info.fits(&hw, None);
            // `max_context` is what a ContextTooLarge verdict tells the caller
            // to fall back to, so a refusal at that very size is a dead end.
            if fit.verdict == FitVerdict::TooLarge {
                continue;
            }
            assert_eq!(
                fit.verdict,
                FitVerdict::Fits,
                "a {size} MiB model on {ram} GiB reported {:?} at its own max_context of {} \
                 — the suggested fallback must be one that works",
                fit.verdict,
                fit.max_context,
            );
        }
    }
}

#[test]
fn a_floor_context_that_does_not_fit_is_reported_as_too_large_not_as_advice() {
    // `fit_context` floors at 2048 tokens deliberately: whether a model that
    // tight should load at all is residency admission's call, not context
    // sizing's. So `max_context` is not always a context that fits, and a
    // `ContextTooLarge` verdict here would send the caller to a smaller window
    // that is also refused — while `Display` promised that window works.
    let dir = tempfile::tempdir().unwrap();
    let info = model(dir.path(), 1_024);
    let fit = info.fits(&machine(2), None);

    assert_eq!(fit.max_context, CONTEXT_FLOOR);
    assert!(
        info.memory_needed(fit.max_context) > fit.available_bytes,
        "this machine is only interesting while even the floor context does not fit",
    );
    assert_eq!(
        fit.verdict,
        FitVerdict::TooLarge,
        "when no context this machine will offer is accepted, the answer is that the model \
         does not fit — not a smaller context that does not fit either",
    );
}

#[test]
fn max_context_never_exceeds_what_the_header_was_trained_for() {
    let dir = tempfile::tempdir().unwrap();
    let info = model(dir.path(), 64);
    let train_context = info.train_context.expect("fixture declares context_length");
    assert_eq!(train_context, 32_768);
    assert!(
        info.max_context(&machine(128)) <= train_context,
        "a roomy machine must not offer more context than the model was trained for",
    );
}

// ── Monotonicity in model size ──────────────────────────────────────────────

#[test]
fn a_larger_model_never_needs_less_memory_at_the_same_context() {
    let dir = tempfile::tempdir().unwrap();
    for &context in CONTEXT_LADDER {
        let mut previous = 0;
        for &size in SIZE_LADDER_MIB {
            let needed = model(dir.path(), size).memory_needed(context);
            assert!(
                needed >= previous,
                "a {size} MiB model was estimated at {needed} bytes at {context} context, less \
                 than the {previous} bytes a smaller model needed — weights are additive",
            );
            previous = needed;
        }
    }
}

#[test]
fn a_larger_model_never_offers_more_context_on_the_same_machine() {
    let dir = tempfile::tempdir().unwrap();
    for &ram in RAM_LADDER {
        let hw = machine(ram);
        let mut previous = u32::MAX;
        for &size in SIZE_LADDER_MIB {
            let max_context = model(dir.path(), size).max_context(&hw);
            assert!(
                max_context <= previous,
                "on {ram} GiB a {size} MiB model offered {max_context} context, more than the \
                 {previous} a smaller model got — bigger weights leave less room for KV",
            );
            previous = max_context;
        }
    }
}

// ── Determinism ─────────────────────────────────────────────────────────────

#[test]
fn the_same_header_and_machine_always_produce_the_same_verdict() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf(
        dir.path(),
        "deterministic.gguf",
        2_048 * 1024 * 1024,
        Some("llama"),
        &arch_dims("llama"),
    );
    let hw = machine(16);
    let first = ModelInfo::read(&path).unwrap().fits(&hw, Some(8_192));
    for _ in 0..4 {
        let again = ModelInfo::read(&path).unwrap().fits(&hw, Some(8_192));
        assert_eq!(
            first, again,
            "re-reading the same file on the same machine changed the verdict — fit sizing must \
             not depend on anything but its inputs",
        );
    }
}

#[test]
fn the_architecture_the_keys_are_prefixed_with_does_not_change_the_estimate() {
    let dir = tempfile::tempdir().unwrap();
    let llama = write_gguf(
        dir.path(),
        "a.gguf",
        1024 * 1024 * 1024,
        Some("llama"),
        &arch_dims("llama"),
    );
    let qwen = write_gguf(
        dir.path(),
        "b.gguf",
        1024 * 1024 * 1024,
        Some("qwen3"),
        &arch_dims("qwen3"),
    );
    // Same dims, different arch tag: the estimate is arithmetic over the dims,
    // so a per-family fudge factor creeping in would show up here.
    assert_eq!(
        ModelInfo::read(&llama).unwrap().memory_needed(8_192),
        ModelInfo::read(&qwen).unwrap().memory_needed(8_192),
    );
}

// ── Missing metadata ────────────────────────────────────────────────────────

#[test]
fn a_header_with_no_architecture_metadata_is_refused_rather_than_guessed() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf(dir.path(), "nameless.gguf", 4 * 1024 * 1024, None, &[]);
    assert!(
        ModelInfo::read(&path).is_err(),
        "a GGUF carrying no architecture, file type or layer count must be refused — sizing it \
         from the file length alone would be a guess presented as a measurement",
    );
}

#[test]
fn a_file_that_is_not_gguf_at_all_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.bin");
    std::fs::write(&path, b"not a gguf file, just bytes").unwrap();
    assert!(ModelInfo::read(&path).is_err());
    assert!(ModelInfo::read(dir.path().join("absent.gguf")).is_err());
}

#[test]
fn a_header_missing_its_layer_counts_never_underestimates_the_weights() {
    let dir = tempfile::tempdir().unwrap();
    // Architecture present (so the header is usable) but no block/head counts,
    // which is what the KV-per-token cost is computed from.
    let path = write_gguf(
        dir.path(),
        "partial.gguf",
        2_048 * 1024 * 1024,
        Some("llama"),
        &[],
    );
    let info = ModelInfo::read(&path).expect("architecture alone is usable metadata");
    assert!(
        info.metadata.block_count.is_none(),
        "fixture must exercise the missing-dims path",
    );
    for &context in CONTEXT_LADDER {
        assert!(
            info.memory_needed(context) > info.file_bytes,
            "the estimate for a header without layer counts fell to or below the raw file size — \
             the fallback must still charge for runtime overhead, not just the weights",
        );
    }

    // The estimator has no dimensions to charge context against here, so it
    // falls back to a flat multiple of the file size. On its own that answer
    // does not move with context at all, and a four-million-token window would
    // be reported as fitting on any machine. A missing header must fail
    // conservatively, so an assumed per-token cost is charged instead.
    let absurd = info.fits(&machine(64), Some(4_000_000));
    assert_ne!(
        absurd.verdict,
        FitVerdict::Fits,
        "a context far past anything a real model is trained for must be refused even when \
         the header gave no dimensions to price it with",
    );
}

#[test]
fn a_header_without_dimensions_still_prices_context() {
    // The property underneath the test above: unknown metadata is charged for,
    // not treated as free. Without this, every context costs the same and the
    // fit answer is whatever the file size alone says.
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf(
        dir.path(),
        "dimensionless.gguf",
        2_048 * 1024 * 1024,
        Some("llama"),
        &[],
    );
    let info = ModelInfo::read(&path).expect("architecture alone is usable metadata");

    assert!(
        info.memory_needed(128_000) > info.memory_needed(4_096),
        "a larger context must cost more even when the cost had to be assumed",
    );
}

#[test]
fn sizing_for_more_conversations_gives_each_a_smaller_window() {
    // Weights are shared between contexts; KV is not. A window picked for one
    // conversation is exceeded the moment a second opens, and llama.cpp
    // reports that as a bare `Decode Error -3` rather than as running out of
    // room — so the arithmetic has to happen here, before the load.
    let dir = tempfile::tempdir().unwrap();
    let info = model(dir.path(), 2_048);
    let hw = machine(64);

    let alone = info.max_context_for(&hw, 1);
    assert_eq!(
        alone,
        info.max_context(&hw),
        "one is what max_context means"
    );

    let mut previous = alone;
    for concurrent in [2usize, 3, 4, 8] {
        let shared = info.max_context_for(&hw, concurrent);
        assert!(
            shared <= previous,
            "{concurrent} conversations were offered {shared} tokens each, more \
             than the {previous} offered to fewer",
        );
        previous = shared;
    }
}

#[test]
fn a_window_sized_for_several_conversations_fits_all_of_them() {
    // The property the sizing exists for: whatever it hands back must still
    // fit once every resident conversation is holding one.
    let dir = tempfile::tempdir().unwrap();
    for &size in SIZE_LADDER_MIB {
        let info = model(dir.path(), size);
        for &ram in RAM_LADDER {
            let hw = machine(ram);
            for concurrent in [1usize, 3, 8] {
                let ctx = info.max_context_for(&hw, concurrent);
                if info.fits(&hw, Some(ctx)).verdict != FitVerdict::Fits {
                    continue; // the model does not fit at all on this machine
                }
                if ctx == CONTEXT_FLOOR {
                    // `fit_context` floors rather than reporting a window too
                    // small to be worth loading, so the floor is not a promise
                    // that anything fits — see the floor test above.
                    continue;
                }
                // Weights and the fixed runtime overhead are paid once for the
                // process; only the KV cache is per conversation. `at_zero` is
                // everything that does not scale, so the difference from it is
                // exactly what each extra conversation adds.
                let at_zero = info.memory_needed(0);
                let per_conversation = info.memory_needed(ctx).saturating_sub(at_zero);
                let all =
                    at_zero.saturating_add(per_conversation.saturating_mul(concurrent as u64));
                assert!(
                    all <= info.fits(&hw, Some(ctx)).available_bytes,
                    "{size} MiB model on {ram} GiB offered {ctx} tokens to each of \
                     {concurrent} conversations, which together need more than the budget",
                );
            }
        }
    }
}
