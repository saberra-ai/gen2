//! iOS runtime memory-budgeting for the shared llama.cpp backend.
//!
//! iOS enforces a per-process memory limit (jetsam) that is *much* smaller
//! than physical RAM, and killing an app that crosses it is silent and abrupt.
//! Before we commit an on-device model load we therefore pre-flight three
//! independent gates:
//!
//! 1. **Available-memory guard** — the on-disk model size × a headroom factor
//!    (weights are resident + KV cache + activations) must fit inside what
//!    `os_proc_available_memory()` reports the OS will *currently* hand this
//!    process before jetsam. This is the iOS-correct API (iOS 13+); physical
//!    RAM / `mach_task_basic_info` do **not** reflect the jetsam budget.
//! 2. **Device floor** — refuse below the A14 / iPhone-12 tier. Derived from
//!    the available-memory budget (a robust, model-name-independent signal).
//! 3. **Model ceiling** — refuse models above ~4B params or heavier than
//!    Q4_K_M quant, read from GGUF metadata / on-disk size *pre-load*.
//!
//! Everything in this module is compiled **only** on iOS. The public entry
//! [`preflight_ios`] is called from the llama loader behind
//! `#[cfg(target_os = "ios")]`; the desktop/flagship build never sees it and
//! is byte-behavior-identical. Refusals return typed [`ExecError`]s (never
//! panics) so the shell can surface an honest "device too small / model too
//! large" status.
//!
//! ⬜ DEVICE-ONLY: the real jetsam ceiling, the true available-memory number,
//! and actual eviction/`mlock` behaviour can only be observed on a physical
//! iPhone — they cannot be verified in a worktree or the simulator. The pure
//! arithmetic (headroom math, quant/param ceiling) is unit-tested off-iOS via
//! the platform-independent helpers below.

use crate::engine::ExecError;
use std::path::Path;

/// Headroom multiplier applied to the on-disk model size to approximate the
/// resident footprint of a *running* model: weights (mmap'd + faulted in when
/// mlock'd) plus KV cache, compute buffers, and activations at our default
/// context. 1.5× is deliberately conservative — a Q4_K_M 3–4B model at an 8k
/// context spends the bulk of its footprint on weights, with the KV cache and
/// scratch buffers adding roughly a third on top. Being conservative here
/// trades a few false "won't fit" refusals for avoiding silent jetsam kills,
/// which is the correct bias on iOS.
pub(crate) const KV_HEADROOM_FACTOR: f64 = 1.5;

/// Minimum available-memory budget (bytes) we require before attempting a
/// load — the A14 / iPhone-12 device floor.
///
/// The iPhone 12 (A14) ships 4 GB physical RAM and hands third-party apps a
/// jetsam budget in the ~2 GB range under the "Extended Virtual Addressing" /
/// increased-memory-limit entitlement (which the flagship holds). Devices
/// below that tier (A12/A13, 3 GB) report a smaller budget and cannot hold a
/// 3–4B Q4 model plus its KV cache without being killed. We gate on the
/// *available-memory budget* rather than the device model string because the
/// budget is the signal that actually predicts a jetsam kill, and it needs no
/// hard-coded device allowlist to maintain.
///
/// 1.6 GiB is chosen to sit above what a sub-A14 device offers a foreground
/// app yet below the A14's entitled budget, so the floor admits iPhone 12+
/// and refuses older tiers. ⬜ The exact per-device number is device-only.
pub(crate) const IOS_DEVICE_FLOOR_BYTES: u64 = 1_600 * 1024 * 1024;

/// Maximum parameter count we allow on iOS: ~4B. Estimated from on-disk size
/// and quant bits-per-weight (see [`estimate_params`]). Above this the model
/// won't hold in the jetsam budget of even an entitled A14-class device.
pub(crate) const IOS_MAX_PARAMS: u64 = 4_400_000_000; // ~4B + slack for 4B nominal

/// Result of reading the quant + estimated size from a GGUF header pre-load.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GgufFacts {
    /// On-disk file size in bytes.
    pub file_size: u64,
    /// `general.file_type` quant enum from GGUF metadata, if present.
    pub file_type: Option<u32>,
}

/// Approximate bits-per-weight for a GGUF `general.file_type` value, used to
/// estimate parameter count from on-disk size. Only the values relevant to the
/// iOS ceiling decision are distinguished precisely; anything heavier than
/// Q4_K_M returns a bits-per-weight ≥ our Q4 reference so the ceiling check
/// treats it as "too heavy" via [`quant_heavier_than_q4km`]. Returns `None`
/// for unknown enums (caller falls back to a conservative default).
///
/// Enum values follow `llama_ftype` in llama.cpp (`ggml.h` / `llama.h`).
pub(crate) fn bits_per_weight(file_type: u32) -> Option<f64> {
    Some(match file_type {
        0 => 32.0, // ALL_F32
        1 => 16.0, // MOSTLY_F16
        2 => 4.5,  // MOSTLY_Q4_0
        3 => 4.5,  // MOSTLY_Q4_1
        7 => 8.5,  // MOSTLY_Q8_0
        8 => 5.5,  // MOSTLY_Q5_0
        9 => 5.5,  // MOSTLY_Q5_1
        10 => 2.6, // MOSTLY_Q2_K
        11 => 3.4, // MOSTLY_Q3_K_S
        12 => 3.9, // MOSTLY_Q3_K_M
        13 => 4.3, // MOSTLY_Q3_K_L
        14 => 4.5, // MOSTLY_Q4_K_S
        15 => 4.8, // MOSTLY_Q4_K_M  (the iOS reference quant)
        16 => 5.5, // MOSTLY_Q5_K_S
        17 => 5.7, // MOSTLY_Q5_K_M
        18 => 6.6, // MOSTLY_Q6_K
        _ => return None,
    })
}

/// Reference bits-per-weight for Q4_K_M — the heaviest quant we permit on iOS.
pub(crate) const Q4KM_BITS_PER_WEIGHT: f64 = 4.8;

/// True if `file_type` denotes a quant *heavier* (more bits/weight) than
/// Q4_K_M. Unknown/absent quant is treated as **not** heavier (we let the
/// size-based param gate do the work instead of falsely refusing).
pub(crate) fn quant_heavier_than_q4km(file_type: Option<u32>) -> bool {
    match file_type.and_then(bits_per_weight) {
        Some(bpw) => bpw > Q4KM_BITS_PER_WEIGHT + 0.01,
        None => false,
    }
}

/// Estimate parameter count from on-disk size and quant. If the quant is known
/// we use its bits-per-weight; otherwise we assume the model is at least as
/// dense as Q4_K_M (a conservative floor that never *under*-counts params for
/// a Q4-or-lighter file, so the ceiling check can't be trivially bypassed by a
/// missing quant tag).
pub(crate) fn estimate_params(facts: &GgufFacts) -> u64 {
    let bpw = facts
        .file_type
        .and_then(bits_per_weight)
        .unwrap_or(Q4KM_BITS_PER_WEIGHT);
    // params ≈ (file_bytes * 8 bits) / bits_per_weight
    ((facts.file_size as f64 * 8.0) / bpw) as u64
}

/// Bytes required to *run* a model of the given on-disk size, including KV
/// cache / activation headroom. Pure arithmetic — unit-tested off-iOS.
pub(crate) fn required_runtime_bytes(file_size: u64) -> u64 {
    (file_size as f64 * KV_HEADROOM_FACTOR) as u64
}

/// The iOS-correct available-memory query: how much memory the OS will let
/// *this process* allocate right now before jetsam terminates it. iOS 13+.
///
/// Returns `None` if the FFI reports 0 (shouldn't happen on a real device but
/// is possible under some sandboxes / the simulator), so callers can degrade
/// to a physical-RAM-independent decision rather than divide by zero.
#[cfg(target_os = "ios")]
pub(crate) fn os_proc_available_memory_bytes() -> Option<u64> {
    // <os/proc.h>, iOS 13+. Not exposed by `libc`, so we FFI it directly.
    unsafe extern "C" {
        fn os_proc_available_memory() -> usize;
    }
    // SAFETY: `os_proc_available_memory` is a leaf C function with no
    // arguments and no pointer parameters; it reads a kernel-maintained
    // per-process counter and returns it by value.
    let avail = unsafe { os_proc_available_memory() } as u64;
    if avail == 0 { None } else { Some(avail) }
}

/// Read GGUF quant (`general.file_type`) + on-disk size without loading the
/// model. Reuses the same lightweight header walk the engine uses for
/// architecture detection. Returns `None` only if the file can't be stat'd;
/// a missing quant key yields `file_type: None` (handled downstream).
pub(crate) fn read_gguf_facts(path: &Path) -> Option<GgufFacts> {
    let file_size = std::fs::metadata(path).ok()?.len();
    let file_type = crate::engine::read_gguf_file_type(path);
    Some(GgufFacts {
        file_size,
        file_type,
    })
}

/// Pre-flight all iOS memory-budget gates for a model at `model_path`.
///
/// Called from the llama loader **before** `LlamaModel::load_from_file` on
/// iOS. Returns `Ok(())` when the model may be loaded, or a typed
/// [`ExecError::OutOfMemory`] describing precisely which gate refused it.
/// `OutOfMemory` maps to `PioError::model_oom` — the exact "device too small /
/// model too large" surface the shell shows.
///
/// The iOS simulator runs on the host Mac and does not enforce the device
/// jetsam budget, so we skip the *available-memory* and *device-floor* gates
/// there (they'd misfire against the Mac's numbers) while keeping the
/// model-ceiling gate active (it's a device-independent policy check).
#[cfg(target_os = "ios")]
pub(crate) fn preflight_ios(model_path: &Path) -> Result<(), ExecError> {
    let facts = read_gguf_facts(model_path).ok_or_else(|| {
        ExecError::OutOfMemory(format!(
            "cannot stat model file to budget iOS memory: {}",
            model_path.display()
        ))
    })?;

    // Gate 3 (device-independent policy): model ceiling — refuse >~4B params
    // or a quant heavier than Q4_K_M. Runs even in the simulator.
    check_model_ceiling(&facts)?;

    // Gates 1 & 2 are jetsam-budget gates and only meaningful on a real
    // device; the simulator borrows the host Mac's (much larger) budget.
    if crate::hardware::is_ios_simulator() {
        return Ok(());
    }

    if let Some(avail) = os_proc_available_memory_bytes() {
        // Gate 2: device floor (A14 / iPhone-12). A sub-floor budget means an
        // older device that can't hold a 3–4B Q4 model + KV cache.
        if avail < IOS_DEVICE_FLOOR_BYTES {
            return Err(ExecError::OutOfMemory(format!(
                "device below the supported floor: iOS reports {avail_mib} MiB available to this \
                 app, but a minimum of {floor_mib} MiB (A14 / iPhone 12 or newer) is required to \
                 run on-device models",
                avail_mib = avail / (1024 * 1024),
                floor_mib = IOS_DEVICE_FLOOR_BYTES / (1024 * 1024),
            )));
        }

        // Gate 1: available-memory guard — required runtime footprint (weights
        // × KV headroom) must fit the current jetsam budget.
        let required = required_runtime_bytes(facts.file_size);
        if required > avail {
            return Err(ExecError::OutOfMemory(format!(
                "model too large for available memory: needs ~{need_mib} MiB (model {model_mib} \
                 MiB × {factor:.1} for KV cache), but iOS reports only {avail_mib} MiB available \
                 to this app — try a smaller model or a lighter quant",
                need_mib = required / (1024 * 1024),
                model_mib = facts.file_size / (1024 * 1024),
                factor = KV_HEADROOM_FACTOR,
                avail_mib = avail / (1024 * 1024),
            )));
        }
    } else {
        // os_proc_available_memory returned 0 — can't budget. Fail closed with
        // an honest error rather than risk a silent jetsam kill.
        return Err(ExecError::OutOfMemory(
            "iOS available-memory query returned 0; cannot safely budget this load".to_string(),
        ));
    }

    Ok(())
}

/// Model-ceiling gate, factored out so it can be unit-tested off-iOS: refuse a
/// model above ~4B params or heavier than Q4_K_M quant. Pure — no FFI.
pub(crate) fn check_model_ceiling(facts: &GgufFacts) -> Result<(), ExecError> {
    if quant_heavier_than_q4km(facts.file_type) {
        return Err(ExecError::OutOfMemory(format!(
            "quant too heavy for iOS: this model uses a quantization heavier than Q4_K_M \
             (file_type={:?}); use a Q4_K_M or lighter build on iOS",
            facts.file_type,
        )));
    }

    let params = estimate_params(facts);
    if params > IOS_MAX_PARAMS {
        return Err(ExecError::OutOfMemory(format!(
            "model too large for iOS: ~{params_b:.1}B parameters estimated (ceiling is ~4B on \
             iOS); use a 3–4B Q4_K_M model",
            params_b = params as f64 / 1e9,
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(file_size: u64, file_type: Option<u32>) -> GgufFacts {
        GgufFacts {
            file_size,
            file_type,
        }
    }

    #[test]
    fn headroom_math_is_1_5x() {
        assert_eq!(required_runtime_bytes(1000), 1500);
        // 2 GiB model → 3 GiB runtime footprint.
        let two_gib = 2u64 * 1024 * 1024 * 1024;
        assert_eq!(required_runtime_bytes(two_gib), 3 * 1024 * 1024 * 1024);
    }

    #[test]
    fn q4km_and_lighter_pass_the_quant_gate() {
        // Q4_K_M (15), Q4_K_S (14), Q4_0 (2), Q3_K_M (12), Q2_K (10) are all
        // <= Q4_K_M weight → not heavier.
        for ft in [2u32, 10, 11, 12, 14, 15] {
            assert!(
                !quant_heavier_than_q4km(Some(ft)),
                "file_type {ft} should NOT be heavier than Q4_K_M"
            );
        }
    }

    #[test]
    fn heavier_quants_fail_the_quant_gate() {
        // Q5_K_M (17), Q6_K (18), Q8_0 (7), F16 (1), F32 (0) are heavier.
        for ft in [0u32, 1, 7, 17, 18] {
            assert!(
                quant_heavier_than_q4km(Some(ft)),
                "file_type {ft} SHOULD be heavier than Q4_K_M"
            );
        }
    }

    #[test]
    fn unknown_quant_is_not_treated_as_heavier() {
        assert!(!quant_heavier_than_q4km(Some(999)));
        assert!(!quant_heavier_than_q4km(None));
    }

    #[test]
    fn estimate_params_for_a_4b_q4km() {
        // A 4B model at Q4_K_M (~4.8 bpw) is ~2.4 GB on disk.
        // 2.4e9 bytes * 8 / 4.8 ≈ 4.0e9 params.
        let f = facts(2_400_000_000, Some(15));
        let p = estimate_params(&f);
        assert!(
            (3_800_000_000..=4_200_000_000).contains(&p),
            "estimated {p} params for a 4B Q4_K_M, expected ~4B"
        );
    }

    #[test]
    fn ceiling_accepts_a_3b_q4km() {
        // 3B Q4_K_M ≈ 1.8 GB on disk → ~3B params, Q4_K_M quant → OK.
        let f = facts(1_800_000_000, Some(15));
        assert!(check_model_ceiling(&f).is_ok());
    }

    #[test]
    fn ceiling_refuses_a_7b_q4km() {
        // 7B Q4_K_M ≈ 4.4 GB on disk → ~7.3B params → over the 4B ceiling.
        let f = facts(4_400_000_000, Some(15));
        let err = check_model_ceiling(&f).unwrap_err();
        assert!(matches!(err, ExecError::OutOfMemory(_)));
        assert!(format!("{err}").contains("too large for iOS"));
    }

    #[test]
    fn ceiling_refuses_a_heavy_quant_even_if_small() {
        // A tiny model but Q6_K → refused on the quant gate.
        let f = facts(500_000_000, Some(18));
        let err = check_model_ceiling(&f).unwrap_err();
        assert!(format!("{err}").contains("quant too heavy"));
    }

    #[test]
    fn ceiling_missing_quant_falls_back_to_q4km_density() {
        // No quant tag: assume Q4_K_M density. 1.8 GB → ~3B params → OK.
        let ok = facts(1_800_000_000, None);
        assert!(check_model_ceiling(&ok).is_ok());
        // 3 GB with no tag → ~5B params (assuming Q4_K_M) → refused.
        let big = facts(3_000_000_000, None);
        assert!(check_model_ceiling(&big).is_err());
    }
}
