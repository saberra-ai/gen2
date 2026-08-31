//! Memory budget derivation.
//!
//! Converts a `MemoryPolicyInput` (raw hardware numbers) into concrete
//! `MemoryBudgets`. All subsystems must read from this module; none should
//! compute their own limits from raw hardware numbers.
//!
//! # Design
//!
//! * `base_budgets_for_tier` returns idealized budgets assuming the machine
//!   has plenty of free RAM.
//! * `effective_budgets` clamps those idealized budgets against what is
//!   actually available right now. This is what callers should normally use.
//!
//! Budget semantics:
//! * `process_soft_limit_mb`  — target ceiling; trigger graceful shedding above this.
//! * `process_hard_limit_mb`  — never exceed; trigger emergency shedding above this.
//! * `search_working_set_mb`  — protected budget for interactive search (top priority).
//! * `kg_derived_state_mb`    — opportunistic; can be evicted first.
//! * `ingestion_peak_mb`      — import / OCR peak; lower-priority than search.
//! * `inference_resident_mb`  — resident model memory (LLM + helpers).
//! * `multimodal_peak_mb`     — image / audio pipeline peak; opportunistic.

use serde::Serialize;

use super::memory_tier::MachineMemoryTier;

/// Raw hardware snapshot passed into policy functions.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct MemoryPolicyInput {
    /// Total physical RAM in MiB as reported by the OS.
    pub total_memory_mb: u64,
    /// Currently available (free + reclaimable) RAM in MiB.
    pub available_memory_mb: u64,
    /// True when running on a mobile platform (iOS / Android).
    pub is_mobile: bool,
}

/// Concrete memory ceilings for every major subsystem.
///
/// All values are in MiB. Hard limit is always ≥ soft limit.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct MemoryBudgets {
    /// Graceful-degradation trigger for total process RSS.
    pub process_soft_limit_mb: u64,
    /// Hard ceiling for total process RSS; emergency shed above this.
    pub process_hard_limit_mb: u64,
    /// Working-set budget for interactive search (protected).
    pub search_working_set_mb: u64,
    /// Opportunistic budget for KG derived state (co-occurrence, PageRank, clusters).
    pub kg_derived_state_mb: u64,
    /// Peak budget for background document import / OCR.
    pub ingestion_peak_mb: u64,
    /// Resident budget for inference runtimes (LLM + helpers combined).
    pub inference_resident_mb: u64,
    /// Peak budget for multimodal (image rasterization / audio) pipelines.
    pub multimodal_peak_mb: u64,
}

// ── Tier detection ──────────────────────────────────────────────────────────

/// Classify `input` into a `MachineMemoryTier`.
///
/// Mobile always maps to `MobileConstrained`. Desktop tiers are
/// assigned by total RAM:
///
/// | Total RAM   | Tier               |
/// |-------------|--------------------|
/// | (mobile)    | MobileConstrained  |
/// | < 8 GiB     | DesktopConstrained |
/// | 8–16 GiB    | DesktopMainstream  |
/// | 16–32 GiB   | DesktopPower       |
/// | > 32 GiB    | Workstation        |
pub fn detect_machine_tier(input: &MemoryPolicyInput) -> MachineMemoryTier {
    if input.is_mobile {
        return MachineMemoryTier::MobileConstrained;
    }
    // Boundaries in MiB (1 GiB = 1024 MiB)
    match input.total_memory_mb {
        0..=8191 => MachineMemoryTier::DesktopConstrained,
        8192..=16383 => MachineMemoryTier::DesktopMainstream,
        16384..=32767 => MachineMemoryTier::DesktopPower,
        _ => MachineMemoryTier::Workstation,
    }
}

// ── Base budgets ────────────────────────────────────────────────────────────

/// Idealized budgets for `tier` assuming ample free RAM.
///
/// These assume search and inference are the protected workloads and that
/// ingestion / KG maintenance / multimodal are opportunistic.
pub fn base_budgets_for_tier(tier: MachineMemoryTier) -> MemoryBudgets {
    match tier {
        MachineMemoryTier::MobileConstrained => MemoryBudgets {
            // Sized for on-device inference shells (pio-nola) that bundle and
            // run a small LLM **and** a Whisper STT model locally. The prior
            // 400 MB inference cap predated that workload and rejected a 1.2B
            // Q4 GGUF (~700 MB resident) with "llm admission denied by
            // residency policy" (`can_admit`: estimate ≤ inference_resident_mb).
            //
            // These are the base/ideal numbers for a high-end phone: recent Pro
            // iPhones ship 8 GB RAM and, with the increased-memory entitlement,
            // grant foreground apps multiple GB. A 4 GB inference budget leaves
            // generous headroom for LFM2 + Whisper co-resident today and a
            // larger model tomorrow. `effective_budgets` clamps everything to
            // `available/2` on smaller devices and scales the sub-budgets
            // (incl. inference) down proportionally — on a 4 GB phone the
            // effective inference budget lands ~1.8 GB, still far above the
            // ~700 MB LLM, and the idle Whisper slot is evicted by
            // `evict_idle_helpers` under pressure.
            process_soft_limit_mb: 4608,
            process_hard_limit_mb: 6144,
            search_working_set_mb: 200,
            kg_derived_state_mb: 100,
            ingestion_peak_mb: 400,
            inference_resident_mb: 4096,
            multimodal_peak_mb: 400,
        },
        MachineMemoryTier::DesktopConstrained => MemoryBudgets {
            process_soft_limit_mb: 1536,
            process_hard_limit_mb: 2048,
            search_working_set_mb: 200,
            kg_derived_state_mb: 100,
            ingestion_peak_mb: 400,
            inference_resident_mb: 800,
            multimodal_peak_mb: 400,
        },
        MachineMemoryTier::DesktopMainstream => MemoryBudgets {
            process_soft_limit_mb: 3072,
            process_hard_limit_mb: 4096,
            search_working_set_mb: 400,
            kg_derived_state_mb: 200,
            ingestion_peak_mb: 800,
            inference_resident_mb: 1536,
            multimodal_peak_mb: 768,
        },
        MachineMemoryTier::DesktopPower => MemoryBudgets {
            process_soft_limit_mb: 6144,
            process_hard_limit_mb: 8192,
            search_working_set_mb: 800,
            kg_derived_state_mb: 400,
            ingestion_peak_mb: 1536,
            inference_resident_mb: 3072,
            multimodal_peak_mb: 1536,
        },
        MachineMemoryTier::Workstation => MemoryBudgets {
            // Workstation = 32+ GB host RAM. Lifted from 12288/16384 to
            // accommodate 31B-class Q4 GGUFs (~17 GB resident on Metal).
            // The prior cap rejected them with "llm admission denied by
            // residency policy" via `can_load_additional_model` checking
            // projected RSS against the soft limit.
            process_soft_limit_mb: 32768,
            process_hard_limit_mb: 49152,
            search_working_set_mb: 1600,
            kg_derived_state_mb: 800,
            ingestion_peak_mb: 3072,
            inference_resident_mb: 24576,
            multimodal_peak_mb: 3072,
        },
    }
}

// ── Effective budgets ───────────────────────────────────────────────────────

/// Derive effective budgets by clamping `base_budgets_for_tier` against
/// the currently available memory.
///
/// Clamping rules:
/// * `process_soft_limit_mb ≤ available * 50%`
/// * `process_hard_limit_mb ≤ available * 75%`
/// * `process_hard_limit_mb ≥ process_soft_limit_mb` (invariant)
///
/// Sub-budgets (search, KG, ingestion, inference, multimodal) are scaled
/// proportionally when the soft limit is clamped, preserving their relative
/// priorities.
pub fn effective_budgets(input: &MemoryPolicyInput) -> MemoryBudgets {
    let tier = detect_machine_tier(input);
    let base = base_budgets_for_tier(tier);

    let avail = input.available_memory_mb;
    if avail == 0 {
        // Degenerate: return base unchanged; governor will classify as Emergency.
        return base;
    }

    // Clamp process limits against available memory.
    let soft_cap = avail / 2;
    let hard_cap = avail * 3 / 4;

    let soft = base.process_soft_limit_mb.min(soft_cap);
    // Hard must be ≥ soft and ≤ hard_cap
    let hard = base.process_hard_limit_mb.min(hard_cap).max(soft);

    // If the soft limit was clamped, scale sub-budgets proportionally.
    if soft < base.process_soft_limit_mb && base.process_soft_limit_mb > 0 {
        let scale_num = soft;
        let scale_den = base.process_soft_limit_mb;
        let scale = |v: u64| (v * scale_num / scale_den).max(1);
        MemoryBudgets {
            process_soft_limit_mb: soft,
            process_hard_limit_mb: hard,
            search_working_set_mb: scale(base.search_working_set_mb),
            kg_derived_state_mb: scale(base.kg_derived_state_mb),
            ingestion_peak_mb: scale(base.ingestion_peak_mb),
            inference_resident_mb: scale(base.inference_resident_mb),
            multimodal_peak_mb: scale(base.multimodal_peak_mb),
        }
    } else {
        MemoryBudgets {
            process_soft_limit_mb: soft,
            process_hard_limit_mb: hard,
            ..base
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn input(total_mb: u64, avail_mb: u64) -> MemoryPolicyInput {
        MemoryPolicyInput {
            total_memory_mb: total_mb,
            available_memory_mb: avail_mb,
            is_mobile: false,
        }
    }

    fn mobile_input(total_mb: u64, avail_mb: u64) -> MemoryPolicyInput {
        MemoryPolicyInput {
            total_memory_mb: total_mb,
            available_memory_mb: avail_mb,
            is_mobile: true,
        }
    }

    // ── Tier detection ────────────────────────────────────────────────────

    #[test]
    fn tier_mobile_always_constrained() {
        // Even a 64 GiB mobile device is MobileConstrained
        let i = mobile_input(65536, 32768);
        assert_eq!(
            detect_machine_tier(&i),
            MachineMemoryTier::MobileConstrained
        );
    }

    #[test]
    fn mobile_inference_budget_admits_bundled_llm() {
        // Regression: pio-nola bundles LFM2-1.2B-Q4 (~700 MB resident) and
        // runs it on-device. The MobileConstrained inference budget must hold
        // it, or `can_admit` rejects with "llm admission denied by residency
        // policy". On iOS the runtime reports available == total RAM, so probe
        // a representative 6 GB-class phone and a tighter 4 GB device.
        const LFM2_RESIDENT_MB: u64 = 700;
        for (total, avail) in [(6144u64, 6144u64), (4096, 4096)] {
            let b = effective_budgets(&mobile_input(total, avail));
            assert!(
                b.inference_resident_mb >= LFM2_RESIDENT_MB,
                "mobile inference budget {} MB < LFM2 {} MB (total={total}, avail={avail})",
                b.inference_resident_mb,
                LFM2_RESIDENT_MB
            );
            // The model must also fit under the process soft limit (the second
            // gate in `can_load_additional_model`, checked from ~0 RSS at boot).
            assert!(
                b.process_soft_limit_mb > LFM2_RESIDENT_MB,
                "mobile soft limit {} MB ≤ LFM2 {} MB",
                b.process_soft_limit_mb,
                LFM2_RESIDENT_MB
            );
        }
    }

    #[test]
    fn tier_8gb_desktop() {
        // 8 GiB = 8192 MiB → DesktopMainstream (boundary is exclusive on lower end)
        let i = input(8192, 4096);
        assert_eq!(
            detect_machine_tier(&i),
            MachineMemoryTier::DesktopMainstream
        );
    }

    #[test]
    fn tier_below_8gb_constrained() {
        // 6 GiB = 6144 MiB → DesktopConstrained
        let i = input(6144, 3000);
        assert_eq!(
            detect_machine_tier(&i),
            MachineMemoryTier::DesktopConstrained
        );
    }

    #[test]
    fn tier_16gb_desktop() {
        let i = input(16384, 8192);
        assert_eq!(detect_machine_tier(&i), MachineMemoryTier::DesktopPower);
    }

    #[test]
    fn tier_32gb_workstation() {
        let i = input(32768, 20000);
        assert_eq!(detect_machine_tier(&i), MachineMemoryTier::Workstation);
    }

    #[test]
    fn tier_64gb_workstation() {
        let i = input(65536, 40000);
        assert_eq!(detect_machine_tier(&i), MachineMemoryTier::Workstation);
    }

    // ── Base budget invariants ─────────────────────────────────────────────

    #[test]
    fn base_budgets_hard_gte_soft_for_all_tiers() {
        use MachineMemoryTier::*;
        for tier in [
            MobileConstrained,
            DesktopConstrained,
            DesktopMainstream,
            DesktopPower,
            Workstation,
        ] {
            let b = base_budgets_for_tier(tier);
            assert!(
                b.process_hard_limit_mb >= b.process_soft_limit_mb,
                "hard < soft for {:?}",
                tier
            );
        }
    }

    #[test]
    fn base_budgets_search_protected() {
        // Search budget must be < soft limit (it's a sub-budget, not the whole thing)
        use MachineMemoryTier::*;
        for tier in [
            MobileConstrained,
            DesktopConstrained,
            DesktopMainstream,
            DesktopPower,
            Workstation,
        ] {
            let b = base_budgets_for_tier(tier);
            assert!(
                b.search_working_set_mb < b.process_soft_limit_mb,
                "search budget >= soft limit for {:?}",
                tier
            );
        }
    }

    // ── Effective budget clamping ──────────────────────────────────────────

    #[test]
    fn effective_budgets_clamp_when_low_available() {
        // Machine has 16 GB total but only 1 GB available
        let i = input(16384, 1024);
        let b = effective_budgets(&i);
        // soft ≤ available * 50% = 512 MiB
        assert!(
            b.process_soft_limit_mb <= 512,
            "soft={}",
            b.process_soft_limit_mb
        );
        // hard ≤ available * 75% = 768 MiB
        assert!(
            b.process_hard_limit_mb <= 768,
            "hard={}",
            b.process_hard_limit_mb
        );
        // invariant: hard >= soft
        assert!(b.process_hard_limit_mb >= b.process_soft_limit_mb);
    }

    #[test]
    fn effective_budgets_no_clamp_when_ample() {
        // Workstation with plenty of free RAM — base budgets should be
        // preserved. `available_memory_mb` must be ≥ 2× soft_limit so
        // the `avail / 2` cap doesn't clip; with the lifted Workstation
        // budget (32768 / 49152) that's ≥ 65536 MiB available.
        let i = input(131072, 98304);
        let b = effective_budgets(&i);
        let base = base_budgets_for_tier(MachineMemoryTier::Workstation);
        assert_eq!(b.process_soft_limit_mb, base.process_soft_limit_mb);
        assert_eq!(b.process_hard_limit_mb, base.process_hard_limit_mb);
    }

    #[test]
    fn effective_budgets_hard_always_gte_soft() {
        // Stress test with various available values
        for avail_mb in [100u64, 500, 1024, 2048, 4096, 8192, 16384] {
            let i = input(16384, avail_mb);
            let b = effective_budgets(&i);
            assert!(
                b.process_hard_limit_mb >= b.process_soft_limit_mb,
                "hard < soft at avail={}",
                avail_mb
            );
        }
    }

    #[test]
    fn effective_budgets_sub_budgets_nonzero_after_clamp() {
        // Extreme pressure: only 200 MiB available
        let i = input(8192, 200);
        let b = effective_budgets(&i);
        assert!(b.search_working_set_mb >= 1);
        assert!(b.kg_derived_state_mb >= 1);
        assert!(b.inference_resident_mb >= 1);
    }
}
