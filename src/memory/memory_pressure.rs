//! Runtime memory pressure classification.
//!
//! Given an estimate of the current process RSS and the effective
//! `MemoryBudgets`, produces a `MemoryPressureLevel` that all subsystems
//! use to decide whether to shed work.
//!
//! # Levels
//!
//! | Level       | Condition                               | Action guideline              |
//! |-------------|-----------------------------------------|-------------------------------|
//! | Normal      | RSS < soft * 80%                        | Full operation                |
//! | Constrained | soft * 80% ≤ RSS < soft                 | Reduce opportunistic work     |
//! | Severe      | soft ≤ RSS < hard                       | Shed non-interactive work     |
//! | Emergency   | RSS ≥ hard                              | Drop everything non-essential |

use serde::Serialize;

use super::memory_policy::MemoryBudgets;

/// Four-level runtime pressure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum MemoryPressureLevel {
    /// Process RSS is well within the soft limit. Full operation allowed.
    Normal,
    /// Approaching the soft limit. Reduce opportunistic / background work.
    Constrained,
    /// Soft limit exceeded. Shed non-interactive work immediately.
    Severe,
    /// Hard limit exceeded or imminent. Drop all non-essential state.
    Emergency,
}

/// Classify the current memory pressure from an RSS estimate and budgets.
///
/// Uses integer arithmetic throughout to avoid floating-point on hot paths.
///
/// Illustrative only — `memory` is crate-internal, so this cannot be a doc
/// test. The same four boundaries are asserted in the unit tests below.
///
/// ```text
/// let budgets = MemoryBudgets {
///     process_soft_limit_mb: 3072,
///     process_hard_limit_mb: 4096,
///     search_working_set_mb: 400,
///     kg_derived_state_mb: 200,
///     ingestion_peak_mb: 800,
///     inference_resident_mb: 1536,
///     multimodal_peak_mb: 768,
/// };
/// assert_eq!(classify_pressure(1000, &budgets), MemoryPressureLevel::Normal);
/// assert_eq!(classify_pressure(2600, &budgets), MemoryPressureLevel::Constrained);
/// assert_eq!(classify_pressure(3500, &budgets), MemoryPressureLevel::Severe);
/// assert_eq!(classify_pressure(4200, &budgets), MemoryPressureLevel::Emergency);
/// ```
pub fn classify_pressure(
    current_process_estimated_mb: u64,
    budgets: &MemoryBudgets,
) -> MemoryPressureLevel {
    let soft = budgets.process_soft_limit_mb;
    let hard = budgets.process_hard_limit_mb;
    let cur = current_process_estimated_mb;

    if cur >= hard {
        MemoryPressureLevel::Emergency
    } else if cur >= soft {
        MemoryPressureLevel::Severe
    } else if cur * 10 >= soft * 8 {
        // cur >= soft * 0.8, using integer multiply to avoid float
        MemoryPressureLevel::Constrained
    } else {
        MemoryPressureLevel::Normal
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn budgets(soft: u64, hard: u64) -> MemoryBudgets {
        MemoryBudgets {
            process_soft_limit_mb: soft,
            process_hard_limit_mb: hard,
            search_working_set_mb: soft / 8,
            kg_derived_state_mb: soft / 16,
            ingestion_peak_mb: soft / 4,
            inference_resident_mb: soft / 2,
            multimodal_peak_mb: soft / 4,
        }
    }

    #[test]
    fn pressure_normal_well_below_soft() {
        let b = budgets(3072, 4096);
        assert_eq!(classify_pressure(0, &b), MemoryPressureLevel::Normal);
        assert_eq!(classify_pressure(1000, &b), MemoryPressureLevel::Normal);
        // 79.9% of soft = 2456 → still Normal
        assert_eq!(classify_pressure(2456, &b), MemoryPressureLevel::Normal);
    }

    #[test]
    fn pressure_constrained_at_80_pct_soft() {
        let b = budgets(3072, 4096);
        // Exactly 80% of 3072 = 2457.6 → 2458
        assert_eq!(
            classify_pressure(2458, &b),
            MemoryPressureLevel::Constrained
        );
        assert_eq!(
            classify_pressure(3071, &b),
            MemoryPressureLevel::Constrained
        );
    }

    #[test]
    fn pressure_severe_at_soft() {
        let b = budgets(3072, 4096);
        assert_eq!(classify_pressure(3072, &b), MemoryPressureLevel::Severe);
        assert_eq!(classify_pressure(4095, &b), MemoryPressureLevel::Severe);
    }

    #[test]
    fn pressure_emergency_at_hard() {
        let b = budgets(3072, 4096);
        assert_eq!(classify_pressure(4096, &b), MemoryPressureLevel::Emergency);
        assert_eq!(classify_pressure(8192, &b), MemoryPressureLevel::Emergency);
    }

    #[test]
    fn pressure_ordering_is_monotonic() {
        use MemoryPressureLevel::*;
        assert!(Normal < Constrained);
        assert!(Constrained < Severe);
        assert!(Severe < Emergency);
    }

    #[test]
    fn pressure_mobile_constrained_tier_budgets() {
        // Synthetic small budget (soft=800, hard=1200) exercising the
        // classify_pressure boundaries — not tied to the MobileConstrained
        // tier constants (see base_budgets_for_tier).
        let b = budgets(800, 1200);
        assert_eq!(classify_pressure(0, &b), MemoryPressureLevel::Normal);
        assert_eq!(classify_pressure(639, &b), MemoryPressureLevel::Normal);
        // 80% of 800 = 640 → inclusive boundary → Constrained
        assert_eq!(classify_pressure(640, &b), MemoryPressureLevel::Constrained);
        assert_eq!(classify_pressure(800, &b), MemoryPressureLevel::Severe);
        assert_eq!(classify_pressure(1200, &b), MemoryPressureLevel::Emergency);
    }

    #[test]
    fn pressure_workstation_tier_budgets() {
        // Workstation: soft=12288, hard=16384
        let b = budgets(12288, 16384);
        assert_eq!(classify_pressure(5000, &b), MemoryPressureLevel::Normal);
        // 80% of 12288 = 9830.4 → 9831
        assert_eq!(
            classify_pressure(9831, &b),
            MemoryPressureLevel::Constrained
        );
        assert_eq!(classify_pressure(12288, &b), MemoryPressureLevel::Severe);
        assert_eq!(classify_pressure(16384, &b), MemoryPressureLevel::Emergency);
    }
}
