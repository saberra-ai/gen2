//! A point-in-time snapshot of machine memory state.
//!
//! `MemorySnapshot` bundles the tier, effective budgets, current pressure
//! level, and raw RSS/available figures into one struct. It is the sole
//! input to `MemoryGovernor` — subsystems never inspect hardware numbers
//! directly.

use serde::Serialize;

use super::memory_policy::MemoryBudgets;
use super::memory_pressure::MemoryPressureLevel;
use super::memory_tier::MachineMemoryTier;

/// Point-in-time view of machine memory state.
///
/// Construct via `MemorySnapshot::new` (policy is applied automatically)
/// or assemble manually for tests.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct MemorySnapshot {
    /// Coarse machine classification (set once at startup).
    pub tier: MachineMemoryTier,
    /// Effective per-subsystem budgets derived from tier + available RAM.
    pub budgets: MemoryBudgets,
    /// Current pressure level derived from `estimated_process_mb` vs budgets.
    pub pressure: MemoryPressureLevel,
    /// Estimated current process RSS in MiB (best-effort).
    pub estimated_process_mb: u64,
    /// Currently available system RAM in MiB at snapshot time.
    pub available_memory_mb: u64,
}

impl MemorySnapshot {
    /// Build a snapshot by running all policy functions over `input`.
    ///
    /// This is the canonical constructor for non-test callsites.
    pub fn new(input: &super::memory_policy::MemoryPolicyInput, estimated_process_mb: u64) -> Self {
        use super::memory_policy::{detect_machine_tier, effective_budgets};
        use super::memory_pressure::classify_pressure;

        let tier = detect_machine_tier(input);
        let budgets = effective_budgets(input);
        let pressure = classify_pressure(estimated_process_mb, &budgets);

        Self {
            tier,
            budgets,
            pressure,
            estimated_process_mb,
            available_memory_mb: input.available_memory_mb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::memory_policy::MemoryPolicyInput;

    fn make_input(total_mb: u64, avail_mb: u64) -> MemoryPolicyInput {
        MemoryPolicyInput {
            total_memory_mb: total_mb,
            available_memory_mb: avail_mb,
            is_mobile: false,
        }
    }

    #[test]
    fn snapshot_new_desktop_mainstream_normal() {
        // 10 GiB total (8192..16383 → DesktopMainstream)
        let input = make_input(10240, 8000);
        let snap = MemorySnapshot::new(&input, 500);
        assert_eq!(snap.tier, MachineMemoryTier::DesktopMainstream);
        assert_eq!(snap.pressure, MemoryPressureLevel::Normal);
        assert_eq!(snap.estimated_process_mb, 500);
    }

    #[test]
    fn snapshot_new_detects_severe_pressure() {
        // 10 GiB machine, ample free RAM → base budgets apply (soft=3072)
        let input = make_input(10240, 8000);
        let snap = MemorySnapshot::new(&input, 3500);
        // 3500 > soft (3072) → Severe
        assert_eq!(snap.pressure, MemoryPressureLevel::Severe);
    }

    #[test]
    fn snapshot_fields_are_consistent() {
        let input = make_input(8192, 4000);
        let snap = MemorySnapshot::new(&input, 0);
        // Budgets hard limit must be >= soft
        assert!(snap.budgets.process_hard_limit_mb >= snap.budgets.process_soft_limit_mb);
        // Available stored correctly
        assert_eq!(snap.available_memory_mb, 4000);
    }
}
