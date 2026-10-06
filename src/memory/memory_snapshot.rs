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
        let mut budgets = effective_budgets(input);
        if !input.is_mobile {
            // OS-visible available memory excludes our resident pages. Add
            // those back when setting a total process ceiling, so allocating
            // weights does not itself make the ceiling fall a second time.
            // Keep one explicit system reserve, not stacked percentages of
            // free RAM for every subsystem. Mobile retains its OS policy.
            let reserve = desktop_system_reserve_mb(input.total_memory_mb);
            let capacity = estimated_process_mb.saturating_add(input.available_memory_mb);
            let process_cap = input.total_memory_mb / 4 * 3;
            budgets.process_soft_limit_mb = capacity.saturating_sub(reserve).min(process_cap);
            budgets.process_hard_limit_mb = capacity
                .saturating_sub(reserve / 2)
                .min(input.total_memory_mb.saturating_sub(reserve / 2))
                .max(budgets.process_soft_limit_mb);
            // Admission estimates include weights, context and working buffers.
            // The process check separately accounts for non-inference RSS.
            budgets.inference_resident_mb = budgets.process_soft_limit_mb;
        }
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

/// Desktop headroom heuristic: 1/16 of physical RAM, bounded to 512–4096 MiB.
/// This is reserved once, outside the model's estimated working set.
pub fn desktop_system_reserve_mb(total_mb: u64) -> u64 {
    (total_mb / 16).clamp(512, 4096)
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
        // Available memory has fallen below the explicit system reserve.
        let input = make_input(10240, 500);
        let snap = MemorySnapshot::new(&input, 3500);
        assert_eq!(snap.pressure, MemoryPressureLevel::Severe);
    }

    #[test]
    fn loading_does_not_shrink_the_desktop_ceiling_again() {
        let before = MemorySnapshot::new(&make_input(16235, 5000), 100);
        let after = MemorySnapshot::new(&make_input(16235, 2000), 3100);
        assert_eq!(
            before.budgets.process_soft_limit_mb,
            after.budgets.process_soft_limit_mb
        );
        assert_eq!(before.budgets.inference_resident_mb, 4086);
        assert!(crate::memory::MemoryGovernor::new(before).can_load_additional_model(3700));
        assert!(crate::memory::MemoryGovernor::new(after).can_load_additional_model(100));
    }

    #[test]
    fn real_pressure_and_process_cap_still_refuse_loads() {
        for available in [0, 200, 1000] {
            let snap = MemorySnapshot::new(&make_input(16235, available), 3000);
            assert!(!crate::memory::MemoryGovernor::new(snap).can_load_additional_model(1));
        }
        let snap = MemorySnapshot::new(&make_input(16384, 15000), 100);
        assert_eq!(snap.budgets.process_soft_limit_mb, 12288);
        assert!(!crate::memory::MemoryGovernor::new(snap).can_load_additional_model(12500));
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
