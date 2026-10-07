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

/// Desktop admission preferences. Paging is opt-in and requires a reliable
/// commit probe; it may make inference slower. Mobile ignores these settings.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct DesktopMemoryPolicy {
    /// Physical headroom in MiB; None uses the automatic reserve. Minimum 256.
    pub system_reserve_mb: Option<u64>,
    /// Extra admission allowance beyond physical headroom, in MiB. Capped at
    /// 1/8 of RAM and 2048 MiB, and disabled below 256 MiB available RAM.
    pub max_paging_mb: u64,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct DesktopMemoryStatus {
    pub system_reserve_mb: u64,
    pub paging_allowance_mb: u64,
    /// Additional memory this process can commit, not pagefile size or RAM.
    pub available_commit_mb: Option<u64>,
}

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
    pub desktop: Option<DesktopMemoryStatus>,
}

impl MemorySnapshot {
    /// Build a snapshot by running all policy functions over `input`.
    ///
    /// This is the canonical constructor for non-test callsites.
    pub fn new(input: &super::memory_policy::MemoryPolicyInput, estimated_process_mb: u64) -> Self {
        Self::with_desktop_policy(
            input,
            estimated_process_mb,
            None,
            DesktopMemoryPolicy::default(),
        )
    }

    pub fn with_desktop_policy(
        input: &super::memory_policy::MemoryPolicyInput,
        estimated_process_mb: u64,
        available_commit_mb: Option<u64>,
        policy: DesktopMemoryPolicy,
    ) -> Self {
        use super::memory_policy::{detect_machine_tier, effective_budgets};
        use super::memory_pressure::classify_pressure;

        let tier = detect_machine_tier(input);
        let mut budgets = effective_budgets(input);
        let mut desktop = None;
        if !input.is_mobile {
            // OS-visible available memory excludes our resident pages. Add
            // those back when setting a total process ceiling, so allocating
            // weights does not itself make the ceiling fall a second time.
            // Keep one explicit system reserve, not stacked percentages of
            // free RAM for every subsystem. Mobile retains its OS policy.
            let reserve = policy
                .system_reserve_mb
                .unwrap_or_else(|| desktop_system_reserve_mb(input.total_memory_mb))
                .max(256)
                .min(input.total_memory_mb);
            let paging = if available_commit_mb.is_some() && input.available_memory_mb >= 256 {
                policy
                    .max_paging_mb
                    .min(input.total_memory_mb / 8)
                    .min(2048)
            } else {
                0
            };
            let capacity = estimated_process_mb
                .saturating_add(input.available_memory_mb)
                .saturating_add(paging);
            // Commit is a separate allocation constraint, never added to RAM.
            // Conservatively price the whole incremental estimate against it.
            let commit_cap = available_commit_mb.map_or(u64::MAX, |commit| {
                estimated_process_mb.saturating_add(commit.saturating_sub(reserve))
            });
            let process_cap = input.total_memory_mb / 4 * 3;
            budgets.process_soft_limit_mb = capacity
                .saturating_sub(reserve)
                .min(process_cap)
                .min(commit_cap);
            budgets.process_hard_limit_mb = capacity
                .saturating_sub(reserve / 2)
                .min(input.total_memory_mb.saturating_sub(reserve / 2))
                .min(commit_cap)
                .max(budgets.process_soft_limit_mb);
            // Admission estimates include weights, context and working buffers.
            // The process check separately accounts for non-inference RSS.
            budgets.inference_resident_mb = budgets.process_soft_limit_mb;
            desktop = Some(DesktopMemoryStatus {
                system_reserve_mb: reserve,
                paging_allowance_mb: paging,
                available_commit_mb,
            });
        }
        let pressure = classify_pressure(estimated_process_mb, &budgets);

        Self {
            tier,
            budgets,
            pressure,
            estimated_process_mb,
            available_memory_mb: input.available_memory_mb,
            desktop,
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

    fn flexible(available: u64, process: u64, commit: Option<u64>) -> MemorySnapshot {
        MemorySnapshot::with_desktop_policy(
            &make_input(16384, available),
            process,
            commit,
            DesktopMemoryPolicy {
                system_reserve_mb: Some(512),
                max_paging_mb: 1024,
            },
        )
    }

    #[test]
    fn bounded_paging_admits_small_image_bundle_with_commit_backing() {
        let strict = MemorySnapshot::new(&make_input(16384, 2400), 100);
        assert!(!crate::memory::MemoryGovernor::new(strict).can_load_additional_model(2718));
        let flexible = flexible(2400, 100, Some(8000));
        assert_eq!(flexible.budgets.process_soft_limit_mb, 3012);
        let governor = crate::memory::MemoryGovernor::new(flexible);
        assert!(governor.can_load_additional_model(2718));
        assert!(!governor.can_load_additional_model(2912));
    }

    #[test]
    fn paging_never_substitutes_for_commit_or_emergency_physical_memory() {
        for commit in [None, Some(0), Some(3000)] {
            assert!(
                !crate::memory::MemoryGovernor::new(flexible(2400, 100, commit))
                    .can_load_additional_model(2718)
            );
        }
        for available in [0, 255] {
            let snap = flexible(available, 3000, Some(8000));
            assert_eq!(snap.desktop.as_ref().unwrap().paging_allowance_mb, 0);
            assert!(!crate::memory::MemoryGovernor::new(snap).can_load_additional_model(1));
        }
    }

    #[test]
    fn paging_is_bounded_and_loaded_pages_are_not_counted_twice() {
        let before = flexible(5000, 100, Some(10000));
        let after = flexible(2000, 3100, Some(7000));
        assert_eq!(
            before.budgets.process_soft_limit_mb,
            after.budgets.process_soft_limit_mb
        );
        let excessive = MemorySnapshot::with_desktop_policy(
            &make_input(8192, 7000),
            100,
            Some(u64::MAX),
            DesktopMemoryPolicy {
                system_reserve_mb: Some(0),
                max_paging_mb: u64::MAX,
            },
        );
        assert_eq!(excessive.budgets.process_soft_limit_mb, 6144);
        let status = excessive.desktop.unwrap();
        assert_eq!(status.paging_allowance_mb, 1024);
        assert_eq!(status.system_reserve_mb, 256);
    }

    #[test]
    fn desktop_options_cannot_change_mobile_budgets() {
        let input = MemoryPolicyInput {
            is_mobile: true,
            ..make_input(8192, 4000)
        };
        let original = MemorySnapshot::new(&input, 500);
        let configured = MemorySnapshot::with_desktop_policy(
            &input,
            500,
            Some(8000),
            DesktopMemoryPolicy {
                system_reserve_mb: Some(256),
                max_paging_mb: 2048,
            },
        );
        assert_eq!(
            serde_json::to_value(original).unwrap(),
            serde_json::to_value(configured).unwrap()
        );
    }
}
