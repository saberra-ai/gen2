//! Memory governor — the single yes/no authority for memory-sensitive operations.
//!
//! All subsystems ask the governor before starting any operation that consumes
//! significant memory. The governor encodes the interactive-first product
//! policy: background/opportunistic work degrades before search and chat are
//! affected.
//!
//! # Policy summary
//!
//! | Decision                       | Normal | Constrained | Severe | Emergency |
//! |-------------------------------|--------|-------------|--------|-----------|
//! | `can_start_import`             | ✓      | ✓           | ✗      | ✗         |
//! | `can_start_ocr`                | ✓      | ✗           | ✗      | ✗         |
//! | `can_start_background_rebuild` | ✓      | ✗           | ✗      | ✗         |
//! | `should_skip_eager_kg`         | ✗      | ✗           | ✓      | ✓         |
//! | `should_shrink_search_candidates` | ✗   | ✓           | ✓      | ✓         |
//! | `can_load_additional_model`    | †      | †           | ✗      | ✗         |
//!
//! † Allowed only if projected RSS stays under the process soft limit.

use super::memory_policy::MemoryBudgets;
use super::memory_pressure::MemoryPressureLevel;
use super::memory_snapshot::MemorySnapshot;

/// Wraps a `MemorySnapshot` and exposes named policy decisions.
///
/// Constructed from a fresh snapshot. Callers should refresh the snapshot
/// (and thus the governor) periodically — it is not self-updating.
#[derive(Clone)]
pub struct MemoryGovernor {
    snapshot: MemorySnapshot,
}

impl MemoryGovernor {
    /// Create a governor from a pre-built snapshot.
    pub fn new(snapshot: MemorySnapshot) -> Self {
        Self { snapshot }
    }

    /// Current pressure level.
    #[inline]
    pub fn pressure(&self) -> MemoryPressureLevel {
        self.snapshot.pressure
    }

    /// Effective per-subsystem budgets.
    #[inline]
    pub fn budgets(&self) -> &MemoryBudgets {
        &self.snapshot.budgets
    }

    /// Expose the full snapshot for reporting / diagnostics.
    #[inline]
    pub fn snapshot(&self) -> &MemorySnapshot {
        &self.snapshot
    }

    // ── Subsystem decisions ──────────────────────────────────────────────────

    /// Allow a text/document import job to start.
    ///
    /// Imports are background work and are shed before interactive search or
    /// chat is affected. Allowed at Normal and Constrained; denied above.
    pub fn can_start_import(&self) -> bool {
        matches!(
            self.snapshot.pressure,
            MemoryPressureLevel::Normal | MemoryPressureLevel::Constrained
        )
    }

    /// Allow an OCR job to start.
    ///
    /// OCR is the most memory-intensive import variant (page rasterization +
    /// ONNX inference). Only allowed at Normal.
    pub fn can_start_ocr(&self) -> bool {
        self.snapshot.pressure == MemoryPressureLevel::Normal
    }

    /// Allow a background index / KG rebuild to start.
    ///
    /// Background rebuilds are purely opportunistic. Only allowed at Normal.
    pub fn can_start_background_rebuild(&self) -> bool {
        self.snapshot.pressure == MemoryPressureLevel::Normal
    }

    /// Whether to skip the eager KG enrichment step during search.
    ///
    /// Eager KG is skipped at Severe and Emergency to protect the interactive
    /// search working set.
    pub fn should_skip_eager_kg(&self) -> bool {
        matches!(
            self.snapshot.pressure,
            MemoryPressureLevel::Severe | MemoryPressureLevel::Emergency
        )
    }

    /// Whether to reduce the number of search candidates materialized.
    ///
    /// Candidate shrinking starts at Constrained — search still works but
    /// uses a smaller working set.
    pub fn should_shrink_search_candidates(&self) -> bool {
        self.snapshot.pressure >= MemoryPressureLevel::Constrained
    }

    /// Whether loading an additional model runtime is safe.
    ///
    /// Returns `false` immediately under Severe / Emergency. At Normal /
    /// Constrained, checks that the projected RSS after admission stays
    /// under the process soft limit (keeping a buffer).
    pub fn can_load_additional_model(&self, estimated_extra_mb: u64) -> bool {
        match self.snapshot.pressure {
            MemoryPressureLevel::Severe | MemoryPressureLevel::Emergency => false,
            _ => {
                let projected = self
                    .snapshot
                    .estimated_process_mb
                    .saturating_add(estimated_extra_mb);
                projected < self.snapshot.budgets.process_soft_limit_mb
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::memory_policy::{MemoryBudgets, MemoryPolicyInput};
    use crate::memory::memory_pressure::MemoryPressureLevel;
    use crate::memory::memory_snapshot::MemorySnapshot;
    use crate::memory::memory_tier::MachineMemoryTier;

    /// Build a snapshot manually, pinning the pressure level so tests are
    /// deterministic regardless of the budget-derivation logic.
    fn governor_at(pressure: MemoryPressureLevel) -> MemoryGovernor {
        // A realistic mainstream-desktop budget
        let budgets = MemoryBudgets {
            process_soft_limit_mb: 3072,
            process_hard_limit_mb: 4096,
            search_working_set_mb: 400,
            kg_derived_state_mb: 200,
            ingestion_peak_mb: 800,
            inference_resident_mb: 1536,
            multimodal_peak_mb: 768,
        };
        // Pick an estimated_process_mb consistent with the requested pressure
        let estimated_process_mb = match pressure {
            MemoryPressureLevel::Normal => 500,
            MemoryPressureLevel::Constrained => 2600, // 80–100% of soft
            MemoryPressureLevel::Severe => 3500,      // soft..hard
            MemoryPressureLevel::Emergency => 5000,   // >= hard
        };
        let snap = MemorySnapshot {
            tier: MachineMemoryTier::DesktopMainstream,
            budgets,
            pressure,
            estimated_process_mb,
            available_memory_mb: 6000,
        };
        MemoryGovernor::new(snap)
    }

    // ── can_start_import ─────────────────────────────────────────────────────

    #[test]
    fn import_allowed_at_normal() {
        assert!(governor_at(MemoryPressureLevel::Normal).can_start_import());
    }

    #[test]
    fn import_allowed_at_constrained() {
        assert!(governor_at(MemoryPressureLevel::Constrained).can_start_import());
    }

    #[test]
    fn import_denied_at_severe() {
        assert!(!governor_at(MemoryPressureLevel::Severe).can_start_import());
    }

    #[test]
    fn import_denied_at_emergency() {
        assert!(!governor_at(MemoryPressureLevel::Emergency).can_start_import());
    }

    // ── can_start_ocr ────────────────────────────────────────────────────────

    #[test]
    fn ocr_allowed_at_normal() {
        assert!(governor_at(MemoryPressureLevel::Normal).can_start_ocr());
    }

    #[test]
    fn ocr_denied_at_constrained() {
        assert!(!governor_at(MemoryPressureLevel::Constrained).can_start_ocr());
    }

    #[test]
    fn ocr_denied_at_severe() {
        assert!(!governor_at(MemoryPressureLevel::Severe).can_start_ocr());
    }

    #[test]
    fn ocr_denied_at_emergency() {
        assert!(!governor_at(MemoryPressureLevel::Emergency).can_start_ocr());
    }

    // ── can_start_background_rebuild ─────────────────────────────────────────

    #[test]
    fn background_rebuild_allowed_at_normal() {
        assert!(governor_at(MemoryPressureLevel::Normal).can_start_background_rebuild());
    }

    #[test]
    fn background_rebuild_denied_at_constrained() {
        assert!(!governor_at(MemoryPressureLevel::Constrained).can_start_background_rebuild());
    }

    #[test]
    fn background_rebuild_denied_at_severe() {
        assert!(!governor_at(MemoryPressureLevel::Severe).can_start_background_rebuild());
    }

    #[test]
    fn background_rebuild_denied_at_emergency() {
        assert!(!governor_at(MemoryPressureLevel::Emergency).can_start_background_rebuild());
    }

    // ── should_skip_eager_kg ─────────────────────────────────────────────────

    #[test]
    fn eager_kg_not_skipped_at_normal() {
        assert!(!governor_at(MemoryPressureLevel::Normal).should_skip_eager_kg());
    }

    #[test]
    fn eager_kg_not_skipped_at_constrained() {
        assert!(!governor_at(MemoryPressureLevel::Constrained).should_skip_eager_kg());
    }

    #[test]
    fn eager_kg_skipped_at_severe() {
        assert!(governor_at(MemoryPressureLevel::Severe).should_skip_eager_kg());
    }

    #[test]
    fn eager_kg_skipped_at_emergency() {
        assert!(governor_at(MemoryPressureLevel::Emergency).should_skip_eager_kg());
    }

    // ── should_shrink_search_candidates ──────────────────────────────────────

    #[test]
    fn search_candidates_not_shrunk_at_normal() {
        assert!(!governor_at(MemoryPressureLevel::Normal).should_shrink_search_candidates());
    }

    #[test]
    fn search_candidates_shrunk_at_constrained() {
        assert!(governor_at(MemoryPressureLevel::Constrained).should_shrink_search_candidates());
    }

    #[test]
    fn search_candidates_shrunk_at_severe() {
        assert!(governor_at(MemoryPressureLevel::Severe).should_shrink_search_candidates());
    }

    #[test]
    fn search_candidates_shrunk_at_emergency() {
        assert!(governor_at(MemoryPressureLevel::Emergency).should_shrink_search_candidates());
    }

    // ── can_load_additional_model ─────────────────────────────────────────────

    #[test]
    fn model_load_allowed_at_normal_with_room() {
        // process=500, soft=3072; adding 1000 → 1500 < 3072
        let g = governor_at(MemoryPressureLevel::Normal);
        assert!(g.can_load_additional_model(1000));
    }

    #[test]
    fn model_load_denied_at_normal_without_room() {
        // process=500, soft=3072; adding 3000 → 3500 >= 3072
        let g = governor_at(MemoryPressureLevel::Normal);
        assert!(!g.can_load_additional_model(3000));
    }

    #[test]
    fn model_load_allowed_at_constrained_with_room() {
        // process=2600, soft=3072; adding 400 → 3000 < 3072
        let g = governor_at(MemoryPressureLevel::Constrained);
        assert!(g.can_load_additional_model(400));
    }

    #[test]
    fn model_load_denied_at_constrained_without_room() {
        // process=2600, soft=3072; adding 600 → 3200 >= 3072
        let g = governor_at(MemoryPressureLevel::Constrained);
        assert!(!g.can_load_additional_model(600));
    }

    #[test]
    fn model_load_denied_at_severe_regardless() {
        // Even a tiny load is denied under Severe
        let g = governor_at(MemoryPressureLevel::Severe);
        assert!(!g.can_load_additional_model(1));
    }

    #[test]
    fn model_load_denied_at_emergency_regardless() {
        let g = governor_at(MemoryPressureLevel::Emergency);
        assert!(!g.can_load_additional_model(0));
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[test]
    fn governor_pressure_accessor() {
        let g = governor_at(MemoryPressureLevel::Severe);
        assert_eq!(g.pressure(), MemoryPressureLevel::Severe);
    }

    #[test]
    fn governor_budgets_accessor() {
        let g = governor_at(MemoryPressureLevel::Normal);
        assert_eq!(g.budgets().process_soft_limit_mb, 3072);
    }

    // ── Full-stack: build from MemoryPolicyInput ──────────────────────────────

    #[test]
    fn governor_from_snapshot_new() {
        let input = MemoryPolicyInput {
            total_memory_mb: 16384,
            available_memory_mb: 8000,
            is_mobile: false,
        };
        let snap = MemorySnapshot::new(&input, 500);
        let g = MemoryGovernor::new(snap);
        // 500 MB on 16 GB machine with 8 GB free → Normal
        assert_eq!(g.pressure(), MemoryPressureLevel::Normal);
        assert!(g.can_start_import());
        assert!(g.can_start_ocr());
        assert!(!g.should_skip_eager_kg());
    }
}
