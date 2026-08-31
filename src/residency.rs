use crate::ResidencyPolicy;
use crate::memory::{MemoryGovernor, MemoryPressureLevel};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum RuntimeKind {
    Llm,
    Embedder,
    Stt,
    Tts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ResidentRuntime {
    pub kind: RuntimeKind,
    pub name: String,
    pub estimated_resident_mb: u64,
    pub last_used_unix_secs: i64,
}

impl ResidentRuntime {
    pub fn new(
        kind: RuntimeKind,
        name: impl Into<String>,
        estimated_resident_mb: u64,
        last_used_unix_secs: i64,
    ) -> Self {
        Self {
            kind,
            name: name.into(),
            estimated_resident_mb,
            last_used_unix_secs,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ResidencyInventory {
    pub llm: Option<ResidentRuntime>,
    pub embedder: Option<ResidentRuntime>,
    pub stt: Option<ResidentRuntime>,
    pub tts: Option<ResidentRuntime>,
}

impl ResidencyInventory {
    pub fn total_estimated_mb(&self) -> u64 {
        [
            self.llm.as_ref(),
            self.embedder.as_ref(),
            self.stt.as_ref(),
            self.tts.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(|runtime| runtime.estimated_resident_mb)
        .sum()
    }

    pub fn loaded_runtime_count(&self) -> usize {
        [
            self.llm.as_ref(),
            self.embedder.as_ref(),
            self.stt.as_ref(),
            self.tts.as_ref(),
        ]
        .into_iter()
        .flatten()
        .count()
    }

    pub fn can_admit(&self, kind: RuntimeKind, extra_mb: u64, governor: &MemoryGovernor) -> bool {
        if self.slot(kind).is_some() {
            return false;
        }
        let projected_total = self.total_estimated_mb().saturating_add(extra_mb);
        projected_total <= governor.budgets().inference_resident_mb
            && governor.can_load_additional_model(extra_mb)
    }

    pub fn admit(&mut self, runtime: ResidentRuntime, governor: &MemoryGovernor) -> bool {
        let kind = runtime.kind;
        if !self.can_admit(kind, runtime.estimated_resident_mb, governor) {
            return false;
        }
        *self.slot_mut(kind) = Some(runtime);
        true
    }

    pub fn unload(&mut self, kind: RuntimeKind) -> Option<ResidentRuntime> {
        self.slot_mut(kind).take()
    }

    pub fn touch(&mut self, kind: RuntimeKind, last_used_unix_secs: i64) {
        if let Some(runtime) = self.slot_mut(kind).as_mut() {
            runtime.last_used_unix_secs = last_used_unix_secs;
        }
    }

    pub fn replace(&mut self, runtime: ResidentRuntime) -> Option<ResidentRuntime> {
        self.slot_mut(runtime.kind).replace(runtime)
    }

    pub fn evict_idle_helpers(
        &mut self,
        now_unix_secs: i64,
        policy: &ResidencyPolicy,
    ) -> Vec<ResidentRuntime> {
        let mut evicted = Vec::new();
        for kind in [RuntimeKind::Embedder, RuntimeKind::Stt, RuntimeKind::Tts] {
            let should_evict = self.slot(kind).as_ref().is_some_and(|runtime| {
                now_unix_secs.saturating_sub(runtime.last_used_unix_secs)
                    >= policy.helper_idle_timeout_secs as i64
            });
            if should_evict && let Some(runtime) = self.unload(kind) {
                evicted.push(runtime);
            }
        }
        evicted
    }

    pub fn unload_for_pressure(
        &mut self,
        governor: &MemoryGovernor,
        active_foreground: Option<RuntimeKind>,
    ) -> Vec<ResidentRuntime> {
        let mut evicted = Vec::new();
        if governor.pressure() >= MemoryPressureLevel::Constrained {
            for kind in [RuntimeKind::Embedder, RuntimeKind::Stt, RuntimeKind::Tts] {
                if active_foreground == Some(kind) {
                    continue;
                }
                if let Some(runtime) = self.unload(kind) {
                    evicted.push(runtime);
                }
            }
        }
        if governor.pressure() >= MemoryPressureLevel::Severe
            && active_foreground != Some(RuntimeKind::Llm)
            && let Some(runtime) = self.unload(RuntimeKind::Llm)
        {
            evicted.push(runtime);
        }
        evicted
    }

    fn slot(&self, kind: RuntimeKind) -> &Option<ResidentRuntime> {
        match kind {
            RuntimeKind::Llm => &self.llm,
            RuntimeKind::Embedder => &self.embedder,
            RuntimeKind::Stt => &self.stt,
            RuntimeKind::Tts => &self.tts,
        }
    }

    fn slot_mut(&mut self, kind: RuntimeKind) -> &mut Option<ResidentRuntime> {
        match kind {
            RuntimeKind::Llm => &mut self.llm,
            RuntimeKind::Embedder => &mut self.embedder,
            RuntimeKind::Stt => &mut self.stt,
            RuntimeKind::Tts => &mut self.tts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{
        MachineMemoryTier, MemoryBudgets, MemoryGovernor, MemoryPressureLevel, MemorySnapshot,
    };

    fn governor() -> MemoryGovernor {
        MemoryGovernor::new(MemorySnapshot {
            tier: MachineMemoryTier::DesktopMainstream,
            budgets: MemoryBudgets {
                process_soft_limit_mb: 3_072,
                process_hard_limit_mb: 4_096,
                search_working_set_mb: 400,
                kg_derived_state_mb: 200,
                ingestion_peak_mb: 800,
                inference_resident_mb: 1_536,
                multimodal_peak_mb: 768,
            },
            pressure: MemoryPressureLevel::Normal,
            estimated_process_mb: 512,
            available_memory_mb: 8_192,
        })
    }

    #[test]
    fn second_llm_admission_denied() {
        let governor = governor();
        let mut inventory = ResidencyInventory::default();
        assert!(inventory.admit(
            ResidentRuntime::new(RuntimeKind::Llm, "a", 1024, 1),
            &governor
        ));
        assert!(!inventory.can_admit(RuntimeKind::Llm, 256, &governor));
    }

    #[test]
    fn second_embedder_admission_denied() {
        let governor = governor();
        let mut inventory = ResidencyInventory::default();
        assert!(inventory.admit(
            ResidentRuntime::new(RuntimeKind::Embedder, "embed-a", 256, 1),
            &governor
        ));
        assert!(!inventory.can_admit(RuntimeKind::Embedder, 128, &governor));
    }

    #[test]
    fn helper_runtime_must_fit_in_inference_budget() {
        let governor = governor();
        let mut inventory = ResidencyInventory::default();
        assert!(inventory.admit(
            ResidentRuntime::new(RuntimeKind::Llm, "llm", 1024, 1),
            &governor
        ));
        assert!(inventory.can_admit(RuntimeKind::Embedder, 256, &governor));
        assert!(!inventory.can_admit(RuntimeKind::Tts, 700, &governor));
    }

    #[test]
    fn helper_runtime_unloads_after_idle_timeout() {
        let mut inventory = ResidencyInventory {
            embedder: Some(ResidentRuntime::new(
                RuntimeKind::Embedder,
                "embed",
                256,
                100,
            )),
            ..Default::default()
        };
        let evicted = inventory.evict_idle_helpers(
            500,
            &ResidencyPolicy {
                helper_idle_timeout_secs: 300,
                llm_swap_requires_unload: true,
            },
        );
        assert_eq!(evicted.len(), 1);
        assert!(inventory.embedder.is_none());
    }

    #[test]
    fn pressure_triggered_unload_removes_helpers() {
        let base_governor = governor();
        let mut inventory = ResidencyInventory {
            llm: Some(ResidentRuntime::new(RuntimeKind::Llm, "llm", 1024, 1)),
            embedder: Some(ResidentRuntime::new(RuntimeKind::Embedder, "embed", 256, 1)),
            ..Default::default()
        };
        let governor = MemoryGovernor::new(MemorySnapshot {
            tier: MachineMemoryTier::DesktopMainstream,
            budgets: base_governor.snapshot().budgets.clone(),
            pressure: MemoryPressureLevel::Constrained,
            estimated_process_mb: 2_800,
            available_memory_mb: 2_048,
        });
        let evicted = inventory.unload_for_pressure(&governor, Some(RuntimeKind::Llm));
        assert_eq!(evicted.len(), 1);
        assert!(inventory.embedder.is_none());
        assert!(inventory.llm.is_some());
    }

    #[test]
    fn severe_pressure_preserves_foreground_runtime() {
        let base_governor = governor();
        let mut inventory = ResidencyInventory {
            llm: Some(ResidentRuntime::new(RuntimeKind::Llm, "llm", 1024, 1)),
            embedder: Some(ResidentRuntime::new(RuntimeKind::Embedder, "embed", 256, 1)),
            ..Default::default()
        };
        let governor = MemoryGovernor::new(MemorySnapshot {
            tier: MachineMemoryTier::DesktopMainstream,
            budgets: base_governor.snapshot().budgets.clone(),
            pressure: MemoryPressureLevel::Severe,
            estimated_process_mb: 3_500,
            available_memory_mb: 1_024,
        });
        let _ = inventory.unload_for_pressure(&governor, Some(RuntimeKind::Llm));
        assert!(inventory.llm.is_some());
        assert!(inventory.embedder.is_none());
    }
}
