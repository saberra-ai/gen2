use super::residency::ResidencyInventory;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidencyStats {
    pub total_estimated_mb: u64,
    pub loaded_runtime_count: usize,
}

impl ResidencyStats {
    pub fn from_inventory(inventory: &ResidencyInventory) -> Self {
        Self {
            total_estimated_mb: inventory.total_estimated_mb(),
            loaded_runtime_count: inventory.loaded_runtime_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen2::residency::{ResidencyInventory, ResidentRuntime, RuntimeKind};

    #[test]
    fn stats_mirror_inventory() {
        let inventory = ResidencyInventory {
            llm: Some(ResidentRuntime::new(RuntimeKind::Llm, "llm", 1024, 1)),
            embedder: Some(ResidentRuntime::new(RuntimeKind::Embedder, "embed", 256, 2)),
            stt: None,
            tts: None,
        };
        let stats = ResidencyStats::from_inventory(&inventory);
        assert_eq!(stats.total_estimated_mb, 1280);
        assert_eq!(stats.loaded_runtime_count, 2);
    }
}
