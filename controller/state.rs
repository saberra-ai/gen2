use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::gen2::Engine;
use crate::gen2::{ResidencyInventory, ResidencyPolicy};

use super::ChatRuntime;
use super::ControllerConfig;
use super::metrics::ControllerMetrics;

/// Mutable controller thread bundle (engine + sessions + cached caps).
pub struct ControllerState {
    pub(super) engine: Engine,
    pub(super) chats: HashMap<String, ChatRuntime>,
    pub(super) residency: ResidencyInventory,
    pub(super) residency_policy: ResidencyPolicy,
    pub(super) caps: crate::gen2::backend::caps::BackendCaps,
    pub(super) config: ControllerConfig,
    pub(super) metrics: Arc<ControllerMetrics>,
}

impl ControllerState {
    pub(crate) fn new(config: ControllerConfig) -> Self {
        let mut engine = Engine::new();
        let caps = engine.backend_caps();
        Self {
            engine,
            chats: HashMap::new(),
            residency: ResidencyInventory::default(),
            residency_policy: ResidencyPolicy::default(),
            caps,
            config,
            metrics: Arc::new(ControllerMetrics::default()),
        }
    }

    pub(super) fn max_active_chats(&self) -> usize {
        self.config.max_active_chats
    }

    pub(super) fn generation_timeout(&self) -> Duration {
        self.config.generation_timeout
    }

    pub(super) fn tick_idle(&self) -> Duration {
        self.config.tick_idle
    }
}

#[cfg(test)]
mod tests {
    use super::{ControllerConfig, ControllerState};
    use crate::gen2::controller::ControllerMetricsSnapshot;

    #[test]
    fn new_controller_state_has_no_active_chats() {
        let state = ControllerState::new(ControllerConfig::default());
        assert!(state.chats.is_empty());
        assert_eq!(state.residency.loaded_runtime_count(), 0);
        assert!(state.residency_policy.llm_swap_requires_unload);
    }

    #[test]
    fn new_controller_state_metrics_start_at_zero() {
        let state = ControllerState::new(ControllerConfig::default());
        assert_eq!(
            state.metrics.snapshot(),
            ControllerMetricsSnapshot::default()
        );
    }
}
