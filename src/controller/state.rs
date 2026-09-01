use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::backend::Engine;
use crate::residency::ResidencyInventory;
use crate::residency_policy::ResidencyPolicy;

use super::ChatRuntime;
use super::ControllerConfig;
use super::metrics::ControllerMetrics;

/// Mutable controller thread bundle (engine + sessions + cached caps).
pub struct ControllerState {
    pub(super) engine: Engine,
    pub(super) chats: HashMap<String, ChatRuntime>,
    pub(super) residency: ResidencyInventory,
    pub(super) residency_policy: ResidencyPolicy,
    pub(super) caps: crate::backend::caps::BackendCaps,
    pub(super) config: ControllerConfig,
    pub(super) metrics: Arc<ControllerMetrics>,
    /// Whole-model on-disk byte size of the currently-loaded primary LLM,
    /// captured from `metadata().len()` at `LoadModel` time. Feeds the
    /// flock fit gate via `RuntimeSnapshot::loaded_model_file_bytes` so a
    /// peer can advertise the real footprint of what it has loaded.
    /// `None` when no model is loaded, or when the model path is a directory
    /// bundle (MLX/ONNX) whose file size isn't a single `metadata().len()`
    /// — honest `None` over a fabricated number (ADR doctrine).
    pub(super) loaded_model_file_bytes: Option<u64>,
    /// Unix seconds of the last observed LLM activity (a chat generating
    /// or a session starting). Drives keepwarm idle-unload — the
    /// residency slot's `last_used` reflects load time, not chat time.
    pub(super) last_llm_activity_unix: i64,
    /// Residency identity (name, estimated MB) of an LLM that keepwarm
    /// idle-unloaded, so wake-on-demand can re-admit the same runtime.
    pub(super) idle_unloaded_llm: Option<(String, u64)>,
}

impl ControllerState {
    #[allow(dead_code)]
    pub(crate) fn new(config: ControllerConfig) -> Self {
        Self::with_engine(Engine::new(), config)
    }

    /// As [`Self::new`], but over an engine the caller built.
    ///
    /// The seam tests use to run the controller against a scripted backend.
    pub(crate) fn with_engine(mut engine: Engine, config: ControllerConfig) -> Self {
        let caps = engine.backend_caps();
        Self {
            engine,
            chats: HashMap::new(),
            residency: ResidencyInventory::default(),
            residency_policy: ResidencyPolicy::default(),
            caps,
            config,
            metrics: Arc::new(ControllerMetrics::default()),
            loaded_model_file_bytes: None,
            last_llm_activity_unix: chrono::Utc::now().timestamp(),
            idle_unloaded_llm: None,
        }
    }

    /// On-disk byte size of `path` for a regular file, or `None` for a
    /// directory bundle (MLX safetensors / ONNX dir) or an unreadable path.
    /// Mirrors the `metadata()` read the engine already does at load
    /// (`gen2::engine::validate_model_file`); we keep only the file case so
    /// the number is real, never a partial directory total.
    pub(super) fn model_file_bytes_of(path: &std::path::Path) -> Option<u64> {
        let md = std::fs::metadata(path).ok()?;
        if md.is_dir() {
            return None;
        }
        Some(md.len())
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
    use crate::controller::ControllerMetricsSnapshot;

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
