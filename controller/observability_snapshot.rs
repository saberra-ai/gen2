//! Unified controller introspection (PR8): policy caps, delivery metrics, and active runtimes.
//!
//! One atomic snapshot for remote status and debugging — complements PR6/PR7 single-field queries.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::gen2::backend::BackendCaps;

use super::config::ControllerConfig;
use super::metrics::ControllerMetricsSnapshot;
use super::runtime_snapshot::ControllerRuntimeSnapshot;
use super::state::ControllerState;

/// Serializable policy row: effective `ControllerConfig` fields plus live counts and backend caps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ControllerPolicySnapshot {
    pub max_active_chats: usize,
    pub generation_timeout_ms: u64,
    pub event_channel_capacity: usize,
    pub tick_idle_ms: u64,
    pub active_chats: usize,
    pub backend_caps: BackendCaps,
}

impl ControllerPolicySnapshot {
    pub fn from_state(state: &ControllerState) -> Self {
        Self::from_config_counts_caps(&state.config, state.chats.len(), state.caps)
    }

    pub fn from_config_counts_caps(
        config: &ControllerConfig,
        active_chats: usize,
        backend_caps: BackendCaps,
    ) -> Self {
        Self {
            max_active_chats: config.max_active_chats,
            generation_timeout_ms: duration_ms(config.generation_timeout),
            event_channel_capacity: config.event_channel_capacity,
            tick_idle_ms: duration_ms(config.tick_idle),
            active_chats,
            backend_caps,
        }
    }
}

fn duration_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Metrics + runtime + policy in one consistent read from the controller thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ControllerObservabilitySnapshot {
    pub policy: ControllerPolicySnapshot,
    pub metrics: ControllerMetricsSnapshot,
    pub runtime: ControllerRuntimeSnapshot,
}

impl Default for ControllerObservabilitySnapshot {
    fn default() -> Self {
        Self {
            policy: ControllerPolicySnapshot::from_config_counts_caps(
                &ControllerConfig::default(),
                0,
                BackendCaps::uninit(),
            ),
            metrics: ControllerMetricsSnapshot::default(),
            runtime: ControllerRuntimeSnapshot::default(),
        }
    }
}

pub(super) fn build_observability_snapshot(
    state: &ControllerState,
) -> ControllerObservabilitySnapshot {
    ControllerObservabilitySnapshot {
        policy: ControllerPolicySnapshot::from_state(state),
        metrics: state.metrics.snapshot(),
        runtime: super::runtime_snapshot::build_runtime_snapshot(state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen2::controller::ControllerConfig;

    #[test]
    fn empty_observability_snapshot_roundtrips_json() {
        let snap = ControllerObservabilitySnapshot::default();
        let json = serde_json::to_string(&snap).expect("serialize");
        let back: ControllerObservabilitySnapshot =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(snap, back);
    }

    #[test]
    fn policy_snapshot_reflects_config() {
        let config = ControllerConfig {
            max_active_chats: 7,
            generation_timeout: Duration::from_secs(30),
            tick_idle: Duration::from_millis(5),
            ..Default::default()
        };
        let caps = BackendCaps::llamacpp();
        let p = ControllerPolicySnapshot::from_config_counts_caps(&config, 2, caps);
        assert_eq!(p.max_active_chats, 7);
        assert_eq!(p.generation_timeout_ms, 30_000);
        assert_eq!(p.tick_idle_ms, 5);
        assert_eq!(p.active_chats, 2);
        assert_eq!(p.backend_caps, caps);
    }
}
