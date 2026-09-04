use std::sync::Arc;
use std::time::Duration;

use crate::advanced::BackendPlugin;
use crate::generation::GenSpec;

use super::SystemTask;

/// Configuration for the inference controller's policy decisions.
///
/// All defaults match the previously hardcoded values — changing this struct's
/// `Default` impl is a behavior change and should be benchmarked.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ControllerConfig {
    /// Maximum number of concurrent chat sessions before LRU eviction kicks in.
    pub max_active_chats: usize,

    /// If a single generation takes longer than this without completing,
    /// the controller emits an error and stops it.
    ///
    /// Validated on M1 MacBook Air 8GB with 7B Q4_K_M — 90s worst case
    /// observed. 120s gives headroom for larger contexts and slower hardware.
    pub generation_timeout: Duration,

    /// Capacity for bounded event channels to callers.
    /// Large enough to absorb burst without blocking the controller,
    /// small enough to bound memory if the receiver stalls.
    pub event_channel_capacity: usize,

    /// How long the controller sleeps between ticks when no chats are active.
    /// Lower = less latency, higher = less CPU waste.
    pub tick_idle: Duration,

    /// Backends brought by the consumer, asked in order before any built-in
    /// routing rule. Empty by default. See [`crate::advanced::plugin`].
    ///
    /// Not serialized: a plugin is a factory, not a value.
    #[serde(skip)]
    pub plugins: Vec<Arc<BackendPlugin>>,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            max_active_chats: 3,
            generation_timeout: Duration::from_secs(120),
            event_channel_capacity: 512,
            tick_idle: Duration::from_millis(2),
            plugins: Vec::new(),
        }
    }
}

impl ControllerConfig {
    /// Sampling defaults for a background task.
    ///
    /// The named tasks get specs tuned for what they are: a title is short and
    /// nearly deterministic, suggestions want some spread, a compaction
    /// summary has to stay faithful across a lot of tokens.
    ///
    /// [`SystemTask::Custom`] gets nothing tuned, because nothing here knows
    /// what it is. Pass your own spec to
    /// [`InferenceHandle::system_infer_with`](crate::InferenceHandle::system_infer_with).
    pub fn system_task_spec(&self, task: &SystemTask) -> GenSpec {
        match task {
            SystemTask::Title => GenSpec {
                max_tokens: Some(50),
                temperature: Some(0.3),
                ..Default::default()
            },
            SystemTask::Suggestions => GenSpec {
                max_tokens: Some(256),
                temperature: Some(0.7),
                ..Default::default()
            },
            // Compaction is quality-sensitive. Lower temperature keeps the
            // summary faithful, the larger budget keeps it from truncating
            // mid-thought. Change these with benchmarks, not intuition.
            SystemTask::Compact => GenSpec {
                max_tokens: Some(512),
                temperature: Some(0.3),
                ..Default::default()
            },
            SystemTask::Summary => GenSpec {
                max_tokens: Some(120),
                temperature: Some(0.3),
                ..Default::default()
            },
            SystemTask::Custom(_) => GenSpec {
                max_tokens: Some(512),
                temperature: Some(0.3),
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_match_original_constants() {
        let config = ControllerConfig::default();
        assert_eq!(config.max_active_chats, 3);
        assert_eq!(config.generation_timeout, Duration::from_secs(120));
        assert_eq!(config.event_channel_capacity, 512);
        assert_eq!(config.tick_idle, Duration::from_millis(2));
        assert!(config.plugins.is_empty());
    }

    #[test]
    fn named_tasks_keep_their_tuning() {
        let config = ControllerConfig::default();

        let title = config.system_task_spec(&SystemTask::Title);
        assert_eq!(title.max_tokens, Some(50));
        assert_eq!(title.temperature, Some(0.3));

        let suggestions = config.system_task_spec(&SystemTask::Suggestions);
        assert_eq!(suggestions.max_tokens, Some(256));
        assert_eq!(suggestions.temperature, Some(0.7));

        let compact = config.system_task_spec(&SystemTask::Compact);
        assert_eq!(compact.max_tokens, Some(512));
        assert_eq!(compact.temperature, Some(0.3));

        let summary = config.system_task_spec(&SystemTask::Summary);
        assert_eq!(summary.max_tokens, Some(120));
        assert_eq!(summary.temperature, Some(0.3));
    }

    /// A custom task is the host's, so its spec is deliberately generic. If
    /// this ever starts matching a named task's tuning, the enum has grown a
    /// domain opinion it has no way to justify.
    #[test]
    fn a_custom_task_gets_nothing_tuned() {
        let config = ControllerConfig::default();
        let a = config.system_task_spec(&SystemTask::custom("triples"));
        let b = config.system_task_spec(&SystemTask::custom("contextual-prefix"));
        assert_eq!(a.max_tokens, b.max_tokens);
        assert_eq!(a.temperature, b.temperature);
        assert_ne!(
            a.max_tokens,
            config.system_task_spec(&SystemTask::Title).max_tokens
        );
    }

    /// `default_gen_spec` is a thin delegate to the default config.
    #[test]
    fn default_gen_spec_delegates_to_config() {
        let config = ControllerConfig::default();
        assert_eq!(
            SystemTask::Title.default_gen_spec().max_tokens,
            config.system_task_spec(&SystemTask::Title).max_tokens
        );
        assert_eq!(
            SystemTask::Compact.default_gen_spec().temperature,
            config.system_task_spec(&SystemTask::Compact).temperature
        );
    }

    #[test]
    fn config_is_clone_debug_serialize() {
        let config = ControllerConfig::default();
        let cloned = config.clone();
        // Debug
        let debug_str = format!("{:?}", cloned);
        assert!(debug_str.contains("ControllerConfig"));
        // Serialize
        let json = serde_json::to_string(&cloned).expect("should serialize");
        assert!(json.contains("max_active_chats"));
    }
}
