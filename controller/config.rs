use std::time::Duration;

use crate::gen2::generation::GenSpec;

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
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            max_active_chats: 3,
            generation_timeout: Duration::from_secs(120),
            event_channel_capacity: 512,
            tick_idle: Duration::from_millis(2),
        }
    }
}

impl ControllerConfig {
    /// Returns the default GenSpec for a system task.
    ///
    /// Uses exhaustive match so adding a new `SystemTask` variant
    /// produces a compile error until defaults are defined.
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
            // Compact: summarization is quality-sensitive. These defaults were tuned
            // for coherent multi-turn summaries. Lower temp = more faithful, higher
            // tokens = less truncation. Change with benchmarks, not intuition.
            SystemTask::Compact => GenSpec {
                max_tokens: Some(512),
                temperature: Some(0.3),
                ..Default::default()
            },
            // Stance & EntityExtract: low temp for structured extraction accuracy.
            SystemTask::Stance | SystemTask::EntityExtract => GenSpec {
                max_tokens: Some(512),
                temperature: Some(0.1),
                ..Default::default()
            },
            SystemTask::Answer => GenSpec {
                max_tokens: Some(1024),
                temperature: Some(0.3),
                ..Default::default()
            },
            SystemTask::Triples => GenSpec {
                max_tokens: Some(1024),
                temperature: Some(0.1),
                ..Default::default()
            },
            SystemTask::TopicLabel => GenSpec {
                max_tokens: Some(100),
                temperature: Some(0.3),
                ..Default::default()
            },
            SystemTask::QueryUnderstand => GenSpec {
                max_tokens: Some(256),
                temperature: Some(0.1),
                ..Default::default()
            },
            SystemTask::Contradiction => GenSpec {
                max_tokens: Some(512),
                temperature: Some(0.2),
                ..Default::default()
            },
            SystemTask::Summary => GenSpec {
                max_tokens: Some(120),
                temperature: Some(0.3),
                ..Default::default()
            },
            // QueryRewrite: short output (one standalone query), low temp for
            // faithful coreference resolution, top_p=0.9 for diversity on
            // pronoun ambiguities. Calibrated on QReCC in Phase 0 probe —
            // 104ms/query on CPU with `resources/model.gguf`.
            SystemTask::QueryRewrite => GenSpec {
                max_tokens: Some(80),
                temperature: Some(0.1),
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
    }

    #[test]
    fn system_task_specs_match_original_values() {
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

        let stance = config.system_task_spec(&SystemTask::Stance);
        assert_eq!(stance.max_tokens, Some(512));
        assert_eq!(stance.temperature, Some(0.1));

        let entity = config.system_task_spec(&SystemTask::EntityExtract);
        assert_eq!(entity.max_tokens, Some(512));
        assert_eq!(entity.temperature, Some(0.1));

        let answer = config.system_task_spec(&SystemTask::Answer);
        assert_eq!(answer.max_tokens, Some(1024));
        assert_eq!(answer.temperature, Some(0.3));

        let triples = config.system_task_spec(&SystemTask::Triples);
        assert_eq!(triples.max_tokens, Some(1024));
        assert_eq!(triples.temperature, Some(0.1));

        let topic = config.system_task_spec(&SystemTask::TopicLabel);
        assert_eq!(topic.max_tokens, Some(100));
        assert_eq!(topic.temperature, Some(0.3));

        let query = config.system_task_spec(&SystemTask::QueryUnderstand);
        assert_eq!(query.max_tokens, Some(256));
        assert_eq!(query.temperature, Some(0.1));

        let contradiction = config.system_task_spec(&SystemTask::Contradiction);
        assert_eq!(contradiction.max_tokens, Some(512));
        assert_eq!(contradiction.temperature, Some(0.2));

        let summary = config.system_task_spec(&SystemTask::Summary);
        assert_eq!(summary.max_tokens, Some(120));
        assert_eq!(summary.temperature, Some(0.3));

        let rewrite = config.system_task_spec(&SystemTask::QueryRewrite);
        assert_eq!(rewrite.max_tokens, Some(80));
        assert_eq!(rewrite.temperature, Some(0.1));
    }

    /// Verify that SystemTask::default_gen_spec() delegates to config correctly.
    #[test]
    fn default_gen_spec_delegates_to_config() {
        let config = ControllerConfig::default();
        // Spot-check a few variants
        assert_eq!(
            SystemTask::Title.default_gen_spec().max_tokens,
            config.system_task_spec(&SystemTask::Title).max_tokens
        );
        assert_eq!(
            SystemTask::Answer.default_gen_spec().temperature,
            config.system_task_spec(&SystemTask::Answer).temperature
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
