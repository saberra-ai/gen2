/// Backend infrastructure capability contract.
///
/// Unlike the modality `Capabilities` bitflags (TEXT/IMAGES/AUDIO), this
/// struct describes what the backend's *runtime machinery* supports.
/// Computed once at engine load and cached — not queried per-call.
///
/// The controller queries this at session creation and branches on the result
/// rather than try-catching `FeatureUnsupported` in the hot loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCaps {
    /// Can save/restore KV cache across sessions (fast resume).
    pub kv_cache: bool,
    /// Tracks how many initial messages were dropped due to context overflow.
    pub context_truncation_tracking: bool,
    /// Detects FFI-level session poisoning (panic during decode).
    pub poison_detection: bool,
    /// Supports embedding generation via the engine.
    pub embedding: bool,
    /// Rough latency tier for first-token. Used by UI to set expectations.
    pub first_token_tier: LatencyTier,
}

/// Rough first-token latency tier. Not a precise measurement — a classification
/// that lets the UI decide whether to show a "thinking…" indicator early.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum LatencyTier {
    /// Local inference with warm KV cache — sub-second first token.
    Fast,
    /// Local inference, cold start — 1-5s first token typical.
    Medium,
    /// Remote API or network round-trip — variable, potentially high.
    Slow,
}

#[allow(dead_code)] // constructors are conditionally used based on backend features
impl BackendCaps {
    /// Capabilities for the LlamaCpp backend.
    pub(crate) fn llamacpp() -> Self {
        Self {
            kv_cache: true,
            context_truncation_tracking: true,
            poison_detection: true,
            embedding: true,
            first_token_tier: LatencyTier::Fast,
        }
    }

    /// Capabilities for the MLX backend (Apple Silicon).
    pub(crate) fn mlx() -> Self {
        Self {
            kv_cache: false,
            context_truncation_tracking: false,
            poison_detection: false,
            embedding: false,
            first_token_tier: LatencyTier::Medium,
        }
    }

    /// Capabilities for the ONNX backend.
    pub(crate) fn onnx() -> Self {
        Self {
            kv_cache: false,
            context_truncation_tracking: false,
            poison_detection: false,
            embedding: false,
            first_token_tier: LatencyTier::Medium,
        }
    }

    /// Capabilities for the external API backend.
    pub(crate) fn external_api() -> Self {
        Self {
            kv_cache: false,
            context_truncation_tracking: false,
            poison_detection: false,
            embedding: true, // most API providers support embeddings
            first_token_tier: LatencyTier::Slow,
        }
    }

    /// Capabilities for the uninitialized engine sentinel.
    pub(crate) fn uninit() -> Self {
        Self {
            kv_cache: false,
            context_truncation_tracking: false,
            poison_detection: false,
            embedding: false,
            first_token_tier: LatencyTier::Slow,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llamacpp_has_full_local_capabilities() {
        let caps = BackendCaps::llamacpp();
        assert!(caps.kv_cache);
        assert!(caps.context_truncation_tracking);
        assert!(caps.poison_detection);
        assert!(caps.embedding);
        assert_eq!(caps.first_token_tier, LatencyTier::Fast);
    }

    #[test]
    fn mlx_has_no_kv_or_poison() {
        let caps = BackendCaps::mlx();
        assert!(!caps.kv_cache);
        assert!(!caps.poison_detection);
        assert_eq!(caps.first_token_tier, LatencyTier::Medium);
    }

    #[test]
    fn external_api_is_slow_with_embeddings() {
        let caps = BackendCaps::external_api();
        assert!(!caps.kv_cache);
        assert!(!caps.poison_detection);
        assert!(caps.embedding);
        assert_eq!(caps.first_token_tier, LatencyTier::Slow);
    }

    #[test]
    fn uninit_has_nothing() {
        let caps = BackendCaps::uninit();
        assert!(!caps.kv_cache);
        assert!(!caps.context_truncation_tracking);
        assert!(!caps.poison_detection);
        assert!(!caps.embedding);
        assert_eq!(caps.first_token_tier, LatencyTier::Slow);
    }

    #[test]
    fn caps_are_copy_and_eq() {
        let a = BackendCaps::llamacpp();
        let b = a; // Copy
        assert_eq!(a, b); // Eq
    }
}
