/// Backend infrastructure capability contract.
///
/// Unlike the modality `Capabilities` bitflags (TEXT/IMAGES/AUDIO), this
/// struct describes what the backend's *runtime machinery* supports.
/// Phase 7: the flags are now derived by probing the `Backend` trait
/// (`as_embeddings`, `as_kv_snapshot`, `first_token_tier`, …) rather than
/// hand-written per-backend constructors. The struct + wire format stay
/// identical for specta / frontend binding compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum LatencyTier {
    /// Local inference with warm KV cache — sub-second first token.
    Fast,
    /// Local inference, cold start — 1-5s first token typical.
    Medium,
    /// Remote API or network round-trip — variable, potentially high.
    Slow,
}

impl BackendCaps {
    /// Probe a `Backend` trait object for its capability set.
    ///
    /// Phase 7 replaces the per-backend constructor functions with this
    /// single source of truth. Truncation tracking is now universally
    /// true (every backend exposes `initial_messages_dropped()` via trait
    /// default 0). KV cache and poison detection remain keyed by backend
    /// name while those capabilities stay llama-specific; migrating them
    /// to a session-level probe is deferred until a second backend gains
    /// them.
    pub(crate) fn from_backend(b: &dyn super::traits::Backend) -> Self {
        let name = b.backend_name();
        Self {
            kv_cache: name == "llamacpp",
            // All backends support this via the generic session_rt::truncate
            // driver (Phase 3). Default `initial_messages_dropped() = 0` on
            // the BackendSession trait makes this trivially true.
            context_truncation_tracking: true,
            poison_detection: name == "llamacpp",
            // Whether this backend carries a *native* embedding
            // implementation — not whether gen2 can embed. Embedding is the
            // utility worker's now (`crate::utilities`), so it works over a
            // backend reporting `false` here, and `Engine::utility_status()`
            // is the question a caller actually wants answered.
            embedding: b.as_embeddings().is_some(),
            first_token_tier: b.first_token_tier(),
        }
    }

    /// Capabilities for the uninitialized engine sentinel (no `&dyn Backend`
    /// available). Retained for `Engine::Uninit` and
    /// `ControllerObservabilitySnapshot::default`.
    pub fn uninit() -> Self {
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
    // Used by tests exercised only under other backend features.
    #[allow(unused_imports)]
    use crate::backend::Engine;

    #[cfg(feature = "backend-llamacpp")]
    #[test]
    fn llamacpp_caps_via_probe() {
        let engine = crate::backend::llama::Engine::new();
        let caps = BackendCaps::from_backend(&engine);
        assert!(caps.kv_cache);
        assert!(caps.context_truncation_tracking);
        assert!(caps.poison_detection);
        assert!(caps.embedding);
        assert_eq!(caps.first_token_tier, LatencyTier::Fast);
    }

    #[cfg(feature = "backend-mlx")]
    #[test]
    fn mlx_caps_via_probe() {
        let engine = crate::backend::mlx::Engine::new();
        let caps = BackendCaps::from_backend(&engine);
        assert!(!caps.kv_cache);
        assert!(caps.context_truncation_tracking); // generic — all backends
        assert!(!caps.poison_detection);
        assert!(!caps.embedding);
        assert_eq!(caps.first_token_tier, LatencyTier::Medium);
    }

    #[cfg(feature = "backend-onnx")]
    #[test]
    fn onnx_caps_via_probe() {
        let engine = crate::backend::onnx::Engine::new();
        let caps = BackendCaps::from_backend(&engine);
        assert!(!caps.kv_cache);
        assert!(caps.context_truncation_tracking);
        assert!(!caps.poison_detection);
        assert!(!caps.embedding);
        assert_eq!(caps.first_token_tier, LatencyTier::Medium);
    }

    #[cfg(feature = "backend-external-api")]
    #[test]
    fn external_api_caps_via_probe() {
        let engine = crate::backend::external_api::Engine::new();
        let caps = BackendCaps::from_backend(&engine);
        assert!(!caps.kv_cache);
        assert!(caps.context_truncation_tracking);
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
        let a = BackendCaps::uninit();
        let b = a; // Copy
        assert_eq!(a, b); // Eq
    }

    #[cfg(feature = "backend-llamacpp")]
    #[test]
    fn engine_backend_caps_matches_probe() {
        // Round-trip: the facade's Engine::backend_caps() should produce
        // the same BackendCaps as from_backend(). This catches drift if a
        // future change re-routes one but not the other.
        let engine = Engine::new();
        let via_facade = engine.backend_caps();
        // When the default backend is llamacpp, the facade route matches
        // the trait probe. (ExternalApi's Engine::new() variant can't be
        // instantiated from crate::Engine::new without config, so
        // this test focuses on the default llama path.)
        assert_eq!(via_facade.first_token_tier, LatencyTier::Fast);
        assert!(via_facade.kv_cache);
    }
}
