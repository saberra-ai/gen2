//! Backend trait system — swappable backend boxes declaring capabilities via
//! trait probes rather than a boolean `BackendCaps` struct.
//!
//! - [`Backend`] is the core contract every backend (local or remote) implements.
//! - [`LocalBackend`] extends [`Backend`] for in-process inference.
//! - [`RemoteBackend`] extends [`Backend`] for out-of-process inference
//!   (external API, P2P/Flock).
//! - Optional capabilities — [`KvSnapshot`], [`Embeddings`], [`Multimodal`] —
//!   are probed via `as_*` methods on [`Backend`] / [`BackendSession`]. Default
//!   `None` = capability unsupported.
//! - [`SessionTokenizer`] powers the generic context-truncation driver in
//!   [`crate::session_rt::truncate`]. Session-scoped because chat templates
//!   depend on the session's persona/settings.

use std::sync::Arc;

use crate::engine::telemetry::HookBus;
use crate::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::generation::{GenSpec, TokenEvent};
use crate::kv::{KvLoadReport, KvLoadSpec, KvSaveSpec, KvSnapshot as KvSnapshotBlob};
use crate::session_rt::SessionSpec;
use crate::types::message::Message;

use super::caps::LatencyTier;
use super::facade::SessionId;

/// Core contract every backend — local or remote — implements.
///
/// Not `Send + Sync` — several backends (llama, MLX) hold non-thread-safe
/// FFI state. Backend state is confined to the controller's `run_loop`.
pub trait Backend: std::fmt::Debug {
    fn backend_name(&self) -> &'static str;

    fn load_model(&self, req: LoadRequest) -> Result<(), ExecError>;
    fn reload_model(&self) -> Result<(), ExecError>;
    fn unload_model(&self);
    fn is_model_loaded(&self) -> bool;

    fn upload_settings(&self, settings: Settings) -> Result<(), ExecError>;
    fn settings(&self) -> Arc<Settings>;
    fn settings_version(&self) -> u64;
    fn hooks(&self) -> Arc<HookBus>;

    fn capabilities(&self) -> Capabilities;
    fn stats(&self) -> ExecutionStats;
    fn first_token_tier(&self) -> LatencyTier;

    /// Architecture string for the currently-loaded bundle (lowercase
    /// `general.architecture` from GGUF, or HF `model_type` for MLX).
    /// Returns `None` when no model is loaded or when the
    /// backend can't surface architecture.
    ///
    /// Used by the chat-event mapper to derive `ChannelMarkers` per
    /// model family — Gemma 4's `<|channel>thought` reasoning markers
    /// only fire when this returns `gemma4`.
    fn bundle_architecture(&self) -> Option<String> {
        None
    }

    fn start_session(&self, spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError>;
    fn end_session(&self, id: SessionId) -> Result<(), ExecError>;

    /// Pre-load weights for a model directory in a background thread so the
    /// next `load_model` call with the same path can skip synchronous disk I/O.
    /// Default no-op — only implemented by backends that support warm loading.
    fn warm_model(&self, _model_dir: std::path::PathBuf) {}

    // Optional-capability upcasts. Default None = unsupported.
    fn as_embeddings(&self) -> Option<&dyn Embeddings> {
        None
    }
    fn as_multimodal(&self) -> Option<&dyn Multimodal> {
        None
    }
}

/// Local-model backend (llama, MLX, LiteRT-LM). Adds a context-size probe used by
/// the generic truncation driver.
pub trait LocalBackend: Backend {
    /// Context-window size in tokens. Zero if no model is loaded.
    fn n_ctx(&self) -> usize;
}

/// Remote-model backend (external API, future P2P/Flock). Context size is
/// informational — may be unknown until the remote reports it.
pub trait RemoteBackend: Backend {
    fn advertised_ctx(&self) -> Option<usize> {
        None
    }
}

/// One inference session. Returned as `Arc<dyn BackendSession>` so the
/// controller can hold heterogeneous sessions in a single map.
///
/// Not `Send + Sync` — the llama backend holds raw FFI pointers via
/// [`llama_cpp_2::context::LlamaContext`]. Sessions are thread-confined to
/// the controller's `run_loop`, which matches existing dispatch enum
/// semantics.
pub trait BackendSession: std::fmt::Debug {
    fn id(&self) -> SessionId;
    fn pause(&self);
    fn resume(&self);
    fn stop(&self);
    fn pull(&self, spec: GenSpec) -> Result<Box<dyn TokenPullerDyn>, ExecError>;
    fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError>;

    /// Optional KV cache probe — `None` unless the backend supports persistent
    /// KV. llama-cpp is the only such backend today.
    fn as_kv_snapshot(&self) -> Option<&dyn KvSnapshot> {
        None
    }
    /// Messages dropped during session creation by the generic truncation
    /// driver. Default 0 for backends that skip truncation.
    fn initial_messages_dropped(&self) -> usize {
        0
    }
    /// Per-session poison signal (FFI-level crash during decode). Default
    /// `false` — only llama surfaces this today.
    fn is_poisoned(&self) -> bool {
        false
    }
    /// Context window the session was actually created with (after any
    /// fit clamp). `0` = the backend doesn't preallocate a fixed context
    /// (MLX grows KV lazily).
    fn ctx_size(&self) -> u32 {
        0
    }
}

/// Object-safe token-stream iterator used by the facade's `TokenPuller` wrapper.
///
/// Not `Send` — llama puller holds raw FFI pointers; thread-confined to the
/// controller's `run_loop`.
pub trait TokenPullerDyn {
    fn next_event(&mut self) -> Option<Result<TokenEvent, ExecError>>;
}

/// KV cache save/load — optional capability on [`BackendSession`].
pub trait KvSnapshot {
    fn save_cache(&self, dst: KvSaveSpec) -> Result<KvSnapshotBlob, ExecError>;
    fn load_cache(&self, src: KvLoadSpec) -> Result<KvLoadReport, ExecError>;
}

/// Embedding generation — optional capability on [`Backend`].
pub trait Embeddings {
    fn load_embedder(&self, req: EmbedLoadRequest) -> Result<(), ExecError>;
    fn is_embedder_loaded(&self) -> bool;
    fn generate_embeddings(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ExecError>;
    fn unload_embedder(&self);
}

/// Multimodal input — optional capability on [`Backend`].
pub trait Multimodal {
    fn supports_images(&self) -> bool;
    fn supports_audio(&self) -> bool;
}

/// Minimal tokenizer contract for the truncation driver. Built per-session
/// because chat templates are session-scoped (see
/// [`crate::backend::llama`] for reference impl).
pub trait SessionTokenizer: Send + Sync {
    /// Apply the session's chat template, tokenize, and return the token count.
    fn count_tokens(&self, messages: &[Message]) -> Result<usize, ExecError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Minimal stub impls to exercise the default methods.
    #[derive(Debug)]
    struct StubBackend;

    impl Backend for StubBackend {
        fn backend_name(&self) -> &'static str {
            "stub"
        }
        fn load_model(&self, _req: LoadRequest) -> Result<(), ExecError> {
            Ok(())
        }
        fn reload_model(&self) -> Result<(), ExecError> {
            Ok(())
        }
        fn unload_model(&self) {}
        fn is_model_loaded(&self) -> bool {
            false
        }
        fn upload_settings(&self, _settings: Settings) -> Result<(), ExecError> {
            Ok(())
        }
        fn settings(&self) -> Arc<Settings> {
            Arc::new(Settings::default())
        }
        fn settings_version(&self) -> u64 {
            0
        }
        fn hooks(&self) -> Arc<HookBus> {
            Arc::new(HookBus::default())
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities::TEXT
        }
        fn stats(&self) -> ExecutionStats {
            ExecutionStats::default()
        }
        fn first_token_tier(&self) -> LatencyTier {
            LatencyTier::Medium
        }
        fn start_session(&self, _spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
            Err(ExecError::ModelNotLoaded)
        }
        fn end_session(&self, _id: SessionId) -> Result<(), ExecError> {
            Ok(())
        }
    }

    #[derive(Debug)]
    struct StubSession;

    impl BackendSession for StubSession {
        fn id(&self) -> SessionId {
            0
        }
        fn pause(&self) {}
        fn resume(&self) {}
        fn stop(&self) {}
        fn pull(&self, _spec: GenSpec) -> Result<Box<dyn TokenPullerDyn>, ExecError> {
            Err(ExecError::ModelNotLoaded)
        }
        fn append_messages(&self, _new_messages: Vec<Message>) -> Result<usize, ExecError> {
            Ok(0)
        }
    }

    #[test]
    fn backend_default_optional_upcasts_return_none() {
        let b = StubBackend;
        assert!(b.as_embeddings().is_none());
        assert!(b.as_multimodal().is_none());
    }

    #[test]
    fn backend_session_defaults() {
        let s = StubSession;
        assert_eq!(s.initial_messages_dropped(), 0);
        assert!(!s.is_poisoned());
        assert!(s.as_kv_snapshot().is_none());
    }
}
