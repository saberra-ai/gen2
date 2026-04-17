//! Backend facade — enum wrapping concrete backends that implement the
//! `Backend` / `BackendSession` / `TokenPullerDyn` traits.
//!
//! Public API is byte-compatible with the former `dispatch.rs`. Instance
//! methods delegate through a single `as_backend()` upcast; `Session` and
//! `TokenPuller` become thin structs holding trait objects.

use std::sync::Arc;

use super::traits::{Backend, BackendSession, TokenPullerDyn};
use crate::gen2::engine::telemetry::HookBus;
use crate::gen2::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::gen2::generation::{GenSpec, TokenEvent};
use crate::gen2::kv::{KvLoadReport, KvLoadSpec, KvSaveSpec, KvSnapshot};
use crate::gen2::session_rt::SessionSpec;
use crate::types::message::Message;

// ─── SessionId ──────────────────────────────────────────────────────────────
pub type SessionId = u64;

// ─── ModelBundle (opaque wrapper — only used for Debug) ─────────────────────
pub enum ModelBundle {
    #[cfg(feature = "backend-llamacpp")]
    LlamaCpp(Arc<super::llama::ModelBundle>),
    #[cfg(feature = "backend-mlx")]
    Mlx(Arc<super::mlx::ModelBundle>),
    #[cfg(feature = "backend-onnx")]
    Onnx(Arc<super::onnx::ModelBundle>),
    /// ExternalApi has no local model bundle — the server manages the model.
    /// This variant exists only so the enum is non-empty when only external-api
    /// is compiled.
    #[cfg(feature = "backend-external-api")]
    ExternalApi,
}

impl std::fmt::Debug for ModelBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(b) => b.fmt(f),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(b) => b.fmt(f),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(b) => b.fmt(f),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi => f.debug_struct("ModelBundle(ExternalApi)").finish(),
        }
    }
}

// ─── Engine ─────────────────────────────────────────────────────────────────

/// Runtime-dispatched inference engine.
///
/// Holds one concrete backend engine internally. The controller creates one
/// `Engine` at startup; `load_model` auto-detects the format and instantiates
/// the right backend.
#[allow(clippy::large_enum_variant)]
pub enum Engine {
    #[cfg(feature = "backend-llamacpp")]
    LlamaCpp(super::llama::Engine),
    #[cfg(feature = "backend-mlx")]
    Mlx(super::mlx::Engine),
    #[cfg(feature = "backend-onnx")]
    Onnx(super::onnx::Engine),
    #[cfg(feature = "backend-external-api")]
    ExternalApi(super::external_api::Engine),
    /// Sentinel used before the first `load_model` call when the default
    /// backend is constructed.
    Uninit,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(e) => e.fmt(f),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(e) => e.fmt(f),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(e) => e.fmt(f),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(e) => e.fmt(f),
            Self::Uninit => f.debug_struct("Engine(Uninit)").finish(),
        }
    }
}

/// Detect which backend to use from the model path:
///  - URL (`http://` or `https://`) → ExternalApi
///  - `.gguf` file → LlamaCpp
///  - directory with `*.safetensors` → MLX (Apple only)
///  - `.onnx` file or dir with `model.onnx` → ONNX
#[allow(unused_variables)]
fn detect_backend(req: &LoadRequest) -> Result<BackendKind, ExecError> {
    let path = &req.model_path;

    // URL detection — external API server
    let path_str = path.to_str().unwrap_or_default();
    if path_str.starts_with("http://") || path_str.starts_with("https://") {
        #[cfg(feature = "backend-external-api")]
        return Ok(BackendKind::ExternalApi);
        #[cfg(not(feature = "backend-external-api"))]
        return Err(ExecError::Other(anyhow::anyhow!(
            "External API backend not compiled"
        )));
    }

    if path.is_file()
        && let Some(ext) = path.extension().and_then(|e| e.to_str())
    {
        match ext {
            "gguf" => return Ok(BackendKind::LlamaCpp),
            "onnx" => return Ok(BackendKind::Onnx),
            _ => {}
        }
    }

    if path.is_dir() {
        // Check for ONNX first (model.onnx in dir)
        if path.join("model.onnx").exists() {
            return Ok(BackendKind::Onnx);
        }
        // Check for safetensors (MLX format)
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Some(ext) = entry.path().extension().and_then(|e| e.to_str())
                    && ext == "safetensors"
                {
                    return Ok(BackendKind::Mlx);
                }
            }
        }
    }

    // Fallback to the default compiled backend
    #[cfg(feature = "backend-llamacpp")]
    return Ok(BackendKind::LlamaCpp);
    #[cfg(all(not(feature = "backend-llamacpp"), feature = "backend-mlx"))]
    return Ok(BackendKind::Mlx);
    #[cfg(all(
        not(feature = "backend-llamacpp"),
        not(feature = "backend-mlx"),
        feature = "backend-onnx"
    ))]
    return Ok(BackendKind::Onnx);
    #[cfg(not(any(
        feature = "backend-llamacpp",
        feature = "backend-mlx",
        feature = "backend-onnx"
    )))]
    Err(ExecError::Other(anyhow::anyhow!("no backend compiled")))
}

#[derive(Debug, Clone, Copy)]
enum BackendKind {
    LlamaCpp,
    Mlx,
    Onnx,
    #[cfg(feature = "backend-external-api")]
    ExternalApi,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Upcast to `&dyn Backend` if a real backend is active. Returns `None`
    /// for `Uninit`. Single source of truth for instance-method dispatch.
    fn as_backend(&self) -> Option<&dyn Backend> {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(e) => Some(e),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(e) => Some(e),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(e) => Some(e),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(e) => Some(e),
            Self::Uninit => None,
        }
    }

    /// Which backend is currently active (for UI display).
    pub fn active_backend_name(&self) -> &'static str {
        self.as_backend()
            .map(|b| b.backend_name())
            .unwrap_or("none")
    }

    /// Which backends are compiled into this binary.
    #[allow(clippy::vec_init_then_push)]
    pub fn available_backends() -> Vec<&'static str> {
        let mut v = Vec::new();
        #[cfg(feature = "backend-llamacpp")]
        v.push("llamacpp");
        #[cfg(feature = "backend-mlx")]
        v.push("mlx");
        #[cfg(feature = "backend-onnx")]
        v.push("onnx");
        #[cfg(feature = "backend-external-api")]
        v.push("external-api");
        v
    }

    /// Detect which backend a model path would use, without loading.
    pub fn detect_backend_for_path(path: &std::path::Path) -> &'static str {
        let req = LoadRequest {
            model_path: path.to_path_buf(),
            ..Default::default()
        };
        match detect_backend(&req) {
            Ok(BackendKind::LlamaCpp) => "llamacpp",
            Ok(BackendKind::Mlx) => "mlx",
            Ok(BackendKind::Onnx) => "onnx",
            #[cfg(feature = "backend-external-api")]
            Ok(BackendKind::ExternalApi) => "external-api",
            Err(_) => "unknown",
        }
    }

    /// Create a new engine with the platform default backend.
    ///
    /// Defaults to llamacpp when available (GGUF is the bundled format).
    /// `load_model` auto-detects from file format and switches backend as needed,
    /// so the initial backend is just a starting point.
    pub fn new() -> Self {
        #[cfg(feature = "backend-llamacpp")]
        {
            Self::LlamaCpp(super::llama::Engine::new())
        }
        #[cfg(all(not(feature = "backend-llamacpp"), feature = "backend-mlx"))]
        {
            return Self::Mlx(super::mlx::Engine::new());
        }
        #[cfg(all(
            not(feature = "backend-llamacpp"),
            not(feature = "backend-mlx"),
            feature = "backend-onnx"
        ))]
        {
            return Self::Onnx(super::onnx::Engine::new());
        }
        #[cfg(not(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-onnx"
        )))]
        Self::Uninit
    }

    /// Load a model, auto-detecting the backend from the file format.
    /// If the detected backend differs from the current one, the engine
    /// is re-initialized to the new backend first.
    pub fn load_model(&mut self, req: LoadRequest) -> Result<(), ExecError> {
        let kind = detect_backend(&req)?;
        self.ensure_backend(kind);
        self.as_backend()
            .ok_or_else(|| ExecError::Other(anyhow::anyhow!("no backend for model format")))?
            .load_model(req)
    }

    /// Switch the engine to the requested backend kind if needed.
    fn ensure_backend(&mut self, kind: BackendKind) {
        let needs_switch = match (&self, kind) {
            #[cfg(feature = "backend-llamacpp")]
            (Self::LlamaCpp(_), BackendKind::LlamaCpp) => false,
            #[cfg(feature = "backend-mlx")]
            (Self::Mlx(_), BackendKind::Mlx) => false,
            #[cfg(feature = "backend-onnx")]
            (Self::Onnx(_), BackendKind::Onnx) => false,
            #[cfg(feature = "backend-external-api")]
            (Self::ExternalApi(_), BackendKind::ExternalApi) => false,
            _ => true,
        };
        if !needs_switch {
            return;
        }
        *self = match kind {
            #[cfg(feature = "backend-llamacpp")]
            BackendKind::LlamaCpp => Self::LlamaCpp(super::llama::Engine::new()),
            #[cfg(feature = "backend-mlx")]
            BackendKind::Mlx => Self::Mlx(super::mlx::Engine::new()),
            #[cfg(feature = "backend-onnx")]
            BackendKind::Onnx => Self::Onnx(super::onnx::Engine::new()),
            #[cfg(feature = "backend-external-api")]
            BackendKind::ExternalApi => Self::ExternalApi(super::external_api::Engine::new()),
            // Backend not compiled in
            #[allow(unreachable_patterns)]
            _ => Self::Uninit,
        };
    }

    pub fn reload_model(&self) -> Result<(), ExecError> {
        self.as_backend()
            .ok_or(ExecError::ModelNotLoaded)?
            .reload_model()
    }

    pub fn load_embedder(&self, req: EmbedLoadRequest) -> Result<(), ExecError> {
        let b = self.as_backend().ok_or(ExecError::ModelNotLoaded)?;
        b.as_embeddings()
            .ok_or(ExecError::FeatureUnsupported("embeddings"))?
            .load_embedder(req)
    }

    pub fn is_embedder_loaded(&self) -> bool {
        self.as_backend()
            .and_then(|b| b.as_embeddings())
            .map(|e| e.is_embedder_loaded())
            .unwrap_or(false)
    }

    pub fn upload_settings(&self, settings: Settings) -> Result<(), ExecError> {
        self.as_backend()
            .ok_or(ExecError::ModelNotLoaded)?
            .upload_settings(settings)
    }

    pub fn settings(&self) -> Arc<Settings> {
        self.as_backend()
            .map(|b| b.settings())
            .unwrap_or_else(|| Arc::new(Settings::default()))
    }

    pub fn settings_version(&self) -> u64 {
        self.as_backend().map(|b| b.settings_version()).unwrap_or(0)
    }

    pub fn hooks(&self) -> Arc<HookBus> {
        self.as_backend()
            .map(|b| b.hooks())
            .unwrap_or_else(|| Arc::new(HookBus::new()))
    }

    #[allow(clippy::arc_with_non_send_sync)]
    pub fn start_session(&self, spec: SessionSpec) -> Result<Arc<Session>, ExecError> {
        let inner = self
            .as_backend()
            .ok_or(ExecError::ModelNotLoaded)?
            .start_session(spec)?;
        Ok(Arc::new(Session(inner)))
    }

    pub fn end_session(&self, id: SessionId) -> Result<(), ExecError> {
        self.as_backend()
            .ok_or(ExecError::ModelNotLoaded)?
            .end_session(id)
    }

    pub fn is_model_loaded(&self) -> bool {
        self.as_backend()
            .map(|b| b.is_model_loaded())
            .unwrap_or(false)
    }

    pub fn capabilities(&self) -> Capabilities {
        self.as_backend()
            .map(|b| b.capabilities())
            .unwrap_or_else(Capabilities::empty)
    }

    /// Infrastructure capability contract for the active backend.
    ///
    /// Unlike `capabilities()` (modality: text/images/audio), this describes
    /// what the backend's runtime machinery supports (KV cache, poisoning, etc.).
    pub fn backend_caps(&self) -> super::caps::BackendCaps {
        // Phase 4: per-variant constructors retained. Phase 7 replaces these
        // with a trait-probe (`BackendCaps::from_backend`).
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(_) => super::caps::BackendCaps::llamacpp(),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(_) => super::caps::BackendCaps::mlx(),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(_) => super::caps::BackendCaps::onnx(),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(_) => super::caps::BackendCaps::external_api(),
            Self::Uninit => super::caps::BackendCaps::uninit(),
        }
    }

    pub fn does_model_support_images(&self) -> bool {
        self.as_backend()
            .and_then(|b| b.as_multimodal())
            .map(|m| m.supports_images())
            .unwrap_or(false)
    }

    pub fn does_model_support_audio(&self) -> bool {
        self.as_backend()
            .and_then(|b| b.as_multimodal())
            .map(|m| m.supports_audio())
            .unwrap_or(false)
    }

    pub fn stats(&self) -> ExecutionStats {
        self.as_backend().map(|b| b.stats()).unwrap_or_default()
    }

    pub fn generate_embeddings(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ExecError> {
        let b = self.as_backend().ok_or(ExecError::ModelNotLoaded)?;
        b.as_embeddings()
            .ok_or(ExecError::FeatureUnsupported("embeddings"))?
            .generate_embeddings(inputs)
    }

    pub fn unload_model(&self) {
        if let Some(b) = self.as_backend() {
            b.unload_model();
        }
    }

    pub fn unload_embedder(&self) {
        if let Some(e) = self.as_backend().and_then(|b| b.as_embeddings()) {
            e.unload_embedder();
        }
    }
}

// ─── Session ────────────────────────────────────────────────────────────────

/// Facade wrapper over a backend-specific session. Holds the trait object so
/// the controller can store heterogeneous sessions in a single map.
pub struct Session(pub(crate) Arc<dyn BackendSession>);

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Session {
    pub fn id(&self) -> SessionId {
        self.0.id()
    }

    pub fn pause(&self) {
        self.0.pause();
    }

    pub fn resume(&self) {
        self.0.resume();
    }

    pub fn stop(&self) {
        self.0.stop();
    }

    pub fn pull(&self, gen_spec: GenSpec) -> Result<TokenPuller, ExecError> {
        Ok(TokenPuller(self.0.pull(gen_spec)?))
    }

    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        self.0.append_messages(new_messages)
    }

    pub fn save_cache(&self, dst: KvSaveSpec) -> Result<KvSnapshot, ExecError> {
        self.0
            .as_kv_snapshot()
            .ok_or(ExecError::FeatureUnsupported("kv cache"))?
            .save_cache(dst)
    }

    pub fn load_cache(&self, src: KvLoadSpec) -> Result<KvLoadReport, ExecError> {
        self.0
            .as_kv_snapshot()
            .ok_or(ExecError::FeatureUnsupported("kv cache"))?
            .load_cache(src)
    }

    /// Messages dropped during initial session creation due to context overflow.
    pub fn initial_messages_dropped(&self) -> usize {
        self.0.initial_messages_dropped()
    }

    /// Returns true if the session's internal state was lost (e.g. due to an
    /// FFI panic). A poisoned session cannot generate further tokens.
    pub fn is_poisoned(&self) -> bool {
        self.0.is_poisoned()
    }
}

// ─── TokenPuller ────────────────────────────────────────────────────────────

/// Facade wrapper over a backend-specific token puller. `Iterator::next`
/// dispatches through the trait object.
pub struct TokenPuller(pub(crate) Box<dyn TokenPullerDyn>);

impl Iterator for TokenPuller {
    type Item = Result<TokenEvent, ExecError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next_event()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ── Engine dispatch tests ──────────────────────────────

    #[test]
    fn uninit_engine_returns_model_not_loaded() {
        let engine = Engine::new();
        assert!(
            !engine.is_model_loaded(),
            "freshly constructed engine should report no model loaded"
        );
    }

    #[test]
    fn format_detection_gguf() {
        let name = Engine::detect_backend_for_path(&PathBuf::from("/tmp/model.gguf"));
        #[cfg(feature = "backend-llamacpp")]
        assert_eq!(name, "llamacpp", ".gguf should map to llamacpp backend");
        #[cfg(not(feature = "backend-llamacpp"))]
        {
            let _ = name;
        }
    }

    #[test]
    fn format_detection_url() {
        let name = Engine::detect_backend_for_path(&PathBuf::from("http://localhost:11434/v1"));
        #[cfg(feature = "backend-external-api")]
        assert_eq!(
            name, "external-api",
            "http:// URL should map to external-api backend"
        );
        #[cfg(not(feature = "backend-external-api"))]
        {
            let _ = name;
        }
    }

    #[test]
    fn format_detection_https_url() {
        let name = Engine::detect_backend_for_path(&PathBuf::from("https://api.openai.com/v1"));
        #[cfg(feature = "backend-external-api")]
        assert_eq!(
            name, "external-api",
            "https:// URL should map to external-api backend"
        );
        #[cfg(not(feature = "backend-external-api"))]
        {
            let _ = name;
        }
    }

    #[test]
    fn active_backend_name_on_fresh_engine() {
        let engine = Engine::new();
        let name = engine.active_backend_name();
        #[cfg(feature = "backend-llamacpp")]
        assert_eq!(name, "llamacpp");
        #[cfg(all(not(feature = "backend-llamacpp"), feature = "backend-mlx"))]
        assert_eq!(name, "mlx");
        #[cfg(not(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-onnx"
        )))]
        assert_eq!(name, "none");
    }

    #[test]
    fn fresh_engine_capabilities_and_stats() {
        let engine = Engine::new();
        let caps = engine.capabilities();
        let stats = engine.stats();
        assert_eq!(stats.decode_tokens, 0);
        let _ = caps;
    }
}
