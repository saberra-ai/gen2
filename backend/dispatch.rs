//! Runtime backend dispatch via enums.
//!
//! Wraps each backend's Engine/Session/TokenPuller in a single enum so the
//! controller and the rest of the app never need to know which backend is active.

use std::sync::Arc;

use crate::gen2::engine::telemetry::HookBus;
use crate::gen2::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::gen2::generation::{GenSpec, TokenEvent};
use crate::gen2::session_rt::SessionSpec;
use crate::generation::model_runner::types::Message;

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
    let path_str = path.to_str().unwrap_or("");
    if path_str.starts_with("http://") || path_str.starts_with("https://") {
        #[cfg(feature = "backend-external-api")]
        return Ok(BackendKind::ExternalApi);
        #[cfg(not(feature = "backend-external-api"))]
        return Err(ExecError::Other(anyhow::anyhow!(
            "External API backend not compiled"
        )));
    }

    if path.is_file() {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext {
                "gguf" => return Ok(BackendKind::LlamaCpp),
                "onnx" => return Ok(BackendKind::Onnx),
                _ => {}
            }
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
                if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                    if ext == "safetensors" {
                        return Ok(BackendKind::Mlx);
                    }
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

impl Engine {
    /// Which backend is currently active (for UI display).
    pub fn active_backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(_) => "llamacpp",
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(_) => "mlx",
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(_) => "onnx",
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(_) => "external-api",
            Self::Uninit => "none",
        }
    }

    /// Which backends are compiled into this binary.
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
    pub fn new() -> Self {
        #[cfg(feature = "backend-llamacpp")]
        {
            return Self::LlamaCpp(super::llama::Engine::new());
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
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(e) => e.load_model(req),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(e) => e.load_model(req),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(e) => e.load_model(req),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(e) => e.load_model(req),
            Self::Uninit => Err(ExecError::Other(anyhow::anyhow!(
                "no backend for model format"
            ))),
        }
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
        dispatch!(self, e => e.reload_model())
    }

    pub fn load_embedder(&self, req: EmbedLoadRequest) -> Result<(), ExecError> {
        dispatch!(self, e => e.load_embedder(req))
    }

    pub fn is_embedder_loaded(&self) -> bool {
        dispatch!(self, e => e.is_embedder_loaded(), false)
    }

    pub fn upload_settings(&self, settings: Settings) -> Result<(), ExecError> {
        dispatch!(self, e => e.upload_settings(settings))
    }

    pub fn settings(&self) -> Arc<Settings> {
        dispatch!(self, e => e.settings(), Arc::new(Settings::default()))
    }

    pub fn settings_version(&self) -> u64 {
        dispatch!(self, e => e.settings_version(), 0)
    }

    pub fn hooks(&self) -> Arc<HookBus> {
        dispatch!(self, e => e.hooks(), Arc::new(HookBus::new()))
    }

    pub fn start_session(&self, spec: SessionSpec) -> Result<Arc<Session>, ExecError> {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(e) => e
                .start_session(spec)
                .map(|s| Arc::new(Session::LlamaCpp(s))),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(e) => e.start_session(spec).map(|s| Arc::new(Session::Mlx(s))),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(e) => e.start_session(spec).map(|s| Arc::new(Session::Onnx(s))),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(e) => e
                .start_session(spec)
                .map(|s| Arc::new(Session::ExternalApi(s))),
            Self::Uninit => Err(ExecError::ModelNotLoaded),
        }
    }

    pub fn end_session(&self, id: SessionId) -> Result<(), ExecError> {
        dispatch!(self, e => e.end_session(id))
    }

    pub fn is_model_loaded(&self) -> bool {
        dispatch!(self, e => e.is_model_loaded(), false)
    }

    pub fn capabilities(&self) -> Capabilities {
        dispatch!(self, e => e.capabilities(), Capabilities::empty())
    }

    pub fn does_model_support_images(&self) -> bool {
        dispatch!(self, e => e.does_model_support_images(), false)
    }

    pub fn does_model_support_audio(&self) -> bool {
        dispatch!(self, e => e.does_model_support_audio(), false)
    }

    pub fn stats(&self) -> ExecutionStats {
        dispatch!(self, e => e.stats(), ExecutionStats::default())
    }

    pub fn generate_embeddings(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ExecError> {
        dispatch!(self, e => e.generate_embeddings(inputs))
    }

    pub fn unload_model(&self) {
        dispatch_void!(self, e => e.unload_model());
    }

    pub fn unload_embedder(&self) {
        dispatch_void!(self, e => e.unload_embedder());
    }
}

/// Dispatch to the active backend, returning the result.
/// For methods that return Result, the Uninit variant returns an error.
macro_rules! dispatch {
    // With a fallback default for non-Result returns
    ($self:expr, $e:ident => $call:expr, $default:expr) => {
        match $self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp($e) => $call,
            #[cfg(feature = "backend-mlx")]
            Self::Mlx($e) => $call,
            #[cfg(feature = "backend-onnx")]
            Self::Onnx($e) => $call,
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi($e) => $call,
            Self::Uninit => $default,
        }
    };
    // For Result-returning methods (Uninit → ModelNotLoaded error)
    ($self:expr, $e:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp($e) => $call,
            #[cfg(feature = "backend-mlx")]
            Self::Mlx($e) => $call,
            #[cfg(feature = "backend-onnx")]
            Self::Onnx($e) => $call,
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi($e) => $call,
            Self::Uninit => Err(ExecError::ModelNotLoaded),
        }
    };
}

macro_rules! dispatch_void {
    ($self:expr, $e:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp($e) => $call,
            #[cfg(feature = "backend-mlx")]
            Self::Mlx($e) => $call,
            #[cfg(feature = "backend-onnx")]
            Self::Onnx($e) => $call,
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi($e) => $call,
            Self::Uninit => {}
        }
    };
}

// Make macros usable before their definition (Rust 2024 allows this)
use dispatch;
use dispatch_void;

// ─── Session ────────────────────────────────────────────────────────────────

pub enum Session {
    #[cfg(feature = "backend-llamacpp")]
    LlamaCpp(Arc<super::llama::Session>),
    #[cfg(feature = "backend-mlx")]
    Mlx(Arc<super::mlx::Session>),
    #[cfg(feature = "backend-onnx")]
    Onnx(Arc<super::onnx::Session>),
    #[cfg(feature = "backend-external-api")]
    ExternalApi(Arc<super::external_api::Session>),
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(s) => write!(f, "Session::LlamaCpp(id={})", s.id),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(s) => write!(f, "Session::Mlx(id={})", s.id),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(s) => write!(f, "Session::Onnx(id={})", s.id),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(s) => write!(f, "Session::ExternalApi(id={})", s.id),
        }
    }
}

/// Dispatch on Session enum variants
macro_rules! session_dispatch {
    ($self:expr, $s:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp($s) => $call,
            #[cfg(feature = "backend-mlx")]
            Self::Mlx($s) => $call,
            #[cfg(feature = "backend-onnx")]
            Self::Onnx($s) => $call,
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi($s) => $call,
        }
    };
}

impl Session {
    pub fn id(&self) -> SessionId {
        session_dispatch!(self, s => s.id)
    }

    pub fn pause(&self) {
        session_dispatch!(self, s => s.pause());
    }

    pub fn resume(&self) {
        session_dispatch!(self, s => s.resume());
    }

    pub fn stop(&self) {
        session_dispatch!(self, s => s.stop());
    }

    pub fn pull(&self, gen_spec: GenSpec) -> Result<TokenPuller, ExecError> {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(s) => s.pull(gen_spec).map(TokenPuller::LlamaCpp),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(s) => s.pull(gen_spec).map(TokenPuller::Mlx),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(s) => s.pull(gen_spec).map(TokenPuller::Onnx),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(s) => s.pull(gen_spec).map(TokenPuller::ExternalApi),
        }
    }

    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        session_dispatch!(self, s => s.append_messages(new_messages))
    }
}


// ─── TokenPuller ────────────────────────────────────────────────────────────

pub enum TokenPuller {
    #[cfg(feature = "backend-llamacpp")]
    LlamaCpp(super::llama::TokenPuller),
    #[cfg(feature = "backend-mlx")]
    Mlx(super::mlx::TokenPuller),
    #[cfg(feature = "backend-onnx")]
    Onnx(super::onnx::TokenPuller),
    #[cfg(feature = "backend-external-api")]
    ExternalApi(super::external_api::RemotePuller),
}

impl Iterator for TokenPuller {
    type Item = Result<TokenEvent, ExecError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(p) => p.next(),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(p) => p.next(),
            #[cfg(feature = "backend-onnx")]
            Self::Onnx(p) => p.next(),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(p) => p.next(),
        }
    }
}
