//! Backend facade — enum wrapping concrete backends that implement the
//! `Backend` / `BackendSession` / `TokenPullerDyn` traits.
//!
//! Public API is byte-compatible with the former `dispatch.rs`. Instance
//! methods delegate through a single `as_backend()` upcast; `Session` and
//! `TokenPuller` become thin structs holding trait objects.

use std::sync::Arc;

use super::traits::{Backend, BackendSession, LocalBackend, TokenPullerDyn};
use crate::advanced::BackendPlugin;
use crate::engine::telemetry::HookBus;
use crate::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::generation::{GenSpec, TokenEvent};
use crate::kv::{KvLoadReport, KvLoadSpec, KvSaveSpec, KvSnapshot};
use crate::session_rt::SessionSpec;
use crate::types::message::Message;

// ─── SessionId ──────────────────────────────────────────────────────────────
pub type SessionId = u64;

// ─── ModelBundle (opaque wrapper — only used for Debug) ─────────────────────
pub enum ModelBundle {
    #[cfg(feature = "backend-llamacpp")]
    LlamaCpp(Arc<super::llama::ModelBundle>),
    #[cfg(feature = "backend-mlx")]
    Mlx(Arc<super::mlx::ModelBundle>),
    /// No bundle at this layer.
    ///
    /// Some backends keep the model somewhere the facade cannot hold it: the
    /// external-api server owns it remotely, LiteRT-LM's engine lives behind
    /// its C ABI, mistral.rs keeps its own loaded state, and a plugin backend
    /// is opaque here by construction.
    /// Unconditional, so the enum stays inhabited no matter which single
    /// backend is compiled — without it, a build of litertlm alone fails to
    /// compile on this `match`.
    None,
}

impl std::fmt::Debug for ModelBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "backend-llamacpp")]
            Self::LlamaCpp(b) => b.fmt(f),
            #[cfg(feature = "backend-mlx")]
            Self::Mlx(b) => b.fmt(f),
            Self::None => f.write_str("ModelBundle(none)"),
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
    /// A backend the consumer registered through
    /// [`BackendPlugin`](crate::advanced::BackendPlugin). Built on the
    /// controller thread from the plugin's factory the first time a path the
    /// plugin claims is loaded, and asked before every built-in rule.
    Plugin(Box<dyn LocalBackend>),
    #[cfg(feature = "backend-external-api")]
    ExternalApi(super::external_api::Engine),
    /// mistral.rs — one backend across GGUF, safetensors, UQFF and HF repos.
    /// Routed only for formats no other compiled backend claims, so adding it
    /// to an existing build does not move anyone's models.
    #[cfg(feature = "backend-mistralrs")]
    MistralRs(super::mistralrs::MistralRsEngine),
    /// LiteRT-LM — Google's on-device runtime, loaded from its C ABI at run
    /// time. Routed only for `.litertlm` bundles, which no other backend
    /// reads, so adding it to an existing build moves nobody's models.
    #[cfg(feature = "backend-litertlm")]
    LiteRtLm(super::litertlm::LiteRtLmEngine),
    /// A scripted backend, for tests that need the runtime to misbehave on
    /// demand. Sticky: `ensure_backend` never switches away from it, so a
    /// `LoadModel` for any path still lands on the script.
    #[cfg(test)]
    Fake(crate::test_support::FakeBackend),
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
            Self::Plugin(e) => e.fmt(f),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(e) => e.fmt(f),
            #[cfg(feature = "backend-mistralrs")]
            Self::MistralRs(e) => e.fmt(f),
            #[cfg(feature = "backend-litertlm")]
            Self::LiteRtLm(e) => e.fmt(f),
            #[cfg(test)]
            Self::Fake(e) => e.fmt(f),
            Self::Uninit => f.debug_struct("Engine(Uninit)").finish(),
        }
    }
}

/// Detect which backend to use from the model path:
///  - a path some registered plugin `claims` → that plugin, before any rule below
///  - URL (`http://` or `https://`) → ExternalApi
///  - `.gguf` file → LlamaCpp
///  - directory with `*.safetensors` → MLX (Apple only)
///  - `.litertlm` bundle → LiteRT-LM
///  - `.onnx` file or dir with `model.onnx` → an error naming the format;
///    no compiled backend reads it
#[allow(unused_variables)]
fn detect_backend(
    req: &LoadRequest,
    plugins: &[Arc<BackendPlugin>],
) -> Result<BackendKind, ExecError> {
    let path = &req.model_path;

    // A registered plugin is asked first, in registration order. First because
    // a consumer that went to the trouble of bringing a backend means it: a
    // plugin claiming `.gguf` takes GGUF away from llama.cpp, deliberately.
    if let Some(i) = plugins.iter().position(|p| (p.claims)(path)) {
        return Ok(BackendKind::Plugin(i));
    }

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
            // GGUF stays with llama.cpp wherever llama.cpp is compiled.
            // Adding mistral.rs to an existing build must not silently move
            // somebody's models to a different implementation.
            #[cfg(feature = "backend-llamacpp")]
            "gguf" => return Ok(BackendKind::LlamaCpp),
            #[cfg(all(not(feature = "backend-llamacpp"), feature = "backend-mistralrs"))]
            "gguf" => return Ok(BackendKind::MistralRs),
            // Neither compiled: keep naming llama.cpp, so the failure a
            // caller sees is the same one they saw before this backend
            // existed.
            #[cfg(all(not(feature = "backend-llamacpp"), not(feature = "backend-mistralrs")))]
            "gguf" => return Ok(BackendKind::LlamaCpp),
            // ONNX had a backend once; it never decoded a token and was
            // removed. Named here rather than falling through, so the error
            // is about the format and not about whatever llama.cpp makes of
            // a file it has never seen.
            "onnx" => return Err(ExecError::FeatureUnsupported(NO_ONNX_BACKEND)),
            // A `.litertlm` bundle is LiteRT-LM's own packaging and nothing
            // else can read it. Deliberately not a fall-through: without the
            // feature this has to say so, because the alternative is handing
            // the file to llama.cpp and reporting whatever it makes of a
            // format it has never seen.
            #[cfg(feature = "backend-litertlm")]
            "litertlm" => return Ok(BackendKind::LiteRtLm),
            #[cfg(not(feature = "backend-litertlm"))]
            "litertlm" => {
                return Err(ExecError::FeatureUnsupported(
                    "`.litertlm` bundles need the `backend-litertlm` feature, \
                     which is not compiled into this build",
                ));
            }
            // UQFF is mistral.rs's own format; nothing else reads it.
            #[cfg(feature = "backend-mistralrs")]
            "uqff" => return Ok(BackendKind::MistralRs),
            _ => {}
        }
    }

    if path.is_dir() {
        // An ONNX bundle directory, for the same reason as the `.onnx` arm.
        if path.join("model.onnx").exists() {
            return Err(ExecError::FeatureUnsupported(NO_ONNX_BACKEND));
        }
        // Check for safetensors (MLX format)
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Some(ext) = entry.path().extension().and_then(|e| e.to_str())
                    && ext == "safetensors"
                {
                    // MLX keeps the Apple path it already had; mistral.rs
                    // takes safetensors only where MLX is not compiled.
                    #[cfg(feature = "backend-mlx")]
                    return Ok(BackendKind::Mlx);
                    #[cfg(all(not(feature = "backend-mlx"), feature = "backend-mistralrs"))]
                    return Ok(BackendKind::MistralRs);
                    #[cfg(all(not(feature = "backend-mlx"), not(feature = "backend-mistralrs")))]
                    return Ok(BackendKind::Mlx);
                }
            }
        }
    }

    // A Hugging Face model directory — config.json and no safetensors that an
    // Apple backend claimed above.
    #[cfg(feature = "backend-mistralrs")]
    if path.is_dir() && path.join("config.json").exists() {
        return Ok(BackendKind::MistralRs);
    }

    // Fallback to the default compiled backend
    #[cfg(feature = "backend-llamacpp")]
    return Ok(BackendKind::LlamaCpp);
    #[cfg(all(not(feature = "backend-llamacpp"), feature = "backend-mlx"))]
    return Ok(BackendKind::Mlx);
    #[cfg(all(
        not(feature = "backend-llamacpp"),
        not(feature = "backend-mlx"),
        feature = "backend-mistralrs"
    ))]
    return Ok(BackendKind::MistralRs);
    // LiteRT-LM is a fallback only where it is the sole local backend. It
    // reads one format, so in any other build a path it cannot open should
    // fail as whatever backend does claim that format.
    #[cfg(all(
        not(feature = "backend-llamacpp"),
        not(feature = "backend-mlx"),
        not(feature = "backend-mistralrs"),
        feature = "backend-litertlm"
    ))]
    return Ok(BackendKind::LiteRtLm);
    #[cfg(not(any(
        feature = "backend-llamacpp",
        feature = "backend-mlx",
        feature = "backend-mistralrs",
        feature = "backend-litertlm"
    )))]
    Err(no_backend_error(path, plugins))
}

/// What a load fails with when nothing — compiled or registered — can take
/// the path.
///
/// Reached only in a build with no local backend feature, which is allowed
/// precisely so a consumer can bring its own; so the message names both
/// ways out rather than just the feature list.
fn no_backend_error(path: &std::path::Path, plugins: &[Arc<BackendPlugin>]) -> ExecError {
    if plugins.is_empty() {
        ExecError::Other(anyhow::anyhow!(
            "no inference backend: this build compiled none (enable a `backend-*` \
             feature) and none was registered with `Engine::builder().backend(..)`, \
             so nothing can load {}",
            path.display()
        ))
    } else {
        let names: Vec<&str> = plugins.iter().map(|p| p.name).collect();
        ExecError::Other(anyhow::anyhow!(
            "no inference backend claims {}: this build compiled none, and the \
             registered plugin backends ({}) declined it",
            path.display(),
            names.join(", ")
        ))
    }
}

/// What a caller sees for a `.onnx` file or a `model.onnx` directory.
const NO_ONNX_BACKEND: &str = "no compiled backend reads ONNX models (`.onnx` file or `model.onnx` \
     directory); the ONNX backend was removed — convert the model to GGUF for \
     llama.cpp, or serve it behind an OpenAI-compatible endpoint";

#[derive(Debug, Clone, Copy)]
enum BackendKind {
    /// Index into the plugin list the detection ran against.
    Plugin(usize),
    LlamaCpp,
    Mlx,
    #[cfg(feature = "backend-mistralrs")]
    MistralRs,
    #[cfg(feature = "backend-litertlm")]
    LiteRtLm,
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
            Self::Plugin(e) => Some(e.as_ref()),
            #[cfg(feature = "backend-external-api")]
            Self::ExternalApi(e) => Some(e),
            #[cfg(feature = "backend-mistralrs")]
            Self::MistralRs(e) => Some(e),
            #[cfg(feature = "backend-litertlm")]
            Self::LiteRtLm(e) => Some(e),
            #[cfg(test)]
            Self::Fake(e) => Some(e),
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
    #[allow(clippy::vec_init_then_push, unused_mut)]
    pub fn available_backends() -> Vec<&'static str> {
        let mut v = Vec::new();
        #[cfg(feature = "backend-llamacpp")]
        v.push("llamacpp");
        #[cfg(feature = "backend-mlx")]
        v.push("mlx");
        #[cfg(feature = "backend-external-api")]
        v.push("external-api");
        #[cfg(feature = "backend-mistralrs")]
        v.push("mistralrs");
        #[cfg(feature = "backend-litertlm")]
        v.push("litertlm");
        v
    }

    /// Whether a zoo bundle naming this backend could actually be served.
    ///
    /// A straight name match against what is compiled. Plugins are not
    /// consulted: they are registered per controller, and the zoo asks about
    /// the build.
    pub fn backend_is_compiled(name: &str) -> bool {
        Self::available_backends().contains(&name)
    }

    /// Detect which backend a model path would use, without loading and with
    /// no plugins registered.
    pub fn detect_backend_for_path(path: &std::path::Path) -> &'static str {
        Self::detect_backend_for_path_with(path, &[])
    }

    /// As [`Self::detect_backend_for_path`], with these plugins asked first.
    /// A plugin's answer is its `name`.
    pub fn detect_backend_for_path_with(
        path: &std::path::Path,
        plugins: &[Arc<BackendPlugin>],
    ) -> &'static str {
        let req = LoadRequest {
            model_path: path.to_path_buf(),
            ..Default::default()
        };
        match detect_backend(&req, plugins) {
            Ok(BackendKind::Plugin(i)) => plugins[i].name,
            Ok(BackendKind::LlamaCpp) => "llamacpp",
            Ok(BackendKind::Mlx) => "mlx",
            #[cfg(feature = "backend-external-api")]
            Ok(BackendKind::ExternalApi) => "external-api",
            #[cfg(feature = "backend-mistralrs")]
            Ok(BackendKind::MistralRs) => "mistralrs",
            #[cfg(feature = "backend-litertlm")]
            Ok(BackendKind::LiteRtLm) => "litertlm",
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
            // Tail expr (not `return`): with llamacpp absent and mlx present,
            // every block after this one is cfg'd out, so this is the
            // function's tail.
            Self::Mlx(super::mlx::Engine::new())
        }
        #[cfg(all(
            not(feature = "backend-llamacpp"),
            not(feature = "backend-mlx"),
            feature = "backend-mistralrs"
        ))]
        {
            // Tail expr, as in the arms above: with every other local backend
            // absent, the blocks after this one are cfg'd out.
            Self::MistralRs(super::mistralrs::MistralRsEngine::new())
        }
        #[cfg(all(
            not(feature = "backend-llamacpp"),
            not(feature = "backend-mlx"),
            not(feature = "backend-mistralrs"),
            feature = "backend-litertlm"
        ))]
        {
            // Tail expr, as in the arms above: with every other local backend
            // absent, the blocks after this one are cfg'd out.
            Self::LiteRtLm(super::litertlm::LiteRtLmEngine::new())
        }
        // No local backend compiled: nothing to instantiate until a load
        // names a path, which a registered plugin may claim.
        #[cfg(not(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-mistralrs",
            feature = "backend-litertlm"
        )))]
        Self::Uninit
    }

    /// Load a model, auto-detecting the backend from the file format.
    /// If the detected backend differs from the current one, the engine
    /// is re-initialized to the new backend first.
    ///
    /// No plugins are consulted; see [`Self::load_model_with`].
    pub fn load_model(&mut self, req: LoadRequest) -> Result<(), ExecError> {
        self.load_model_with(req, &[])
    }

    /// As [`Self::load_model`], asking `plugins` (in order) before any
    /// built-in rule. A plugin that `claims` the path is built from its
    /// factory here, on the calling thread, and becomes the active backend.
    pub fn load_model_with(
        &mut self,
        req: LoadRequest,
        plugins: &[Arc<BackendPlugin>],
    ) -> Result<(), ExecError> {
        // A scripted backend answers for every path, so format detection —
        // which reads the filesystem — must not run and must not veto a path
        // that was never meant to exist.
        #[cfg(test)]
        if let Self::Fake(fake) = self {
            return fake.load_model(req);
        }
        let kind = detect_backend(&req, plugins)?;
        self.ensure_backend(kind, plugins);
        self.as_backend()
            .ok_or_else(|| ExecError::Other(anyhow::anyhow!("no backend for model format")))?
            .load_model(req)
    }

    /// Switch the engine to the requested backend kind if needed.
    fn ensure_backend(&mut self, kind: BackendKind, plugins: &[Arc<BackendPlugin>]) {
        let needs_switch = match (&self, kind) {
            // A scripted backend answers for every path. Switching away from
            // it on the first `LoadModel` would quietly replace the fake with
            // a real backend and make the test meaningless.
            #[cfg(test)]
            (Self::Fake(_), _) => false,
            // The same plugin again keeps its instance (and its loaded
            // weights); a different plugin, or a built-in, is a switch.
            (Self::Plugin(active), BackendKind::Plugin(i)) => {
                active.backend_name() != plugins[i].name
            }
            #[cfg(feature = "backend-llamacpp")]
            (Self::LlamaCpp(_), BackendKind::LlamaCpp) => false,
            #[cfg(feature = "backend-mlx")]
            (Self::Mlx(_), BackendKind::Mlx) => false,
            #[cfg(feature = "backend-external-api")]
            (Self::ExternalApi(_), BackendKind::ExternalApi) => false,
            #[cfg(feature = "backend-mistralrs")]
            (Self::MistralRs(_), BackendKind::MistralRs) => false,
            #[cfg(feature = "backend-litertlm")]
            (Self::LiteRtLm(_), BackendKind::LiteRtLm) => false,
            _ => true,
        };
        if !needs_switch {
            return;
        }
        *self = match kind {
            BackendKind::Plugin(i) => Self::Plugin((plugins[i].make)()),
            #[cfg(feature = "backend-llamacpp")]
            BackendKind::LlamaCpp => Self::LlamaCpp(super::llama::Engine::new()),
            #[cfg(feature = "backend-mlx")]
            BackendKind::Mlx => Self::Mlx(super::mlx::Engine::new()),
            #[cfg(feature = "backend-external-api")]
            BackendKind::ExternalApi => Self::ExternalApi(super::external_api::Engine::new()),
            #[cfg(feature = "backend-mistralrs")]
            BackendKind::MistralRs => Self::MistralRs(super::mistralrs::MistralRsEngine::new()),
            #[cfg(feature = "backend-litertlm")]
            BackendKind::LiteRtLm => Self::LiteRtLm(super::litertlm::LiteRtLmEngine::new()),
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
    ///
    /// Phase 7: single-source-of-truth via `BackendCaps::from_backend` trait
    /// probe; no more per-variant constructors.
    pub fn backend_caps(&self) -> super::caps::BackendCaps {
        match self.as_backend() {
            None => super::caps::BackendCaps::uninit(),
            Some(b) => super::caps::BackendCaps::from_backend(b),
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

    /// Pre-load weights for `model_dir` in a background thread.
    /// No-op if the current backend doesn't support warm loading.
    pub fn warm_model(&self, model_dir: std::path::PathBuf) {
        if let Some(b) = self.as_backend() {
            b.warm_model(model_dir);
        }
    }

    /// Architecture string for the currently-loaded bundle (lowercase
    /// `general.architecture` from GGUF, or HF `model_type` for MLX /
    /// LiteRT-LM). Returns `None` when no model is loaded.
    pub fn bundle_architecture(&self) -> Option<String> {
        self.as_backend().and_then(|b| b.bundle_architecture())
    }

    /// Context window of the loaded model, when the backend can say. See
    /// [`Backend::context_window`].
    pub fn context_window(&self) -> Option<u32> {
        self.as_backend().and_then(|b| b.context_window())
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

    /// Context window the session was created with (after the fit
    /// clamp); `0` when the backend doesn't preallocate one.
    pub fn ctx_size(&self) -> u32 {
        self.0.ctx_size()
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
        #[cfg(all(
            not(feature = "backend-llamacpp"),
            not(feature = "backend-mlx"),
            feature = "backend-mistralrs"
        ))]
        assert_eq!(name, "mistralrs");
        #[cfg(all(
            not(feature = "backend-llamacpp"),
            not(feature = "backend-mlx"),
            not(feature = "backend-mistralrs"),
            feature = "backend-litertlm"
        ))]
        assert_eq!(name, "litertlm");
        #[cfg(not(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-mistralrs",
            feature = "backend-litertlm"
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

    /// Adding mistral.rs must not move anybody's models.
    ///
    /// The precedence rule exists because a backend feature is additive: a
    /// build that already sent GGUF to llama.cpp must keep doing so when
    /// mistral.rs is compiled in beside it, or upgrading the feature list
    /// silently changes which implementation runs a user's model.
    #[test]
    #[cfg(all(feature = "backend-llamacpp", feature = "backend-mistralrs"))]
    fn gguf_stays_with_llamacpp_when_both_are_compiled() {
        assert_eq!(
            Engine::detect_backend_for_path(std::path::Path::new("/models/model.gguf")),
            "llamacpp"
        );
    }

    /// And with llama.cpp absent, mistral.rs picks GGUF up rather than leaving
    /// the caller with a backend that cannot open it.
    #[test]
    #[cfg(all(not(feature = "backend-llamacpp"), feature = "backend-mistralrs"))]
    fn gguf_falls_to_mistralrs_when_llamacpp_is_absent() {
        assert_eq!(
            Engine::detect_backend_for_path(std::path::Path::new("/models/model.gguf")),
            "mistralrs"
        );
    }

    /// A format only mistral.rs reads is always its own.
    #[test]
    #[cfg(feature = "backend-mistralrs")]
    fn uqff_is_always_mistralrs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.uqff");
        std::fs::write(&path, b"not really a uqff, only the extension matters").unwrap();
        assert_eq!(Engine::detect_backend_for_path(&path), "mistralrs");
    }

    /// Two backends, two formats, no collision.
    ///
    /// This is the check that adding LiteRT-LM to a working llama.cpp build
    /// changes nothing: the GGUF a user already had keeps going to llama.cpp,
    /// and only the format nothing else reads goes to the new backend.
    #[test]
    #[cfg(all(feature = "backend-llamacpp", feature = "backend-litertlm"))]
    fn each_format_goes_to_the_backend_that_can_read_it() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("foo.gguf");
        let bundle = dir.path().join("foo.litertlm");
        std::fs::write(&gguf, b"only the extension matters here").unwrap();
        std::fs::write(&bundle, b"only the extension matters here").unwrap();

        assert_eq!(
            Engine::detect_backend_for_path(&gguf),
            "llamacpp",
            "adding a backend must not move a model that already worked"
        );
        assert_eq!(
            Engine::detect_backend_for_path(&bundle),
            "litertlm",
            "a `.litertlm` bundle is readable by nothing else"
        );
    }

    /// A `.litertlm` bundle without the feature says so, rather than being
    /// handed to a backend that has never seen the format.
    ///
    /// The alternative is llama.cpp reporting a parse failure on a file it was
    /// never meant to open, which sends the caller looking at their model
    /// instead of at their feature list.
    #[test]
    #[cfg(not(feature = "backend-litertlm"))]
    fn a_litertlm_bundle_without_the_backend_names_the_missing_feature() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("foo.litertlm");
        std::fs::write(&path, b"only the extension matters here").unwrap();

        let req = LoadRequest {
            model_path: path,
            ..Default::default()
        };
        let err = detect_backend(&req, &[]).expect_err("the format is not compiled in");
        let text = err.to_string();
        assert!(
            text.contains("backend-litertlm"),
            "the error should name the feature to enable, got: {text}"
        );
    }

    /// A `.onnx` file or a `model.onnx` directory fails naming the format.
    ///
    /// The ONNX backend was removed on 2026-09-04 without ever decoding a
    /// token. Nothing left in the crate reads the format, and the error has
    /// to say that in every build — not fall through to llama.cpp reporting
    /// a bad magic number on a file it was never meant to open.
    #[test]
    fn an_onnx_model_fails_naming_the_format_in_every_build() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.onnx");
        std::fs::write(&file, b"only the name matters here").unwrap();

        for path in [file.clone(), dir.path().to_path_buf()] {
            let req = LoadRequest {
                model_path: path.clone(),
                ..Default::default()
            };
            match detect_backend(&req, &[]) {
                Err(ExecError::FeatureUnsupported(msg)) => assert!(
                    msg.contains("ONNX") && msg.contains("no compiled backend"),
                    "{}: the error should say no backend reads ONNX, got: {msg}",
                    path.display()
                ),
                other => panic!(
                    "{}: expected FeatureUnsupported, got {other:?}",
                    path.display()
                ),
            }
        }
        assert_eq!(Engine::detect_backend_for_path(&file), "unknown");
    }

    // ── Plugin routing ─────────────────────────────────────────

    /// A plugin over the scripted backend, claiming one extension.
    fn scripted_plugin(
        name: &'static str,
        claims: fn(&std::path::Path) -> bool,
    ) -> Arc<BackendPlugin> {
        let script = crate::test_support::Script::new();
        Arc::new(BackendPlugin {
            name,
            claims,
            make: Box::new(move || Box::new(script.backend())),
        })
    }

    fn claims_fake(path: &std::path::Path) -> bool {
        path.extension().is_some_and(|e| e == "fake")
    }

    fn claims_gguf(path: &std::path::Path) -> bool {
        path.extension().is_some_and(|e| e == "gguf")
    }

    /// A path only the plugin claims lands on the plugin, whatever is
    /// compiled in — and the path need not exist, because the plugin's
    /// `claims` runs before any rule that reads the filesystem.
    #[test]
    fn a_plugin_takes_the_path_it_claims() {
        let plugins = [scripted_plugin("scripted", claims_fake)];
        let path = std::path::Path::new("/nowhere/model.fake");
        assert_eq!(
            Engine::detect_backend_for_path_with(path, &plugins),
            "scripted"
        );

        let mut engine = Engine::new();
        let req = LoadRequest {
            model_path: path.to_path_buf(),
            ..Default::default()
        };
        engine
            .load_model_with(req, &plugins)
            .expect("the scripted plugin loads anything it claims");
        assert!(matches!(engine, Engine::Plugin(_)), "got {engine:?}");
        assert_eq!(engine.active_backend_name(), "fake");
        assert!(engine.is_model_loaded());
    }

    /// A plugin is asked before the built-in rules, so one claiming `.gguf`
    /// takes GGUF away from llama.cpp even when llama.cpp is compiled.
    #[test]
    fn a_plugin_outranks_builtin_routing() {
        let plugins = [scripted_plugin("gguf-plugin", claims_gguf)];
        let path = std::path::Path::new("/nowhere/model.gguf");
        assert_eq!(
            Engine::detect_backend_for_path_with(path, &plugins),
            "gguf-plugin"
        );
        // And without it, the built-in answer is unchanged.
        assert_ne!(Engine::detect_backend_for_path(path), "gguf-plugin");
    }

    /// Loading a second path the same plugin claims keeps the instance. A
    /// path the plugin declines is never handed to it: where a built-in
    /// backend is compiled, the engine switches to that; where none is, the
    /// detection itself fails and the engine is left exactly as it was — the
    /// same as any path nothing can route.
    #[test]
    fn the_same_plugin_is_kept_across_loads_and_a_declined_path_is_not_handed_to_it() {
        let plugins = [scripted_plugin("scripted", claims_fake)];
        let mut engine = Engine::new();
        for name in ["a.fake", "b.fake"] {
            let req = LoadRequest {
                model_path: std::path::PathBuf::from("/nowhere").join(name),
                ..Default::default()
            };
            engine.load_model_with(req, &plugins).unwrap();
        }
        let Engine::Plugin(backend) = &engine else {
            panic!("expected the plugin, got {engine:?}");
        };
        assert!(backend.is_model_loaded());

        let req = LoadRequest {
            model_path: "/nowhere/model.other".into(),
            ..Default::default()
        };
        let outcome = engine.load_model_with(req, &plugins);
        #[cfg(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-mistralrs",
            feature = "backend-litertlm"
        ))]
        {
            let _ = outcome;
            assert!(
                !matches!(engine, Engine::Plugin(_)),
                "a path the plugin declined must go to the built-in backend, got {engine:?}"
            );
        }
        #[cfg(not(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-mistralrs",
            feature = "backend-litertlm"
        )))]
        {
            let err = outcome.expect_err("nothing compiled can route the path");
            assert!(err.to_string().contains("scripted"), "{err}");
            assert!(
                matches!(engine, Engine::Plugin(_)),
                "a load nothing could route leaves the engine as it was, got {engine:?}"
            );
        }
    }

    /// With no backend compiled and no plugin registered, the error says both.
    #[test]
    fn the_no_backend_error_names_both_ways_out() {
        let path = std::path::Path::new("/nowhere/model.bin");
        let text = no_backend_error(path, &[]).to_string();
        assert!(text.contains("backend-*"), "{text}");
        assert!(text.contains("backend(..)"), "{text}");
        let text = no_backend_error(path, &[scripted_plugin("scripted", claims_fake)]).to_string();
        assert!(text.contains("scripted"), "{text}");
    }
}
