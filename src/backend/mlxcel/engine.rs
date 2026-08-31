//! [`Backend`] impl for the mlxcel backend — [`MlxcelEngine`].
//!
//! Mirrors the shape of [`super::super::mlx::Engine`]: the engine owns
//! engine-level [`Settings`] and hands sessions out as `Arc<dyn BackendSession>`.
//! The one structural difference is thread-confinement: the `!Send` MLX model +
//! tokenizer live on a dedicated [`ModelWorker`] thread (see [`super::worker`]),
//! not inline, because mlxcel's C++ state is bound to its creating thread. The
//! engine holds only the `Send` worker handle + settings.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::backend::caps::LatencyTier;
use crate::backend::traits::{Backend, BackendSession, LocalBackend};
use crate::engine::telemetry::{HookBus, HookEvent};
use crate::engine::{Capabilities, ExecError, ExecutionStats, LoadRequest, Settings};
use crate::session_rt::SessionSpec;
use crate::session_rt::media_util::messages_have_images;

use super::session::MlxcelSession;
use super::worker::{LoadInfo, ModelWorker};

// `pub` (capped to crate by the `pub(crate) mod mlxcel`) so it can back the
// `pub Engine::Mlxcel` facade variant without a `private_interfaces` warning —
// mirrors `mlx::Engine` / `llama::Engine`.
pub struct MlxcelEngine {
    /// The dedicated MLX worker thread (owns the `!Send` model + tokenizer).
    worker: Arc<ModelWorker>,
    /// Facts about the currently-loaded model; `None` until `load_model`.
    loaded: RwLock<Option<LoadInfo>>,
    /// The model directory of the last successful load (for `reload_model`).
    last_load: RwLock<Option<LoadRequest>>,
    settings: ArcSwap<Settings>,
    settings_version: AtomicU64,
    next_session_id: AtomicU64,
    hooks: Arc<HookBus>,
}

impl std::fmt::Debug for MlxcelEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine(mlxcel)")
            .field("loaded", &self.loaded.read().is_some())
            .field(
                "settings_version",
                &self.settings_version.load(Ordering::SeqCst),
            )
            .finish()
    }
}

impl MlxcelEngine {
    /// Construct an mlxcel engine and spawn its worker thread. The worker
    /// initializes the MLX runtime; no model is loaded until `load_model`.
    #[allow(dead_code)] // constructed by the controller/facade in a later slice
    pub(crate) fn new() -> Self {
        Self {
            worker: Arc::new(ModelWorker::spawn()),
            loaded: RwLock::new(None),
            last_load: RwLock::new(None),
            settings: ArcSwap::from_pointee(Settings::default()),
            settings_version: AtomicU64::new(0),
            next_session_id: AtomicU64::new(0),
            hooks: Arc::new(HookBus::default()),
        }
    }

    /// PROFILE-ONLY (S5 perf verify): run one greedy `generate_streaming` pass in
    /// the given `on_token` mode and return the timing. Drives the worker's
    /// [`profile_blocking`](super::worker::ModelWorker::profile_blocking); the
    /// model must already be loaded. Used exclusively by the decode-profile
    /// captest — not on any user path.
    #[cfg(test)]
    pub(crate) fn profile(
        &self,
        prompt: &str,
        max_tokens: usize,
        mode: super::worker::ProfileMode,
    ) -> Result<super::worker::ProfileRun, ExecError> {
        self.worker
            .profile_blocking(prompt.to_string(), max_tokens, mode)
    }
}

impl Backend for MlxcelEngine {
    fn backend_name(&self) -> &'static str {
        "mlxcel"
    }

    fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let info = self.worker.load(req.model_path.clone())?;
        let meta = crate::bundle::ModelMeta {
            n_ctx: info.n_ctx as u32,
            n_layer: info.num_layers as u32,
            architecture: info.architecture.clone(),
            ..Default::default()
        };
        *self.loaded.write() = Some(info);
        *self.last_load.write() = Some(req);

        tracing::info!("mlxcel.load_model.ok");
        self.hooks.emit(HookEvent::EngineLoadOk {
            caps_text: true,
            caps_images: false,
            caps_audio: false,
            meta,
        });
        Ok(())
    }

    fn reload_model(&self) -> Result<(), ExecError> {
        let req = self
            .last_load
            .read()
            .clone()
            .ok_or(ExecError::ModelNotLoaded)?;
        self.load_model(req)
    }

    fn unload_model(&self) {
        self.worker.unload();
        *self.loaded.write() = None;
    }

    fn is_model_loaded(&self) -> bool {
        self.loaded.read().is_some()
    }

    fn upload_settings(&self, settings: Settings) -> Result<(), ExecError> {
        settings.validate()?;
        self.settings.store(Arc::new(settings));
        self.settings_version.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn settings(&self) -> Arc<Settings> {
        self.settings.load_full()
    }

    fn settings_version(&self) -> u64 {
        self.settings_version.load(Ordering::SeqCst)
    }

    fn hooks(&self) -> Arc<HookBus> {
        self.hooks.clone()
    }

    fn capabilities(&self) -> Capabilities {
        // Text-only in the tracer-bullet slice (VLM prefill is a later slice).
        if self.is_model_loaded() {
            Capabilities::TEXT
        } else {
            Capabilities::empty()
        }
    }

    fn stats(&self) -> ExecutionStats {
        // Per-run throughput stats are a later slice (S5 perf-verify times
        // decode directly). Tracer-bullet returns the default.
        ExecutionStats::default()
    }

    fn first_token_tier(&self) -> LatencyTier {
        LatencyTier::Medium
    }

    fn bundle_architecture(&self) -> Option<String> {
        self.loaded
            .read()
            .as_ref()
            .and_then(|i| i.architecture.clone())
    }

    fn start_session(&self, spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        if !self.is_model_loaded() {
            return Err(ExecError::ModelNotLoaded);
        }

        // Fail closed on image inputs — mlxcel is text-only in this slice, so
        // reject rather than silently flatten images to markdown (mirrors the
        // mlx backend's multimodal contract).
        if messages_have_images(&spec.messages) {
            return Err(ExecError::FeatureUnsupported("images"));
        }

        // Resolve effective settings: session overrides inherit engine defaults.
        let base = self.settings();
        let settings = if let Some(mut overrides) = spec.overrides.clone() {
            overrides.inherit_missing(base.as_ref());
            overrides
        } else {
            (*base).clone()
        };

        // Thread the loaded model's REAL chat template + bos/eos into the
        // session so `build_prompt` renders the model's actual template
        // (gemma `<start_of_turn>` etc.), not the naive role-tagged concat.
        // Mirrors how `mlx::Engine::start_session` hands the bundle (which
        // carries `chat_template_str`/`bos_str`/`eos_str`) to `mlx::Session`.
        let (chat_template, bos_str, eos_str) = {
            let guard = self.loaded.read();
            match guard.as_ref() {
                Some(info) => (
                    info.chat_template.clone(),
                    info.bos_str.clone(),
                    info.eos_str.clone(),
                ),
                // `is_model_loaded()` above guarantees Some; defensive fallback.
                None => (None, None, None),
            }
        };

        let id = self.next_session_id.fetch_add(1, Ordering::SeqCst);
        let session = MlxcelSession::new(
            id,
            self.worker.clone(),
            settings,
            spec.messages,
            chat_template,
            bos_str,
            eos_str,
        );
        Ok(Arc::new(session) as Arc<dyn BackendSession>)
    }

    fn end_session(&self, _id: u64) -> Result<(), ExecError> {
        // Sessions hold no worker-side state (the worker is stateless between
        // generations in this slice), so there's nothing to tear down.
        Ok(())
    }
}

impl LocalBackend for MlxcelEngine {
    fn n_ctx(&self) -> usize {
        self.loaded.read().as_ref().map(|i| i.n_ctx).unwrap_or(0)
    }
}
