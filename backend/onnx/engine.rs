//! ONNX Runtime inference engine.

use arc_swap::{ArcSwap, ArcSwapOption};
use dashmap::DashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::gen2::engine::telemetry::{HookBus, HookEvent};
use crate::gen2::engine::{Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings};
use crate::gen2::backend::common::tokenizer::HfTokenizer;
use super::bundle::ModelBundle;
use super::session::{Session, SessionId};
use crate::gen2::session_rt::SessionSpec;
use crate::gen2::bundle::ModelMeta;

use parking_lot::{Mutex, RwLock};

pub struct Engine {
    bundle: ArcSwapOption<ModelBundle>,
    sessions: DashMap<SessionId, ()>,
    settings: ArcSwap<Settings>,
    last_load: RwLock<Option<LoadRequest>>,
    settings_version: AtomicU64,
    next_session_id: AtomicU64,
    hooks: Arc<HookBus>,
    load_guard: parking_lot::Mutex<()>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine(ONNX)")
            .field("sessions", &self.sessions.len())
            .field(
                "settings_version",
                &self.settings_version.load(Ordering::SeqCst),
            )
            .field("has_bundle", &self.bundle.load_full().is_some())
            .finish()
    }
}

impl Engine {
    pub fn new() -> Self {
        Self {
            bundle: ArcSwapOption::from(None),
            sessions: DashMap::new(),
            settings: ArcSwap::from_pointee(Settings::default()),
            last_load: RwLock::new(None),
            settings_version: AtomicU64::new(0),
            next_session_id: AtomicU64::new(1),
            hooks: Arc::new(HookBus::new()),
            load_guard: parking_lot::Mutex::new(()),
        }
    }

    pub fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let _g = self.load_guard.lock();

        let model_path = &req.model_path;

        // For ONNX, model_path points to a directory containing model.onnx + tokenizer.json
        // or directly to a .onnx file (tokenizer.json in same dir).
        let (onnx_path, model_dir) = if model_path.is_dir() {
            (model_path.join("model.onnx"), model_path.as_path())
        } else {
            (
                model_path.clone(),
                model_path.parent().unwrap_or(model_path.as_path()),
            )
        };

        let threads = req.ctx_params.threads;
        let ort_session = super::loader::build_session(&onnx_path, threads)?;
        let num_layers = super::loader::detect_num_layers(&ort_session);

        let tokenizer = HfTokenizer::from_dir(model_dir)
            .map_err(|e| ExecError::Other(e))?;

        let meta = ModelMeta {
            model_uuid: String::new(),
            n_ctx: req.ctx_params.n_ctx.unwrap_or(4096),
            n_layer: num_layers as u32,
        };

        let caps = Capabilities::TEXT;

        let bundle = ModelBundle {
            session: Mutex::new(ort_session),
            tokenizer,
            capabilities: caps.clone(),
            meta: meta.clone(),
            num_layers,
        };

        self.sessions.clear();
        self.bundle.store(Some(Arc::new(bundle)));
        *self.last_load.write() = Some(req);

        tracing::info!("engine.load_model.ok (ONNX, {} layers)", num_layers);
        self.hooks.emit(HookEvent::EngineLoadOk {
            caps_text: true,
            caps_images: false,
            caps_audio: false,
            meta,
        });
        Ok(())
    }

    pub fn reload_model(&self) -> Result<(), ExecError> {
        let req = self
            .last_load
            .read()
            .clone()
            .ok_or(ExecError::ModelNotLoaded)?;
        self.load_model(req)
    }

    pub fn load_embedder(&self, _req: EmbedLoadRequest) -> Result<(), ExecError> {
        Err(ExecError::Unimplemented)
    }

    pub fn is_embedder_loaded(&self) -> bool {
        false
    }

    pub fn upload_settings(&self, settings: Settings) -> Result<(), ExecError> {
        settings.validate()?;
        self.settings.store(Arc::new(settings));
        self.settings_version.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn settings(&self) -> Arc<Settings> {
        self.settings.load_full()
    }

    pub fn settings_version(&self) -> u64 {
        self.settings_version.load(Ordering::SeqCst)
    }

    pub fn hooks(&self) -> Arc<HookBus> {
        self.hooks.clone()
    }

    pub fn start_session(&self, spec: SessionSpec) -> Result<Arc<Session>, ExecError> {
        let bundle = self.bundle.load_full().ok_or(ExecError::ModelNotLoaded)?;
        let base_settings = self.settings();
        let settings = if let Some(mut overrides) = spec.overrides.clone() {
            overrides.inherit_missing(base_settings.as_ref());
            overrides
        } else {
            (*base_settings).clone()
        };

        let id = self.next_session_id.fetch_add(1, Ordering::SeqCst);
        let session = Arc::new(Session::new(
            id,
            bundle.clone(),
            self.hooks.clone(),
            settings,
            spec.messages,
            spec.persona.as_ref(),
        )?);
        self.sessions.insert(id, ());
        Ok(session)
    }

    pub fn end_session(&self, id: SessionId) -> Result<(), ExecError> {
        if self.sessions.remove(&id).is_some() {
            Ok(())
        } else {
            Err(ExecError::InvalidArg("unknown session id"))
        }
    }

    pub fn is_model_loaded(&self) -> bool {
        self.bundle.load_full().is_some()
    }

    pub fn capabilities(&self) -> Capabilities {
        self.bundle
            .load_full()
            .as_deref()
            .map(|b| b.capabilities.clone())
            .unwrap_or_else(Capabilities::empty)
    }

    pub fn does_model_support_images(&self) -> bool {
        self.capabilities().contains(Capabilities::IMAGES)
    }

    pub fn does_model_support_audio(&self) -> bool {
        self.capabilities().contains(Capabilities::AUDIO)
    }

    pub fn stats(&self) -> ExecutionStats {
        ExecutionStats::default()
    }

    pub fn generate_embeddings(&self, _inputs: &[String]) -> Result<Vec<Vec<f32>>, ExecError> {
        Err(ExecError::Unimplemented)
    }

    pub fn unload_model(&self) {
        self.bundle.store(None);
    }

    pub fn unload_embedder(&self) {
        // no-op
    }
}
