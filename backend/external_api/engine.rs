//! External API inference engine.
//!
//! Manages connection to an OpenAI-compatible server and dispatches
//! sessions that stream tokens via SSE.

use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::gen2::bundle::ModelMeta;
use crate::gen2::engine::telemetry::{HookBus, HookEvent};
use crate::gen2::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::gen2::session_rt::SessionSpec;
use super::session::{Session, SessionId};

pub struct Engine {
    server_url: RwLock<String>,
    model_id: RwLock<String>,
    api_key: RwLock<Option<String>>,
    api_format: RwLock<String>,
    client: reqwest::blocking::Client,
    loaded: AtomicBool,
    sessions: DashMap<SessionId, ()>,
    settings: ArcSwap<Settings>,
    last_load: RwLock<Option<LoadRequest>>,
    settings_version: AtomicU64,
    next_session_id: AtomicU64,
    hooks: Arc<HookBus>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine(ExternalApi)")
            .field("server_url", &*self.server_url.read())
            .field("model_id", &*self.model_id.read())
            .field("loaded", &self.loaded.load(Ordering::SeqCst))
            .field("sessions", &self.sessions.len())
            .finish()
    }
}

impl Engine {
    pub fn new() -> Self {
        Self {
            server_url: RwLock::new(String::new()),
            model_id: RwLock::new(String::new()),
            api_key: RwLock::new(None),
            api_format: RwLock::new("openai".into()),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .read_timeout(std::time::Duration::from_secs(2))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
            loaded: AtomicBool::new(false),
            sessions: DashMap::new(),
            settings: ArcSwap::from_pointee(Settings::default()),
            last_load: RwLock::new(None),
            settings_version: AtomicU64::new(0),
            next_session_id: AtomicU64::new(1),
            hooks: Arc::new(HookBus::new()),
        }
    }

    /// Load model from an external server.
    ///
    /// The `model_path` in `LoadRequest` is expected to be a URL like
    /// `http://localhost:11434/v1`. The model_id is derived from the URL
    /// path or from the config's `external_model_id`.
    pub fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let url_str = req
            .model_path
            .to_str()
            .ok_or_else(|| ExecError::Other(anyhow::anyhow!("invalid URL path")))?
            .to_string();

        // Parse server URL: strip trailing slash for consistency
        let base_url = url_str.trim_end_matches('/').to_string();

        // Apply API key and format from the load request
        if let Some(ref key) = req.api_key {
            *self.api_key.write() = Some(key.clone());
        }
        if let Some(ref fmt) = req.api_format {
            *self.api_format.write() = fmt.clone();
        }

        let api_format = self.api_format.read().clone();
        let api_key = self.api_key.read().clone();

        // Validate connectivity (Anthropic doesn't have /models)
        if api_format == "anthropic" {
            tracing::info!("external_api: configured for Anthropic at {}", base_url);
        } else {
        let models_url = format!("{}/models", base_url);
        let mut check = self.client.get(&models_url);
        if let Some(ref key) = api_key {
            check = check.header("Authorization", format!("Bearer {}", key));
        }
        match check.send() {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!(
                    "external_api: connected to {} (models endpoint OK)",
                    base_url
                );
            }
            Ok(resp) => {
                // Some servers don't have /models but still work; warn and proceed
                tracing::warn!(
                    "external_api: /models returned {} — proceeding anyway",
                    resp.status()
                );
            }
            Err(e) => {
                return Err(ExecError::Other(anyhow::anyhow!(
                    "cannot connect to external server at {}: {}",
                    models_url,
                    e
                )));
            }
        }
        } // end openai connectivity check

        *self.server_url.write() = base_url;
        // Default model_id — can be overridden via config
        *self.model_id.write() = String::new();
        self.loaded.store(true, Ordering::SeqCst);
        self.sessions.clear();
        *self.last_load.write() = Some(req);

        let meta = ModelMeta::default();
        self.hooks.emit(HookEvent::EngineLoadOk {
            caps_text: true,
            caps_images: false,
            caps_audio: false,
            meta,
        });

        tracing::info!("engine.load_model.ok (ExternalApi)");
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
        if !self.loaded.load(Ordering::SeqCst) {
            return Err(ExecError::ModelNotLoaded);
        }

        let base_settings = self.settings();
        let settings = if let Some(mut overrides) = spec.overrides.clone() {
            overrides.inherit_missing(base_settings.as_ref());
            overrides
        } else {
            (*base_settings).clone()
        };

        let id = self.next_session_id.fetch_add(1, Ordering::SeqCst);
        let server_url = self.server_url.read().clone();
        let model_id = self.model_id.read().clone();
        let api_key = self.api_key.read().clone();
        let api_format = self.api_format.read().clone();

        let session = Arc::new(Session::new(
            id,
            server_url,
            model_id,
            api_key,
            api_format,
            self.client.clone(),
            self.hooks.clone(),
            settings,
            spec.messages,
        ));
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
        self.loaded.load(Ordering::SeqCst)
    }

    pub fn capabilities(&self) -> Capabilities {
        if self.loaded.load(Ordering::SeqCst) {
            Capabilities::TEXT
        } else {
            Capabilities::empty()
        }
    }

    pub fn does_model_support_images(&self) -> bool {
        false
    }

    pub fn does_model_support_audio(&self) -> bool {
        false
    }

    pub fn stats(&self) -> ExecutionStats {
        ExecutionStats::default()
    }

    pub fn generate_embeddings(&self, _inputs: &[String]) -> Result<Vec<Vec<f32>>, ExecError> {
        Err(ExecError::Unimplemented)
    }

    pub fn unload_model(&self) {
        self.loaded.store(false, Ordering::SeqCst);
        *self.server_url.write() = String::new();
        *self.model_id.write() = String::new();
        *self.api_key.write() = None;
        *self.api_format.write() = "openai".into();
    }

    pub fn unload_embedder(&self) {
        // no-op
    }

    /// Set the model_id (called from config layer when user provides external_model_id).
    pub fn set_model_id(&self, id: String) {
        *self.model_id.write() = id;
    }

    pub fn set_api_key(&self, key: Option<String>) {
        *self.api_key.write() = key;
    }

    pub fn set_api_format(&self, format: String) {
        *self.api_format.write() = format;
    }
}
