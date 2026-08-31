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

use super::session::{Session, SessionId};
use crate::bundle::ModelMeta;
use crate::engine::telemetry::{HookBus, HookEvent};
use crate::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::session_rt::SessionSpec;

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
        // Default model_id — overridden by config's external_model_id (the Tauri
        // settings layer calls set_model_id). In daemon/headless mode that wiring
        // isn't present, so honor a PIO_EXTERNAL_MODEL_ID env fallback too —
        // otherwise an empty model_id sends "default" upstream (session.rs), which
        // model-specific servers like ollama reject as not-found.
        *self.model_id.write() = std::env::var("PIO_EXTERNAL_MODEL_ID").unwrap_or_default();
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

// ─── Trait impls (Phase 2) ─────────────────────────────────────────────────

use crate::backend::caps::LatencyTier;
use crate::backend::traits::{Backend, BackendSession, Embeddings, RemoteBackend};

impl Backend for Engine {
    fn backend_name(&self) -> &'static str {
        "external_api"
    }
    fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        Engine::load_model(self, req)
    }
    fn reload_model(&self) -> Result<(), ExecError> {
        Engine::reload_model(self)
    }
    fn unload_model(&self) {
        Engine::unload_model(self)
    }
    fn is_model_loaded(&self) -> bool {
        Engine::is_model_loaded(self)
    }
    fn upload_settings(&self, settings: Settings) -> Result<(), ExecError> {
        Engine::upload_settings(self, settings)
    }
    fn settings(&self) -> Arc<Settings> {
        Engine::settings(self)
    }
    fn settings_version(&self) -> u64 {
        Engine::settings_version(self)
    }
    fn hooks(&self) -> Arc<HookBus> {
        Engine::hooks(self)
    }
    fn capabilities(&self) -> Capabilities {
        Engine::capabilities(self)
    }
    fn stats(&self) -> ExecutionStats {
        Engine::stats(self)
    }
    fn first_token_tier(&self) -> LatencyTier {
        LatencyTier::Slow
    }
    fn start_session(&self, spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        let s = Engine::start_session(self, spec)?;
        Ok(s as Arc<dyn BackendSession>)
    }
    fn end_session(&self, id: SessionId) -> Result<(), ExecError> {
        Engine::end_session(self, id)
    }
    fn as_embeddings(&self) -> Option<&dyn Embeddings> {
        Some(self)
    }
}

impl RemoteBackend for Engine {
    fn advertised_ctx(&self) -> Option<usize> {
        None
    }
}

impl Embeddings for Engine {
    fn load_embedder(&self, req: EmbedLoadRequest) -> Result<(), ExecError> {
        Engine::load_embedder(self, req)
    }
    fn is_embedder_loaded(&self) -> bool {
        Engine::is_embedder_loaded(self)
    }
    fn generate_embeddings(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ExecError> {
        Engine::generate_embeddings(self, inputs)
    }
    fn unload_embedder(&self) {
        Engine::unload_embedder(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{GenSpec, TokenEvent};
    use crate::session_rt::SessionSpec;
    use crate::types::message::{Message, MessageBody, MessageContent};

    fn user_message(text: &str) -> Message {
        Message {
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.into()),
            },
            name: None,
        }
    }

    // ── 1. Load model from URL (OpenAI format) ──────────────────────

    #[test]
    fn load_model_from_url() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("GET", "/v1/models")
            .with_status(200)
            .with_body(r#"{"data":[]}"#)
            .create();

        let engine = Engine::new();
        let req = LoadRequest {
            model_path: std::path::PathBuf::from(format!("{}/v1", server.url())),
            api_key: Some("test".into()),
            api_format: Some("openai".into()),
            ..Default::default()
        };
        engine.load_model(req).expect("load_model should succeed");
        assert!(engine.is_model_loaded());
    }

    // ── 2. Anthropic format skips /models connectivity check ────────

    #[test]
    fn load_model_anthropic_skips_models_check() {
        // No /models mock — if it tried to hit /models, the request would fail.
        // Anthropic format should skip that check entirely.
        let server = mockito::Server::new();

        let engine = Engine::new();
        let req = LoadRequest {
            model_path: std::path::PathBuf::from(format!("{}/v1", server.url())),
            api_key: Some("test-key".into()),
            api_format: Some("anthropic".into()),
            ..Default::default()
        };
        engine
            .load_model(req)
            .expect("anthropic load should skip /models and succeed");
        assert!(engine.is_model_loaded());
    }

    // ── 3. OpenAI SSE token streaming ───────────────────────────────

    #[test]
    fn openai_sse_token_streaming() {
        let mut server = mockito::Server::new();

        // Mock the /models endpoint for load_model
        let _models_mock = server
            .mock("GET", "/v1/models")
            .with_status(200)
            .with_body(r#"{"data":[]}"#)
            .create();

        // SSE body with two tokens + stop + [DONE]
        let sse_body = "\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
\n\
data: [DONE]\n\
";

        let _completions_mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create();

        // Load model
        let engine = Engine::new();
        let req = LoadRequest {
            model_path: std::path::PathBuf::from(format!("{}/v1", server.url())),
            api_key: Some("test-key".into()),
            api_format: Some("openai".into()),
            ..Default::default()
        };
        engine.load_model(req).unwrap();

        // Start session with a user message
        let spec = SessionSpec {
            messages: vec![user_message("Hello")],
            ..Default::default()
        };
        let session = engine.start_session(spec).unwrap();

        // Pull tokens
        let puller = session.pull(GenSpec::default()).unwrap();
        let mut tokens = Vec::new();
        for event in puller {
            match event.unwrap() {
                TokenEvent::Token(tok) => tokens.push(tok.text),
                TokenEvent::Eos => break,
                TokenEvent::Stopped => break,
                _ => {}
            }
        }

        assert_eq!(tokens, vec!["Hello", " world"]);
    }

    // ── 4. Auth headers — OpenAI format ─────────────────────────────

    #[test]
    fn auth_headers_openai() {
        let mut server = mockito::Server::new();

        let _models_mock = server
            .mock("GET", "/v1/models")
            .with_status(200)
            .with_body(r#"{"data":[]}"#)
            .create();

        // Verify the POST has the correct Authorization header
        let _completions_mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header("Authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body("data: [DONE]\n\n")
            .create();

        let engine = Engine::new();
        let req = LoadRequest {
            model_path: std::path::PathBuf::from(format!("{}/v1", server.url())),
            api_key: Some("test-key".into()),
            api_format: Some("openai".into()),
            ..Default::default()
        };
        engine.load_model(req).unwrap();

        let spec = SessionSpec {
            messages: vec![user_message("Hi")],
            ..Default::default()
        };
        let session = engine.start_session(spec).unwrap();
        let puller = session.pull(GenSpec::default()).unwrap();

        // Consume the puller to trigger the HTTP request
        for event in puller {
            match event {
                Ok(TokenEvent::Eos) | Ok(TokenEvent::Stopped) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }

        // If mockito matched the header, the mock was satisfied.
        // An unmatched mock would have returned 501.
        _completions_mock.assert();
    }

    // ── 5. Auth headers — Anthropic format ──────────────────────────

    #[test]
    fn auth_headers_anthropic() {
        let mut server = mockito::Server::new();

        // No /models mock needed — Anthropic skips it.
        // Verify the POST has x-api-key and anthropic-version headers.
        let _messages_mock = server
            .mock("POST", "/v1/messages")
            .match_header("x-api-key", "test-key")
            .match_header("anthropic-version", "2023-06-01")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
            .create();

        let engine = Engine::new();
        let req = LoadRequest {
            model_path: std::path::PathBuf::from(format!("{}/v1", server.url())),
            api_key: Some("test-key".into()),
            api_format: Some("anthropic".into()),
            ..Default::default()
        };
        engine.load_model(req).unwrap();

        let spec = SessionSpec {
            messages: vec![user_message("Hi")],
            ..Default::default()
        };
        let session = engine.start_session(spec).unwrap();
        let puller = session.pull(GenSpec::default()).unwrap();

        for event in puller {
            match event {
                Ok(TokenEvent::Eos) | Ok(TokenEvent::Stopped) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }

        _messages_mock.assert();
    }
}
