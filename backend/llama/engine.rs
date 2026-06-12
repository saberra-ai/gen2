use arc_swap::{ArcSwap, ArcSwapOption};
use dashmap::DashMap;
use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
};

use super::bundle::ModelBundle;
use super::embedder::LlamaEmbedder;
use super::session::{Session, SessionId};
use crate::gen2::engine::telemetry::{HookBus, HookEvent};
use crate::gen2::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::gen2::session_rt::SessionSpec;
use crate::gen2::session_rt::media_util::messages_have_images;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use once_cell::sync::OnceCell;
use parking_lot::RwLock;

static BACKEND: OnceCell<Arc<LlamaBackend>> = OnceCell::new();

fn get_backend() -> Result<Arc<LlamaBackend>, ExecError> {
    BACKEND
        .get_or_try_init(|| {
            let inner = LlamaBackend::init().map_err(|e| ExecError::Other(e.into()))?;
            Ok(Arc::new(inner))
        })
        .map(Arc::clone)
}

pub struct Engine {
    bundle: ArcSwapOption<ModelBundle>,
    // Keep a lightweight registry to track IDs without holding non-Send Session types.
    sessions: DashMap<SessionId, ()>,
    settings: ArcSwap<Settings>,
    last_load: RwLock<Option<LoadRequest>>,
    settings_version: AtomicU64,
    next_session_id: AtomicU64,
    hooks: Arc<HookBus>,
    embedder: ArcSwapOption<LlamaEmbedder>,
    load_guard: parking_lot::Mutex<()>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("sessions", &self.sessions.len())
            .field(
                "settings_version",
                &self.settings_version.load(Ordering::SeqCst),
            )
            .field("has_bundle", &self.bundle.load_full().is_some())
            .field("has_embedder", &self.embedder.load_full().is_some())
            .finish()
    }
}

impl Engine {
    /// Construct an engine. Bundle/session internals arrive in later milestones.
    pub fn new() -> Self {
        // Route llama/ggml native logs away from stdout/stderr to avoid Windows GUI stderr assertions
        // Tauri on Windows often runs without a console; using tracing avoids touching closed stdio.
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));
        Self {
            bundle: ArcSwapOption::from(None),
            sessions: DashMap::new(),
            settings: ArcSwap::from_pointee(Settings::default()),
            last_load: RwLock::new(None),
            settings_version: AtomicU64::new(0),
            next_session_id: AtomicU64::new(1),
            hooks: Arc::new(HookBus::new()),
            embedder: ArcSwapOption::from(None),
            load_guard: parking_lot::Mutex::new(()),
        }
    }

    // 1) dynamically load models / optional mmproj — stubbed.
    pub fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let _g = self.load_guard.lock();
        // tracing::info!("engine.load_model.start", path=%req.model_path.display());
        // self.hooks.emit(HookEvent::EngineLoadStart { path: req.model_path.display().to_string() });
        let backend = get_backend()?;

        let bundle = super::loader::build_bundle(&backend, &req)?;
        let caps = bundle.capabilities.clone();
        let meta = bundle.meta.clone();
        // Drop session registry so callers can't keep using stale sessions after a reload.
        self.sessions.clear();
        self.bundle.store(Some(Arc::new(bundle)));
        *self.last_load.write() = Some(req);
        tracing::info!("engine.load_model.ok");
        self.hooks.emit(HookEvent::EngineLoadOk {
            caps_text: true,
            caps_images: caps.contains(Capabilities::IMAGES),
            caps_audio: caps.contains(Capabilities::AUDIO),
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

    pub fn load_embedder(&self, req: EmbedLoadRequest) -> Result<(), ExecError> {
        let backend = get_backend()?;
        let embedder = super::loader::build_embedder(&backend, &req)?;
        self.embedder.store(Some(Arc::new(embedder)));
        Ok(())
    }

    pub fn is_embedder_loaded(&self) -> bool {
        self.embedder.load_full().is_some()
    }

    // 2) upload settings
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

    // 3) start session — create context, prefill, sampler, and session state.
    pub fn start_session(&self, spec: SessionSpec) -> Result<Arc<Session>, ExecError> {
        let bundle = self.bundle.load_full().ok_or(ExecError::ModelNotLoaded)?;
        let backend = get_backend()?;
        let base_settings = self.settings();
        let settings = if let Some(mut overrides) = spec.overrides.clone() {
            overrides.inherit_missing(base_settings.as_ref());
            overrides
        } else {
            (*base_settings).clone()
        };
        {
            if messages_have_images(&spec.messages)
                && !bundle.capabilities.contains(Capabilities::IMAGES)
            {
                return Err(ExecError::FeatureUnsupported("images"));
            }
        }
        let id = self.next_session_id.fetch_add(1, Ordering::SeqCst);
        #[allow(clippy::arc_with_non_send_sync)]
        let session = Arc::new(Session::new(
            id,
            bundle.clone(),
            backend,
            self.hooks.clone(),
            settings.clone(),
            spec.messages,
            spec.persona.as_ref(),
        )?);
        if let Some(cache) = spec.cache.clone() {
            let _ = session.load_cache(cache)?; // apply cache best-effort here; strict/lenient is handled inside
        }
        self.sessions.insert(id, ());
        Ok(session)
    }
    // For now, targeted stopping is handled by higher layers (chat-scoped flags).

    pub fn end_session(&self, id: SessionId) -> Result<(), ExecError> {
        // Remove from the lightweight registry.
        if self.sessions.remove(&id).is_some() {
            // Optional: emit a hook if you have/introduce one.
            // self.hooks.emit(HookEvent::SessionEnded { id });
            Ok(())
        } else {
            Err(ExecError::InvalidArg("unknown session id"))
        }
    }

    // 6) utils
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

    // stats placeholder
    pub fn stats(&self) -> ExecutionStats {
        ExecutionStats::default()
    }

    pub fn generate_embeddings(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ExecError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let embedder = self
            .embedder
            .load_full()
            .ok_or(ExecError::EmbedderNotLoaded)?;
        let slices: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
        embedder.embed(&slices, false).map_err(ExecError::Other)
    }

    pub fn unload_model(&self) {
        self.bundle.store(None); // drops when last session releases
    }
    pub fn unload_embedder(&self) {
        self.embedder.store(None);
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Trait impls (Phase 2) ─────────────────────────────────────────────────
//
// Each method forwards to the existing inherent method of the same name. The
// facade's enum dispatch remains in charge until Phase 4 flips it.

use crate::gen2::backend::caps::LatencyTier;
use crate::gen2::backend::traits::{Backend, BackendSession, Embeddings, LocalBackend, Multimodal};

impl Backend for Engine {
    fn backend_name(&self) -> &'static str {
        "llamacpp"
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
        LatencyTier::Fast
    }
    fn bundle_architecture(&self) -> Option<String> {
        self.bundle
            .load_full()
            .and_then(|b| b.meta.architecture.clone())
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
    fn as_multimodal(&self) -> Option<&dyn Multimodal> {
        Some(self)
    }
}

impl LocalBackend for Engine {
    fn n_ctx(&self) -> usize {
        self.bundle
            .load_full()
            .map(|b| b.meta.n_ctx as usize)
            .unwrap_or(0)
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

impl Multimodal for Engine {
    fn supports_images(&self) -> bool {
        Engine::does_model_support_images(self)
    }
    fn supports_audio(&self) -> bool {
        Engine::does_model_support_audio(self)
    }
}

// Drop guard — consumed when session lifetime is active
#[allow(dead_code)]
struct SessionGuard {
    id: SessionId,
    engine: Weak<Engine>,
}
impl Drop for SessionGuard {
    fn drop(&mut self) {
        if let Some(engine) = self.engine.upgrade() {
            let _ = engine.sessions.remove(&self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen2::generation::GenSpec;
    use crate::gen2::generation::TokenEvent;
    use crate::gen2::session_rt::SessionSpec;
    use crate::gen2::{Message, MessageBody, MessageContent};
    use std::path::PathBuf;

    #[test]
    fn default_engine_state() {
        let e = Engine::new();
        assert!(!e.is_model_loaded());
        assert!(!e.does_model_support_images());
        assert!(!e.does_model_support_audio());
    }

    #[test]
    fn reload_without_load_fails() {
        let e = Engine::new();
        let err = e.reload_model().unwrap_err();
        assert!(matches!(err, ExecError::ModelNotLoaded));
    }

    #[test]
    fn embedding_without_load_fails() {
        let e = Engine::new();
        assert!(!e.is_embedder_loaded());
        let err = e.generate_embeddings(&["hello".to_string()]).unwrap_err();
        matches!(err, ExecError::EmbedderNotLoaded);
    }

    #[test]
    fn settings_validation() {
        let e = Engine::new();
        let mut s = Settings::default();
        s.sampling.temperature = Some(0.7);
        s.sampling.top_p = Some(0.9);
        s.sampling.top_k = Some(50);
        e.upload_settings(s).expect("valid settings");
        assert_eq!(e.settings_version(), 1);
    }

    #[test]
    fn settings_invalid_temp() {
        let e = Engine::new();
        let mut s = Settings::default();
        s.sampling.temperature = Some(3.0);
        let err = e.upload_settings(s).unwrap_err();
        matches!(err, ExecError::SettingsError(_));
    }

    // Helper: read model path from TEST_MODEL_PATH; return None if not set.
    fn test_model_path() -> Option<PathBuf> {
        // std::env::var("TEST_MODEL_PATH").ok().map(Into::into)
        // Some("C:/Users/vlope/Downloads/gemma-3-4b-it-qat-Q4_K_S.gguf".into())
        Some("/Users/victorlopez/Downloads/gemma-3-4b-it-qat-Q4_K_M.gguf".into())
    }

    // Ignored by default: requires a local GGUF model.
    #[test]
    #[ignore]
    fn load_model_from_env() -> Result<(), Box<dyn std::error::Error>> {
        let Some(model_path) = test_model_path() else {
            return Ok(());
        };
        let e = Engine::new();
        e.load_model(LoadRequest {
            model_path,
            ..Default::default()
        })?;
        assert!(e.is_model_loaded());
        assert!(e.capabilities().contains(Capabilities::TEXT));
        Ok(())
    }

    // Ignored by default: end-to-end prompt → tokens pull.
    #[test]
    #[ignore]
    fn simple_generation_smoke() -> Result<(), Box<dyn std::error::Error>> {
        let Some(model_path) = test_model_path() else {
            return Ok(());
        };
        let e = Engine::new();
        // Keep context small to avoid large KV allocations on CI/dev
        let mut s = Settings::default();
        s.system.ctx_size = Some(512);
        s.system.batch_size = Some(16);
        s.system.threads = Some(2);
        s.system.threads_batch = Some(2);
        e.upload_settings(s)?;
        e.load_model(LoadRequest {
            model_path,
            ..Default::default()
        })?;

        let msgs = vec![Message {
            name: None,
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText("Hello".into()),
            },
        }];
        let s = e.start_session(SessionSpec {
            messages: msgs,
            ..Default::default()
        })?;
        let mut puller = s.pull(GenSpec {
            max_tokens: Some(1000),
            ..Default::default()
        })?;

        let mut got_any = false;
        let mut steps = 0u32;
        let mut result = String::with_capacity(1024); // preallocate a bit
        for ev in puller.by_ref() {
            steps += 1;
            match ev? {
                TokenEvent::Token(tok) if !tok.text.is_empty() => {
                    got_any = true;
                    result.push_str(tok.text.as_str());
                }
                TokenEvent::Eos | TokenEvent::Stopped => break,
                TokenEvent::Paused => continue,
                _ => {}
            }
            if steps > 128 {
                break;
            }
        }
        assert!(got_any, "no tokens produced");
        Ok(())
    }

    fn test_mmproj_path() -> Option<PathBuf> {
        // std::env::var("TEST_MMPROJ_PATH").ok().map(Into::into)
        // Some("C:/Users/vlope/Downloads/mmproj-F16.gguf".into())
        Some("/Users/victorlopez/Downloads/mmproj-F16.gguf".into())
    }

    fn test_image_path() -> Option<PathBuf> {
        // std::env::var("TEST_IMAGE_PATH").ok().map(Into::into)
        // Some("C:/Users/vlope/Downloads/cat.jfif".into())
        Some("/Users/victorlopez/Downloads/profile.jpg".into())
    }

    /// Minimal MTMD smoke test: load mmproj, run one image+text turn.
    #[test]
    #[ignore]
    fn multimodal_image_smoke() -> Result<(), Box<dyn std::error::Error>> {
        use crate::types::message::{MessageChunk, Url};
        let (Some(model_path), Some(mmproj_path), Some(image_path)) =
            (test_model_path(), test_mmproj_path(), test_image_path())
        else {
            eprintln!(
                "set TEST_MODEL_PATH, TEST_MMPROJ_PATH, and TEST_IMAGE_PATH to run this test"
            );
            return Ok(());
        };

        let e = Engine::new();
        let mut s = Settings::default();
        s.system.ctx_size = Some(512);
        s.system.batch_size = Some(16);
        s.system.threads = Some(2);
        s.system.threads_batch = Some(2);
        e.upload_settings(s)?;
        e.load_model(LoadRequest {
            model_path,
            mmproj_path: Some(mmproj_path),
            ..Default::default()
        })?;
        assert!(e.does_model_support_images());

        let img_url = format!("file://{}", image_path.display());
        let msgs = vec![Message {
            name: None,
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::MultipleChunks(vec![
                    MessageChunk::Text {
                        text: "Describe this photo please, write as if you're a weeb:".into(),
                    },
                    MessageChunk::ImageUrl {
                        image_url: Url { url: img_url },
                    },
                ]),
            },
        }];

        let s = e.start_session(SessionSpec {
            messages: msgs,
            ..Default::default()
        })?;
        let mut puller = s.pull(GenSpec {
            max_tokens: Some(128),
            ..Default::default()
        })?;

        let mut saw_media = false;
        let mut got_any = false;
        let mut steps = 0u32;
        let mut result = String::with_capacity(1024); // preallocate a bit
        for ev in puller.by_ref() {
            steps += 1;
            match ev? {
                TokenEvent::Token(tok) if !tok.text.is_empty() => {
                    result.push_str(tok.text.as_str());
                    got_any = true;
                }
                TokenEvent::MediaBoundary(_) => {
                    saw_media = true;
                }
                TokenEvent::Eos | TokenEvent::Stopped => break,
                TokenEvent::Paused => continue,
                _ => {}
            }
            if steps > 128 {
                break;
            }
        }
        println!("{}", result);
        assert!(saw_media, "no media boundary observed");
        assert!(got_any, "no tokens produced");
        Ok(())
    }
    #[test]
    #[ignore]
    fn kv_cache_persist_and_reload() -> Result<(), Box<dyn std::error::Error>> {
        use tempfile::NamedTempFile;
        let Some(model_path) = test_model_path() else {
            return Ok(());
        };

        // Phase 1: engine A loads, generates, and saves a KV cache (from a fresh session)
        let e1 = Engine::new();
        let mut s = Settings::default();
        s.system.ctx_size = Some(512);
        s.system.batch_size = Some(16);
        s.system.threads = Some(2);
        s.system.threads_batch = Some(2);
        e1.upload_settings(s)?;
        e1.load_model(LoadRequest {
            model_path: model_path.clone(),
            ..Default::default()
        })?;

        // Generate briefly to confirm engine is healthy
        let msgs = vec![Message {
            name: None,
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText("Hello".into()),
            },
        }];
        let s_gen = e1.start_session(SessionSpec {
            messages: msgs.clone(),
            ..Default::default()
        })?;
        let mut p = s_gen.pull(GenSpec {
            max_tokens: Some(8),
            ..Default::default()
        })?;
        let mut seen = 0usize;
        for ev in p.by_ref() {
            if matches!(ev?, TokenEvent::Token(_)) {
                seen += 1;
                if seen >= 2 {
                    break;
                }
            }
        }
        assert!(seen >= 1);

        // Create a fresh session with same messages and save its KV (post-prefill)
        let s_cache = e1.start_session(SessionSpec {
            messages: msgs.clone(),
            ..Default::default()
        })?;
        let tmp = NamedTempFile::new()?;
        let kv_path = tmp.path().to_path_buf();
        let snap = s_cache.save_cache(crate::gen2::kv::KvSaveSpec::ToPath(kv_path.clone()))?;
        assert!(snap.tokens_covered > 0);

        drop(s_gen);
        drop(s_cache);
        drop(e1); // simulate engine shutdown

        // Phase 2: engine B loads, starts a session, and applies the saved KV cache
        let e2 = Engine::new();
        let mut s2 = Settings::default();
        s2.system.ctx_size = Some(512);
        s2.system.batch_size = Some(16);
        s2.system.threads = Some(2);
        s2.system.threads_batch = Some(2);
        e2.upload_settings(s2)?;
        e2.load_model(LoadRequest {
            model_path,
            ..Default::default()
        })?;
        let s2 = e2.start_session(SessionSpec {
            messages: msgs,
            ..Default::default()
        })?;
        let report = s2.load_cache(crate::gen2::kv::KvLoadSpec::Strict(kv_path))?;
        assert!(report.loaded, "kv cache failed to load strictly");
        Ok(())
    }

    /// Verify that generation stops at or before max_tokens.
    #[test]
    #[ignore]
    fn generation_respects_max_tokens() -> Result<(), Box<dyn std::error::Error>> {
        let Some(model_path) = test_model_path() else {
            return Ok(());
        };
        let e = Engine::new();
        let mut s = Settings::default();
        s.system.ctx_size = Some(512);
        s.system.batch_size = Some(16);
        s.system.threads = Some(2);
        s.system.threads_batch = Some(2);
        e.upload_settings(s)?;
        e.load_model(LoadRequest {
            model_path,
            ..Default::default()
        })?;

        let msgs = vec![Message {
            name: None,
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText("Hello".into()),
            },
        }];
        let session = e.start_session(SessionSpec {
            messages: msgs,
            ..Default::default()
        })?;
        let mut puller = session.pull(GenSpec {
            max_tokens: Some(5),
            ..Default::default()
        })?;

        let mut token_count = 0usize;
        for ev in puller.by_ref() {
            match ev? {
                TokenEvent::Token(_) => token_count += 1,
                TokenEvent::Eos | TokenEvent::Stopped => break,
                _ => {}
            }
        }
        assert!(
            token_count <= 5,
            "expected at most 5 tokens, got {}",
            token_count
        );
        Ok(())
    }

    /// Verify that calling stop() on the session yields TokenEvent::Stopped.
    #[test]
    #[ignore]
    fn generation_stop_flag() -> Result<(), Box<dyn std::error::Error>> {
        let Some(model_path) = test_model_path() else {
            return Ok(());
        };
        let e = Engine::new();
        let mut s = Settings::default();
        s.system.ctx_size = Some(512);
        s.system.batch_size = Some(16);
        s.system.threads = Some(2);
        s.system.threads_batch = Some(2);
        e.upload_settings(s)?;
        e.load_model(LoadRequest {
            model_path,
            ..Default::default()
        })?;

        let msgs = vec![Message {
            name: None,
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText("Hello".into()),
            },
        }];
        let session = e.start_session(SessionSpec {
            messages: msgs,
            ..Default::default()
        })?;
        let mut puller = session.pull(GenSpec {
            max_tokens: Some(1000),
            ..Default::default()
        })?;

        // Consume first token then request stop
        let mut got_first = false;
        let mut got_stopped = false;
        for ev in puller.by_ref() {
            match ev? {
                TokenEvent::Token(_) if !got_first => {
                    got_first = true;
                    session.stop();
                }
                TokenEvent::Stopped => {
                    got_stopped = true;
                    break;
                }
                TokenEvent::Eos => break,
                _ => {}
            }
        }
        assert!(got_first, "should have produced at least one token");
        assert!(got_stopped, "should have received TokenEvent::Stopped");
        Ok(())
    }

    /// Pulling twice from the same session should fail with "session already consumed".
    #[test]
    #[ignore]
    fn session_consumed_on_pull() -> Result<(), Box<dyn std::error::Error>> {
        let Some(model_path) = test_model_path() else {
            return Ok(());
        };
        let e = Engine::new();
        let mut s = Settings::default();
        s.system.ctx_size = Some(512);
        s.system.batch_size = Some(16);
        s.system.threads = Some(2);
        s.system.threads_batch = Some(2);
        e.upload_settings(s)?;
        e.load_model(LoadRequest {
            model_path,
            ..Default::default()
        })?;

        let msgs = vec![Message {
            name: None,
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText("Hello".into()),
            },
        }];
        let session = e.start_session(SessionSpec {
            messages: msgs,
            ..Default::default()
        })?;

        // First pull should succeed
        let _puller = session.pull(GenSpec::default())?;

        // Second pull should fail — state was consumed
        match session.pull(GenSpec::default()) {
            Err(err) => {
                let msg = format!("{}", err);
                assert!(
                    msg.contains("session already consumed"),
                    "unexpected error: {}",
                    msg
                );
            }
            Ok(_) => panic!("expected error on second pull, but got Ok"),
        }
        Ok(())
    }

    /// Embedder with empty input should return Ok(vec![]).
    #[test]
    #[ignore]
    fn embedder_empty_input() -> Result<(), Box<dyn std::error::Error>> {
        let Some(model_path) = test_model_path() else {
            return Ok(());
        };
        let e = Engine::new();
        e.load_model(LoadRequest {
            model_path: model_path.clone(),
            ..Default::default()
        })?;
        // Load embedder with the same model path (may or may not support it,
        // but the empty-input short-circuit should fire before backend errors)
        let _ = e.load_embedder(EmbedLoadRequest { model_path });

        let result = e.generate_embeddings(&[])?;
        assert!(result.is_empty(), "expected empty vec for empty input");
        Ok(())
    }
}
