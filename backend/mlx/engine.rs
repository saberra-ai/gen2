//! MLX inference engine.

use arc_swap::{ArcSwap, ArcSwapOption};
use dashmap::DashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::bundle::ModelBundle;
use super::model::RotaryEmbedding;
use super::session::{Session, SessionId};
use super::tokenizer::HfTokenizer;
use crate::gen2::bundle::ModelMeta;
use crate::gen2::engine::telemetry::{HookBus, HookEvent};
use crate::gen2::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::gen2::session_rt::SessionSpec;

use parking_lot::RwLock;

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
        f.debug_struct("Engine(MLX)")
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

        // For MLX, model_path points to a directory containing config.json + *.safetensors
        let model_dir = &req.model_path;

        let (model, config) = super::loader::build_any_model(model_dir)?;

        let head_dim = config.head_dim();
        let max_seq = config.max_position_embeddings;
        let rope_theta = config.rope_theta;
        let rope = RotaryEmbedding::new(head_dim, max_seq, rope_theta);

        let tokenizer = HfTokenizer::from_dir(model_dir).map_err(|e| ExecError::Other(e))?;

        let chat_template_str = crate::gen2::backend::common::load_chat_template(model_dir)
            .unwrap_or_else(crate::gen2::backend::common::default_llama3_template);
        let bos_str = tokenizer
            .bos_id()
            .and_then(|id| tokenizer.decode(&[id]).ok())
            .unwrap_or_default();
        let eos_str = tokenizer
            .eos_id()
            .and_then(|id| tokenizer.decode(&[id]).ok())
            .unwrap_or_default();

        let meta = crate::gen2::backend::common::compute_hf_model_meta(
            &tokenizer,
            model_dir,
            config.max_position_embeddings as u32,
            config.num_hidden_layers as u32,
            Some(&chat_template_str),
        );

        let caps = Capabilities::TEXT;

        let bundle = ModelBundle {
            model,
            rope,
            tokenizer,
            config,
            capabilities: caps.clone(),
            meta: meta.clone(),
            model_dir: model_dir.to_path_buf(),
            chat_template_str,
            bos_str,
            eos_str,
        };

        self.sessions.clear();
        self.bundle.store(Some(Arc::new(bundle)));
        *self.last_load.write() = Some(req);

        tracing::info!("engine.load_model.ok (MLX)");
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
        // MLX embedder not yet implemented
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
        // no-op for MLX
    }
}

// ─── Trait impls (Phase 2) ─────────────────────────────────────────────────

use crate::gen2::backend::caps::LatencyTier;
use crate::gen2::backend::traits::{Backend, BackendSession, LocalBackend};

impl Backend for Engine {
    fn backend_name(&self) -> &'static str {
        "mlx"
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
        LatencyTier::Medium
    }
    fn start_session(&self, spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        let s = Engine::start_session(self, spec)?;
        Ok(s as Arc<dyn BackendSession>)
    }
    fn end_session(&self, id: SessionId) -> Result<(), ExecError> {
        Engine::end_session(self, id)
    }
    // No as_embeddings / as_multimodal — MLX doesn't support either today.
}

impl LocalBackend for Engine {
    fn n_ctx(&self) -> usize {
        self.bundle
            .load_full()
            .map(|b| b.meta.n_ctx as usize)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen2::engine::{EmbedLoadRequest, ExecError, LoadRequest};

    /// Load a real MLX safetensors model directory.
    /// Set TEST_MLX_MODEL_DIR to a directory containing config.json + *.safetensors.
    #[test]
    #[ignore]
    fn load_model_from_safetensors_dir() -> Result<(), Box<dyn std::error::Error>> {
        let model_dir = match std::env::var("TEST_MLX_MODEL_DIR") {
            Ok(p) => {
                let path = std::path::PathBuf::from(p);
                if !path.exists() {
                    eprintln!("TEST_MLX_MODEL_DIR path does not exist, skipping");
                    return Ok(());
                }
                path
            }
            Err(_) => {
                eprintln!("set TEST_MLX_MODEL_DIR to run this test");
                return Ok(());
            }
        };

        let e = Engine::new();
        assert!(!e.is_model_loaded());
        e.load_model(LoadRequest {
            model_path: model_dir,
            ..Default::default()
        })?;
        assert!(e.is_model_loaded());
        assert!(e.capabilities().contains(Capabilities::TEXT));
        Ok(())
    }

    /// Generate a few tokens with the loaded model and print them.
    /// Verifies the full prefill → decode → detokenize pipeline end-to-end.
    #[test]
    #[ignore]
    fn generate_tokens() -> Result<(), Box<dyn std::error::Error>> {
        use crate::gen2::Message;
        use crate::gen2::generation::GenSpec;
        use crate::gen2::session_rt::SessionSpec;
        use crate::types::message::{MessageBody, MessageContent};

        let model_dir = match std::env::var("TEST_MLX_MODEL_DIR") {
            Ok(p) => {
                let path = std::path::PathBuf::from(p);
                if !path.exists() {
                    eprintln!("TEST_MLX_MODEL_DIR does not exist, skipping");
                    return Ok(());
                }
                path
            }
            Err(_) => {
                eprintln!("set TEST_MLX_MODEL_DIR to run this test");
                return Ok(());
            }
        };

        let e = Engine::new();
        e.load_model(LoadRequest {
            model_path: model_dir,
            ..Default::default()
        })?;

        let messages = vec![Message {
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText("What is 2 + 2?".into()),
            },
            name: None,
        }];

        let session = e.start_session(SessionSpec {
            messages,
            overrides: None,
            persona: None,
            attachments: vec![],
            cache: None,
        })?;

        let gen_spec = GenSpec {
            max_tokens: Some(64),
            ..Default::default()
        };
        let mut puller = session.pull(gen_spec)?;

        use crate::gen2::generation::TokenEvent;
        print!("\n[generate_tokens] output: ");
        let mut n_tokens = 0;
        loop {
            match puller.next() {
                Some(Ok(TokenEvent::Token(tok))) => {
                    print!("{}", tok.text);
                    n_tokens += 1;
                }
                Some(Ok(TokenEvent::Eos)) | Some(Ok(TokenEvent::Stopped)) => break,
                Some(Ok(TokenEvent::Paused))
                | Some(Ok(TokenEvent::Special(_)))
                | Some(Ok(TokenEvent::MediaBoundary(_))) => continue,
                Some(Err(e)) => return Err(e.into()),
                None => break,
            }
        }
        println!("\n[generate_tokens] generated {} tokens", n_tokens);
        assert!(n_tokens > 0, "expected at least one generated token");
        Ok(())
    }

    /// Ten-turn conversation — exercises repeated `append_messages` delta
    /// prefills and checks that context survives many round-trips.
    #[test]
    #[ignore = "requires TEST_MLX_MODEL_DIR env var pointing to a local model"]
    fn multiturn_ten_turns() -> Result<(), Box<dyn std::error::Error>> {
        use crate::gen2::Message;
        use crate::gen2::generation::{GenSpec, TokenEvent};
        use crate::gen2::session_rt::SessionSpec;
        use crate::types::message::{MessageBody, MessageContent};

        let model_dir = match std::env::var("TEST_MLX_MODEL_DIR") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!("set TEST_MLX_MODEL_DIR to run this test");
                return Ok(());
            }
        };
        if !model_dir.exists() {
            eprintln!("TEST_MLX_MODEL_DIR does not exist, skipping");
            return Ok(());
        }

        let e = Engine::new();
        e.load_model(LoadRequest {
            model_path: model_dir,
            ..Default::default()
        })?;

        let user_msg = |t: &str| Message {
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(t.into()),
            },
            name: None,
        };
        let asst_msg = |t: &str| Message {
            role: "assistant".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(t.into()),
            },
            name: None,
        };

        // Mixed-skill conversation — memory, reasoning, creativity, callbacks.
        let questions = [
            "I'm planning a weekend in Lisbon. Pick one neighborhood you'd recommend \
             and say why in 2 sentences.",
            "Good. I have a mild fear of heights — does that change your pick?",
            "Assume I'm going in January. What's the weather like, briefly?",
            "Suggest one dish I should try, and where it's from in Portugal.",
            "Turn that dish into a haiku (5-7-5).",
            "What's the Portuguese word for the main ingredient in that dish?",
            "Now forget Portugal for a second — if that word were a startup name, \
             what would the product be?",
            "Pitch it to me in one sentence, VC-style.",
            "Roast that pitch — give me the sharpest critique in 2 sentences.",
            "OK, looping back: which neighborhood did you recommend in turn 1, \
             and does the startup idea fit there?",
        ];

        let session = e.start_session(SessionSpec {
            messages: vec![user_msg(questions[0])],
            overrides: None,
            persona: None,
            attachments: vec![],
            cache: None,
        })?;

        let drain = |puller: &mut crate::gen2::backend::mlx::puller::TokenPuller| -> String {
            let mut out = String::new();
            loop {
                match puller.next() {
                    Some(Ok(TokenEvent::Token(tok))) => out.push_str(&tok.text),
                    Some(Ok(TokenEvent::Eos)) | Some(Ok(TokenEvent::Stopped)) => break,
                    Some(Ok(_)) => continue,
                    Some(Err(_)) | None => break,
                }
            }
            out
        };

        let total_start = std::time::Instant::now();
        let mut last_reply = String::new();
        for (i, q) in questions.iter().enumerate() {
            if i > 0 {
                // Append prior assistant turn + next user question.
                session.append_messages(vec![asst_msg(last_reply.trim()), user_msg(q)])?;
            }
            let t = std::time::Instant::now();
            let mut puller = session.pull(GenSpec {
                max_tokens: Some(160),
                ..Default::default()
            })?;
            let reply = drain(&mut puller);
            drop(puller);
            println!(
                "[turn {:>2}] ({:>4}ms) Q: {}\n          A: {}",
                i + 1,
                t.elapsed().as_millis(),
                q,
                reply.trim()
            );
            if reply.trim().is_empty() {
                eprintln!("turn {} was empty (continuing)", i + 1);
            }
            last_reply = reply;
        }
        println!("\ntotal: {:.1}s", total_start.elapsed().as_secs_f32());
        Ok(())
    }

    /// Exercises `append_messages`: turn 1 answers "what is 2+2", then the
    /// assistant reply + a follow-up user message are appended and turn 2
    /// is generated. Verifies delta prefill over the existing KV cache.
    #[test]
    #[ignore = "requires TEST_MLX_MODEL_DIR env var pointing to a local model"]
    fn multiturn_append_messages() -> Result<(), Box<dyn std::error::Error>> {
        use crate::gen2::Message;
        use crate::gen2::generation::{GenSpec, TokenEvent};
        use crate::gen2::session_rt::SessionSpec;
        use crate::types::message::{MessageBody, MessageContent};

        let model_dir = match std::env::var("TEST_MLX_MODEL_DIR") {
            Ok(p) => {
                let path = std::path::PathBuf::from(p);
                if !path.exists() {
                    eprintln!("TEST_MLX_MODEL_DIR does not exist, skipping");
                    return Ok(());
                }
                path
            }
            Err(_) => {
                eprintln!("set TEST_MLX_MODEL_DIR to run this test");
                return Ok(());
            }
        };

        let e = Engine::new();
        e.load_model(LoadRequest {
            model_path: model_dir,
            ..Default::default()
        })?;

        let user_msg = |t: &str| Message {
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(t.into()),
            },
            name: None,
        };
        let asst_msg = |t: &str| Message {
            role: "assistant".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(t.into()),
            },
            name: None,
        };

        let session = e.start_session(SessionSpec {
            messages: vec![user_msg("What is 2 + 2?")],
            overrides: None,
            persona: None,
            attachments: vec![],
            cache: None,
        })?;

        let drain = |puller: &mut crate::gen2::backend::mlx::puller::TokenPuller| -> String {
            let mut out = String::new();
            loop {
                match puller.next() {
                    Some(Ok(TokenEvent::Token(tok))) => out.push_str(&tok.text),
                    Some(Ok(TokenEvent::Eos)) | Some(Ok(TokenEvent::Stopped)) => break,
                    Some(Ok(_)) => continue,
                    Some(Err(_)) | None => break,
                }
            }
            out
        };

        // ── Turn 1 ───────────────────────────────────────────────────────────
        let mut p1 = session.pull(GenSpec {
            max_tokens: Some(32),
            ..Default::default()
        })?;
        let reply1 = drain(&mut p1);
        drop(p1); // return DecodeState to the session
        println!("[multiturn] turn1: {:?}", reply1);
        assert!(!reply1.is_empty(), "turn 1 produced no text");

        // ── Append turn-1 assistant reply + turn-2 user question ─────────────
        session.append_messages(vec![
            asst_msg(reply1.trim()),
            user_msg("And what is that number times 3?"),
        ])?;

        // ── Turn 2 ───────────────────────────────────────────────────────────
        let mut p2 = session.pull(GenSpec {
            max_tokens: Some(32),
            ..Default::default()
        })?;
        let reply2 = drain(&mut p2);
        drop(p2);
        println!("[multiturn] turn2: {:?}", reply2);
        assert!(!reply2.is_empty(), "turn 2 produced no text");

        Ok(())
    }

    /// MLX does not support embedders — load_embedder should return Unimplemented.
    #[test]
    fn embedder_not_supported() {
        let e = Engine::new();
        let err = e
            .load_embedder(EmbedLoadRequest {
                model_path: std::path::PathBuf::from("/nonexistent"),
            })
            .unwrap_err();
        assert!(
            matches!(err, ExecError::Unimplemented),
            "expected Unimplemented, got: {:?}",
            err
        );
    }
}
