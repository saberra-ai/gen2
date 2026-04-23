//! MLX inference engine.

use arc_swap::{ArcSwap, ArcSwapOption};
use dashmap::DashMap;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::bundle::ModelBundle;
use super::model::RotaryEmbedding;
use super::session::{PrefixCacheEntry, Session, SessionId, hash_str};
use super::tokenizer::HfTokenizer;
use crate::gen2::engine::telemetry::{HookBus, HookEvent};
use crate::gen2::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::gen2::session_rt::SessionSpec;

use parking_lot::RwLock;

// ─── PrefixCacheLru ───────────────────────────────────────────────────────────

/// Maximum number of distinct system-prompt prefixes held in the KV cache.
///
/// Each entry holds full KV tensors at `cur_pos` tokens — at ~512 prefix tokens
/// on a 7B model this is ~50 MB per entry, so 4 entries ≈ 200 MB Metal RAM in
/// the worst case. Invalidated on model reload.
const PREFIX_LRU_CAP: usize = 4;

/// Small LRU of [`PrefixCacheEntry`] keyed by `PrefixCacheEntry::key`.
///
/// Front of the deque is most-recently-used. Linear scan is fine at this size.
struct PrefixCacheLru {
    entries: VecDeque<PrefixCacheEntry>,
}

impl PrefixCacheLru {
    fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(PREFIX_LRU_CAP),
        }
    }

    /// Return a reference to the entry matching `key`, moving it to the front.
    fn touch(&mut self, key: u64) -> Option<&PrefixCacheEntry> {
        let pos = self.entries.iter().position(|e| e.key == key)?;
        if pos != 0 {
            let entry = self.entries.remove(pos)?;
            self.entries.push_front(entry);
        }
        self.entries.front()
    }

    /// Insert or replace `entry`, evicting the LRU entry if at capacity.
    fn insert(&mut self, entry: PrefixCacheEntry) {
        // Replace existing entry with the same key (dedup).
        if let Some(pos) = self.entries.iter().position(|e| e.key == entry.key) {
            self.entries.remove(pos);
        } else if self.entries.len() >= PREFIX_LRU_CAP {
            self.entries.pop_back();
        }
        self.entries.push_front(entry);
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

// ─── WarmSlot ─────────────────────────────────────────────────────────────────

/// Pre-loaded model bundle waiting for the next `load_model` call.
///
/// Built in a background thread by `Engine::warm_model`; consumed and cleared
/// by `Engine::load_model` when the paths match, skipping the synchronous
/// weight-loading step entirely.
struct WarmSlot {
    model_dir: PathBuf,
    bundle: Arc<ModelBundle>,
}

// ─── Engine ───────────────────────────────────────────────────────────────────

pub struct Engine {
    bundle: ArcSwapOption<ModelBundle>,
    /// Pre-loaded bundle for the next `load_model` call.
    warm_slot: Arc<RwLock<Option<WarmSlot>>>,
    /// Cached KV state for recently seen system prompt + persona combinations.
    /// Up to [`PREFIX_LRU_CAP`] distinct prefixes, LRU-evicted.
    /// Cleared on model reload.
    prefix_cache: parking_lot::Mutex<PrefixCacheLru>,
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

// ─── Bundle builder ───────────────────────────────────────────────────────────

/// Build a [`ModelBundle`] from a model directory (config + safetensors).
///
/// Used by both `load_model` (cold path) and the `warm_model` background thread.
fn build_bundle_from_dir(model_dir: &Path) -> Result<ModelBundle, ExecError> {
    let (model, config) = super::loader::build_any_model(model_dir)?;

    let head_dim = config.head_dim();
    let max_seq = config.max_position_embeddings;
    let rope_theta = config.rope_theta;
    let rope = RotaryEmbedding::new(head_dim, max_seq, rope_theta);

    let tokenizer = HfTokenizer::from_dir(model_dir).map_err(ExecError::Other)?;

    let chat_template_str = crate::gen2::backend::common::load_chat_template(model_dir)
        .unwrap_or_else(crate::gen2::backend::common::default_llama3_template);
    // Decode keeping specials: we want `{{ bos_token }}` in the chat template
    // to expand to the literal `<bos>` string. `decode(..., skip_special=true)`
    // would silently strip it, producing a prompt without the leading BOS —
    // which caused Gemma 4 to see a different embedding at position 0 and
    // produce catastrophically wrong logits on step 1 ("--- own neighborhood
    // neighborhood neighborhood …").
    let bos_str = tokenizer
        .bos_id()
        .and_then(|id| tokenizer.decode_keep_specials(&[id]).ok())
        .unwrap_or_default();
    let eos_str = tokenizer
        .eos_id()
        .and_then(|id| tokenizer.decode_keep_specials(&[id]).ok())
        .unwrap_or_default();

    let meta = crate::gen2::backend::common::compute_hf_model_meta(
        &tokenizer,
        model_dir,
        config.max_position_embeddings as u32,
        config.num_hidden_layers as u32,
        Some(&chat_template_str),
    );

    Ok(ModelBundle {
        model,
        rope,
        tokenizer,
        config,
        capabilities: Capabilities::TEXT,
        meta,
        model_dir: model_dir.to_path_buf(),
        chat_template_str,
        bos_str,
        eos_str,
    })
}

// ─── Engine impl ──────────────────────────────────────────────────────────────

impl Engine {
    pub fn new() -> Self {
        Self {
            bundle: ArcSwapOption::from(None),
            warm_slot: Arc::new(RwLock::new(None)),
            prefix_cache: parking_lot::Mutex::new(PrefixCacheLru::new()),
            sessions: DashMap::new(),
            settings: ArcSwap::from_pointee(Settings::default()),
            last_load: RwLock::new(None),
            settings_version: AtomicU64::new(0),
            next_session_id: AtomicU64::new(1),
            hooks: Arc::new(HookBus::new()),
            load_guard: parking_lot::Mutex::new(()),
        }
    }

    /// Pre-load a model in a background thread so the next `load_model` for the
    /// same path skips synchronous weight I/O entirely.
    ///
    /// No-op if the path is already loaded or already warming.
    pub fn warm_model(&self, model_dir: PathBuf) {
        // Already loaded with this path — nothing to warm.
        if let Some(b) = self.bundle.load_full() {
            if b.model_dir == model_dir {
                return;
            }
        }
        // Warm slot already holds this path or is loading it.
        {
            let slot = self.warm_slot.read();
            if let Some(s) = slot.as_ref() {
                if s.model_dir == model_dir {
                    return;
                }
            }
        }

        let slot_arc = Arc::clone(&self.warm_slot);
        std::thread::spawn(move || {
            tracing::info!(path = %model_dir.display(), "mlx warm_model: starting background load");
            match build_bundle_from_dir(&model_dir) {
                Ok(bundle) => {
                    *slot_arc.write() = Some(WarmSlot {
                        model_dir: model_dir.clone(),
                        bundle: Arc::new(bundle),
                    });
                    tracing::info!(path = %model_dir.display(), "mlx warm_model: slot ready");
                }
                Err(e) => {
                    tracing::warn!(path = %model_dir.display(), err = %e, "mlx warm_model: failed");
                }
            }
        });
    }

    pub fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let _g = self.load_guard.lock();
        let model_dir = &req.model_path;

        // Fast path: warm slot has a pre-loaded bundle for this path.
        let bundle = {
            let mut slot = self.warm_slot.write();
            if let Some(s) = slot.as_ref() {
                if s.model_dir == *model_dir {
                    tracing::info!("engine.load_model: warm slot hit — skipping disk I/O");
                    slot.take().map(|s| s.bundle)
                } else {
                    None
                }
            } else {
                None
            }
        };

        let bundle = match bundle {
            Some(b) => b,
            None => {
                let raw = build_bundle_from_dir(model_dir)?;
                Arc::new(raw)
            }
        };

        let meta = bundle.meta.clone();
        self.sessions.clear();
        self.prefix_cache.lock().clear(); // invalidate on model change
        self.bundle.store(Some(bundle));
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

        // Build a cache key from the model + system prompt + persona.
        // This is the "fixed prefix" shared across all sessions with the same config.
        let system_text = settings.prompt.system_prompt.as_deref().unwrap_or("");
        let persona_id = spec.persona.as_ref().map(|p| p.id.as_str()).unwrap_or("");
        let prefix_key = hash_str(&format!(
            "{}|{}|{}",
            bundle.model_dir.display(),
            system_text,
            persona_id
        ));

        let cached_prefix = {
            let mut guard = self.prefix_cache.lock();
            guard.touch(prefix_key).map(|e| e.clone_into_state(0))
        };

        let id = self.next_session_id.fetch_add(1, Ordering::SeqCst);
        let (session, new_prefix) = Session::new_with_prefix(
            id,
            bundle.clone(),
            self.hooks.clone(),
            settings,
            spec.messages,
            spec.persona.as_ref(),
            prefix_key,
            cached_prefix,
            spec.thinking.as_enable_thinking(),
        )?;

        // Populate/replace the prefix cache if the session built a fresh one.
        if let Some(entry) = new_prefix {
            self.prefix_cache.lock().insert(entry);
        }

        self.sessions.insert(id, ());
        Ok(Arc::new(session))
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

    #[cfg(test)]
    pub(super) fn prefix_cache_len(&self) -> usize {
        self.prefix_cache.lock().entries.len()
    }

    #[cfg(test)]
    pub(super) fn prefix_cache_contains(&self, key: u64) -> bool {
        self.prefix_cache
            .lock()
            .entries
            .iter()
            .any(|e| e.key == key)
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
    fn warm_model(&self, model_dir: std::path::PathBuf) {
        Engine::warm_model(self, model_dir)
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
    use super::super::session::CachePolicy;
    use super::*;
    use crate::gen2::engine::{EmbedLoadRequest, ExecError, LoadRequest};

    // ─── PrefixCacheLru tests ────────────────────────────────────────────────
    //
    // Stub entries hold an empty KV vec so the LRU can be exercised without
    // loading an MLX model. Only the `key` field drives LRU behaviour.

    fn stub_entry(key: u64) -> PrefixCacheEntry {
        PrefixCacheEntry {
            key,
            kv: vec![],
            cur_pos: 0,
            policy: CachePolicy {
                evict_trigger: 0,
                evict_to: 0,
            },
        }
    }

    fn keys(lru: &PrefixCacheLru) -> Vec<u64> {
        lru.entries.iter().map(|e| e.key).collect()
    }

    #[test]
    fn lru_touch_miss_returns_none() {
        let mut lru = PrefixCacheLru::new();
        assert!(lru.touch(42).is_none());
        lru.insert(stub_entry(1));
        assert!(lru.touch(2).is_none());
    }

    #[test]
    fn lru_touch_hit_moves_to_front() {
        let mut lru = PrefixCacheLru::new();
        lru.insert(stub_entry(1));
        lru.insert(stub_entry(2));
        lru.insert(stub_entry(3));
        // Order after inserts: [3, 2, 1] (front=MRU).
        assert_eq!(keys(&lru), vec![3, 2, 1]);

        // Touching 1 (oldest) must move it to the front.
        assert!(lru.touch(1).is_some());
        assert_eq!(keys(&lru), vec![1, 3, 2]);

        // Touching the already-front entry must be a no-op on order.
        assert!(lru.touch(1).is_some());
        assert_eq!(keys(&lru), vec![1, 3, 2]);
    }

    #[test]
    fn lru_insert_evicts_least_recent_at_capacity() {
        let mut lru = PrefixCacheLru::new();
        for i in 1..=(PREFIX_LRU_CAP as u64) {
            lru.insert(stub_entry(i));
        }
        assert_eq!(lru.entries.len(), PREFIX_LRU_CAP);

        // Insert a new key — the oldest (key=1, at the back) must be evicted.
        lru.insert(stub_entry(99));
        assert_eq!(lru.entries.len(), PREFIX_LRU_CAP);
        assert!(
            lru.touch(99).is_some(),
            "newly inserted key must be present"
        );
        // Key 1 was LRU and should have been dropped.
        assert!(
            !lru.entries.iter().any(|e| e.key == 1),
            "LRU entry should have been evicted; got {:?}",
            keys(&lru)
        );
    }

    #[test]
    fn lru_insert_dedups_same_key() {
        let mut lru = PrefixCacheLru::new();
        lru.insert(stub_entry(1));
        lru.insert(stub_entry(2));
        lru.insert(stub_entry(1)); // duplicate key
        // Must not grow — duplicate key replaces, doesn't add.
        assert_eq!(lru.entries.len(), 2);
        // And the duplicate moves to the front.
        assert_eq!(keys(&lru), vec![1, 2]);
    }

    #[test]
    fn lru_touch_promotion_protects_against_eviction() {
        let mut lru = PrefixCacheLru::new();
        for i in 1..=(PREFIX_LRU_CAP as u64) {
            lru.insert(stub_entry(i));
        }
        // Make key=1 MRU by touching it.
        lru.touch(1);
        // Insert a new key — key=2 should now be LRU, not key=1.
        lru.insert(stub_entry(99));
        assert!(lru.touch(1).is_some(), "touched key must survive eviction");
        assert!(
            !lru.entries.iter().any(|e| e.key == 2),
            "previously-promoted-away entry should have been evicted"
        );
    }

    #[test]
    fn lru_clear_empties_all_entries() {
        let mut lru = PrefixCacheLru::new();
        lru.insert(stub_entry(1));
        lru.insert(stub_entry(2));
        lru.clear();
        assert_eq!(lru.entries.len(), 0);
        assert!(lru.touch(1).is_none());
        // After clear, LRU must still be usable.
        lru.insert(stub_entry(3));
        assert_eq!(keys(&lru), vec![3]);
    }

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
            ..Default::default()
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
                | Some(Ok(TokenEvent::ToolCall(_)))
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
            ..Default::default()
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
            ..Default::default()
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

    /// Exercises the multi-entry prefix LRU with a real MLX bundle.
    ///
    /// Starts sessions with three distinct system prompts, confirms all three
    /// entries coexist, then re-uses the first prompt and asserts the cache
    /// size stays at 3 (dedup) and the oldest entry is still present (proving
    /// the old single-slot behaviour is gone).
    #[test]
    #[ignore = "requires TEST_MLX_MODEL_DIR env var pointing to a local model"]
    fn prefix_lru_holds_multiple_prefixes() -> Result<(), Box<dyn std::error::Error>> {
        use crate::gen2::Message;
        use crate::gen2::backend::mlx::puller::TokenPuller;
        use crate::gen2::engine::Settings;
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
            model_path: model_dir.clone(),
            ..Default::default()
        })?;
        assert_eq!(
            e.prefix_cache_len(),
            0,
            "fresh engine must have an empty prefix cache"
        );

        // Build three distinct system prompts → three distinct prefix keys.
        //
        // The prefix cache keys on `role: "system"` Messages at the head of the
        // chain (NOT on Settings::prompt::system_prompt, which gets fused into
        // the rendered prompt instead of producing a separate prefix segment).
        let prompts = [
            "You are a terse calculator.",
            "You are a cheerful tour guide for Lisbon.",
            "You respond only in haiku.",
        ];
        let keys: Vec<u64> = prompts
            .iter()
            .map(|p| {
                hash_str(&format!(
                    "{}|{}|{}",
                    model_dir.display(),
                    p,
                    "" // no persona
                ))
            })
            .collect();

        let sys_msg = |t: &str| Message {
            role: "system".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(t.into()),
            },
            name: None,
        };
        let user_msg = |t: &str| Message {
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(t.into()),
            },
            name: None,
        };

        let drain = |puller: &mut TokenPuller| -> String {
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

        // Start a short session per prompt and drain a couple tokens so the
        // prefill path runs and populates the LRU.
        //
        // `settings.prompt.system_prompt` drives the cache key, but the engine
        // only attempts the two-phase prefill when `sys_messages` (taken from
        // the message chain) is non-empty. Setting both keeps the two in sync.
        for (i, prompt) in prompts.iter().enumerate() {
            let mut settings = Settings::default();
            settings.prompt.system_prompt = Some((*prompt).to_string());
            let session = e.start_session(SessionSpec {
                messages: vec![sys_msg(prompt), user_msg("hi")],
                overrides: Some(settings),
                ..Default::default()
            })?;
            let mut p = session.pull(GenSpec {
                max_tokens: Some(4),
                ..Default::default()
            })?;
            let _ = drain(&mut p);
            drop(p);
            assert_eq!(
                e.prefix_cache_len(),
                i + 1,
                "after {} distinct prompts cache should hold {} entries",
                i + 1,
                i + 1
            );
            assert!(
                e.prefix_cache_contains(keys[i]),
                "key for prompt {} should be present",
                i
            );
        }

        // All three prefixes coexist — this fails under the old single-slot cache.
        for (i, key) in keys.iter().enumerate() {
            assert!(
                e.prefix_cache_contains(*key),
                "prompt {} evicted — single-slot behaviour regressed",
                i
            );
        }

        // Re-use the first prompt: cache hit, dedup — size must stay at 3.
        let mut settings = Settings::default();
        settings.prompt.system_prompt = Some(prompts[0].to_string());
        let session = e.start_session(SessionSpec {
            messages: vec![sys_msg(prompts[0]), user_msg("hi again")],
            overrides: Some(settings),
            ..Default::default()
        })?;
        let mut p = session.pull(GenSpec {
            max_tokens: Some(4),
            ..Default::default()
        })?;
        let _ = drain(&mut p);
        drop(p);
        assert_eq!(
            e.prefix_cache_len(),
            3,
            "re-using an existing prefix must not grow the cache"
        );
        for (i, key) in keys.iter().enumerate() {
            assert!(
                e.prefix_cache_contains(*key),
                "prompt {} missing after dedup insert",
                i
            );
        }

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
