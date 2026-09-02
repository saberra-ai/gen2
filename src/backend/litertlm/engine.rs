//! The [`Backend`] implementation.
//!
//! Lifecycle only: which model is loaded, which sessions exist, what settings
//! apply. Everything about how inference happens is LiteRT-LM's, and nothing
//! about LiteRT-LM reaches past this module.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::backend::caps::LatencyTier;
use crate::backend::facade::SessionId;
use crate::backend::{Backend, BackendSession, LocalBackend};
use crate::engine::telemetry::HookBus;
use crate::engine::{Capabilities, ExecError, ExecutionStats, LoadRequest, Settings};
use crate::generation::GenSpec;
use crate::session_rt::SessionSpec;
use crate::types::message::Message;

use super::capabilities::ModelFacts;
use super::convert;
use super::ffi::{ConversationSetup, OwnedConversation, OwnedEngine, Runtime};
use super::session::LiteRtLmSession;

/// The accelerator LiteRT-LM should try first.
///
/// gen2 has no per-backend accelerator setting and is not gaining one, so this
/// is the backend's own default rather than something a caller configures.
/// `cpu` because it is the one backend every shipped `.litertlm` supports; a
/// bundle built for GPU or NPU still runs, just without that acceleration.
const DEFAULT_BACKEND: &str = "cpu";

/// A model, and everything derived from it at load time.
struct Loaded {
    engine: Arc<OwnedEngine>,
    facts: ModelFacts,
}

pub(crate) struct LiteRtLmEngine {
    runtime: RwLock<Option<Arc<Runtime>>>,
    loaded: RwLock<Option<Loaded>>,
    /// The request that produced the current model, so `reload_model` can
    /// repeat it without the caller restating anything.
    last_load: RwLock<Option<LoadRequest>>,
    settings: RwLock<Arc<Settings>>,
    settings_version: AtomicU64,
    next_session_id: AtomicU64,
    hooks: Arc<HookBus>,
}

impl std::fmt::Debug for LiteRtLmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiteRtLmEngine")
            .field("loaded", &self.loaded.read().is_some())
            .finish_non_exhaustive()
    }
}

impl LiteRtLmEngine {
    pub(crate) fn new() -> Self {
        Self {
            runtime: RwLock::new(None),
            loaded: RwLock::new(None),
            last_load: RwLock::new(None),
            settings: RwLock::new(Arc::new(Settings::default())),
            settings_version: AtomicU64::new(0),
            next_session_id: AtomicU64::new(1),
            hooks: Arc::new(HookBus::default()),
        }
    }

    /// Load the runtime once and keep it. Loading it per model would mean
    /// unloading a library other sessions still hold function pointers into.
    fn runtime(&self) -> Result<Arc<Runtime>, ExecError> {
        if let Some(rt) = self.runtime.read().as_ref() {
            return Ok(Arc::clone(rt));
        }
        let rt = Runtime::load()?;
        *self.runtime.write() = Some(Arc::clone(&rt));
        Ok(rt)
    }
}

impl Default for LiteRtLmEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for LiteRtLmEngine {
    fn backend_name(&self) -> &'static str {
        "litertlm"
    }

    fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let runtime = self.runtime()?;
        let path = req.model_path.to_string_lossy().into_owned();

        // The request's own context wins over the engine's settings: it is
        // what this load was asked for, and `Engine::builder().context(n)`
        // travels here rather than through `Settings`.
        let settings = self.settings();
        let ctx_size = req.ctx_params.n_ctx.or(settings.system.ctx_size);
        let threads = req.ctx_params.threads.or(settings.system.threads);

        // What the model can do, asked before the engine is built so a
        // conversation is never configured against a guess.
        let facts = ModelFacts::probe(&runtime, &path, ctx_size)?;

        let engine = OwnedEngine::create(
            Arc::clone(&runtime),
            &path,
            DEFAULT_BACKEND,
            Some(facts.max_context_tokens as i32),
            threads.map(|n| n as i32),
            None,
        )?;

        // The `Arc` is shared with every session opened on this model, not
        // across threads: `Backend` is deliberately not `Send`, and the
        // controller confines all of it to one thread. Same reason the llama
        // backend allows this.
        #[allow(clippy::arc_with_non_send_sync)]
        let engine = Arc::new(engine);
        *self.loaded.write() = Some(Loaded { engine, facts });
        *self.last_load.write() = Some(req);
        Ok(())
    }

    fn reload_model(&self) -> Result<(), ExecError> {
        let last = self.last_load.read().clone();
        let req = last.ok_or(ExecError::ModelNotLoaded)?;
        self.load_model(req)
    }

    fn unload_model(&self) {
        // Idempotent: unloading twice, or unloading what was never loaded, is
        // ordinary during error recovery. The runtime itself stays loaded.
        *self.loaded.write() = None;
    }

    fn is_model_loaded(&self) -> bool {
        self.loaded.read().is_some()
    }

    fn upload_settings(&self, settings: Settings) -> Result<(), ExecError> {
        *self.settings.write() = Arc::new(settings);
        self.settings_version.fetch_add(1, Ordering::Release);
        Ok(())
    }

    fn settings(&self) -> Arc<Settings> {
        Arc::clone(&self.settings.read())
    }

    fn settings_version(&self) -> u64 {
        self.settings_version.load(Ordering::Acquire)
    }

    fn hooks(&self) -> Arc<HookBus> {
        Arc::clone(&self.hooks)
    }

    fn capabilities(&self) -> Capabilities {
        // Text only until a multimodal path is proven end to end. Advertising
        // images before that would send a caller's guard the wrong way, and
        // the conformance suite checks that the probe and the bitset agree.
        match self.loaded.read().as_ref() {
            Some(_) => Capabilities::TEXT,
            None => Capabilities::empty(),
        }
    }

    fn stats(&self) -> ExecutionStats {
        // LiteRT-LM reports benchmark information per conversation rather than
        // per engine, and a fabricated number here would be worse than none.
        ExecutionStats::default()
    }

    fn first_token_tier(&self) -> LatencyTier {
        // Built for on-device inference, and the shipped bundles are small and
        // quantized. Medium rather than Fast because prefill on a CPU backend
        // is not instant.
        LatencyTier::Medium
    }

    fn start_session(&self, spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        let guard = self.loaded.read();
        let loaded = guard.as_ref().ok_or(ExecError::ModelNotLoaded)?;
        let engine = Arc::clone(&loaded.engine);
        let ctx_size = loaded.facts.max_context_tokens;
        drop(guard);

        let id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        let settings = spec
            .overrides
            .clone()
            .unwrap_or_else(|| (*self.settings()).clone());
        let tools_json = spec
            .tools
            .as_ref()
            .map(|(t, _)| convert::tools_json(t))
            .filter(|json| json != "[]");

        // The last message is the turn to generate from, not context. A
        // conversation opened with the whole transcript has already consumed
        // the prompt, and the first `pull` would have nothing left to send.
        let (history, prompt) = match spec.messages.split_last() {
            Some((last, earlier)) => (earlier, vec![last.clone()]),
            None => (&spec.messages[..], Vec::new()),
        };

        let conversation = open_conversation(
            &engine,
            &settings,
            &GenSpec::default(),
            tools_json.as_deref(),
            history,
        )?;

        // `Arc<dyn BackendSession>` is the trait's own return type; the
        // session is thread-confined like every other backend's.
        #[allow(clippy::arc_with_non_send_sync)]
        Ok(Arc::new(LiteRtLmSession::new(
            id,
            engine,
            conversation,
            settings,
            tools_json,
            history.to_vec(),
            prompt,
            ctx_size,
        )))
    }

    fn end_session(&self, _id: SessionId) -> Result<(), ExecError> {
        // Sessions are owned by whoever holds the `Arc`; there is no registry
        // here to remove one from, and inventing one would be bookkeeping with
        // nothing behind it.
        Ok(())
    }
}

/// Open a conversation holding exactly `history`.
///
/// Shared with the session, which rebuilds through it when a turn cannot be
/// delivered incrementally — so a rebuilt conversation is configured exactly
/// like a fresh one, rather than by a second copy of this logic that drifts.
pub(super) fn open_conversation(
    engine: &OwnedEngine,
    settings: &Settings,
    spec: &GenSpec,
    tools_json: Option<&str>,
    history: &[Message],
) -> Result<OwnedConversation, ExecError> {
    let (system, rest) = convert::leading_system(history);
    OwnedConversation::create(
        engine,
        ConversationSetup {
            system_message: system.as_deref(),
            messages_json: &convert::messages_json(rest),
            tools_json,
            sampler: convert::sampler_of(settings, spec),
            max_output_tokens: settings
                .stopping
                .max_tokens
                .map(|n| n.min(i32::MAX as usize) as i32),
            // gen2 has no thinking switch of its own to forward, so the
            // model's own default stands.
            enable_thinking: None,
            constrained_decoding: spec.grammar.is_some(),
        },
    )
}

impl LocalBackend for LiteRtLmEngine {
    /// The context the loaded model was actually configured with.
    ///
    /// Zero when nothing is loaded, and never a guess otherwise — see
    /// [`ModelFacts::probe`], which refuses a load it cannot answer this for.
    /// The controller plans truncation against this number; a backend that
    /// invents one makes every one of those decisions against a fiction.
    fn n_ctx(&self) -> usize {
        self.loaded
            .read()
            .as_ref()
            .map(|l| l.facts.max_context_tokens as usize)
            .unwrap_or(0)
    }
}
