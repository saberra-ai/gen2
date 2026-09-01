//! The [`Backend`] implementation.
//!
//! Lifecycle only: which model is loaded, which sessions exist, what settings
//! apply. Everything about how inference happens is mistral.rs's, and nothing
//! about mistral.rs reaches past this module.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use mistralrs::blocking::BlockingModel;
use parking_lot::RwLock;

use crate::backend::caps::LatencyTier;
use crate::backend::facade::SessionId;
use crate::backend::{Backend, BackendSession, LocalBackend};
use crate::engine::telemetry::HookBus;
use crate::engine::{Capabilities, ExecError, ExecutionStats, LoadRequest, Settings};
use crate::session_rt::SessionSpec;

use super::loader;
use super::session::MistralRsSession;

/// Context to report when the model has not said otherwise.
///
/// mistral.rs sizes and pages its own KV cache, so this is what gen2's
/// truncation driver plans against rather than a limit being enforced here.
const ASSUMED_CONTEXT: usize = 8192;

pub(crate) struct MistralRsEngine {
    model: RwLock<Option<Arc<BlockingModel>>>,
    /// The request that produced the current model, so `reload_model` can
    /// repeat it without the caller restating anything.
    last_load: RwLock<Option<LoadRequest>>,
    settings: RwLock<Arc<Settings>>,
    settings_version: AtomicU64,
    next_session_id: AtomicU64,
    hooks: Arc<HookBus>,
}

impl std::fmt::Debug for MistralRsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralRsEngine")
            .field("loaded", &self.model.read().is_some())
            .finish_non_exhaustive()
    }
}

impl MistralRsEngine {
    pub(crate) fn new() -> Self {
        Self {
            model: RwLock::new(None),
            last_load: RwLock::new(None),
            settings: RwLock::new(Arc::new(Settings::default())),
            settings_version: AtomicU64::new(0),
            next_session_id: AtomicU64::new(1),
            hooks: Arc::new(HookBus::default()),
        }
    }
}

impl Default for MistralRsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for MistralRsEngine {
    fn backend_name(&self) -> &'static str {
        "mistralrs"
    }

    fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let model = loader::load(&req)?;
        *self.model.write() = Some(Arc::new(model));
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
        // ordinary during error recovery.
        *self.model.write() = None;
    }

    fn is_model_loaded(&self) -> bool {
        self.model.read().is_some()
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
        if self.is_model_loaded() {
            Capabilities::TEXT
        } else {
            Capabilities::empty()
        }
    }

    fn stats(&self) -> ExecutionStats {
        // mistral.rs reports usage per response rather than per engine, and a
        // fabricated number here would be worse than none.
        ExecutionStats::default()
    }

    fn first_token_tier(&self) -> LatencyTier {
        LatencyTier::Medium
    }

    fn start_session(&self, spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        let model = self.model.read().clone().ok_or(ExecError::ModelNotLoaded)?;
        let id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        let settings = spec
            .overrides
            .clone()
            .unwrap_or_else(|| (*self.settings()).clone());
        let tools = spec
            .tools
            .as_ref()
            .map(|(t, _)| t.clone())
            .unwrap_or_default();

        Ok(Arc::new(MistralRsSession::new(
            id,
            model,
            settings,
            spec.messages,
            tools,
        )))
    }

    fn end_session(&self, _id: SessionId) -> Result<(), ExecError> {
        // Sessions are owned by whoever holds the `Arc`; there is no registry
        // here to remove one from, and inventing one would be bookkeeping with
        // nothing behind it.
        Ok(())
    }
}

impl LocalBackend for MistralRsEngine {
    fn n_ctx(&self) -> usize {
        self.settings
            .read()
            .system
            .ctx_size
            .map(|n| n as usize)
            .unwrap_or(ASSUMED_CONTEXT)
    }
}
