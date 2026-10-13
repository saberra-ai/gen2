use parking_lot::RwLock;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::{config::Facts, ffi, session::OnnxSession};
use crate::backend::common::{
    chat_template::ChatTemplate, load_chat_template, tokenizer::HfTokenizer,
};
use crate::backend::{Backend, BackendSession, LatencyTier, LocalBackend, SessionId};
use crate::engine::telemetry::HookBus;
use crate::engine::{Capabilities, ExecError, ExecutionStats, LoadRequest, Settings};
use crate::session_rt::SessionSpec;
use crate::types::message::TokenizerConfigToken;

pub(super) struct Bundle {
    pub model: Arc<ffi::Model>,
    pub tokenizer: HfTokenizer,
    pub template: ChatTemplate,
    pub facts: Facts,
}

pub struct OnnxEngine {
    runtime: RwLock<Option<Arc<ffi::Runtime>>>,
    bundle: RwLock<Option<Arc<Bundle>>>,
    last_load: RwLock<Option<LoadRequest>>,
    settings: RwLock<Arc<Settings>>,
    settings_version: AtomicU64,
    next_id: AtomicU64,
    hooks: Arc<HookBus>,
    stats: Arc<RwLock<ExecutionStats>>,
}
impl std::fmt::Debug for OnnxEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEngine")
            .field("loaded", &self.is_model_loaded())
            .finish_non_exhaustive()
    }
}
impl OnnxEngine {
    pub(crate) fn new() -> Self {
        Self {
            runtime: RwLock::new(None),
            bundle: RwLock::new(None),
            last_load: RwLock::new(None),
            settings: RwLock::new(Arc::new(Settings::default())),
            settings_version: AtomicU64::new(0),
            next_id: AtomicU64::new(1),
            hooks: Arc::new(HookBus::default()),
            stats: Arc::new(RwLock::new(ExecutionStats::default())),
        }
    }
}
impl Default for OnnxEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for OnnxEngine {
    fn backend_name(&self) -> &'static str {
        "onnx"
    }
    fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let settings = self.settings();
        let facts = Facts::read(
            &req.model_path,
            req.ctx_params.n_ctx.or(settings.system.ctx_size),
        )?;
        let tokenizer = HfTokenizer::from_dir(&facts.directory).map_err(ExecError::Other)?;
        let template = load_chat_template(&facts.directory).ok_or_else(|| {
            ExecError::InvalidModelFile(
                "ONNX bundle needs a chat template in tokenizer_config.json or chat_template.jinja"
                    .into(),
            )
        })?;
        // Validate the template before the shared parser's infallible constructor.
        let mut env = minijinja::Environment::new();
        let generation_tags =
            regex::Regex::new(r"\{\%-?\s*(?:end)?generation\s*-?\%\}").expect("static regex");
        let normalized = generation_tags
            .replace_all(&template, "")
            .replace("[::-1]", "|reverse");
        env.add_template("chat", &normalized)
            .map_err(|e| ExecError::TemplateError(e.to_string()))?;
        let special = |id: Option<u32>| -> Result<Option<TokenizerConfigToken>, ExecError> {
            id.map(|id| {
                tokenizer
                    .decode_keep_specials(&[id])
                    .map(TokenizerConfigToken::String)
                    .map_err(ExecError::Other)
            })
            .transpose()
        };
        let template = ChatTemplate::new(
            template,
            special(tokenizer.bos_id())?,
            special(tokenizer.eos_id())?,
        );
        let existing = self.runtime.read().clone();
        let runtime = match existing {
            Some(runtime) => runtime,
            None => ffi::Runtime::load()?,
        };
        *self.runtime.write() = Some(runtime.clone());
        let model = ffi::Model::load(
            runtime,
            &facts.directory,
            req.ctx_params.threads.or(settings.system.threads),
            req.model_params.gpu_layers.or(settings.system.gpu_layers) == Some(0),
        )?;
        // Native handles are confined to the controller thread. Arc gives
        // sessions shared ownership without asserting Send or Sync.
        #[allow(clippy::arc_with_non_send_sync)]
        let bundle = Arc::new(Bundle {
            model: Arc::new(model),
            tokenizer,
            template,
            facts,
        });
        *self.bundle.write() = Some(bundle);
        *self.last_load.write() = Some(req);
        Ok(())
    }
    fn reload_model(&self) -> Result<(), ExecError> {
        let request = self
            .last_load
            .read()
            .clone()
            .ok_or(ExecError::ModelNotLoaded)?;
        self.load_model(request)
    }
    fn unload_model(&self) {
        *self.bundle.write() = None;
    }
    fn is_model_loaded(&self) -> bool {
        self.bundle.read().is_some()
    }
    fn upload_settings(&self, settings: Settings) -> Result<(), ExecError> {
        settings.validate()?;
        *self.settings.write() = Arc::new(settings);
        self.settings_version.fetch_add(1, Ordering::Release);
        Ok(())
    }
    fn settings(&self) -> Arc<Settings> {
        self.settings.read().clone()
    }
    fn settings_version(&self) -> u64 {
        self.settings_version.load(Ordering::Acquire)
    }
    fn hooks(&self) -> Arc<HookBus> {
        self.hooks.clone()
    }
    fn capabilities(&self) -> Capabilities {
        if self.is_model_loaded() {
            Capabilities::TEXT
        } else {
            Capabilities::empty()
        }
    }
    fn stats(&self) -> ExecutionStats {
        self.stats.read().clone()
    }
    fn first_token_tier(&self) -> LatencyTier {
        LatencyTier::Medium
    }
    fn bundle_architecture(&self) -> Option<String> {
        self.bundle
            .read()
            .as_ref()
            .map(|b| b.facts.architecture.clone())
    }
    fn context_window(&self) -> Option<u32> {
        self.bundle.read().as_ref().map(|b| b.facts.context)
    }
    fn start_session(&self, spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        let bundle = self
            .bundle
            .read()
            .clone()
            .ok_or(ExecError::ModelNotLoaded)?;
        let mut settings = spec.overrides.clone().unwrap_or_default();
        settings.inherit_missing(&self.settings());
        #[allow(clippy::arc_with_non_send_sync)]
        Ok(Arc::new(OnnxSession::new(
            self.next_id.fetch_add(1, Ordering::Relaxed),
            bundle,
            settings,
            spec,
            self.hooks.clone(),
            self.stats.clone(),
        )?))
    }
    fn end_session(&self, _id: SessionId) -> Result<(), ExecError> {
        Ok(())
    }
}
impl LocalBackend for OnnxEngine {
    fn n_ctx(&self) -> usize {
        self.context_window().unwrap_or(0) as usize
    }
}
