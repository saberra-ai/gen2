//! [`Runtime`] — what owns this machine's inference resources.
//!
//! The happy path needs no runtime at all:
//!
//! ```no_run
//! fn main() -> gen2::Result<()> {
//!     let model = gen2::load("qwen3-8b.gguf")?;
//!
//!     println!(
//!         "{}",
//!         model.generate("Why is the sky blue?").text()?
//!     );
//!
//!     Ok(())
//! }
//! ```
//!
//! A generation is configured on the builder [`Model::generate`] returns:
//!
//! ```no_run
//! # let model = gen2::load("qwen3-8b.gguf")?;
//! let response = model
//!     .generate("Write a haiku about local inference")
//!     .temperature(0.8)
//!     .max_tokens(64)
//!     .run()?;
//!
//! println!("{}", response.text());
//! println!("{:?}", response.usage());
//! # Ok::<(), gen2::Error>(())
//! ```
//!
//! Hold a [`Runtime`] yourself to load more than one model, or to reach a
//! remote OpenAI-compatible endpoint the same way:
//!
//! ```no_run
//! use gen2::Runtime;
//!
//! let runtime = Runtime::new()?;
//! let local = runtime.load("qwen3-8b.gguf")?;
//! let remote = runtime
//!     .openai()
//!     .base_url("http://localhost:11434/v1")
//!     .model("qwen3:8b")
//!     .connect()?;
//!
//! for model in [&local, &remote] {
//!     println!("{}", model.generate("hello").text()?);
//! }
//! # Ok::<(), gen2::Error>(())
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::controller::ControllerConfig;
use crate::engine::Settings;
use crate::hardware::HardwareProfile;

use super::engine::{Engine, EngineBuilder};
use super::error::{Error, Result};
use super::fit;
use super::model::{Model, ModelId, ModelSourceKind};

/// Owns backends, loaded weights, and the registry of models built on them.
///
/// Cheap to clone; every clone is the same runtime. Dropping the last handle
/// — and the last [`Model`] from it — shuts the backends down.
///
/// # What a runtime is
///
/// Each loaded model runs on its own controller loop: a runtime with N models
/// holds N engines. Weights are a cache the runtime manages (api_spec.md
/// §4.2): a [`Model`] handle stays valid after its weights are evicted —
/// by [`Runtime::evict`], or automatically when loading or restoring
/// another model would exceed the memory budget, least recently used first
/// — and its next turn restores them. The controls are on this type and
/// documented as advanced (§4.5); the types they return are under
/// [`gen2::advanced::runtime`](crate::advanced::runtime).
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

/// Shared state behind every handle to one runtime.
pub(crate) struct RuntimeInner {
    config: ControllerConfig,
    settings: Settings,
    /// Every model loaded so far, by id. Entries are never removed: a
    /// [`Model`] carries its own `Arc` to the same entry, and an evicted
    /// model keeps its entry so it can be restored.
    models: Mutex<HashMap<ModelId, Arc<Loaded>>>,
    next_id: AtomicU64,
    /// Serialises admission: one load, restore, or eviction decides at a
    /// time, so two turns racing to restore two models cannot both pass a
    /// check the other's weights then break.
    admission: Mutex<()>,
    /// Resident-memory budget in MB, when fixed rather than read from the
    /// memory governor. Tests pin it; a real runtime asks the machine.
    budget_mb: Option<u64>,
    /// Models evicted to make room, over the runtime's life.
    evictions: AtomicU64,
}

/// One model the runtime has loaded.
pub(crate) struct Loaded {
    pub(crate) engine: Engine,
    pub(crate) name: Option<String>,
    pub(crate) source: ModelSourceKind,
    /// The file's header, when it was a readable GGUF. What
    /// [`Model::capabilities`] reads tool support from.
    pub(crate) header: Option<fit::ModelInfo>,
    /// Host memory the weights are estimated to hold while resident, in MB
    /// — what the runtime's ledger counts against the budget. Zero for a
    /// remote model, which holds nothing here.
    pub(crate) estimated_mb: u64,
    /// Whether the weights are resident, as this layer last left them. The
    /// engine is asked before a restore, so a controller-side unload is
    /// noticed too.
    resident: AtomicBool,
    /// The last turn, generation, or preload — what "least recently used"
    /// orders by. `None` until first use, which sorts before any use.
    last_used: Mutex<Option<Instant>>,
}

impl Loaded {
    pub(crate) fn new(
        engine: Engine,
        name: Option<String>,
        source: ModelSourceKind,
        header: Option<fit::ModelInfo>,
        estimated_mb: u64,
    ) -> Self {
        Self {
            engine,
            name,
            source,
            header,
            estimated_mb,
            resident: AtomicBool::new(true),
            last_used: Mutex::new(None),
        }
    }

    /// Mark the model used now.
    pub(crate) fn touch(&self) {
        if let Ok(mut t) = self.last_used.lock() {
            *t = Some(Instant::now());
        }
    }

    fn last_used(&self) -> Option<Instant> {
        self.last_used.lock().ok().and_then(|t| *t)
    }

    /// Whether the weights are resident. A remote model always is: there
    /// is nothing on this machine to evict.
    pub(crate) fn is_resident(&self) -> bool {
        !self.source.is_local() || self.resident.load(Ordering::SeqCst)
    }

    fn set_resident(&self, resident: bool) {
        self.resident.store(resident, Ordering::SeqCst);
    }
}

impl Runtime {
    /// A runtime with default policy.
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    /// Configure a runtime. The knobs are deliberately few.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// Load a local model — a GGUF file, or a bundle directory a compiled
    /// backend reads — or one from the Hugging Face Hub by reference
    /// (`hf:owner/repo[:QUANT]`, see [`gen2::hf`](crate::hf)). The backend
    /// is chosen from what is there.
    ///
    /// Returns once the weights are resident. A path that does not exist, or
    /// is not a model, fails here rather than at first use; so does a
    /// reference the Hub cannot serve.
    pub fn load(&self, path: impl AsRef<Path>) -> Result<Model> {
        let path = path.as_ref();
        match super::hf::resolve_model_path(path)? {
            Some(download) => self.load_downloaded(download),
            None => self.load_local(path, None, ModelSourceKind::LocalFile),
        }
    }

    /// Load a model from the Hugging Face Hub, in typed form.
    ///
    /// The string form goes through [`Runtime::load`]; this takes an
    /// [`HfModel`](crate::hf::HfModel) built by hand — a progress hook, a
    /// cache directory of its own. Downloads what is not cached, then loads
    /// as [`Runtime::load`] would; the repo's projector, when it has one,
    /// rides along as the vision projector.
    pub fn load_hf(&self, model: super::hf::HfModel) -> Result<Model> {
        self.load_downloaded(model.download()?)
    }

    fn load_downloaded(&self, download: super::hf::HfDownload) -> Result<Model> {
        let source = ModelSourceKind::HuggingFace {
            repo: download.resolved.repo,
            file: download.resolved.file,
        };
        self.load_local(&download.model, download.mmproj, source)
    }

    fn load_local(
        &self,
        path: &Path,
        mmproj: Option<PathBuf>,
        source: ModelSourceKind,
    ) -> Result<Model> {
        let estimated_mb = crate::residency_policy::estimate_resident_mb_for_path_offloaded(
            path,
            self.inner.settings.system.gpu_layers,
        );
        // Make room before the weights are read, under the admission lock
        // so nothing else is admitted on the strength of the same free
        // memory. The engine's own admission check is the final say; this
        // only evicts what it can to let that succeed.
        let engine = {
            let _admission = self.inner.admission.lock();
            self.make_room(estimated_mb, None);
            let mut builder = Engine::builder()
                .model(path)
                .settings(self.inner.settings.clone())
                .config(self.inner.config.clone());
            if let Some(mmproj) = mmproj {
                builder = builder.mmproj(mmproj);
            }
            builder.build()?
        };
        let header = fit::ModelInfo::read(path).ok();
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty());
        let model = self.register(Loaded::new(engine, name, source, header, estimated_mb));
        model.loaded().touch();
        Ok(model)
    }

    /// An engine builder carrying this runtime's settings and controller
    /// config, for the auxiliary engines (embedder, reranker) that share
    /// them.
    pub(crate) fn engine_builder(&self) -> EngineBuilder {
        Engine::builder()
            .settings(self.inner.settings.clone())
            .config(self.inner.config.clone())
    }

    /// A model served by an OpenAI-compatible endpoint.
    ///
    /// Works for the hosted API and for local servers that speak the same
    /// protocol — Ollama, llama-server, vLLM, LM Studio. Local servers need no
    /// key. Requires the `backend-external-api` feature to connect.
    ///
    /// ```no_run
    /// # let runtime = gen2::Runtime::new()?;
    /// let gpt = runtime
    ///     .openai()
    ///     .base_url("https://api.openai.com/v1")
    ///     .api_key(std::env::var("OPENAI_API_KEY").unwrap_or_default())
    ///     .model("gpt-5-mini")
    ///     .connect()?;
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn openai(&self) -> RemoteModelBuilder {
        RemoteModelBuilder::new(self.clone(), RemoteFormat::OpenAi)
    }

    /// A model served by an Anthropic-compatible endpoint. Same shape as
    /// [`Runtime::openai`], different wire format.
    pub fn anthropic(&self) -> RemoteModelBuilder {
        RemoteModelBuilder::new(self.clone(), RemoteFormat::Anthropic)
    }

    /// The models this runtime has loaded, oldest first.
    pub fn models(&self) -> Vec<ModelId> {
        let mut ids: Vec<ModelId> = self
            .inner
            .models
            .lock()
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default();
        ids.sort();
        ids
    }

    // ── Advanced runtime controls (api_spec.md §4.5) ────────────────────
    //
    // Below the happy path: a normal consumer loads models and runs turns,
    // and residency takes care of itself. These are for the caller who
    // wants to see or steer it.

    /// The machine this runtime detected: RAM, cores, GPU backend, VRAM.
    ///
    /// Advanced (api_spec.md §4.5). Detected once per process and cached.
    pub fn hardware(&self) -> HardwareProfile {
        HardwareProfile::cached().clone()
    }

    /// What is resident right now, model by model.
    ///
    /// Advanced (api_spec.md §4.5). A snapshot: it is stale as soon as a
    /// turn runs, and is for display and tests, not for deciding whether a
    /// model can be used — every model can, resident or not.
    pub fn residency(&self) -> ResidencySnapshot {
        let mut models: Vec<ModelResidency> = self
            .inner
            .models
            .lock()
            .map(|m| {
                m.iter()
                    .map(|(id, loaded)| ModelResidency {
                        id: *id,
                        name: loaded.name.clone(),
                        local: loaded.source.is_local(),
                        resident: loaded.is_resident(),
                        estimated_mb: loaded.estimated_mb,
                        idle_for: loaded.last_used().map(|t| t.elapsed()),
                    })
                    .collect()
            })
            .unwrap_or_default();
        models.sort_by_key(|m| m.id);
        ResidencySnapshot { models }
    }

    /// Make `model`'s weights resident now rather than on its next turn.
    ///
    /// Advanced (api_spec.md §4.5). A no-op when they already are, or for a
    /// remote model. May evict the least recently used other model to make
    /// room, as a turn would.
    pub fn preload(&self, model: &Model) -> Result<()> {
        let loaded = self.own(model)?;
        self.restore(model.id(), loaded)?;
        loaded.touch();
        Ok(())
    }

    /// Release `model`'s weights. The handle stays valid: its next turn or
    /// generation restores them.
    ///
    /// Advanced (api_spec.md §4.5). A no-op when they are not resident, or
    /// for a remote model. Conversations the model held are rebuilt from
    /// their sessions on that next turn, at the cost of one prefill each.
    pub fn evict(&self, model: &Model) -> Result<()> {
        let loaded = self.own(model)?;
        let _admission = self.inner.admission.lock();
        Self::unload(loaded)
    }

    /// Aggregate counters over every model this runtime holds.
    ///
    /// Advanced (api_spec.md §4.5).
    pub fn stats(&self) -> RuntimeStats {
        let residency = self.residency();
        let active_sessions = self
            .inner
            .models
            .lock()
            .map(|m| {
                m.values()
                    .filter_map(|loaded| {
                        loaded
                            .engine
                            .controller()
                            .get_controller_runtime_snapshot()
                            .ok()
                    })
                    .map(|snapshot| snapshot.chats.len())
                    .sum()
            })
            .unwrap_or(0);
        RuntimeStats {
            models: residency.models.len(),
            resident_models: residency.models.iter().filter(|m| m.resident).count(),
            estimated_resident_mb: residency.resident_mb(),
            active_sessions,
            evictions: self.inner.evictions.load(Ordering::SeqCst),
        }
    }

    /// Restore `loaded`'s weights if they are not resident, evicting the
    /// least recently used other model first when the budget needs it.
    /// The lazy half of api_spec.md §4.2; every turn passes through here.
    pub(crate) fn restore(&self, id: ModelId, loaded: &Arc<Loaded>) -> Result<()> {
        if !loaded.source.is_local() {
            return Ok(());
        }
        // Fast path, unlocked: the common turn on a resident model.
        if loaded.is_resident() && loaded.engine.is_model_loaded() {
            return Ok(());
        }
        let _admission = self.inner.admission.lock();
        if loaded.is_resident() && loaded.engine.is_model_loaded() {
            return Ok(());
        }
        // Either evicted here, or unloaded below the facade (the
        // controller's idle unload, say). Same restore.
        loaded.set_resident(false);
        self.make_room(loaded.estimated_mb, Some(id));
        loaded.engine.reload_model()?;
        loaded.set_resident(true);
        Ok(())
    }

    /// Evict least-recently-used resident models, other than `keep`, until
    /// `extra_mb` more fits the budget or nothing evictable is left.
    ///
    /// Called with the admission lock held. Best effort: when nothing can
    /// be evicted, the load or restore proceeds and the engine's own
    /// admission check has the final say.
    fn make_room(&self, extra_mb: u64, keep: Option<ModelId>) {
        loop {
            if self.can_admit(extra_mb) {
                return;
            }
            let victim = self.least_recently_used(keep);
            let Some(victim) = victim else {
                return;
            };
            if Self::unload(&victim).is_ok() {
                self.inner.evictions.fetch_add(1, Ordering::SeqCst);
            } else {
                // A model that will not unload cannot make room; try no
                // further, rather than spin on it.
                return;
            }
        }
    }

    /// Whether `extra_mb` more resident memory fits: the runtime's ledger
    /// against the budget, and the machine's memory governor.
    fn can_admit(&self, extra_mb: u64) -> bool {
        let ledger = self.residency().resident_mb();
        let projected = ledger.saturating_add(extra_mb);
        match self.inner.budget_mb {
            Some(budget) => projected <= budget,
            None => {
                let governor = crate::memory::current_memory_governor();
                projected <= governor.budgets().inference_resident_mb
                    && governor.can_load_additional_model(extra_mb)
            }
        }
    }

    /// The resident local model used longest ago, other than `keep`.
    fn least_recently_used(&self, keep: Option<ModelId>) -> Option<Arc<Loaded>> {
        let models = self.inner.models.lock().ok()?;
        models
            .iter()
            .filter(|(id, loaded)| {
                Some(**id) != keep && loaded.source.is_local() && loaded.is_resident()
            })
            .min_by_key(|(id, loaded)| (loaded.last_used(), **id))
            .map(|(_, loaded)| Arc::clone(loaded))
    }

    /// Unload one model's weights. Called with the admission lock held.
    fn unload(loaded: &Arc<Loaded>) -> Result<()> {
        if !loaded.source.is_local() || !loaded.is_resident() {
            return Ok(());
        }
        loaded.engine.unload_model()?;
        loaded.set_resident(false);
        Ok(())
    }

    /// `model`'s entry, if it belongs to this runtime.
    fn own<'m>(&self, model: &'m Model) -> Result<&'m Arc<Loaded>> {
        if !model.belongs_to(&self.inner) {
            return Err(Error::InvalidRequest(format!(
                "{} belongs to another runtime",
                model.id()
            )));
        }
        Ok(model.loaded())
    }

    pub(crate) fn from_inner(inner: Arc<RuntimeInner>) -> Self {
        Self { inner }
    }

    fn register(&self, loaded: Loaded) -> Model {
        let id = ModelId(self.inner.next_id.fetch_add(1, Ordering::SeqCst));
        let loaded = Arc::new(loaded);
        if let Ok(mut models) = self.inner.models.lock() {
            models.insert(id, Arc::clone(&loaded));
        }
        Model::new(Arc::clone(&self.inner), id, loaded)
    }

    /// A model over a scripted backend, in a fresh runtime.
    ///
    /// The seam the facade's own tests use: the response mapping, the
    /// handle semantics, and what reaches the backend are contracts that do
    /// not need real weights to prove.
    #[cfg(test)]
    pub(crate) fn scripted(script: crate::test_support::Script) -> Model {
        let runtime = Self::new().expect("a default runtime always builds");
        runtime.scripted_in(script, 0)
    }

    /// A scripted model in this runtime, `estimated_mb` on the ledger.
    #[cfg(test)]
    pub(crate) fn scripted_in(
        &self,
        script: crate::test_support::Script,
        estimated_mb: u64,
    ) -> Model {
        // Through the same admission as `load`, so eviction on load is
        // testable without weights.
        let engine = {
            let _admission = self.inner.admission.lock();
            self.make_room(estimated_mb, None);
            Engine::scripted_with_config(script, self.inner.config.clone())
        };
        let model = self.register(Loaded::new(
            engine,
            Some("scripted".into()),
            ModelSourceKind::LocalFile,
            None,
            estimated_mb,
        ));
        model.loaded().touch();
        model
    }
}

/// One model's residency, from [`Runtime::residency`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelResidency {
    /// Which model.
    pub id: ModelId,
    /// Its display name, as [`ModelInfo::name`](super::model::ModelInfo::name).
    pub name: Option<String>,
    /// Whether the weights live on this machine at all.
    pub local: bool,
    /// Whether they are loaded right now. Always true for a remote model.
    pub resident: bool,
    /// Host memory the weights are estimated to hold while resident, in
    /// MB. An estimate from the file size and the GPU offload, not a
    /// measurement.
    pub estimated_mb: u64,
    /// Time since the model last ran a turn or was preloaded; `None` if it
    /// never has.
    pub idle_for: Option<Duration>,
}

/// What a [`Runtime`] has resident (api_spec.md §4.5).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ResidencySnapshot {
    /// Every model the runtime holds, by id, resident or not.
    pub models: Vec<ModelResidency>,
}

impl ResidencySnapshot {
    /// Whether `id`'s weights are resident. False for an unknown id.
    pub fn is_resident(&self, id: ModelId) -> bool {
        self.models.iter().any(|m| m.id == id && m.resident)
    }

    /// Ids of the models whose weights are resident, ascending.
    pub fn resident(&self) -> Vec<ModelId> {
        self.models
            .iter()
            .filter(|m| m.resident)
            .map(|m| m.id)
            .collect()
    }

    /// Estimated MB held by the resident local models.
    pub fn resident_mb(&self) -> u64 {
        self.models
            .iter()
            .filter(|m| m.local && m.resident)
            .map(|m| m.estimated_mb)
            .sum()
    }
}

/// Aggregate counters from [`Runtime::stats`] (api_spec.md §4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct RuntimeStats {
    /// Models loaded into the runtime, resident or not.
    pub models: usize,
    /// Of those, how many have their weights resident.
    pub resident_models: usize,
    /// Estimated MB the resident local models hold.
    pub estimated_resident_mb: u64,
    /// Conversations the engines currently hold runtime state for, summed.
    pub active_sessions: usize,
    /// Models evicted to make room for another, over the runtime's life.
    /// Explicit [`Runtime::evict`] calls are not counted.
    pub evictions: u64,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("models", &self.models())
            .finish_non_exhaustive()
    }
}

/// Builds a [`Runtime`].
///
/// Small on purpose: nothing about backends or controllers belongs here.
#[derive(Default)]
pub struct RuntimeBuilder {
    config: ControllerConfig,
    settings: Option<Settings>,
    budget_mb: Option<u64>,
}

impl RuntimeBuilder {
    /// How many conversations each model keeps warm at once before the
    /// least recently used is evicted from the cache (it is rebuilt on its
    /// next turn, at the cost of one re-read).
    pub fn max_active_sessions(mut self, n: usize) -> Self {
        self.config.max_active_chats = n.max(1);
        self
    }

    /// Sampling, stopping, and prompt settings every model starts from.
    pub fn settings(mut self, settings: Settings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// Register a backend built outside the crate. A path the plugin claims
    /// is routed to it ahead of every built-in rule; see
    /// [`gen2::advanced::plugin`](crate::advanced::plugin).
    pub fn backend(mut self, plugin: crate::advanced::BackendPlugin) -> Self {
        self.config.plugins.push(Arc::new(plugin));
        self
    }

    /// Pin the resident-memory budget instead of asking the machine, so
    /// eviction is decided by arithmetic the test controls.
    #[cfg(test)]
    pub(crate) fn resident_budget_mb(mut self, mb: u64) -> Self {
        self.budget_mb = Some(mb);
        self
    }

    /// Build it. Starts nothing: backends start when a model is loaded.
    pub fn build(self) -> Result<Runtime> {
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                config: self.config,
                settings: self.settings.unwrap_or_default(),
                models: Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(0),
                admission: Mutex::new(()),
                budget_mb: self.budget_mb,
                evictions: AtomicU64::new(0),
            }),
        })
    }
}

impl std::fmt::Debug for RuntimeBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeBuilder")
            .field("max_active_sessions", &self.config.max_active_chats)
            .finish_non_exhaustive()
    }
}

/// Which remote wire format a [`RemoteModelBuilder`] speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoteFormat {
    OpenAi,
    Anthropic,
}

/// Connects a [`Model`] served by a remote endpoint. See
/// [`Runtime::openai`].
#[must_use = "a RemoteModelBuilder does nothing until .connect() is called"]
pub struct RemoteModelBuilder {
    runtime: Runtime,
    format: RemoteFormat,
    base_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
}

impl RemoteModelBuilder {
    fn new(runtime: Runtime, format: RemoteFormat) -> Self {
        Self {
            runtime,
            format,
            base_url: None,
            api_key: None,
            model: None,
        }
    }

    /// The API root — `https://api.openai.com/v1`, or
    /// `http://localhost:11434/v1` for Ollama. Required.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// The bearer token. Optional: a local server needs none.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// The model the endpoint should serve — `"gpt-5-mini"`, `"qwen3:8b"`.
    /// Required: providers that serve several reject a request naming none.
    pub fn model(mut self, name: impl Into<String>) -> Self {
        self.model = Some(name.into());
        self
    }

    /// Probe the endpoint and register the model.
    ///
    /// Fails here — not at first generation — when the endpoint cannot be
    /// reached, so a wrong URL is reported as such.
    pub fn connect(self) -> Result<Model> {
        let base_url = self.base_url.ok_or_else(|| {
            Error::InvalidRequest("a remote model needs .base_url(..), the API root".into())
        })?;
        let model = self.model.ok_or_else(|| {
            Error::InvalidRequest(
                "a remote model needs .model(..), the name the endpoint serves it under".into(),
            )
        })?;
        let key = self.api_key.unwrap_or_default();
        let inner = &self.runtime.inner;
        let builder = Engine::builder()
            .settings(inner.settings.clone())
            .config(inner.config.clone())
            .remote_model(&model);
        let builder = match self.format {
            RemoteFormat::OpenAi => builder.openai(base_url, key),
            RemoteFormat::Anthropic => builder.anthropic(base_url, key),
        };
        let engine = builder.build()?;
        Ok(self.runtime.register(Loaded::new(
            engine,
            Some(model),
            ModelSourceKind::Remote,
            None,
            0,
        )))
    }
}

impl std::fmt::Debug for RemoteModelBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteModelBuilder")
            .field("format", &self.format)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "<set>"))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::output::FinishReason;
    use crate::engine::Capabilities;
    use crate::test_support::{Script, Step};
    use crate::types::message::{FunctionDefinition, ToolSpec};

    #[test]
    fn a_model_generates_text_and_keeps_nothing() {
        let model = Runtime::scripted(Script::new().say(["hi ", "there"]));
        let first = model.generate("x").text().expect("should generate");
        let second = model.generate("x").text().expect("should generate again");
        assert_eq!(first, "hi there");
        assert_eq!(
            second, first,
            "each generation starts fresh — a second call must not continue the first"
        );
    }

    #[test]
    fn run_returns_a_structured_response() {
        let model = Runtime::scripted(Script::new().say(["done"]));
        let response = model
            .generate("x")
            .max_tokens(64)
            .run()
            .expect("should generate");
        assert_eq!(response.text(), "done");
        assert_eq!(*response.finish_reason(), FinishReason::Stop);
        assert!(response.tool_calls().is_empty());
        assert_eq!(response.reasoning(), None);
    }

    #[test]
    fn a_tool_call_comes_back_structured_with_its_own_finish_reason() {
        let model = Runtime::scripted(Script::new().program([
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ]));
        let response = model
            .generate("Weather in Paris?")
            .tools([ToolSpec {
                r#type: "function".into(),
                function: FunctionDefinition {
                    name: "get_weather".into(),
                    description: Some("Current weather".into()),
                    arguments: serde_json::json!({"type": "object"}),
                },
            }])
            .run()
            .expect("should generate");
        assert_eq!(*response.finish_reason(), FinishReason::ToolCall);
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Paris");
        assert_eq!(response.text(), "");
    }

    #[test]
    fn tools_and_system_reach_the_backend() {
        let script = Script::new().say(["ok"]);
        let model = Runtime::scripted(script.clone());
        model
            .generate("x")
            .system("Be terse.")
            .tools([ToolSpec {
                r#type: "function".into(),
                function: FunctionDefinition {
                    name: "read".into(),
                    description: None,
                    arguments: serde_json::json!({}),
                },
            }])
            .text()
            .expect("should generate");
        assert_eq!(script.tools_seen(), vec![vec!["read".to_string()]]);
        assert!(
            script.seen().iter().any(|m| m.contains("Be terse.")),
            "the system prompt must reach the backend, saw: {:?}",
            script.seen()
        );
    }

    #[test]
    fn sampling_knobs_reach_the_backend() {
        let script = Script::new().say(["ok"]);
        let model = Runtime::scripted(script.clone());
        model
            .generate("x")
            .temperature(0.3)
            .max_tokens(7)
            .seed(9)
            .top_p(0.5)
            .top_k(3)
            .text()
            .expect("should generate");
        let spec = script.specs_seen().pop().expect("one turn ran");
        assert_eq!(spec.temperature, Some(0.3));
        assert_eq!(spec.max_tokens, Some(7));
        assert_eq!(spec.seed, Some(9));
        assert_eq!(spec.top_p, Some(0.5));
        assert_eq!(spec.top_k, Some(3));
    }

    #[test]
    fn clones_are_the_same_model() {
        let model = Runtime::scripted(Script::new().say(["a"]));
        let other = model.clone();
        assert_eq!(model.id(), other.id());
        assert_eq!(model.info(), other.info());
        // And usable from another thread.
        let text = std::thread::spawn(move || other.generate("x").text())
            .join()
            .expect("thread should not panic")
            .expect("should generate");
        assert_eq!(text, "a");
    }

    #[test]
    fn info_and_capabilities_describe_the_loaded_model() {
        let model = Runtime::scripted(
            Script::new()
                .context(2048)
                .capable_of(Capabilities::TEXT | Capabilities::IMAGES),
        );
        let info = model.info();
        assert_eq!(info.context_window, Some(2048));
        assert_eq!(info.source, ModelSourceKind::LocalFile);
        assert!(info.local);
        assert_eq!(info.name.as_deref(), Some("scripted"));

        let caps = model.capabilities();
        assert!(caps.text);
        assert!(caps.images);
        assert!(!caps.audio);
        assert!(
            !caps.reasoning,
            "nothing about a scripted backend says it reasons"
        );
    }

    #[test]
    fn a_model_hands_back_the_runtime_it_lives_in() {
        let model = Runtime::scripted(Script::new());
        assert_eq!(model.runtime().models(), vec![model.id()]);
    }

    #[test]
    fn a_runtime_lists_what_it_loaded() {
        let runtime = Runtime::builder()
            .max_active_sessions(2)
            .build()
            .expect("builds");
        assert!(runtime.models().is_empty());
        let a = runtime.scripted_in(Script::new(), 0);
        let b = runtime.scripted_in(Script::new(), 0);
        assert_ne!(a.id(), b.id());
        assert_eq!(runtime.models(), vec![a.id(), b.id()]);
    }

    #[test]
    fn a_failed_load_is_an_error_and_starts_no_lingering_controller() {
        // `gen2::load` builds an engine, which starts the loop before
        // loading. If the load fails, the half-built engine must drop and
        // join it — otherwise every typo'd path leaks a thread.
        for _ in 0..8 {
            assert!(crate::load("/nonexistent/model.gguf").is_err());
        }
        let runtime = Runtime::new().expect("builds");
        assert!(runtime.load("/nonexistent/model.gguf").is_err());
        assert!(
            runtime.models().is_empty(),
            "a model that failed to load must not be registered"
        );
    }

    #[test]
    fn a_remote_model_needs_a_url_and_a_name_before_it_connects() {
        let runtime = Runtime::new().expect("builds");
        let err = runtime.openai().model("m").connect().unwrap_err();
        assert!(err.to_string().contains("base_url"), "got: {err}");
        let err = runtime
            .openai()
            .base_url("http://127.0.0.1:1/v1")
            .connect()
            .unwrap_err();
        assert!(err.to_string().contains("model"), "got: {err}");
    }
}
