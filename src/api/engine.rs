//! [`Engine`] — load a model, run turns, shut down cleanly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::backend::common::grammar::GrammarSpec;
use crate::controller::{
    ControllerCmd, ControllerConfig, ControllerHandle, start_controller_joinable,
};
use crate::engine::{Capabilities, LoadOutcome, Settings};
use crate::generation::GenSpec;
use crate::hardware::HardwareProfile;

use super::agent::Agent;
use super::agent_spawned::OwnedAgent;
use super::chat::Chat;
use super::error::{Error, Result};
use super::fit::ModelInfo;
use super::inference::Inference;
use super::session::Session;
use super::spawned::OwnedChat;

/// A running inference engine.
///
/// Owns the controller loop and the backend it holds. Build one with
/// [`Engine::builder`], then start turns with [`Engine::chat`] or
/// [`Engine::infer`].
///
/// Shutting down is automatic: dropping an `Engine` stops the loop and waits
/// for it to release the backend. Use [`Engine::shutdown`] instead when you
/// want to observe teardown rather than let it happen silently.
pub struct Engine {
    handle: ControllerHandle,
    join: Option<JoinHandle<()>>,
    /// How many of each live session's messages the engine has already been
    /// given, so a follow-up sends only what's new.
    ///
    /// Keyed by session id and pruned when a session is dropped —
    /// `Session::opened` is the authority on whether a conversation is live, so
    /// a missing entry here just means "send everything", never a wrong answer.
    sent_through: Mutex<HashMap<String, usize>>,
    /// Sampling defaults every turn starts from.
    defaults: GenSpec,
    /// Bumped every time the chat model changes.
    ///
    /// A session's cached prefill belongs to the model that produced it, so a
    /// swap has to invalidate every live session. Sessions record the
    /// generation they were opened against and reopen when it moves.
    generation: std::sync::atomic::AtomicU64,
    /// The script behind a scripted engine, for tests that assert on what the
    /// backend was shown rather than on what the facade believes.
    #[cfg(test)]
    script: Option<crate::test_support::Script>,
}

impl Engine {
    /// Configure an engine.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    /// Load a model from a path with default settings.
    ///
    /// Shorthand for `Engine::builder().model(path).build()`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::builder().model(path).build()
    }

    /// Start a turn in a conversation you own.
    ///
    /// The reply is appended to `session`, and the engine keeps that
    /// conversation's warm KV cache keyed to it — so a follow-up resends
    /// nothing and re-prefills nothing.
    pub fn chat<'a>(&'a self, session: &'a mut Session) -> Chat<'a> {
        Chat::new(self, session)
    }

    /// Start an agent: tools it can call, and a loop that runs until it
    /// answers or hits a budget.
    ///
    /// Unlike [`Chat::on_tool`](super::Chat::on_tool), the agent owns dispatch
    /// — it resolves the tool the model named, validates the arguments, and
    /// routes failures. You register tools, not a `match`.
    pub fn agent<'a>(&'a self, session: &'a mut Session) -> Agent<'a> {
        Agent::new(self, session)
    }

    /// An agent that runs off the calling thread.
    ///
    /// The shape a UI needs: `spawn()` returns immediately, updates stream
    /// back, and the steering handle can cut a generation short — which the
    /// borrowed [`Engine::agent`] cannot, having no owned engine to ask.
    pub fn agent_owned(self: &Arc<Self>, session: Session) -> OwnedAgent {
        OwnedAgent::new(Arc::clone(self), session)
    }

    /// Run one prompt against a throwaway conversation.
    ///
    /// For when there is nothing to keep: classification, extraction, a title.
    /// Use [`Engine::chat`] with a [`Session`] whenever a later turn might
    /// reference this one.
    pub fn infer(&self, text: impl Into<String>) -> Inference<'_> {
        Inference::new(self, text.into())
    }

    /// Sort text into one of a fixed set of labels.
    ///
    /// The label set is enforced by the decoder, and the returned string is
    /// always one the caller supplied.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), gen2::Error> {
    /// # let engine = gen2::Engine::load("model.gguf")?;
    /// let label = engine
    ///     .classify("The service was fantastic")
    ///     .labels(["positive", "negative", "neutral"])
    ///     .label()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn classify(&self, text: impl Into<String>) -> super::classify::Classify<'_> {
        super::classify::Classify::new(self, text.into())
    }

    /// Read a typed value out of unstructured text.
    ///
    /// The JSON schema the model decodes under is generated from `T`, and the
    /// reply is parsed back into `T`, so the constraint and the parser cannot
    /// drift apart.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), gen2::Error> {
    /// # let engine = gen2::Engine::load("model.gguf")?;
    /// #[derive(serde::Deserialize, schemars::JsonSchema)]
    /// struct Invoice {
    ///     vendor: String,
    ///     total: f64,
    /// }
    ///
    /// let invoice: Invoice = engine.extract("Acme Ltd — total $1,240.00").value()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract<T>(&self, text: impl Into<String>) -> super::extract::Extract<'_, T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        super::extract::Extract::new(self, text.into())
    }

    /// Start a turn on a conversation the engine takes ownership of, so it can
    /// be [`spawn`](OwnedChat::spawn)ed onto a worker thread.
    ///
    /// The session comes back on [`Update::Done`](super::Update::Done).
    pub fn chat_owned(self: &Arc<Self>, session: Session) -> OwnedChat {
        OwnedChat::new(Arc::clone(self), session)
    }

    /// Replace the sampling and prompt settings for subsequent turns.
    pub fn apply_settings(&self, settings: Settings) -> Result<()> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::ApplySettings { settings, resp })?;
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(Error::Load)
    }

    /// Load a chat model, replacing whatever is loaded.
    ///
    /// The engine stays up — sessions, tools, and settings survive. What does
    /// not survive is the cached prefill: it belongs to the previous model, so
    /// every live session reopens on its next turn and pays one re-read. That
    /// happens automatically; there is nothing to remember.
    ///
    /// Returns once the new weights are resident.
    ///
    /// # A failed load leaves no model
    ///
    /// The old model is torn down before the new one is read, so a load that
    /// fails part-way leaves the engine with nothing loaded. The path is
    /// checked first, which catches a missing or non-model file cleanly, but a
    /// failure during load — out of memory, corrupt weights — cannot be undone.
    /// Check [`Engine::is_model_loaded`] after a failure you did not expect.
    /// Returns what the load actually did.
    ///
    /// A failing load is retried with progressively safer configurations —
    /// without the vision projector, then on the CPU — so success alone does
    /// not mean the model is running the way it was asked for. Check
    /// [`LoadOutcome::as_requested`] when that matters, or ignore it when
    /// "running somehow" is the goal.
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<LoadOutcome> {
        self.load_model_with(path, None, Settings::default())
    }

    /// [`Engine::load_model`], with a projector and explicit settings.
    pub fn load_model_with(
        &self,
        path: impl AsRef<Path>,
        mmproj: Option<&Path>,
        settings: Settings,
    ) -> Result<LoadOutcome> {
        let path = path.as_ref();

        // Check the file before asking the controller to load it. A load that
        // fails part-way leaves *nothing* loaded — it tears the old model down
        // first — so a typo in a path would otherwise cost you the model you
        // had. This catches a missing file or a non-model, which is most of it;
        // a failure during load (out of memory, corrupt weights) still ends
        // with no model loaded, and there is no undo for that from here.
        let is_url = path
            .to_str()
            .is_some_and(|p| p.starts_with("http://") || p.starts_with("https://"));
        if !is_url {
            crate::engine::validate_model_file(path)?;
        }

        let (resp, rx) = channel();
        self.send(ControllerCmd::LoadModel {
            model_path: path.to_path_buf(),
            mmproj_path: mmproj.map(Path::to_path_buf),
            settings,
            api_key: None,
            api_format: None,
            resp,
        })?;
        let outcome = rx
            .recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(Error::Load)?;

        // Only after a successful load: a failed swap leaves the old model in
        // place, and invalidating sessions against a model still loaded would
        // cost a re-prefill for nothing.
        self.generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Per-session message counts describe a conversation the new model has
        // never seen.
        if let Ok(mut m) = self.sent_through.lock() {
            m.clear();
        }
        Ok(outcome)
    }

    /// What the loaded model can accept.
    ///
    /// Empty when nothing is loaded. Ask before sending images rather than
    /// discovering it at generation time — though the turn builders check this
    /// themselves, so a mistake surfaces as [`Error::Unsupported`] rather than
    /// a backend failure.
    pub fn capabilities(&self) -> Capabilities {
        let (resp, rx) = channel();
        if self.send(ControllerCmd::GetCapabilities { resp }).is_err() {
            return Capabilities::empty();
        }
        rx.recv().unwrap_or_else(|_| Capabilities::empty())
    }

    /// Whether the loaded model accepts images.
    pub fn supports_images(&self) -> bool {
        self.capabilities().contains(Capabilities::IMAGES)
    }

    /// Whether the loaded model accepts audio.
    pub fn supports_audio(&self) -> bool {
        self.capabilities().contains(Capabilities::AUDIO)
    }

    /// Drop the loaded model, freeing its memory. The engine stays up.
    ///
    /// Live sessions reopen on their next turn, as they do after a swap — a
    /// cached prefill cannot outlive the weights that produced it.
    pub fn unload_model(&self) -> Result<()> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::UnloadModel { resp })?;
        rx.recv().map_err(|_| Error::ControllerGone)?;
        self.generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Ok(mut m) = self.sent_through.lock() {
            m.clear();
        }
        Ok(())
    }

    /// Re-read the current model from disk.
    ///
    /// For picking up a file that changed underneath you. Same invalidation as
    /// a swap: the weights are new even if the path isn't.
    pub fn reload_model(&self) -> Result<()> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::ReloadModel { resp })?;
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(Error::Load)?;
        self.generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Ok(mut m) = self.sent_through.lock() {
            m.clear();
        }
        Ok(())
    }

    /// How many times the chat model has been swapped.
    ///
    /// Sessions use this to notice a swap; exposed because a caller displaying
    /// "model changed" wants the same signal.
    pub fn model_generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Load an embedding model, replacing any already loaded.
    ///
    /// Independent of the chat model: an engine can hold both, or only one.
    /// `kind` forces an embedder family (`"qwen3"`); `None` detects it from the
    /// filename.
    pub fn load_embedder(&self, path: impl AsRef<Path>, kind: Option<String>) -> Result<()> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::LoadEmbedder {
            model_path: path.as_ref().to_path_buf(),
            kind,
            resp,
        })?;
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(Error::Load)
    }

    /// Load a reranking model.
    ///
    /// A cross-encoder — it reads a query and a document together and scores
    /// how well they match. Independent of the chat model: loading one does
    /// not disturb a conversation, and reranking does not stop generation.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), gen2::Error> {
    /// # let engine = gen2::Engine::load("model.gguf")?;
    /// engine.load_reranker("/models/bge-reranker-v2-m3-Q4_K_M.gguf")?;
    /// let ranked = engine.rerank("how do I cancel?", &[
    ///     "Our office hours are 9-5.".to_string(),
    ///     "To cancel, open Settings and choose Close account.".to_string(),
    /// ])?;
    /// assert_eq!(ranked[0].index, 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_reranker(&self, path: impl AsRef<Path>) -> Result<()> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::LoadReranker {
            model_path: path.as_ref().to_path_buf(),
            resp,
        })?;
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(Error::Load)
    }

    /// Score `documents` against `query`, best first.
    ///
    /// Each result carries the document's original position, because the
    /// returned order is not the order passed in. The document text is not
    /// copied back — the caller already has it.
    ///
    /// An empty list returns an empty result without touching the model.
    pub fn rerank(
        &self,
        query: impl Into<String>,
        documents: &[String],
    ) -> Result<Vec<crate::utilities::RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let (resp, rx) = channel();
        self.send(ControllerCmd::Rerank {
            query: query.into(),
            documents: documents.to_vec(),
            resp,
        })?;
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(|message| Error::Generation {
                code: "rerank_failed".into(),
                message,
            })
    }

    /// Whether a reranking model is loaded.
    pub fn is_reranker_loaded(&self) -> bool {
        self.utility_status()
            .map(|s| s.reranker.is_some())
            .unwrap_or(false)
    }

    /// Whether an embedding model is loaded.
    pub fn is_embedder_loaded(&self) -> bool {
        let (resp, rx) = channel();
        if self.send(ControllerCmd::IsEmbedderLoaded { resp }).is_err() {
            return false;
        }
        rx.recv().unwrap_or(false)
    }

    /// Embed one string.
    ///
    /// Prefer [`Engine::embed`] for several at once — batching is what makes
    /// embedding a corpus fast.
    pub fn embed_one(&self, input: impl Into<String>) -> Result<Vec<f32>> {
        let mut out = self.embed(&[input.into()])?;
        out.pop().ok_or_else(|| Error::Generation {
            code: "embedding_failed".into(),
            message: "embedder returned no vectors".into(),
        })
    }

    /// Embed one or more strings with the loaded embedder.
    ///
    /// Returns one vector per input, in order. Load an embedder first, with
    /// [`EngineBuilder::embedder`] or [`Engine::load_embedder`].
    pub fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::GenerateEmbeddings {
            inputs: inputs.to_vec(),
            resp,
        })?;
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(|e| Error::Generation {
                code: "embedding_failed".into(),
                message: e,
            })
    }

    /// Whether a model is currently loaded.
    ///
    /// False if the controller is gone — nothing is loaded in a dead engine.
    pub fn is_model_loaded(&self) -> bool {
        let (resp, rx) = channel();
        if self.send(ControllerCmd::IsModelLoaded { resp }).is_err() {
            return false;
        }
        rx.recv().unwrap_or(false)
    }

    /// Stop a running generation. The stream ends with [`Finish::Stopped`].
    ///
    /// [`Finish::Stopped`]: super::Finish::Stopped
    pub fn stop(&self, chat_id: impl Into<String>) -> Result<()> {
        self.send(ControllerCmd::StopChat {
            chat_id: chat_id.into(),
        })
    }

    /// Pause a running generation, keeping its session warm.
    pub fn pause(&self, chat_id: impl Into<String>) -> Result<()> {
        self.send(ControllerCmd::PauseChat {
            chat_id: chat_id.into(),
        })
    }

    /// Resume a paused generation.
    pub fn resume(&self, chat_id: impl Into<String>) -> Result<()> {
        self.send(ControllerCmd::ResumeChat {
            chat_id: chat_id.into(),
        })
    }

    /// Hint that a model directory is about to be used, so it can be paged in
    /// ahead of the first real request. Fire-and-forget.
    pub fn warm(&self, model_dir: impl Into<PathBuf>) {
        let _ = self.send(ControllerCmd::WarmModel {
            model_dir: model_dir.into(),
        });
    }

    /// Stop the engine and wait for the backend to be released.
    ///
    /// Equivalent to dropping it, except failures surface instead of being
    /// swallowed.
    pub fn shutdown(mut self) -> Result<()> {
        self.stop_and_join()
    }

    /// The lower-level controller handle, for what this facade doesn't cover.
    pub fn controller(&self) -> &ControllerHandle {
        &self.handle
    }

    pub(crate) fn send(&self, cmd: ControllerCmd) -> Result<()> {
        self.handle.send(cmd).map_err(|_| Error::ControllerGone)
    }

    /// Send a raw controller command.
    ///
    /// Tests only, and only for the ones that need to start work without
    /// waiting for it — proving a helper request does not block chat
    /// scheduling means firing one and then doing something else, which no
    /// blocking public method can express.
    #[cfg(test)]
    pub(crate) fn send_for_test(&self, cmd: ControllerCmd) -> Result<()> {
        self.send(cmd)
    }

    /// Which auxiliary runtimes are loaded.
    ///
    /// Separate from [`Engine::capabilities`], which describes what the chat
    /// model can accept. An embedder being resident says nothing about
    /// whether the loaded model can read an image.
    pub fn utility_status(&self) -> Result<crate::utilities::UtilityStatus> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::GetUtilityStatus { resp })?;
        rx.recv().map_err(|_| Error::ControllerGone)
    }

    pub(crate) fn event_channel_capacity(&self) -> usize {
        self.handle.config().event_channel_capacity
    }

    /// How many of `session_id`'s messages the engine already has.
    pub(crate) fn sent_through(&self, session_id: &str) -> usize {
        self.sent_through
            .lock()
            .ok()
            .and_then(|m| m.get(session_id).copied())
            .unwrap_or(0)
    }

    /// Record that the engine now has `count` of `session_id`'s messages.
    pub(crate) fn mark_sent(&self, session_id: &str, count: usize) {
        if let Ok(mut m) = self.sent_through.lock() {
            // Dropping a Session cannot reach in here to clean up, and
            // `Engine::infer` mints one per call, so this is bounded rather
            // than left to grow for the life of the process. Well above
            // `max_active_chats`, since a dropped entry only costs the next
            // turn a resend.
            const MAX_TRACKED: usize = 1024;
            if m.len() >= MAX_TRACKED && !m.contains_key(session_id) {
                m.clear();
            }
            m.insert(session_id.to_string(), count);
        }
    }

    /// Forget a conversation's cached bookkeeping.
    ///
    /// Called when a [`Session`] is finished with. Not required for
    /// correctness — a missing entry just means the next turn resends
    /// everything — but without it the map grows for the life of the process.
    pub fn forget(&self, session: &Session) {
        if let Ok(mut m) = self.sent_through.lock() {
            m.remove(session.id());
        }
    }

    /// Sampling defaults every turn starts from.
    pub(crate) fn default_gen_spec(&self) -> GenSpec {
        self.defaults.clone()
    }

    fn stop_and_join(&mut self) -> Result<()> {
        // Ignore the send error: if the loop is already gone there is nothing
        // to ask, and the join below is what actually matters.
        let _ = self.handle.send(ControllerCmd::Shutdown);
        match self.join.take() {
            Some(join) => join.join().map_err(|_| Error::Generation {
                code: "controller_panicked".into(),
                message: "the controller loop panicked during shutdown".into(),
            }),
            None => Ok(()),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Not just tidiness: the loop holds the backend's native context, and
        // a process that exits while it is still running tears down ggml's
        // statics underneath it and aborts. Joining here is what makes that
        // impossible to get wrong.
        let _ = self.stop_and_join();
    }
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").finish_non_exhaustive()
    }
}

/// Builds an [`Engine`].
#[derive(Default)]
pub struct EngineBuilder {
    model_path: Option<PathBuf>,
    mmproj_path: Option<PathBuf>,
    settings: Option<Settings>,
    config: Option<ControllerConfig>,
    api_key: Option<String>,
    api_format: Option<String>,
    embedder_path: Option<PathBuf>,
    reranker_path: Option<PathBuf>,
    embedder_kind: Option<String>,
    context: ContextChoice,
    defaults: GenSpec,
    plugins: Vec<Arc<crate::advanced::BackendPlugin>>,
}

/// How the context window is decided.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum ContextChoice {
    /// Whatever the backend picks. No fit check.
    #[default]
    Backend,
    /// This exact size; fail if it doesn't fit.
    Exact(u32),
    /// The largest the machine can give.
    Auto,
}

impl EngineBuilder {
    /// The model to load — a GGUF file, an MLX model directory, or a
    /// `.litertlm` bundle. The
    /// backend is chosen from what's there; you never name one.
    pub fn model(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// A multimodal projector, for vision models that need one alongside the
    /// weights.
    pub fn mmproj(mut self, path: impl AsRef<Path>) -> Self {
        self.mmproj_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sampling, stopping, and prompt settings. Defaults are sensible; set
    /// this to override them wholesale.
    pub fn settings(mut self, settings: Settings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// Controller policy: how many chats may run at once, event buffering,
    /// generation timeout.
    pub fn config(mut self, config: ControllerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Bring your own backend.
    ///
    /// A path the plugin claims loads through it, ahead of every built-in
    /// rule; every other path is routed as before. Register several and they
    /// are asked in this order. See [`crate::advanced::plugin`] for how to
    /// write one.
    pub fn backend(mut self, plugin: crate::advanced::BackendPlugin) -> Self {
        self.plugins.push(Arc::new(plugin));
        self
    }

    /// Serve from an OpenAI-compatible endpoint instead of local weights.
    ///
    /// `base_url` is the API root — `https://api.openai.com/v1`, or any
    /// compatible server (`http://localhost:11434/v1` for Ollama). The backend
    /// is selected from the URL, so this must be one.
    pub fn openai(mut self, base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        self.model_path = Some(PathBuf::from(base_url.into()));
        self.api_key = Some(api_key.into());
        self.api_format = Some("openai".into());
        self
    }

    /// Serve from an Anthropic-compatible endpoint instead of local weights.
    ///
    /// `base_url` is the API root, e.g. `https://api.anthropic.com/v1`.
    pub fn anthropic(mut self, base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        self.model_path = Some(PathBuf::from(base_url.into()));
        self.api_key = Some(api_key.into());
        self.api_format = Some("anthropic".into());
        self
    }

    /// Size the context window to the machine.
    ///
    /// Reads the model's header, measures the hardware, and picks the largest
    /// context that fits. Beats guessing: too small wastes the model, too large
    /// fails at load time or thrashes.
    ///
    /// GGUF only — other formats keep the backend's default.
    pub fn auto_context(mut self) -> Self {
        self.context = ContextChoice::Auto;
        self
    }

    /// Use exactly this context window.
    ///
    /// [`build`](Self::build) fails with [`Error::WontFit`] if the machine
    /// can't supply it, rather than discovering it during load.
    pub fn context(mut self, tokens: u32) -> Self {
        self.context = ContextChoice::Exact(tokens);
        self
    }

    /// Load an embedding model alongside (or instead of) the chat model.
    ///
    /// An engine with only an embedder is valid — [`Engine::embed`] works and
    /// generation returns [`Error::ModelNotLoaded`].
    pub fn embedder(mut self, path: impl AsRef<Path>) -> Self {
        self.embedder_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Load a reranking model alongside the chat model.
    pub fn reranker(mut self, path: impl AsRef<Path>) -> Self {
        self.reranker_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Force an embedder family (`"qwen3"`) instead of detecting it from the
    /// filename.
    pub fn embedder_kind(mut self, kind: impl Into<String>) -> Self {
        self.embedder_kind = Some(kind.into());
        self
    }

    /// Cap tokens for every turn unless it overrides this.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.defaults.max_tokens = Some(n);
        self
    }

    /// Sampling temperature for every turn unless it overrides this.
    pub fn temperature(mut self, t: f32) -> Self {
        self.defaults.temperature = Some(t);
        self
    }

    /// Seed for every turn unless it overrides this.
    pub fn seed(mut self, seed: u64) -> Self {
        self.defaults.seed = Some(seed);
        self
    }

    /// Decode deterministically by default: temperature 0 with a fixed seed.
    ///
    /// Set here, every turn is reproducible without repeating `.greedy()` at
    /// each call site.
    pub fn greedy(mut self) -> Self {
        self.defaults.temperature = Some(0.0);
        self.defaults.seed = Some(self.defaults.seed.unwrap_or(0));
        self
    }

    /// Constrain every turn's output to a grammar unless it overrides this.
    ///
    /// Useful when an engine exists to produce one shape — a classifier, an
    /// extractor. A turn can still pass its own [`Chat::grammar`], or drop the
    /// default with [`Chat::unconstrained`], because loading the weights is the
    /// expensive part and one engine should be able to serve several shapes.
    pub fn grammar(mut self, grammar: GrammarSpec) -> Self {
        self.defaults.grammar = Some(grammar);
        self
    }

    /// Start the controller and load the model.
    ///
    /// Returns once the weights are resident and the engine is ready to
    /// generate — no separate "is it loaded yet" step.
    pub fn build(self) -> Result<Engine> {
        if self.model_path.is_none() && self.embedder_path.is_none() {
            return Err(Error::Load(
                "nothing to load — call .model(path), .embedder(path), .openai(..), \
                 or .anthropic(..)"
                    .into(),
            ));
        }

        // Preflight the fit before starting anything, so a model that cannot
        // run fails with a verdict instead of a load error.
        let mut config = self.config.unwrap_or_default();
        config.plugins.extend(self.plugins);
        let mut settings = self.settings.unwrap_or_default();

        // An explicit context is honoured whether or not the preflight below
        // can run. `ModelInfo::read` understands GGUF, so without this
        // `.context(n)` did nothing at all for a safetensors bundle, a
        // `.litertlm` file, or anything else it cannot parse — accepted,
        // documented, and silently dropped. The preflight still clamps it when
        // the model *is* readable, which is why this is set first.
        if let ContextChoice::Exact(n) = self.context {
            settings.system.ctx_size = Some(n);
        }

        if self.context != ContextChoice::Backend
            && let Some(path) = self.model_path.as_deref()
            && let Ok(info) = ModelInfo::read(path)
        {
            let hw = HardwareProfile::detect();
            let wanted = match self.context {
                ContextChoice::Exact(n) => Some(n),
                // Sized for as many conversations as the controller will keep
                // resident, not for one. Every live conversation holds its own
                // KV cache, so a window picked for a single chat is exceeded
                // as soon as a second one opens.
                _ => Some(info.max_context_for(&hw, config.max_active_chats)),
            };
            let fit = info.fits(&hw, wanted);
            if !fit.ok() {
                return Err(Error::WontFit(Box::new(fit)));
            }
            settings.system.ctx_size = Some(fit.context);
        }

        let (handle, join) = start_controller_joinable(config);
        let engine = Engine {
            handle,
            join: Some(join),
            sent_through: Mutex::new(HashMap::new()),
            defaults: self.defaults,
            generation: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            script: None,
        };

        // On any failure below, `engine` drops here — which stops and joins the
        // loop we just started, so a failed load leaks no thread.
        if let Some(model_path) = self.model_path {
            let (resp, rx) = channel();
            engine.send(ControllerCmd::LoadModel {
                model_path,
                mmproj_path: self.mmproj_path,
                settings,
                api_key: self.api_key,
                api_format: self.api_format,
                resp,
            })?;
            rx.recv()
                .map_err(|_| Error::ControllerGone)?
                .map_err(Error::Load)?;
        }

        if let Some(reranker_path) = self.reranker_path {
            engine.load_reranker(reranker_path)?;
        }
        if let Some(embedder_path) = self.embedder_path {
            engine.load_embedder(embedder_path, self.embedder_kind)?;
        }

        Ok(engine)
    }
}

#[cfg(test)]
impl Engine {
    /// An engine over a scripted backend, with a model already loaded.
    ///
    /// The seam agent tests use. Real model behaviour is probabilistic — an
    /// agent test that needs the model to emit two tool calls can only hope it
    /// does — so the loop's own contracts (dispatch, budgets, approval,
    /// steering, scheduling) are tested against a script instead, and the live
    /// tests prove a real backend implements the same contract.
    pub(crate) fn scripted(script: crate::test_support::Script) -> Self {
        Self::scripted_with_config(script, ControllerConfig::default())
    }

    /// As [`Self::scripted`], with controller policy the test chooses — the
    /// way to make eviction happen without opening dozens of conversations.
    pub(crate) fn scripted_with_config(
        script: crate::test_support::Script,
        config: ControllerConfig,
    ) -> Self {
        Self::scripted_with(script, config, None)
    }

    /// As [`Self::scripted_with_config`], with the utility worker supplied.
    ///
    /// The seam for proving a busy helper does not stop chat tokens.
    pub(crate) fn scripted_with(
        script: crate::test_support::Script,
        config: ControllerConfig,
        utilities: Option<crate::utilities::UtilityWorker>,
    ) -> Self {
        let kept = script.clone();
        let (handle, join) = crate::controller::start_controller_with_engine_and_utilities(
            config,
            script.into_engine_factory(),
            utilities,
        );
        let engine = Engine {
            handle,
            join: Some(join),
            sent_through: Mutex::new(HashMap::new()),
            defaults: GenSpec::default(),
            generation: std::sync::atomic::AtomicU64::new(0),
            script: Some(kept),
        };
        let (resp, rx) = channel();
        engine
            .send(ControllerCmd::LoadModel {
                model_path: "/scripted/model.gguf".into(),
                mmproj_path: None,
                settings: Default::default(),
                api_key: None,
                api_format: None,
                resp,
            })
            .expect("the scripted controller should accept a load");
        rx.recv()
            .expect("the scripted controller should answer")
            .expect("the scripted backend should load");
        engine
    }

    /// The script driving this engine's backend.
    pub(crate) fn script(&self) -> &crate::test_support::Script {
        self.script
            .as_ref()
            .expect("only a scripted engine has one")
    }
}

impl std::fmt::Debug for EngineBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineBuilder")
            .field("model_path", &self.model_path)
            .field("mmproj_path", &self.mmproj_path)
            .finish_non_exhaustive()
    }
}

/// Build the event channel a turn streams over.
pub(crate) fn event_channel(
    capacity: usize,
) -> (
    std::sync::mpsc::SyncSender<crate::controller::ControllerEvent>,
    std::sync::mpsc::Receiver<crate::controller::ControllerEvent>,
) {
    sync_channel(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An app runs inference off its UI thread, so `Arc<Engine>` has to be
    /// shareable. If this stops compiling, every multi-threaded consumer breaks.
    #[test]
    fn engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
        assert_send_sync::<std::sync::Arc<Engine>>();
    }

    #[test]
    #[cfg(feature = "backend-external-api")]
    fn openai_builder_selects_the_external_api_backend() {
        // Backend detection keys off the path being a URL. An earlier version
        // put the model *name* here, which silently fell through to the default
        // local backend instead of talking to the API at all.
        let b = Engine::builder().openai("https://api.openai.com/v1", "sk-test");
        let detected = crate::backend::Engine::detect_backend_for_path(
            b.model_path.as_ref().unwrap().as_path(),
        );
        assert_eq!(detected, "external-api", "got {detected}");
    }

    #[test]
    fn building_with_nothing_to_load_is_an_error_not_a_panic() {
        let err = Engine::builder().build().unwrap_err();
        assert!(
            err.to_string().contains("nothing to load"),
            "error should say what is missing, got: {err}"
        );
    }

    #[test]
    fn a_failed_build_starts_no_lingering_controller() {
        // The builder starts the loop before loading. If the load fails, the
        // half-built Engine must drop and join it — otherwise every failed
        // load leaks a thread holding a backend.
        for _ in 0..8 {
            assert!(
                Engine::builder()
                    .model("/nonexistent/model.gguf")
                    .build()
                    .is_err()
            );
        }
    }
}
