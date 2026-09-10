//! [`Model`] — a logical inference target.

use std::sync::Arc;

use super::engine::Engine;
use super::generation::Generation;
use super::input::Input;
use super::runtime::{Loaded, Runtime, RuntimeInner};
use super::session::Session;
use super::turn::Turn;
use crate::engine::GpuOffload;

/// A model you ask to generate.
///
/// Obtained from [`Runtime::load`](super::Runtime::load),
/// [`Runtime::openai`](super::Runtime::openai), or [`gen2::load`](crate::load).
/// Cheap to clone and safe to share across threads; every clone is the same
/// model. A `Model` owns no conversation — one-shot generation uses a
/// conversation it discards, and a persistent one is a [`Session`], run
/// through [`Model::turn`].
///
/// The handle keeps its runtime alive, so a model from [`gen2::load`](crate::load)
/// needs nothing else held.
#[derive(Clone)]
pub struct Model {
    runtime: Arc<RuntimeInner>,
    id: ModelId,
    loaded: Arc<Loaded>,
}

impl Model {
    pub(crate) fn new(runtime: Arc<RuntimeInner>, id: ModelId, loaded: Arc<Loaded>) -> Self {
        Self {
            runtime,
            id,
            loaded,
        }
    }

    /// Generate a reply to `input`, keeping nothing afterwards.
    ///
    /// Returns a builder; nothing runs until [`Generation::run`] or
    /// [`Generation::text`].
    ///
    /// ```no_run
    /// # let model = gen2::load("m.gguf")?;
    /// let text = model.generate("Why is the sky blue?").max_tokens(64).text()?;
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn generate(&self, input: impl Into<Input>) -> Generation<'_> {
        Generation::new(self, input.into())
    }

    /// One invocation of this model against a conversation (api_spec.md §11).
    ///
    /// Returns a builder: stage messages, set controls, then
    /// [`Turn::run`] or [`Turn::stream`]. The reply is appended to the
    /// session. A turn with no new message is valid — it is how a session
    /// continues after tool results were pushed.
    ///
    /// ```no_run
    /// # let model = gen2::load("m.gguf")?;
    /// let mut session = gen2::Session::new().with_system("Be concise.");
    /// let first = model.turn(&mut session).user("Explain CRDTs").run()?;
    /// let second = model.turn(&mut session).user("Now compare them to Raft").run()?;
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn turn<'a>(&'a self, session: &'a mut Session) -> Turn<'a> {
        Turn::new(self, session)
    }

    /// What this model is.
    pub fn info(&self) -> ModelInfo {
        let snapshot = self
            .engine()
            .controller()
            .get_controller_runtime_snapshot()
            .ok();
        let header = self.loaded.header.as_ref();
        ModelInfo {
            id: self.id,
            name: self.loaded.name.clone(),
            architecture: snapshot
                .as_ref()
                .and_then(|s| s.loaded_model_architecture.clone())
                .or_else(|| header.and_then(|h| h.architecture.clone())),
            context_window: snapshot.as_ref().and_then(|s| s.loaded_model_context),
            offload: snapshot.and_then(|s| s.loaded_model_offload),
            source: self.loaded.source.clone(),
            local: self.loaded.source.is_local(),
        }
    }

    /// What this model can be asked to do.
    pub fn capabilities(&self) -> ModelCapabilities {
        let engine = self.engine();
        let caps = engine.capabilities();
        let local = self.loaded.source.is_local();
        let architecture = self
            .loaded
            .header
            .as_ref()
            .and_then(|h| h.architecture.clone())
            .or_else(|| {
                engine
                    .controller()
                    .get_controller_runtime_snapshot()
                    .ok()
                    .and_then(|s| s.loaded_model_architecture)
            });
        ModelCapabilities {
            text: true,
            images: caps.contains(crate::engine::Capabilities::IMAGES),
            audio: caps.contains(crate::engine::Capabilities::AUDIO),
            // Local backends render tool declarations through the chat
            // template; the header says whether the template has a place for
            // them. The remote backend does not put `tools` on the wire yet.
            tools: local
                && self
                    .loaded
                    .header
                    .as_ref()
                    .map(|h| h.supports_tools)
                    .unwrap_or(true),
            reasoning: architecture
                .as_deref()
                .is_some_and(is_reasoning_architecture),
            // Grammar-constrained decoding is llama.cpp's; nothing else
            // compiled in enforces a schema, and the remote request carries
            // no `response_format`.
            structured_output: local && cfg!(feature = "backend-llamacpp"),
        }
    }

    /// This model's id within its runtime.
    pub fn id(&self) -> ModelId {
        self.id
    }

    /// The engine serving this model.
    pub(crate) fn engine(&self) -> &Engine {
        &self.loaded.engine
    }

    /// The runtime's entry for this model.
    pub(crate) fn loaded(&self) -> &Arc<Loaded> {
        &self.loaded
    }

    /// Whether this handle came from the runtime behind `inner`.
    pub(crate) fn belongs_to(&self, inner: &Arc<RuntimeInner>) -> bool {
        Arc::ptr_eq(&self.runtime, inner)
    }

    /// Make sure the weights are resident before a turn (api_spec.md §4.2).
    ///
    /// A handle stays valid after [`Runtime::evict`](super::Runtime::evict)
    /// or an automatic eviction; the next use restores the model, evicting
    /// the least recently used other model first if the budget needs it.
    /// Marks the model used either way, for that ordering.
    pub(crate) fn ensure_resident(&self) -> super::error::Result<()> {
        Runtime::from_inner(Arc::clone(&self.runtime)).restore(self.id, &self.loaded)?;
        self.loaded.touch();
        Ok(())
    }

    /// The runtime this model belongs to — the one [`gen2::load`](crate::load)
    /// made privately, or the one it was loaded into.
    pub fn runtime(&self) -> Runtime {
        Runtime::from_inner(Arc::clone(&self.runtime))
    }
}

/// Architectures whose chat template the crate knows to carry a reasoning
/// channel: Qwen3's `<think>` and Gemma 4's thought channel are the two the
/// backends' templates and `ThinkingMode` are written against.
fn is_reasoning_architecture(arch: &str) -> bool {
    matches!(arch, "qwen3" | "qwen3moe" | "gemma4")
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("id", &self.id)
            .field("name", &self.loaded.name)
            .field("source", &self.loaded.source)
            .finish_non_exhaustive()
    }
}

/// Identifies a model within its [`Runtime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(pub(crate) u64);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "model-{}", self.0)
    }
}

/// What a model is, without backend internals.
///
/// Format, quantization, and weight size for a local file are on
/// [`gen2::advanced::fit::ModelInfo`](crate::advanced::fit::ModelInfo), read from the header.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelInfo {
    /// Its id within the runtime.
    pub id: ModelId,
    /// A display name: the file stem for a local model, the served name for
    /// a remote one.
    pub name: Option<String>,
    /// Architecture family — `"llama"`, `"qwen3"`, `"gemma3"` — when known.
    pub architecture: Option<String>,
    /// The context window the backend allocated, in tokens. `None` for a
    /// remote model whose provider did not advertise one.
    pub context_window: Option<u32>,
    /// Where the model runs.
    pub source: ModelSourceKind,
    /// Whether inference happens on this machine.
    pub local: bool,
    /// Where the weights are: layers on the GPU, and which GPU, as the load
    /// placed them. `None` for a remote model, or a backend that does not
    /// report placement. On Apple silicon the default build reports the
    /// Metal backend (`"MTL"`, ggml's name for it) — no feature flag is
    /// involved.
    pub offload: Option<GpuOffload>,
}

/// Where a model's weights are.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelSourceKind {
    /// A file (or bundle directory) on this machine.
    LocalFile,
    /// A file fetched from the Hugging Face Hub by `hf:` reference and
    /// cached on this machine — see [`gen2::hf`](crate::hf).
    HuggingFace {
        /// `owner/name`.
        repo: String,
        /// The file within the repo that was loaded.
        file: String,
    },
    /// An OpenAI- or Anthropic-compatible endpoint.
    Remote,
}

impl ModelSourceKind {
    /// Whether inference happens on this machine.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::LocalFile | Self::HuggingFace { .. })
    }
}

/// What a model can be asked to do.
///
/// Runtime machinery — KV snapshots, poison detection — is not here; this is
/// the caller's view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ModelCapabilities {
    /// Generates text. Always true.
    pub text: bool,
    /// Accepts images on the input.
    pub images: bool,
    /// Accepts audio on the input.
    pub audio: bool,
    /// Can be offered tools and asked to call them.
    pub tools: bool,
    /// Exposes a reasoning channel the response separates from the reply.
    pub reasoning: bool,
    /// Output can be constrained to a schema during decoding.
    pub structured_output: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A UI generates off its main thread, so a `Model` has to cross threads
    /// and be shared. If this stops compiling, every such consumer breaks.
    #[test]
    fn model_is_clone_send_and_sync() {
        fn assert_all<T: Clone + Send + Sync + 'static>() {}
        assert_all::<Model>();
    }

    #[test]
    fn reasoning_is_claimed_only_for_families_the_crate_knows() {
        assert!(is_reasoning_architecture("qwen3"));
        assert!(is_reasoning_architecture("gemma4"));
        assert!(!is_reasoning_architecture("llama"));
        assert!(!is_reasoning_architecture("qwen2"));
    }
}
