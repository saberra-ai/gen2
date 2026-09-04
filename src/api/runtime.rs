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
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::controller::ControllerConfig;
use crate::engine::Settings;

use super::engine::Engine;
use super::error::{Error, Result};
use super::fit;
use super::model::{Model, ModelId, ModelSourceKind};

/// Owns backends, loaded weights, and the registry of models built on them.
///
/// Cheap to clone; every clone is the same runtime. Dropping the last handle
/// — and the last [`Model`] from it — shuts the backends down.
///
/// # What a runtime is today
///
/// Each loaded model runs on its own controller loop: a runtime with N models
/// holds N engines, each with its own weights resident. Nothing is shared or
/// evicted between them yet — a `Model` handle stays valid for as long as it
/// exists, and its weights stay loaded for as long as that. Residency
/// policy across models (evict, restore on use, preload) is S2.4.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

/// Shared state behind every handle to one runtime.
pub(crate) struct RuntimeInner {
    config: ControllerConfig,
    settings: Settings,
    /// Every model loaded so far, by id. Entries are never removed today —
    /// see the note on [`Runtime`]. Held so a runtime can enumerate what it
    /// owns; each [`Model`] carries its own `Arc` to the same entry.
    models: Mutex<HashMap<ModelId, Arc<Loaded>>>,
    next_id: AtomicU64,
}

/// One model the runtime has loaded.
pub(crate) struct Loaded {
    pub(crate) engine: Engine,
    pub(crate) name: Option<String>,
    pub(crate) source: ModelSourceKind,
    /// The file's header, when it was a readable GGUF. What
    /// [`Model::capabilities`] reads tool support from.
    pub(crate) header: Option<fit::ModelInfo>,
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
    /// backend reads. The backend is chosen from what is there.
    ///
    /// Returns once the weights are resident. A path that does not exist, or
    /// is not a model, fails here rather than at first use.
    pub fn load(&self, path: impl AsRef<Path>) -> Result<Model> {
        let path = path.as_ref();
        let engine = Engine::builder()
            .model(path)
            .settings(self.inner.settings.clone())
            .config(self.inner.config.clone())
            .build()?;
        let header = fit::ModelInfo::read(path).ok();
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty());
        Ok(self.register(Loaded {
            engine,
            name,
            source: ModelSourceKind::LocalFile,
            header,
        }))
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

    // S2.4: `hardware()`, `residency()`, `preload`, `evict`, `stats` — the
    // advanced runtime controls of spec §4.5 — arrive with multi-model
    // residency, under `gen2::advanced`.

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
        runtime.register(Loaded {
            engine: Engine::scripted(script),
            name: Some("scripted".into()),
            source: ModelSourceKind::LocalFile,
            header: None,
        })
    }
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

    /// Build it. Starts nothing: backends start when a model is loaded.
    pub fn build(self) -> Result<Runtime> {
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                config: self.config,
                settings: self.settings.unwrap_or_default(),
                models: Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(0),
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
        Ok(self.runtime.register(Loaded {
            engine,
            name: Some(model),
            source: ModelSourceKind::Remote,
            header: None,
        }))
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
        let a = runtime.register(Loaded {
            engine: Engine::scripted(Script::new()),
            name: None,
            source: ModelSourceKind::LocalFile,
            header: None,
        });
        let b = runtime.register(Loaded {
            engine: Engine::scripted(Script::new()),
            name: None,
            source: ModelSourceKind::LocalFile,
            header: None,
        });
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
