//! Gen2 inference controller (session runtimes, dispatch, delivery).
//!
//! Runtime contract (lifecycle, teardown, delivery, snapshots): repository file
//! `docs/gen2-controller-runtime-contract.md`.
mod commands;
mod config;
mod lifecycle;
pub(crate) mod metrics;
mod observability;
mod observability_snapshot;
mod runtime_snapshot;
mod scheduler;
mod state;
mod state_transitions;

// Sibling modules expose pub(super) items already — callers use
// super::lifecycle::* / super::observability::* directly.

pub use config::ControllerConfig;
pub use metrics::ControllerMetricsSnapshot;
pub use observability_snapshot::{ControllerObservabilitySnapshot, ControllerPolicySnapshot};
pub use runtime_snapshot::{
    ActiveChatSnapshot, ControllerRuntimeSnapshot, RuntimeLifecycleSnapshot,
};
pub use state::ControllerState;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel};
use std::thread;
use std::time::{Duration, Instant};

use crate::ExecutionStats;
use crate::engine::Settings;
use crate::generation::GenSpec;
use crate::types::message::Message;

/// A background inference workload: ephemeral, fire-and-forget, hidden from
/// the user.
///
/// Each runs as its own short-lived session on the controller. Prompt in,
/// tokens back, session cleaned up on completion.
///
/// The named variants are the ones any assistant has. Everything domain
/// specific is [`SystemTask::Custom`] — the label namespaces the ephemeral
/// session id and shows up in observability, and nothing here reads meaning
/// into it. Pair a custom task with the sampling it wants through
/// [`InferenceHandle::system_infer_with`]; its
/// [`default_gen_spec`](SystemTask::default_gen_spec) is deliberately plain.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[non_exhaustive]
pub enum SystemTask {
    /// Name a conversation from its history.
    Title,
    /// Propose follow-up prompts.
    Suggestions,
    /// Summarise a conversation down to fit the window again.
    Compact,
    /// Short topic line for a conversation list.
    Summary,
    /// Anything else the host runs in the background.
    Custom(std::borrow::Cow<'static, str>),
}

impl SystemTask {
    /// A host-defined background task, labelled for session ids and traces.
    ///
    /// ```
    /// # use gen2::SystemTask;
    /// let task = SystemTask::custom("triples");
    /// assert!(task.session_id().starts_with("triples-"));
    /// ```
    pub fn custom(label: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        Self::Custom(label.into())
    }
}

/// Primary user chat vs an internal system inference workload.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[non_exhaustive]
pub enum WorkloadKind {
    PrimaryChat,
    SystemTask(SystemTask),
}

/// Why a generation completed successfully from the controller's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[non_exhaustive]
pub enum CompletionReason {
    Eos,
    StoppedByUser,
    Evicted,
    ReceiverDropped,
    ModelReloaded,
    ControllerShutdown,
}

/// Why a generation failed or was aborted as an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[allow(dead_code)]
#[non_exhaustive]
pub enum FailureReason {
    Timeout,
    GenerationError,
    SessionPoisoned,
    StartSessionFailed,
    PullerCreateFailed,
}

/// Explicit lifecycle for a single chat runtime (no implicit bool matrix).
#[allow(clippy::large_enum_variant)]
pub(super) enum ChatRunState {
    /// Session may exist but no generation is active (reserved for future use).
    Idle,
    Generating {
        puller: crate::generation::TokenPuller,
        gen_started: Instant,
        last_gen_spec: GenSpec,
    },
    Paused {
        last_gen_spec: GenSpec,
    },
    Completed(CompletionReason),
    Failed(FailureReason),
}

impl ChatRunState {
    pub(super) fn is_generating(&self) -> bool {
        matches!(self, Self::Generating { .. })
    }

    pub(super) fn is_paused(&self) -> bool {
        matches!(self, Self::Paused { .. })
    }

    pub(super) fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed(_) | Self::Failed(_))
    }

    /// `StartChat` / `ContinueChat` / `ResumeChat` may call `Session::pull` only when not mid-flight.
    #[allow(dead_code)]
    pub(super) fn allows_user_pull_start(&self) -> bool {
        !self.is_generating()
    }

    /// Monotonic start instant for the active puller, if any.
    pub(super) fn generation_started_at(&self) -> Option<Instant> {
        match self {
            Self::Generating { gen_started, .. } => Some(*gen_started),
            _ => None,
        }
    }
}

impl WorkloadKind {
    pub(super) fn is_system_task(&self) -> bool {
        matches!(self, Self::SystemTask(_))
    }
}

impl SystemTask {
    /// A fresh session id for one run of this task.
    pub fn session_id(&self) -> String {
        format!("{}-{}", self.label(), uuid::Uuid::new_v4())
    }

    /// What this task calls itself, in session ids and traces.
    pub fn label(&self) -> &str {
        match self {
            Self::Title => "title",
            Self::Suggestions => "suggestions",
            Self::Compact => "compact",
            Self::Summary => "summary",
            Self::Custom(label) => label,
        }
    }

    /// Sensible generation defaults for each task type.
    ///
    /// Delegates to [`ControllerConfig::system_task_spec`] with default config.
    /// Callers can override by passing their own `GenSpec` to
    /// `InferenceHandle::system_infer_with`.
    pub fn default_gen_spec(&self) -> GenSpec {
        ControllerConfig::default().system_task_spec(self)
    }
}

/// Commands accepted by the inference controller thread.
///
/// Grouped into four categories:
/// - **Model lifecycle** — load/reload models and apply settings
/// - **Status queries** — synchronous checks on loaded state
/// - **Chat operations** — start, continue, pause, stop generation
/// - **Utility** — embeddings, system inference, shutdown
#[non_exhaustive]
pub enum ControllerCmd {
    // ── Model lifecycle ──────────────────────────────────────────────
    /// Load or reload the primary LLM from disk with the given settings.
    LoadModel {
        model_path: PathBuf,
        mmproj_path: Option<PathBuf>,
        settings: Settings,
        api_key: Option<String>,
        api_format: Option<String>,
        resp: Sender<Result<(), String>>,
    },
    /// Apply new sampling/system settings without reloading the model.
    ApplySettings {
        settings: Settings,
        resp: Sender<Result<(), String>>,
    },
    /// Load or reload the embedding model.
    LoadEmbedder {
        model_path: PathBuf,
        /// Optional explicit embedder family override (e.g. `"qwen3"`). `None`
        /// → detect from the filename (defaults to EmbeddingGemma). Keeps the
        /// default path unchanged; lets opt-in callers force Qwen3-Embedding.
        kind: Option<String>,
        resp: Sender<Result<(), String>>,
    },

    // ── Status queries ───────────────────────────────────────────────
    /// Check whether the primary LLM is loaded and ready.
    IsModelLoaded { resp: Sender<bool> },
    /// What the loaded model can accept — text, images, audio.
    GetCapabilities {
        resp: Sender<crate::engine::Capabilities>,
    },
    /// Drop the loaded model, freeing its memory. The engine stays up.
    UnloadModel { resp: Sender<()> },
    /// Re-read the current model from disk.
    ReloadModel { resp: Sender<Result<(), String>> },
    /// Which backend is currently active in the controller's engine (the facade
    /// `active_backend_name()` — e.g. `"llamacpp"`, `"mlx"`, `"mlxcel"`). Proves
    /// routing end-to-end: after `LoadModel` a caller can assert the model
    /// actually landed on the intended backend, not a fallback (S6 mlxcel
    /// go/no-go). Returns `"none"` when no model is loaded.
    GetActiveBackendName { resp: Sender<&'static str> },
    /// Check whether the embedding model is loaded.
    IsEmbedderLoaded { resp: Sender<bool> },
    /// Check whether the multimodal projector is loaded (image support).
    IsMmprojLoaded { resp: Sender<bool> },
    /// Check whether a chat session is active for the given chat_id.
    IsChatLoaded { chat_id: String, resp: Sender<bool> },
    /// PR6: snapshot of controller delivery / termination counters (local engine thread).
    GetControllerMetrics {
        resp: Sender<ControllerMetricsSnapshot>,
    },
    /// PR7: active chat runtimes (ids, workload, lifecycle) — no payloads.
    GetControllerRuntimeSnapshot {
        resp: Sender<ControllerRuntimeSnapshot>,
    },
    /// PR8: policy caps + PR6 metrics + PR7 runtime in one consistent snapshot.
    GetControllerObservabilitySnapshot {
        resp: Sender<ControllerObservabilitySnapshot>,
    },

    // ── Chat operations ──────────────────────────────────────────────
    /// Start a new chat session with full message history.
    StartChat {
        chat_id: String,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        /// Reasoning-channel policy for this session. Pinned at
        /// session start; `append_messages` (continuation) reuses
        /// whatever mode was chosen here. `Auto` preserves the
        /// chat-template default.
        thinking: crate::generation::ThinkingMode,
        /// Canonical id of the model this request targets (the locally
        /// selected/loaded model). Threaded into the host's router as
        /// `required_model` (peer match) and used to resolve the model's
        /// footprint by id from the catalog for the fit gate. `None` ⇒
        /// legacy capability-only routing. See
        /// the host's model-footprint resolver.
        model_id: Option<String>,
        /// Whole-model on-disk byte footprint for this request, resolved by the
        /// send site (which can reach a model catalog)
        /// via the host's model-footprint resolver. This
        /// is the precise size for the fit gate across **all** model kinds
        /// — catalog GGUF, local file, and directory bundle (MLX/ONNX), whose
        /// summed size only the store-bearing resolver can produce. Threaded
        /// through to the host's router as the caller-supplied `model_size_bytes`,
        /// which the host's dispatcher
        /// prefers over its sync catalog/local-loaded fallbacks. `None` ⇒ the
        /// seam falls back to catalog-by-id or the local-loaded footprint.
        model_size_bytes: Option<u64>,
        /// Tools for template rendering + the parser's name gate.
        /// `(tools, tool_prompt)`; `None` = no tool calling.
        tools: Option<(Vec<crate::types::message::ToolSpec>, String)>,
        tx: SyncSender<ControllerEvent>,
    },
    /// Continue an existing chat session with newly appended messages.
    ContinueChat {
        chat_id: String,
        new_messages: Vec<Message>,
        gen_spec: GenSpec,
        /// Canonical model id for remote routing (see
        /// [`ControllerCmd::StartChat::model_id`]).
        model_id: Option<String>,
        /// Store-resolved whole-model footprint (see
        /// [`ControllerCmd::StartChat::model_size_bytes`]).
        model_size_bytes: Option<u64>,
        tx: SyncSender<ControllerEvent>,
    },
    /// Abort and remove a chat session.
    StopChat { chat_id: String },
    /// Pause token generation for a chat (session stays in memory).
    PauseChat { chat_id: String },
    /// Resume a paused chat session.
    ResumeChat { chat_id: String },

    // ── System inference ─────────────────────────────────────────────
    /// Fire-and-forget ephemeral inference for system-level decisions.
    /// Prompt in, tokens streamed back, session auto-cleaned.
    SystemInfer {
        task: SystemTask,
        chat_id: String,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        /// Reasoning-channel policy for the ephemeral session. `Auto`
        /// (the construction default) preserves the model chat-template's
        /// own behaviour; `Off` forces a direct answer on thinking-trained
        /// models (e.g. DiffusionGemma's `enable_thinking=false` empty-thought
        /// prefill) so the reply is just the answer, no scaffold.
        thinking: crate::generation::ThinkingMode,
        /// Canonical model id for remote routing (see
        /// [`ControllerCmd::StartChat::model_id`]). Internal system-inference
        /// callers pass `None` (they ride whatever model the controller has
        /// loaded); the production chat path threads the selected model id.
        model_id: Option<String>,
        /// Store-resolved whole-model footprint (see
        /// [`ControllerCmd::StartChat::model_size_bytes`]). Internal
        /// system-inference callers pass `None` (they ride the loaded model).
        model_size_bytes: Option<u64>,
        required_node: Option<String>,
        tx: SyncSender<ControllerEvent>,
    },

    // ── Utility ──────────────────────────────────────────────────────
    /// Generate embedding vectors for a batch of text inputs.
    GenerateEmbeddings {
        inputs: Vec<String>,
        resp: Sender<Result<Vec<Vec<f32>>, String>>,
    },
    /// Pre-load weights for a model directory in a background thread so the
    /// next `LoadModel` for the same path skips synchronous disk I/O.
    /// Fire-and-forget — no response channel.
    WarmModel { model_dir: PathBuf },
    /// Shut down the controller loop and clean up all sessions.
    Shutdown,
}

/// Events emitted by the controller back to command callers during generation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ControllerEvent {
    /// A newly generated token fragment.
    Token(String),
    MediaBoundary(crate::generation::MediaBoundary),
    /// A structured tool-call extracted by the cross-backend tool-call
    /// parser (see `gen2/backend/common/tool_calls.rs`). The payload's
    /// `arguments` field is raw JSON text; downstream consumers validate
    /// against their own tool schema.
    ToolCall(crate::generation::ToolCall),
    /// End-of-sequence reached (generation complete).
    Eos,
    /// Generation was stopped by user request.
    Stopped,
    /// An error occurred during generation. Carries the error code for
    /// downstream routing (auto-retry, navigate-to-settings, etc.).
    Error {
        code: String,
        message: String,
    },
    /// Final execution statistics for the completed generation.
    FinalStats(ExecutionStats),
    /// Context was truncated — N old messages dropped to fit context window.
    ContextTruncated(usize),
    /// Context was compacted — N old messages replaced by a summary.
    ContextCompacted {
        compacted: usize,
        strategy: String,
    },
}

#[derive(Clone)]
pub struct ControllerHandle {
    tx: Sender<ControllerCmd>,
    config: ControllerConfig,
}

impl ControllerHandle {
    pub fn send(&self, cmd: ControllerCmd) -> Result<(), String> {
        self.tx.send(cmd).map_err(|e| e.to_string())
    }

    /// The controller configuration. Use `config().event_channel_capacity`
    /// when creating event channels to match the controller's policy.
    pub fn config(&self) -> &ControllerConfig {
        &self.config
    }

    /// Blocking read of [`ControllerMetricsSnapshot`] from the controller thread.
    pub fn get_controller_metrics(&self) -> Result<ControllerMetricsSnapshot, String> {
        let (tx, rx) = channel();
        self.send(ControllerCmd::GetControllerMetrics { resp: tx })?;
        rx.recv().map_err(|e| e.to_string())
    }

    /// Blocking read of active runtime rows (sorted `chat_id`, no message payloads).
    pub fn get_controller_runtime_snapshot(&self) -> Result<ControllerRuntimeSnapshot, String> {
        let (tx, rx) = channel();
        self.send(ControllerCmd::GetControllerRuntimeSnapshot { resp: tx })?;
        rx.recv().map_err(|e| e.to_string())
    }

    /// Whole-model on-disk byte size of the currently-loaded primary LLM, or
    /// `None` when no model is loaded (or the model is a directory bundle).
    ///
    /// Issues the runtime-snapshot command and waits **with a short timeout**
    /// for the reply — one in-process round-trip to the controller thread.
    /// Sources the VRAM/RAM fit gate from the local device's loaded
    /// model (Part A of VRAM-aware routing); see
    /// the host's route resolver.
    ///
    /// The timeout (not the unbounded `get_controller_runtime_snapshot`) is
    /// deliberate: this runs on the dispatch hot path, where a wedged or
    /// not-yet-servicing controller thread must NOT be able to block the
    /// dispatcher indefinitely. The reply is a tiny struct the controller can
    /// build without I/O (the size was stat'd once at `LoadModel`), so the real
    /// round-trip is sub-millisecond; the budget is generous. On send error,
    /// timeout, or disconnect we return `None` — honest "size unknown" →
    /// the fit gate degrades to legacy capability-only routing rather than
    /// stalling or surfacing a transport error.
    pub fn loaded_model_file_bytes(&self) -> Option<u64> {
        /// Hot-path budget for the controller round-trip. Sub-ms in practice;
        /// this bounds worst-case dispatch latency if the controller is wedged.
        const QUERY_TIMEOUT: Duration = Duration::from_millis(250);
        let (tx, rx) = channel();
        self.send(ControllerCmd::GetControllerRuntimeSnapshot { resp: tx })
            .ok()?;
        rx.recv_timeout(QUERY_TIMEOUT)
            .ok()
            .and_then(|snap| snap.loaded_model_file_bytes)
    }

    /// Blocking read of unified observability (policy + metrics + runtime).
    pub fn get_controller_observability_snapshot(
        &self,
    ) -> Result<ControllerObservabilitySnapshot, String> {
        let (tx, rx) = channel();
        self.send(ControllerCmd::GetControllerObservabilitySnapshot { resp: tx })?;
        rx.recv().map_err(|e| e.to_string())
    }

    /// Fire-and-forget: pre-load weights for `model_dir` in a background thread.
    /// The next `load_model` call for the same path will skip synchronous I/O.
    pub fn warm_model(&self, model_dir: PathBuf) {
        let _ = self.tx.send(ControllerCmd::WarmModel { model_dir });
    }

    /// Create a `ControllerHandle` from a raw sender. Intended for testing
    /// in downstream crates.
    #[doc(hidden)]
    pub fn new_for_test(tx: Sender<ControllerCmd>) -> Self {
        Self {
            tx,
            config: ControllerConfig::default(),
        }
    }
}

/// A controller running somewhere other than this process.
///
/// gen2 dispatches [`ControllerCmd`]s and reads [`ControllerEvent`]s back. It
/// has no opinion about where that happens: another machine on the LAN, a
/// worker process, a socket, a test double. Implement this over your transport
/// and wrap it with [`InferenceHandle::remote`], and everything built on
/// `InferenceHandle` works unchanged.
///
/// Events come back the way they do locally, over the `tx` channel inside the
/// command you were handed. `send` returns as soon as the command is on its
/// way; a failure to *deliver* is the error, and a failure during generation
/// arrives as [`ControllerEvent::Error`].
pub trait RemoteDispatch: Send + Sync + 'static {
    /// Deliver one command. The `Err` describes why it could not be sent.
    fn send(&self, cmd: ControllerCmd) -> Result<(), String>;

    /// What this dispatch calls itself, for traces and
    /// [`InferenceHandle::placement`]. Defaults to `"remote"`.
    fn label(&self) -> &str {
        "remote"
    }

    /// Policy this dispatch runs under. Defaults to
    /// [`ControllerConfig::default`], which is what a caller who has no
    /// particular opinion should leave it as.
    fn config(&self) -> &ControllerConfig {
        &DEFAULT_REMOTE_CONFIG
    }

    /// Hint that a model is about to be needed. Fire-and-forget; the default
    /// ignores it.
    fn warm_model(&self, model_dir: PathBuf) {
        let _ = model_dir;
    }
}

/// Default policy for a [`RemoteDispatch`] that does not supply its own.
static DEFAULT_REMOTE_CONFIG: std::sync::LazyLock<ControllerConfig> =
    std::sync::LazyLock::new(ControllerConfig::default);

/// Where the work sent through an [`InferenceHandle`] actually runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement<'a> {
    /// This process, on this machine.
    Local,
    /// Behind a [`RemoteDispatch`], which calls itself this.
    Remote(&'a str),
}

/// A controller to talk to, local or not.
///
/// Everything above this layer takes an `InferenceHandle` and stops caring
/// where inference runs.
///
/// ```
/// # use gen2::{ControllerCmd, InferenceHandle, Placement, RemoteDispatch};
/// struct OverTheWire;
/// impl RemoteDispatch for OverTheWire {
///     fn send(&self, _cmd: ControllerCmd) -> Result<(), String> { Ok(()) }
///     fn label(&self) -> &str { "workshop-mac" }
/// }
///
/// let handle = InferenceHandle::remote(OverTheWire);
/// assert_eq!(handle.placement(), Placement::Remote("workshop-mac"));
/// ```
#[derive(Clone)]
pub enum InferenceHandle {
    /// A controller loop on this machine.
    Local(ControllerHandle),
    /// A controller reached through a host-supplied transport.
    Remote(std::sync::Arc<dyn RemoteDispatch>),
}

impl std::fmt::Debug for InferenceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("InferenceHandle")
            .field(&self.placement())
            .finish()
    }
}

impl InferenceHandle {
    /// Wrap a host transport as a handle.
    pub fn remote(dispatch: impl RemoteDispatch) -> Self {
        Self::Remote(std::sync::Arc::new(dispatch))
    }

    /// Send one command to whichever controller this handle points at.
    pub fn send(&self, cmd: ControllerCmd) -> Result<(), String> {
        match self {
            Self::Local(h) => h.send(cmd),
            Self::Remote(h) => h.send(cmd),
        }
    }

    /// Where work sent through this handle runs.
    ///
    /// A host that reports provenance to its user reads this. gen2 states the
    /// fact and stops there: what a given remote *means* — another of your
    /// machines, a rented box, someone else's cloud — is the host's to know and
    /// the host's to say.
    pub fn placement(&self) -> Placement<'_> {
        match self {
            Self::Local(_) => Placement::Local,
            Self::Remote(h) => Placement::Remote(h.label()),
        }
    }

    /// The policy this handle's controller runs under.
    pub fn config(&self) -> &ControllerConfig {
        match self {
            Self::Local(h) => h.config(),
            Self::Remote(h) => h.config(),
        }
    }

    /// Fire-and-forget hint that a model is about to be needed.
    pub fn warm_model(&self, model_dir: PathBuf) {
        match self {
            Self::Local(h) => h.warm_model(model_dir),
            Self::Remote(h) => h.warm_model(model_dir),
        }
    }

    /// Fire a system-level inference task using the controller's configured GenSpec.
    ///
    /// Prompt in, full text out, session auto-cleaned. This is the primary
    /// entry point for internal LLM decision-making.
    pub async fn system_infer(
        &self,
        task: SystemTask,
        chat_id: impl Into<String>,
        messages: Vec<Message>,
    ) -> Result<String, crate::engine::ExecError> {
        let gen_spec = self.config().system_task_spec(&task);
        self.system_infer_with(task, chat_id, messages, gen_spec)
            .await
    }

    /// Single-prompt convenience: wraps `prompt` as a user message, generates
    /// a unique session ID, and runs the task with default GenSpec.
    pub async fn system_prompt(
        &self,
        task: SystemTask,
        prompt: impl Into<String>,
    ) -> Result<String, crate::engine::ExecError> {
        let messages = vec![Message::user(prompt)];
        let session_id = task.session_id();
        self.system_infer(task, session_id, messages).await
    }

    /// Fire a system-level inference task with a custom GenSpec.
    pub async fn system_infer_with(
        &self,
        task: SystemTask,
        chat_id: impl Into<String>,
        messages: Vec<Message>,
        gen_spec: GenSpec,
    ) -> Result<String, crate::engine::ExecError> {
        self.system_infer_with_route(task, chat_id, messages, gen_spec, None, None, None)
            .await
    }

    /// As [`Self::system_infer_with`], but carries routing hints for a
    /// [`RemoteDispatch`]: which model the target must already have, how big it
    /// is, and which node to insist on. Never loads or transfers model files —
    /// it only lets the target pick an exact-model route it already has. The
    /// local arm ignores all three.
    #[allow(clippy::too_many_arguments)]
    pub async fn system_infer_with_route(
        &self,
        task: SystemTask,
        chat_id: impl Into<String>,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        model_id: Option<String>,
        model_size_bytes: Option<u64>,
        required_node: Option<String>,
    ) -> Result<String, crate::engine::ExecError> {
        use crate::engine::ExecError;

        let capacity = self.config().event_channel_capacity;
        let (tx, rx) = std::sync::mpsc::sync_channel::<ControllerEvent>(capacity);
        let cmd = ControllerCmd::SystemInfer {
            task,
            chat_id: chat_id.into(),
            messages,
            gen_spec,
            thinking: crate::generation::ThinkingMode::default(),
            model_id,
            model_size_bytes,
            required_node,
            tx,
        };
        self.send(cmd)
            .map_err(|e| ExecError::Generation(e.to_string()))?;

        tokio::task::spawn_blocking(move || {
            let mut out = String::with_capacity(512);
            while let Ok(ev) = rx.recv() {
                match ev {
                    ControllerEvent::Token(t) => out.push_str(&t),
                    ControllerEvent::Eos | ControllerEvent::Stopped => break,
                    ControllerEvent::Error { code, message } => {
                        return Err(ExecError::Coded { code, message });
                    }
                    _ => {}
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| ExecError::Generation(e.to_string()))?
    }

    /// Streaming variant of [`Self::system_infer`]: invokes `on_token` for each
    /// generated token as it arrives (instead of only returning the joined
    /// text), and still returns the full text. Use it to show a background
    /// task's output as it lands rather than after it finishes.
    pub async fn system_infer_streaming<F>(
        &self,
        task: SystemTask,
        chat_id: impl Into<String>,
        messages: Vec<Message>,
        mut on_token: F,
    ) -> Result<String, crate::engine::ExecError>
    where
        F: FnMut(&str) + Send + 'static,
    {
        use crate::engine::ExecError;

        let gen_spec = self.config().system_task_spec(&task);
        let capacity = self.config().event_channel_capacity;
        let (tx, rx) = std::sync::mpsc::sync_channel::<ControllerEvent>(capacity);
        let cmd = ControllerCmd::SystemInfer {
            task,
            chat_id: chat_id.into(),
            messages,
            gen_spec,
            thinking: crate::generation::ThinkingMode::default(),
            // Internal system inference — no per-request model fence.
            model_id: None,
            model_size_bytes: None,
            required_node: None,
            tx,
        };
        self.send(cmd)
            .map_err(|e| ExecError::Generation(e.to_string()))?;

        tokio::task::spawn_blocking(move || {
            let mut out = String::with_capacity(512);
            while let Ok(ev) = rx.recv() {
                match ev {
                    ControllerEvent::Token(t) => {
                        on_token(&t);
                        out.push_str(&t);
                    }
                    ControllerEvent::Eos | ControllerEvent::Stopped => break,
                    ControllerEvent::Error { code, message } => {
                        return Err(ExecError::Coded { code, message });
                    }
                    _ => {}
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| ExecError::Generation(e.to_string()))?
    }

    /// Streaming system inference with a caller-supplied [`GenSpec`], instead
    /// of the task's default. Same contract as
    /// [`Self::system_infer_streaming`] — the difference is that `gen_spec`
    /// overrides the per-task sampling defaults, which is what a host with its
    /// own settings surface needs so those settings actually change output.
    pub async fn system_infer_streaming_with<F>(
        &self,
        task: SystemTask,
        chat_id: impl Into<String>,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        thinking: crate::generation::ThinkingMode,
        mut on_token: F,
    ) -> Result<String, crate::engine::ExecError>
    where
        F: FnMut(&str) + Send + 'static,
    {
        use crate::engine::ExecError;

        let capacity = self.config().event_channel_capacity;
        let (tx, rx) = std::sync::mpsc::sync_channel::<ControllerEvent>(capacity);
        let cmd = ControllerCmd::SystemInfer {
            task,
            chat_id: chat_id.into(),
            messages,
            gen_spec,
            thinking,
            // Internal system inference — no per-request model fence.
            model_id: None,
            model_size_bytes: None,
            required_node: None,
            tx,
        };
        self.send(cmd)
            .map_err(|e| ExecError::Generation(e.to_string()))?;

        tokio::task::spawn_blocking(move || {
            let mut out = String::with_capacity(512);
            while let Ok(ev) = rx.recv() {
                match ev {
                    ControllerEvent::Token(t) => {
                        on_token(&t);
                        out.push_str(&t);
                    }
                    ControllerEvent::Eos | ControllerEvent::Stopped => break,
                    ControllerEvent::Error { code, message } => {
                        return Err(ExecError::Coded { code, message });
                    }
                    _ => {}
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| ExecError::Generation(e.to_string()))?
    }

    /// Blocking read of delivery / termination counters from the active backend.
    ///
    /// For the local controller, mirrors [`ControllerHandle::get_controller_metrics`].
    /// For a remote route, forwards the same query the dispatch answers for status.
    pub fn get_controller_metrics(&self) -> Result<ControllerMetricsSnapshot, String> {
        let (tx, rx) = channel();
        self.send(ControllerCmd::GetControllerMetrics { resp: tx })?;
        rx.recv().map_err(|e| e.to_string())
    }

    /// Blocking read of active inference runtimes from the active backend.
    pub fn get_controller_runtime_snapshot(&self) -> Result<ControllerRuntimeSnapshot, String> {
        let (tx, rx) = channel();
        self.send(ControllerCmd::GetControllerRuntimeSnapshot { resp: tx })?;
        rx.recv().map_err(|e| e.to_string())
    }

    /// Blocking read of unified observability from the active backend.
    pub fn get_controller_observability_snapshot(
        &self,
    ) -> Result<ControllerObservabilitySnapshot, String> {
        let (tx, rx) = channel();
        self.send(ControllerCmd::GetControllerObservabilitySnapshot { resp: tx })?;
        rx.recv().map_err(|e| e.to_string())
    }
}

/// Start the inference controller with explicit configuration.
pub fn start_controller_with_config(config: ControllerConfig) -> ControllerHandle {
    start_controller_joinable(config).0
}

/// As [`start_controller_with_config`], but also hands back the loop's
/// `JoinHandle`.
///
/// The loop owns the backend's native context. A process that exits while it is
/// still running tears down ggml's statics underneath it and aborts, so anything
/// that wants a clean shutdown has to be able to *wait* for the loop to finish,
/// not just ask it to stop. `Engine` uses this to join on drop.
pub(crate) fn start_controller_joinable(
    config: ControllerConfig,
) -> (ControllerHandle, thread::JoinHandle<()>) {
    tracing::debug!(?config, "starting inference controller");
    let (tx, rx): (Sender<ControllerCmd>, Receiver<ControllerCmd>) = channel();
    let handle_config = config.clone();
    let join = thread::spawn(move || run_loop(rx, config));
    (
        ControllerHandle {
            tx,
            config: handle_config,
        },
        join,
    )
}

/// Start the inference controller with a custom max-active-chats limit.
/// All other policy values use defaults.
pub fn start_controller_with_limit(max_active: usize) -> ControllerHandle {
    start_controller_with_config(ControllerConfig {
        max_active_chats: max_active,
        ..ControllerConfig::default()
    })
}

/// Start the inference controller with default configuration.
pub fn start_controller() -> ControllerHandle {
    start_controller_with_config(ControllerConfig::default())
}

/// Start a controller over an engine the caller builds.
///
/// `build` runs on the controller's own thread, because [`Backend`] is not
/// `Send` — backends hold non-thread-safe FFI state, which is exactly why the
/// loop owns one and nobody else touches it. A test that wants to inspect the
/// backend keeps a handle to the `Send + Sync` half (see
/// [`Script`](crate::test_support::Script)) rather than to the backend itself.
#[cfg(test)]
pub(crate) fn start_controller_with_engine(
    config: ControllerConfig,
    build: Box<dyn FnOnce() -> crate::backend::Engine + Send>,
) -> (ControllerHandle, thread::JoinHandle<()>) {
    let (tx, rx): (Sender<ControllerCmd>, Receiver<ControllerCmd>) = channel();
    let handle_config = config.clone();
    let join = thread::spawn(move || run_loop_with_engine(rx, config, build()));
    (
        ControllerHandle {
            tx,
            config: handle_config,
        },
        join,
    )
}

/// Default event channel capacity — matches `ControllerConfig::default().event_channel_capacity`.
///
/// Kept as a constant for backward compatibility with external crates and tests
/// that create their own channels. Prefer `handle.config().event_channel_capacity`
/// when a `ControllerHandle` or `InferenceHandle` is available.
pub const EVENT_CHANNEL_CAPACITY: usize = 512;

/// One active session + outbound events + explicit run state.
pub(super) struct ChatRuntime {
    pub(super) session: Arc<crate::session_rt::Session>,
    pub(super) tx: SyncSender<ControllerEvent>,
    pub(super) workload: WorkloadKind,
    pub(super) last_used: Instant,
    /// Last spec used for `Session::pull` (Resume / Continue paths).
    pub(super) last_gen_spec: GenSpec,
    pub(super) state: ChatRunState,
    /// Generic session health — aggregates backend poison signal + decode
    /// panics + consecutive errors. Drives `FailureReason::SessionPoisoned`.
    pub(super) health: crate::backend::SessionHealth,
}

impl ChatRuntime {
    /// `true` when this runtime should receive scheduler ticks.
    pub(super) fn should_tick(&self) -> bool {
        self.state.is_generating()
    }
}

fn run_loop(rx: Receiver<ControllerCmd>, config: ControllerConfig) {
    run_loop_with_engine(rx, config, crate::backend::Engine::new())
}

fn run_loop_with_engine(
    rx: Receiver<ControllerCmd>,
    config: ControllerConfig,
    engine: crate::backend::Engine,
) {
    let tick_busy = Duration::from_millis(0);
    let mut state = ControllerState::with_engine(engine, config);

    'outer: loop {
        if state.chats.is_empty() {
            match rx.recv() {
                Err(_) => break,
                Ok(ControllerCmd::Shutdown) => break,
                Ok(cmd) => {
                    if let ControlFlow::Break = commands::dispatch_cmd(cmd, &mut state) {
                        break;
                    }
                }
            }
            continue;
        }

        match rx.recv_timeout(state.tick_idle()) {
            Ok(ControllerCmd::Shutdown) => break 'outer,
            Ok(cmd) => {
                if let ControlFlow::Break = commands::dispatch_cmd(cmd, &mut state) {
                    break 'outer;
                }
            }
            Err(_timeout) => {}
        }

        commands::tick_active_chats(&mut state, tick_busy);
    }

    for (_id, chat) in state.chats.drain() {
        lifecycle::terminate_runtime_owned(
            &state.engine,
            chat,
            lifecycle::RuntimeOutcome::Completed(CompletionReason::ControllerShutdown),
            state.metrics.as_ref(),
        );
    }
}

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests_by_invariant;

pub(super) enum ControlFlow {
    Continue,
    Break,
}

#[cfg(test)]
mod tests {
    use super::lifecycle::{RuntimeOutcome, terminate_runtime};
    use super::observability::EmitResult;
    use super::*;
    use crate::backend::Engine;
    use std::collections::HashMap;
    use std::sync::mpsc::sync_channel;

    #[test]
    fn bounded_channel_drops_on_full() {
        let (tx, rx) = sync_channel::<ControllerEvent>(EVENT_CHANNEL_CAPACITY);

        // Fill the channel to capacity.
        for _ in 0..EVENT_CHANNEL_CAPACITY {
            tx.try_send(ControllerEvent::Stopped)
                .expect("channel should accept up to capacity");
        }

        // One more event should return Full (not block).
        let result = match tx.try_send(ControllerEvent::Stopped) {
            Ok(()) => EmitResult::Sent,
            Err(std::sync::mpsc::TrySendError::Full(_)) => EmitResult::Full,
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => EmitResult::Disconnected,
        };
        assert!(
            matches!(result, EmitResult::Full),
            "expected EmitResult::Full when channel is at capacity"
        );

        // Verify the receiver still has exactly EVENT_CHANNEL_CAPACITY events.
        let mut count = 0;
        while rx.try_recv().is_ok() {
            count += 1;
        }
        assert_eq!(count, EVENT_CHANNEL_CAPACITY);
    }

    #[test]
    fn try_emit_disconnected() {
        let (tx, rx) = sync_channel::<ControllerEvent>(EVENT_CHANNEL_CAPACITY);

        // Drop the receiver so the channel is disconnected.
        drop(rx);

        let result = match tx.try_send(ControllerEvent::Stopped) {
            Ok(()) => EmitResult::Sent,
            Err(std::sync::mpsc::TrySendError::Full(_)) => EmitResult::Full,
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => EmitResult::Disconnected,
        };
        assert!(
            matches!(result, EmitResult::Disconnected),
            "expected EmitResult::Disconnected when receiver is dropped"
        );
    }

    #[test]
    fn chat_run_state_helpers_classify_lifecycle() {
        let spec = GenSpec::default();
        assert!(!ChatRunState::Idle.is_generating());
        assert!(!ChatRunState::Idle.is_paused());
        assert!(!ChatRunState::Idle.is_terminal());

        let paused = ChatRunState::Paused {
            last_gen_spec: spec.clone(),
        };
        assert!(!paused.is_generating());
        assert!(paused.is_paused());
        assert!(!paused.is_terminal());

        assert!(ChatRunState::Completed(CompletionReason::Eos).is_terminal());
        assert!(ChatRunState::Failed(FailureReason::Timeout).is_terminal());
    }

    #[test]
    fn workload_kind_detects_system_tasks() {
        assert!(!WorkloadKind::PrimaryChat.is_system_task());
        assert!(WorkloadKind::SystemTask(SystemTask::Title).is_system_task());
    }

    /// Every `CompletionReason` / `FailureReason` variant is constructible;
    /// later PRs wire them into `terminate_runtime` without adding new variants blindly.
    #[test]
    fn terminal_reason_variants_are_exhaustive_for_refactors() {
        let reasons = [
            CompletionReason::Eos,
            CompletionReason::StoppedByUser,
            CompletionReason::Evicted,
            CompletionReason::ReceiverDropped,
            CompletionReason::ModelReloaded,
            CompletionReason::ControllerShutdown,
        ];
        for r in reasons {
            let ChatRunState::Completed(inner) = ChatRunState::Completed(r) else {
                unreachable!();
            };
            assert_eq!(inner, r);
        }
        let failures = [
            FailureReason::Timeout,
            FailureReason::GenerationError,
            FailureReason::SessionPoisoned,
            FailureReason::StartSessionFailed,
            FailureReason::PullerCreateFailed,
        ];
        for f in failures {
            let ChatRunState::Failed(inner) = ChatRunState::Failed(f) else {
                unreachable!();
            };
            assert_eq!(inner, f);
        }
    }

    /// `terminate_runtime` on a missing id must not panic (StopChat unknown id relies on this).
    #[test]
    fn terminate_runtime_missing_chat_is_no_op() {
        let engine = Engine::new();
        let mut chats: HashMap<String, ChatRuntime> = HashMap::new();
        let metrics = metrics::ControllerMetrics::default();
        terminate_runtime(
            &engine,
            &mut chats,
            "does-not-exist",
            RuntimeOutcome::Completed(CompletionReason::StoppedByUser),
            &metrics,
        );
        assert!(chats.is_empty());
    }

    /// Shutdown runs `terminate_runtime_owned` for each remaining runtime — must not panic.
    #[test]
    fn shutdown_drains_active_chat_without_panic() {
        let handle = start_controller();
        let (tx, _rx) = sync_channel::<ControllerEvent>(EVENT_CHANNEL_CAPACITY);
        handle
            .send(ControllerCmd::StartChat {
                chat_id: "drain-me".into(),
                messages: vec![],
                gen_spec: GenSpec::default(),
                thinking: Default::default(),
                model_id: None,
                model_size_bytes: None,
                tools: None,
                tx,
            })
            .expect("start chat");
        handle.send(ControllerCmd::Shutdown).expect("shutdown");
        std::thread::sleep(std::time::Duration::from_millis(100));
        let (resp_tx, _resp_rx) = std::sync::mpsc::channel();
        assert!(
            handle
                .send(ControllerCmd::IsModelLoaded { resp: resp_tx })
                .is_err(),
            "controller thread should have exited after shutdown"
        );
    }

    #[test]
    fn try_emit_sent() {
        let (tx, rx) = sync_channel::<ControllerEvent>(EVENT_CHANNEL_CAPACITY);

        let result = match tx.try_send(ControllerEvent::Stopped) {
            Ok(()) => EmitResult::Sent,
            Err(std::sync::mpsc::TrySendError::Full(_)) => EmitResult::Full,
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => EmitResult::Disconnected,
        };
        assert!(
            matches!(result, EmitResult::Sent),
            "expected EmitResult::Sent for a successful send"
        );

        // Verify the receiver got the event.
        let event = rx.try_recv().expect("receiver should have one event");
        assert!(
            matches!(event, ControllerEvent::Stopped),
            "expected ControllerEvent::Stopped"
        );

        // No additional events should be present.
        assert!(
            rx.try_recv().is_err(),
            "receiver should be empty after consuming the single event"
        );
    }

    #[test]
    fn is_model_loaded_before_load() {
        let handle = start_controller();
        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsModelLoaded { resp: tx })
            .expect("send should succeed");
        let loaded = rx.recv().expect("should receive a response");
        assert!(!loaded, "no model should be loaded on a fresh controller");
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    #[test]
    fn get_controller_metrics_via_handle_returns_snapshot() {
        let handle = start_controller();
        let snap = handle.get_controller_metrics().expect("controller metrics");
        assert_eq!(snap, ControllerMetricsSnapshot::default());
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    #[test]
    fn get_controller_runtime_snapshot_via_handle_empty() {
        let handle = start_controller();
        let snap = handle
            .get_controller_runtime_snapshot()
            .expect("runtime snapshot");
        assert!(snap.chats.is_empty());
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    #[test]
    fn get_controller_observability_snapshot_via_handle_matches_defaults() {
        let handle = start_controller();
        let snap = handle
            .get_controller_observability_snapshot()
            .expect("observability snapshot");
        assert_eq!(snap.metrics, ControllerMetricsSnapshot::default());
        assert!(snap.runtime.chats.is_empty());
        assert_eq!(snap.policy.active_chats, 0);
        assert_eq!(
            snap.policy.max_active_chats,
            ControllerConfig::default().max_active_chats
        );
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    #[test]
    fn is_embedder_loaded_before_load() {
        let handle = start_controller();
        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsEmbedderLoaded { resp: tx })
            .expect("send should succeed");
        let loaded = rx.recv().expect("should receive a response");
        assert!(
            !loaded,
            "no embedder should be loaded on a fresh controller"
        );
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    #[test]
    fn is_chat_loaded_unknown_id() {
        let handle = start_controller();
        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsChatLoaded {
                chat_id: "nonexistent".to_string(),
                resp: tx,
            })
            .expect("send should succeed");
        let loaded = rx.recv().expect("should receive a response");
        assert!(!loaded, "unknown chat_id should not be loaded");
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    #[test]
    fn shutdown_then_send_fails() {
        let handle = start_controller();
        handle
            .send(ControllerCmd::Shutdown)
            .expect("shutdown send should succeed");
        // Give the controller thread a moment to process shutdown.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let (tx, _rx) = std::sync::mpsc::channel();
        let result = handle.send(ControllerCmd::IsModelLoaded { resp: tx });
        assert!(
            result.is_err(),
            "sending after shutdown should fail because the channel is closed"
        );
    }

    #[test]
    fn stop_chat_unknown_id_no_panic() {
        let handle = start_controller();
        // Stopping a chat that was never started should not panic or kill the controller.
        handle
            .send(ControllerCmd::StopChat {
                chat_id: "unknown-chat-id".to_string(),
            })
            .expect("stop unknown chat should succeed");
        // Verify the controller is still alive by querying it.
        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsModelLoaded { resp: tx })
            .expect("send should succeed after StopChat on unknown id");
        let loaded = rx.recv().expect("controller should still respond");
        assert!(!loaded, "no model should be loaded");
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    #[test]
    fn controller_handle_clone_works() {
        let handle = start_controller();
        let cloned = handle.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        cloned
            .send(ControllerCmd::IsModelLoaded { resp: tx })
            .expect("send via cloned handle should succeed");
        let loaded = rx
            .recv()
            .expect("should receive a response via cloned handle");
        assert!(!loaded, "no model should be loaded");
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    // ── Step 3a: Controller command dispatch tests ────────────────────

    /// Create a ControllerHandle, send IsModelLoaded, assert returns false.
    #[test]
    fn status_queries_before_load() {
        let handle = start_controller();
        // IsModelLoaded
        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsModelLoaded { resp: tx })
            .expect("send should succeed");
        assert!(
            !rx.recv().unwrap(),
            "model should not be loaded on fresh controller"
        );
        // IsEmbedderLoaded
        let (tx2, rx2) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsEmbedderLoaded { resp: tx2 })
            .expect("send should succeed");
        assert!(
            !rx2.recv().unwrap(),
            "embedder should not be loaded on fresh controller"
        );
        // IsMmprojLoaded
        let (tx3, rx3) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsMmprojLoaded { resp: tx3 })
            .expect("send should succeed");
        assert!(
            !rx3.recv().unwrap(),
            "mmproj should not be loaded on fresh controller"
        );
        // IsChatLoaded
        let (tx4, rx4) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsChatLoaded {
                chat_id: "any-id".to_string(),
                resp: tx4,
            })
            .expect("send should succeed");
        assert!(
            !rx4.recv().unwrap(),
            "no chat should be loaded on fresh controller"
        );
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    /// Spawn controller thread, send Shutdown, verify thread exits cleanly
    /// (subsequent sends fail because the channel is closed).
    #[test]
    fn shutdown_command_exits_loop() {
        let handle = start_controller();
        handle
            .send(ControllerCmd::Shutdown)
            .expect("shutdown send should succeed");
        // Give the controller thread time to process shutdown and exit.
        std::thread::sleep(std::time::Duration::from_millis(100));
        // The controller thread has exited, so the receiving end of the
        // channel is dropped. Any subsequent send should fail.
        let (tx, _rx) = std::sync::mpsc::channel();
        let result = handle.send(ControllerCmd::IsModelLoaded { resp: tx });
        assert!(result.is_err(), "channel should be closed after shutdown");
    }

    /// Send StopChat for nonexistent chat_id "999" — no panic, controller
    /// stays alive and responsive.
    #[test]
    fn stop_chat_without_active_chat() {
        let handle = start_controller();
        handle
            .send(ControllerCmd::StopChat {
                chat_id: "999".to_string(),
            })
            .expect("stop nonexistent chat should not panic");
        // Verify controller is still alive.
        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsModelLoaded { resp: tx })
            .expect("controller should still be responsive");
        let loaded = rx.recv().expect("should receive response");
        assert!(!loaded);
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    #[test]
    fn apply_settings_without_model() {
        let handle = start_controller();
        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::ApplySettings {
                settings: Settings::default(),
                resp: tx,
            })
            .expect("send should succeed");
        let result = rx.recv().expect("should receive a response");
        // Cold start: `backend::facade::Engine::new` eagerly instantiates a
        // backend ONLY for llamacpp/mlx/onnx — for those, `as_backend()` is
        // Some and `upload_settings` succeeds even with no model file loaded.
        // `backend-external-api` (and the no-backend build) start `Uninit` and
        // instantiate lazily on `load_model`, so `upload_settings` short-circuits
        // to `Err(ModelNotLoaded)` in the facade. Either way the controller must
        // respond without panicking — that is the boundary guarantee. Keep this
        // cfg split in lockstep with `Engine::new`'s feature cascade.
        #[cfg(not(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-onnx"
        )))]
        assert!(
            result.is_err(),
            "applying settings with no eagerly-initialized backend should return Err(ModelNotLoaded)"
        );
        #[cfg(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-onnx"
        ))]
        assert!(
            result.is_ok(),
            "applying default settings on an eagerly-initialized backend should succeed"
        );
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    // ── Chaos: lifecycle race unit tests ─────────────────────────────

    /// Pause → Stop → Resume on the same chat — the resume targets a removed
    /// session and must not panic or corrupt controller state.
    #[test]
    fn pause_then_stop_then_resume_no_panic() {
        let handle = start_controller();
        let chat_id = "psr-unit".to_string();

        let (tx, _rx) = sync_channel::<ControllerEvent>(EVENT_CHANNEL_CAPACITY);
        handle
            .send(ControllerCmd::StartChat {
                chat_id: chat_id.clone(),
                messages: vec![],
                gen_spec: crate::generation::GenSpec::default(),
                thinking: Default::default(),
                model_id: None,
                model_size_bytes: None,
                tools: None,
                tx,
            })
            .expect("start should succeed");
        handle
            .send(ControllerCmd::PauseChat {
                chat_id: chat_id.clone(),
            })
            .expect("pause should succeed");
        handle
            .send(ControllerCmd::StopChat {
                chat_id: chat_id.clone(),
            })
            .expect("stop should succeed");
        handle
            .send(ControllerCmd::ResumeChat { chat_id })
            .expect("resume on removed chat should not fail at send level");

        std::thread::sleep(std::time::Duration::from_millis(50));

        // Controller must still be responsive
        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsModelLoaded { resp: tx })
            .expect("controller should still be alive");
        let _ = rx.recv().expect("should get response");
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    /// The remote seam is generic: a host transport that knows nothing about
    /// gen2's internals is a first-class handle.
    #[test]
    fn a_host_transport_is_a_first_class_handle() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Counting(AtomicUsize);
        impl RemoteDispatch for Counting {
            fn send(&self, _cmd: ControllerCmd) -> Result<(), String> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn label(&self) -> &str {
                "workshop-mac"
            }
        }

        let dispatch = std::sync::Arc::new(Counting(AtomicUsize::new(0)));
        let handle = InferenceHandle::Remote(dispatch.clone());

        assert_eq!(handle.placement(), Placement::Remote("workshop-mac"));
        // A transport with no opinion inherits the default policy rather than
        // being handed the local controller's.
        assert_eq!(
            handle.config().event_channel_capacity,
            ControllerConfig::default().event_channel_capacity
        );

        let (resp, _rx) = std::sync::mpsc::channel();
        handle.send(ControllerCmd::IsModelLoaded { resp }).unwrap();
        handle.warm_model(PathBuf::from("/models/whatever"));
        assert_eq!(
            dispatch.0.load(Ordering::Relaxed),
            1,
            "warm_model defaults to a no-op"
        );
    }

    /// A local handle says so, and says nothing about hardware ownership —
    /// that reading belongs to the host.
    #[test]
    fn a_local_handle_reports_local_placement() {
        let handle = InferenceHandle::Local(start_controller());
        assert_eq!(handle.placement(), Placement::Local);
        let _ = handle.send(ControllerCmd::Shutdown);
    }

    /// A custom task carries its own label into the session id, and gets plain
    /// sampling rather than something tuned for a task gen2 cannot see.
    #[test]
    fn a_custom_system_task_is_labelled_but_untuned() {
        let task = SystemTask::custom("triples");
        assert_eq!(task.label(), "triples");
        assert!(task.session_id().starts_with("triples-"));
        assert_ne!(
            task.default_gen_spec().max_tokens,
            SystemTask::Title.default_gen_spec().max_tokens,
            "a named task keeps its tuning; a custom one must not inherit it"
        );
    }

    /// Flood the controller with 1000 StopChat commands for random UUIDs.
    #[test]
    fn flood_stop_unknown_ids() {
        let handle = start_controller();
        for i in 0..1000 {
            let _ = handle.send(ControllerCmd::StopChat {
                chat_id: format!("unknown-{i}-{}", uuid::Uuid::new_v4()),
            });
        }
        // Give controller time to process the flood
        std::thread::sleep(std::time::Duration::from_millis(200));

        let (tx, rx) = std::sync::mpsc::channel();
        handle
            .send(ControllerCmd::IsModelLoaded { resp: tx })
            .expect("controller should survive 1000 unknown stops");
        let loaded = rx.recv().expect("should receive response");
        assert!(!loaded);
        let _ = handle.send(ControllerCmd::Shutdown);
    }
}
