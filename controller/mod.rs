mod commands;
mod config;
pub(crate) mod metrics;
mod scheduler;
mod state;
mod state_transitions;

pub use config::ControllerConfig;
pub use metrics::ControllerMetricsSnapshot;
pub use state::ControllerState;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel};
use std::thread;
use std::time::{Duration, Instant};

use crate::gen2::engine::{HookEvent, HookListener, LoadRequest};
use crate::gen2::generation::GenSpec;
use crate::gen2::session_rt::SessionSpec;
use crate::gen2::{Engine, ExecutionStats, Settings};
use crate::types::message::Message;

/// System-level inference tasks — ephemeral, fire-and-forget, hidden from users.
///
/// Each variant runs as an ephemeral session on the controller: prompt in,
/// tokens streamed back, session auto-cleaned on completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum SystemTask {
    /// Generate a chat title from conversation history.
    Title,
    /// Generate follow-up suggestions for a chat.
    Suggestions,
    /// Generate a context-compaction summary.
    Compact,
    /// Grounded answer synthesis from evidence chunks.
    Answer,
    /// LLM-powered triple (subject, predicate, object) extraction.
    Triples,
    /// Entity stance extraction from text.
    Stance,
    /// LLM-powered named entity recognition.
    EntityExtract,
    /// Topic cluster labeling from conversation samples.
    TopicLabel,
    /// Intent classification + query rewriting.
    QueryUnderstand,
    /// Evidence contradiction detection.
    Contradiction,
    /// Chat sidebar summary — topic extraction from recent messages.
    Summary,
}

/// Primary user chat vs an internal system inference workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WorkloadKind {
    PrimaryChat,
    SystemTask(SystemTask),
}

/// Why a generation completed successfully from the controller's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CompletionReason {
    Eos,
    StoppedByUser,
    Evicted,
    ReceiverDropped,
    ModelReloaded,
    ControllerShutdown,
}

/// Why a generation failed or was aborted as an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum FailureReason {
    Timeout,
    GenerationError,
    SessionPoisoned,
    StartSessionFailed,
    PullerCreateFailed,
}

/// Terminal classification for [`terminate_runtime`] / [`terminate_runtime_owned`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeOutcome {
    Completed(CompletionReason),
    Failed(FailureReason),
}

/// Explicit lifecycle for a single chat runtime (no implicit bool matrix).
#[allow(clippy::large_enum_variant)]
pub(super) enum ChatRunState {
    /// Session may exist but no generation is active (reserved for future use).
    Idle,
    Generating {
        puller: crate::gen2::generation::TokenPuller,
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

pub(super) fn runtime_outcome_from_state(chat: &ChatRuntime) -> RuntimeOutcome {
    match &chat.state {
        ChatRunState::Completed(r) => RuntimeOutcome::Completed(*r),
        ChatRunState::Failed(f) => RuntimeOutcome::Failed(*f),
        _ => RuntimeOutcome::Completed(CompletionReason::StoppedByUser),
    }
}

/// End engine session bookkeeping for a session id (no `ControllerEvent`).
pub(super) fn deregister_and_end_session(engine: &Engine, sid: u64) {
    engine.hooks().deregister(sid);
    let _ = engine.end_session(sid);
}

/// Stop inference, notify UI, drop hooks, and end the engine session — one path.
fn finalize_runtime_session(
    engine: &Engine,
    mut chat: ChatRuntime,
    metrics: &metrics::ControllerMetrics,
) {
    let sid = chat.session.id();
    chat.session.stop();
    metrics::emit_must_deliver(metrics, &chat.tx, ControllerEvent::Stopped);
    deregister_and_end_session(engine, sid);
}

/// Remove a runtime from the map (if present) and finalize its engine session once.
pub(super) fn terminate_runtime(
    engine: &Engine,
    chats: &mut HashMap<String, ChatRuntime>,
    chat_id: &str,
    outcome: RuntimeOutcome,
    metrics: &metrics::ControllerMetrics,
) {
    if let Some(chat) = chats.remove(chat_id) {
        terminate_runtime_owned(engine, chat, outcome, metrics);
    }
}

/// Finalize an owned runtime: record terminal outcome if not already terminal, then teardown.
pub(super) fn terminate_runtime_owned(
    engine: &Engine,
    mut chat: ChatRuntime,
    outcome: RuntimeOutcome,
    metrics: &metrics::ControllerMetrics,
) {
    if !chat.state.is_terminal() {
        chat.state = match outcome {
            RuntimeOutcome::Completed(r) => ChatRunState::Completed(r),
            RuntimeOutcome::Failed(f) => ChatRunState::Failed(f),
        };
    }
    use std::sync::atomic::Ordering;
    match outcome {
        RuntimeOutcome::Completed(CompletionReason::Evicted) => {
            metrics.evictions.fetch_add(1, Ordering::Relaxed);
        }
        RuntimeOutcome::Completed(CompletionReason::ModelReloaded) => {
            metrics
                .model_reload_terminations
                .fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
    finalize_runtime_session(engine, chat, metrics);
}

impl SystemTask {
    /// Generate a unique session ID for this task type.
    pub fn session_id(&self) -> String {
        format!("sabra-{}-{}", self.suffix(), uuid::Uuid::new_v4())
    }

    /// Suffix used to namespace ephemeral session IDs.
    fn suffix(&self) -> &'static str {
        match self {
            Self::Title => "title",
            Self::Suggestions => "suggestions",
            Self::Compact => "compact",
            Self::Answer => "answer",
            Self::Triples => "triples",
            Self::Stance => "stance",
            Self::EntityExtract => "entities",
            Self::TopicLabel => "topic",
            Self::QueryUnderstand => "query",
            Self::Contradiction => "contradiction",
            Self::Summary => "summary",
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
        resp: Sender<Result<(), String>>,
    },

    // ── Status queries ───────────────────────────────────────────────
    /// Check whether the primary LLM is loaded and ready.
    IsModelLoaded { resp: Sender<bool> },
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

    // ── Chat operations ──────────────────────────────────────────────
    /// Start a new chat session with full message history.
    StartChat {
        chat_id: String,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        tx: SyncSender<ControllerEvent>,
    },
    /// Continue an existing chat session with newly appended messages.
    ContinueChat {
        chat_id: String,
        new_messages: Vec<Message>,
        gen_spec: GenSpec,
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
        tx: SyncSender<ControllerEvent>,
    },

    // ── Utility ──────────────────────────────────────────────────────
    /// Generate embedding vectors for a batch of text inputs.
    GenerateEmbeddings {
        inputs: Vec<String>,
        resp: Sender<Result<Vec<Vec<f32>>, String>>,
    },
    /// Shut down the controller loop and clean up all sessions.
    Shutdown,
}

/// Events emitted by the controller back to command callers during generation.
#[derive(Debug, Clone)]
pub enum ControllerEvent {
    /// A newly generated token fragment.
    Token(String),
    MediaBoundary(crate::gen2::generation::MediaBoundary),
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

    /// Create a `ControllerHandle` from a raw sender. Intended for testing
    /// in downstream crates (pio-daemon SSE tests, etc.).
    #[doc(hidden)]
    pub fn new_for_test(tx: Sender<ControllerCmd>) -> Self {
        Self {
            tx,
            config: ControllerConfig::default(),
        }
    }
}

/// Lazily-initialized default config for remote/flock handles that don't
/// carry their own config. Using `LazyLock` avoids allocation on every call.
#[cfg(any(feature = "p2p-client", feature = "flock"))]
static DEFAULT_CONFIG: std::sync::LazyLock<ControllerConfig> =
    std::sync::LazyLock::new(ControllerConfig::default);

/// Unified handle that dispatches to either a local or remote controller.
///
/// The API layer uses this everywhere — it doesn't need to know whether
/// inference is running locally or on a remote peer.
#[derive(Clone)]
pub enum InferenceHandle {
    Local(ControllerHandle),
    #[cfg(feature = "p2p-client")]
    Remote(std::sync::Arc<crate::p2p::client::ResilientRemoteHandle>),
    #[cfg(feature = "flock")]
    Flock(std::sync::Arc<crate::p2p::flock::handle::FlockHandle>),
    /// Flock **gateway** consumer (v=2 invite lease) — not peer pool routing.
    #[cfg(feature = "flock")]
    RegisteredFlockGateway(std::sync::Arc<crate::p2p::flock::RegisteredFlockInferenceHandle>),
}

impl InferenceHandle {
    pub fn send(&self, cmd: ControllerCmd) -> Result<(), String> {
        match self {
            Self::Local(h) => h.send(cmd),
            #[cfg(feature = "p2p-client")]
            Self::Remote(h) => h.send(cmd),
            #[cfg(feature = "flock")]
            Self::Flock(h) => h.send(cmd),
            #[cfg(feature = "flock")]
            Self::RegisteredFlockGateway(h) => h.send(cmd),
        }
    }

    /// Current liveness state of the remote peer, or `Alive` for local.
    #[cfg(feature = "p2p-client")]
    pub fn liveness(&self) -> crate::p2p::heartbeat::Liveness {
        match self {
            Self::Local(_) => crate::p2p::heartbeat::Liveness::Alive,
            Self::Remote(h) => h.liveness(),
            #[cfg(feature = "flock")]
            Self::Flock(h) => h.liveness(),
            #[cfg(feature = "flock")]
            Self::RegisteredFlockGateway(_) => crate::p2p::heartbeat::Liveness::Alive,
        }
    }

    /// The controller config driving this handle's policy decisions.
    /// Returns default config for remote/flock handles.
    pub fn config(&self) -> &ControllerConfig {
        match self {
            Self::Local(h) => h.config(),
            #[cfg(feature = "p2p-client")]
            Self::Remote(_) => &DEFAULT_CONFIG,
            #[cfg(feature = "flock")]
            Self::Flock(_) => &DEFAULT_CONFIG,
            #[cfg(feature = "flock")]
            Self::RegisteredFlockGateway(_) => &DEFAULT_CONFIG,
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
    ) -> Result<String, crate::error::PioError> {
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
    ) -> Result<String, crate::error::PioError> {
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
    ) -> Result<String, crate::error::PioError> {
        use crate::error::{ErrorCode, PioError};

        let capacity = self.config().event_channel_capacity;
        let (tx, rx) = std::sync::mpsc::sync_channel::<ControllerEvent>(capacity);
        let cmd = ControllerCmd::SystemInfer {
            task,
            chat_id: chat_id.into(),
            messages,
            gen_spec,
            tx,
        };
        self.send(cmd).map_err(PioError::generation)?;

        tokio::task::spawn_blocking(move || {
            let mut out = String::with_capacity(512);
            while let Ok(ev) = rx.recv() {
                match ev {
                    ControllerEvent::Token(t) => out.push_str(&t),
                    ControllerEvent::Eos | ControllerEvent::Stopped => break,
                    ControllerEvent::Error { code, message } => {
                        return Err(PioError {
                            code: ErrorCode::from_snake_case(&code),
                            message,
                        });
                    }
                    _ => {}
                }
            }
            Ok(out)
        })
        .await
        .map_err(PioError::generation)?
    }

    /// Blocking read of delivery / termination counters from the active backend.
    ///
    /// For the local controller, mirrors [`ControllerHandle::get_controller_metrics`].
    /// For remote or flock routes, forwards the same wire query the server handles for status.
    pub fn get_controller_metrics(&self) -> Result<ControllerMetricsSnapshot, String> {
        let (tx, rx) = channel();
        self.send(ControllerCmd::GetControllerMetrics { resp: tx })?;
        rx.recv().map_err(|e| e.to_string())
    }
}

/// Start the inference controller with explicit configuration.
pub fn start_controller_with_config(config: ControllerConfig) -> ControllerHandle {
    tracing::debug!(?config, "starting inference controller");
    let (tx, rx): (Sender<ControllerCmd>, Receiver<ControllerCmd>) = channel();
    let handle_config = config.clone();
    thread::spawn(move || run_loop(rx, config));
    ControllerHandle {
        tx,
        config: handle_config,
    }
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

/// Default event channel capacity — matches `ControllerConfig::default().event_channel_capacity`.
///
/// Kept as a constant for backward compatibility with external crates and tests
/// that create their own channels. Prefer `handle.config().event_channel_capacity`
/// when a `ControllerHandle` or `InferenceHandle` is available.
pub const EVENT_CHANNEL_CAPACITY: usize = 512;

/// One active session + outbound events + explicit run state.
pub(super) struct ChatRuntime {
    pub(super) session: Arc<crate::gen2::session_rt::Session>,
    pub(super) tx: SyncSender<ControllerEvent>,
    pub(super) workload: WorkloadKind,
    pub(super) last_used: Instant,
    /// Last spec used for `Session::pull` (Resume / Continue paths).
    pub(super) last_gen_spec: GenSpec,
    pub(super) state: ChatRunState,
}

impl ChatRuntime {
    /// `true` when this runtime should receive scheduler ticks.
    pub(super) fn should_tick(&self) -> bool {
        self.state.is_generating()
    }
}

/// Enter [`ChatRunState::Generating`] with a fresh puller (Start / Continue / Resume).
pub(super) fn attach_generating_puller(
    chat: &mut ChatRuntime,
    puller: crate::gen2::generation::TokenPuller,
    pull_spec: GenSpec,
) {
    let last_gen_spec = pull_spec.clone();
    chat.last_gen_spec = last_gen_spec;
    chat.state = ChatRunState::Generating {
        puller,
        gen_started: Instant::now(),
        last_gen_spec: pull_spec,
    };
}

/// Hook listener forwarding final stats for a specific session id back to the UI channel.
pub(super) struct Forwarder {
    sid: u64,
    tx: SyncSender<ControllerEvent>,
    metrics: Arc<metrics::ControllerMetrics>,
}
impl std::fmt::Debug for Forwarder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Forwarder")
    }
}
impl HookListener for Forwarder {
    fn on_event(&self, ev: &HookEvent) {
        if let HookEvent::FinalStats { session_id, stats } = ev
            && *session_id == self.sid
        {
            metrics::emit_must_deliver(
                self.metrics.as_ref(),
                &self.tx,
                ControllerEvent::FinalStats(stats.clone()),
            );
        }
    }
}

/// Result of bounded emit — tells the caller whether the receiver is still alive.
pub(super) enum EmitResult {
    Sent,
    Full,         // channel full, event dropped (receiver alive but slow)
    Disconnected, // receiver gone — stop wasting compute
}

pub(super) fn apply_generation_defaults(engine: &Engine, mut spec: GenSpec) -> GenSpec {
    let defaults = engine.settings();
    defaults.apply_to_gen_spec(&mut spec);
    spec
}

pub(super) fn build_load_request(
    model_path: PathBuf,
    mmproj_path: Option<PathBuf>,
    settings: &Settings,
) -> LoadRequest {
    let mut req = LoadRequest {
        model_path,
        mmproj_path,
        ..Default::default()
    };
    req.ctx_params.n_ctx = settings.system.ctx_size;
    req.ctx_params.threads = settings.system.threads;
    req.model_params.gpu_layers = settings.system.gpu_layers;
    req
}

/// Evict the least recently used runtime (smallest `last_used`).
/// Delegates target selection to `scheduler::pick_eviction_target`.
pub(super) fn evict_lru(chats: &mut HashMap<String, ChatRuntime>) -> Option<(String, ChatRuntime)> {
    let views = scheduler::views(chats);
    scheduler::pick_eviction_target(&views).and_then(|k| chats.remove_entry(&k))
}

fn run_loop(rx: Receiver<ControllerCmd>, config: ControllerConfig) {
    let tick_busy = Duration::from_millis(0);
    let mut state = ControllerState::new(config);

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
        terminate_runtime_owned(
            &state.engine,
            chat,
            RuntimeOutcome::Completed(CompletionReason::ControllerShutdown),
            state.metrics.as_ref(),
        );
    }
}

pub(super) enum ControlFlow {
    Continue,
    Break,
}

/// Shared logic for ephemeral (auto-cleaning) sessions: title generation,
/// compaction summaries, suggestions, etc.
#[allow(clippy::too_many_arguments)]
pub(super) fn start_ephemeral(
    engine: &mut Engine,
    chats: &mut HashMap<String, ChatRuntime>,
    chat_id: String,
    task: SystemTask,
    suffix: &str,
    messages: Vec<Message>,
    gen_spec: GenSpec,
    metrics: Arc<metrics::ControllerMetrics>,
    tx: SyncSender<ControllerEvent>,
) {
    // Ensure unique chat_id (don't collide with live chats).
    let base = format!("{chat_id}::{suffix}");
    let chat_id = if chats.contains_key(&base) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_millis();
        format!("{base}:{ts}")
    } else {
        base
    };

    let gen_spec = apply_generation_defaults(engine, gen_spec);

    match engine.start_session(SessionSpec {
        messages,
        ..Default::default()
    }) {
        Err(e) => {
            metrics::emit_must_deliver(
                metrics.as_ref(),
                &tx,
                ControllerEvent::Error {
                    code: "generation_error".into(),
                    message: format!("{:?}", e),
                },
            );
        }
        Ok(session) => {
            let sid = session.id();
            engine.hooks().register_with_id(
                sid,
                Arc::new(Forwarder {
                    sid,
                    tx: tx.clone(),
                    metrics: metrics.clone(),
                }),
            );
            match session.pull(gen_spec.clone()) {
                Err(e) => {
                    metrics::emit_must_deliver(
                        metrics.as_ref(),
                        &tx,
                        ControllerEvent::Error {
                            code: "generation_error".into(),
                            message: format!("{:?}", e),
                        },
                    );
                    deregister_and_end_session(engine, sid);
                }
                Ok(puller) => {
                    let pull_spec = gen_spec;
                    let mut runtime = ChatRuntime {
                        session,
                        tx,
                        workload: WorkloadKind::SystemTask(task),
                        last_used: Instant::now(),
                        last_gen_spec: pull_spec.clone(),
                        state: ChatRunState::Idle,
                    };
                    attach_generating_puller(&mut runtime, puller, pull_spec);
                    let _ = chats.insert(chat_id, runtime);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let snap = handle
            .get_controller_metrics()
            .expect("controller metrics");
        assert_eq!(snap, ControllerMetricsSnapshot::default());
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
        // With no backend features, the engine is Uninit and returns Err.
        // With a backend (e.g. backend-llamacpp), upload_settings may succeed
        // even without a model file loaded. Either way, the controller must
        // respond without panicking — that is the boundary guarantee.
        #[cfg(not(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-onnx",
            feature = "backend-external-api"
        )))]
        assert!(
            result.is_err(),
            "applying settings without any backend should return Err(ModelNotLoaded)"
        );
        #[cfg(any(
            feature = "backend-llamacpp",
            feature = "backend-mlx",
            feature = "backend-onnx",
            feature = "backend-external-api"
        ))]
        assert!(
            result.is_ok(),
            "applying default settings on an initialized backend should succeed"
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
                gen_spec: crate::gen2::generation::GenSpec::default(),
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
