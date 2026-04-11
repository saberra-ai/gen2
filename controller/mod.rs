mod config;
mod scheduler;

pub use config::ControllerConfig;

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel};
use std::thread;
use std::time::{Duration, Instant};

use crate::gen2::engine::{EmbedLoadRequest, HookEvent, HookListener, LoadRequest};
use crate::gen2::generation::{GenSpec, TokenEvent};
use crate::gen2::session_rt::SessionSpec;
use crate::gen2::{Engine, ExecutionStats, Settings};
use crate::types::message::Message;

/// System-level inference tasks — ephemeral, fire-and-forget, hidden from users.
///
/// Each variant runs as an ephemeral session on the controller: prompt in,
/// tokens streamed back, session auto-cleaned on completion.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

struct ChatStream {
    session: Arc<crate::gen2::session_rt::Session>,
    puller: Option<crate::gen2::generation::TokenPuller>,
    tx: SyncSender<ControllerEvent>,
    paused: bool,
    finished: bool,
    ephemeral: bool,
    last_used: Instant,     // ⬅ track most recent access
    gen_started: Instant,   // when the current generation began (for timeout)
    last_gen_spec: GenSpec, // stored for puller recreation on resume
}

/// Hook listener forwarding final stats for a specific session id back to the UI channel.
struct Forwarder {
    sid: u64,
    tx: SyncSender<ControllerEvent>,
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
            try_emit(&self.tx, ControllerEvent::FinalStats(stats.clone()));
        }
    }
}

/// Result of try_emit — tells the caller whether the receiver is still alive.
enum EmitResult {
    Sent,
    Full,         // channel full, event dropped (receiver alive but slow)
    Disconnected, // receiver gone — stop wasting compute
}

/// Try to send an event on a bounded channel without blocking.
fn try_emit(tx: &SyncSender<ControllerEvent>, event: ControllerEvent) -> EmitResult {
    match tx.try_send(event) {
        Ok(()) => EmitResult::Sent,
        Err(TrySendError::Full(_)) => {
            tracing::trace!("event channel full, dropping event");
            EmitResult::Full
        }
        Err(TrySendError::Disconnected(_)) => {
            tracing::debug!("event channel disconnected, receiver dropped");
            EmitResult::Disconnected
        }
    }
}

fn apply_generation_defaults(engine: &Engine, mut spec: GenSpec) -> GenSpec {
    let defaults = engine.settings();
    defaults.apply_to_gen_spec(&mut spec);
    spec
}

fn build_load_request(
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

fn stop_notify_and_end(engine: &Engine, mut chat: ChatStream) {
    let sid = chat.session.id();
    chat.session.stop();
    try_emit(&chat.tx, ControllerEvent::Stopped);
    engine.hooks().deregister(sid);
    let _ = engine.end_session(sid);
}

/// Evict the least recently used ChatStream (smallest `last_used`).
/// Delegates target selection to `scheduler::pick_eviction_target`.
fn evict_lru(chats: &mut HashMap<String, ChatStream>) -> Option<(String, ChatStream)> {
    let views = scheduler::views(chats);
    scheduler::pick_eviction_target(&views).and_then(|k| chats.remove_entry(&k))
}

fn run_loop(rx: Receiver<ControllerCmd>, config: ControllerConfig) {
    let mut engine = Engine::new();
    let mut chats: HashMap<String, ChatStream> = HashMap::new();
    let tick_idle = config.tick_idle;
    let tick_busy = Duration::from_millis(0);
    let max_active = config.max_active_chats;
    let generation_timeout = config.generation_timeout;
    // Cached backend capabilities — updated on model load.
    let mut caps = engine.backend_caps();

    'outer: loop {
        // If there are no active chats, block waiting for a command.
        if chats.is_empty() {
            match rx.recv() {
                Err(_) => break, // controller dropped
                Ok(ControllerCmd::Shutdown) => break,
                Ok(cmd) => {
                    if let ControlFlow::Break =
                        dispatch_cmd(cmd, &mut engine, &mut chats, max_active, &mut caps)
                    {
                        break;
                    }
                }
            }
            continue;
        }

        // With active chats, interleave command handling with generation ticks.
        // Use a short timeout so we don't busy-spin but also keep latency low.
        match rx.recv_timeout(tick_idle) {
            Ok(ControllerCmd::Shutdown) => break 'outer,
            Ok(cmd) => {
                if let ControlFlow::Break =
                    dispatch_cmd(cmd, &mut engine, &mut chats, max_active, &mut caps)
                {
                    break 'outer;
                }
            }
            Err(_timeout) => {
                // no command; proceed to tick chats
            }
        }

        // Priority scheduling: primary chats tick before ephemeral.
        let sched_views = scheduler::views(&chats);
        let keys = scheduler::tick_order(&sched_views);
        let mut to_remove: Vec<String> = Vec::new();

        for id in keys {
            if let Some(chat) = chats.get_mut(&id) {
                if chat.paused || chat.finished {
                    continue;
                }
                // ── Generation timeout ──────────────────────────────────
                if chat.gen_started.elapsed() > generation_timeout {
                    tracing::warn!(
                        chat_id = %id,
                        elapsed_secs = chat.gen_started.elapsed().as_secs(),
                        "generation timed out"
                    );
                    try_emit(
                        &chat.tx,
                        ControllerEvent::Error {
                            code: "timeout".into(),
                            message: format!(
                                "generation timed out after {}s",
                                generation_timeout.as_secs()
                            ),
                        },
                    );
                    chat.finished = true;
                    chat.puller = None;
                    if chat.ephemeral {
                        to_remove.push(id.clone());
                    }
                    continue;
                }
                // one step
                let Some(puller) = chat.puller.as_mut() else {
                    continue;
                };
                // Wrap puller.next() in catch_unwind to detect FFI panics.
                // On panic the DecodeState is lost — session becomes poisoned.
                let step = catch_unwind(AssertUnwindSafe(|| puller.next()));
                // Helper: emit event and stop generation if receiver is gone.
                let mut receiver_dead = false;
                let mut emit = |event: ControllerEvent| {
                    if let EmitResult::Disconnected = try_emit(&chat.tx, event) {
                        receiver_dead = true;
                    }
                };

                match step {
                    Err(_panic) => {
                        emit(ControllerEvent::Error {
                            code: "session_poisoned".into(),
                            message: "inference panic: session state lost; restart chat".into(),
                        });
                        chat.finished = true;
                        chat.puller = None;
                    }
                    Ok(Some(Ok(TokenEvent::Token(tok)))) => {
                        if !tok.text.is_empty() {
                            emit(ControllerEvent::Token(tok.text));
                            chat.last_used = Instant::now();
                        }
                    }
                    Ok(Some(Ok(TokenEvent::Eos))) => {
                        emit(ControllerEvent::Eos);
                        chat.finished = true;
                        chat.puller = None;
                    }
                    Ok(Some(Ok(TokenEvent::Stopped))) => {
                        emit(ControllerEvent::Stopped);
                        chat.finished = true;
                        chat.puller = None;
                    }
                    Ok(Some(Ok(TokenEvent::MediaBoundary(boundary)))) => {
                        emit(ControllerEvent::MediaBoundary(boundary));
                    }
                    Ok(Some(Ok(TokenEvent::Paused))) => { /* handled by paused flag */ }
                    Ok(Some(Ok(TokenEvent::Special(_)))) => { /* no-op for now */ }
                    Ok(Some(Err(e))) => {
                        let (code, msg) = if caps.poison_detection && chat.session.is_poisoned() {
                            (
                                "session_poisoned",
                                format!("session state lost (possible FFI crash): {:?}", e),
                            )
                        } else {
                            ("generation_error", format!("{:?}", e))
                        };
                        emit(ControllerEvent::Error {
                            code: code.into(),
                            message: msg,
                        });
                        chat.finished = true;
                        chat.puller = None;
                    }
                    Ok(None) => {
                        emit(ControllerEvent::Stopped);
                        chat.finished = true;
                        chat.puller = None;
                    }
                }
                // If the receiver is gone, stop wasting compute on this chat
                if receiver_dead {
                    chat.finished = true;
                    chat.puller = None;
                }
                if scheduler::should_cleanup(&chat.schedule_view()) {
                    to_remove.push(id.clone());
                }
            }
        }

        for id in to_remove.drain(..) {
            if let Some(chat) = chats.remove(&id) {
                stop_notify_and_end(&engine, chat);
            }
        }
        // If we're still busy (have chats), yield very briefly to avoid hogging CPU.
        if !chats.is_empty() {
            thread::sleep(tick_busy);
        }
    }

    // Drain & notify any remaining chats on shutdown
    for (_id, chat) in chats.drain() {
        stop_notify_and_end(&engine, chat);
    }
}

enum ControlFlow {
    Continue,
    Break,
}

/// Shared logic for ephemeral (auto-cleaning) sessions: title generation,
/// compaction summaries, suggestions, etc.
fn start_ephemeral(
    engine: &mut Engine,
    chats: &mut HashMap<String, ChatStream>,
    chat_id: String,
    suffix: &str,
    messages: Vec<Message>,
    gen_spec: GenSpec,
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
            try_emit(
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
                }),
            );
            match session.pull(gen_spec.clone()) {
                Err(e) => {
                    try_emit(
                        &tx,
                        ControllerEvent::Error {
                            code: "generation_error".into(),
                            message: format!("{:?}", e),
                        },
                    );
                }
                Ok(puller) => {
                    let _ = chats.insert(
                        chat_id,
                        ChatStream {
                            session,
                            puller: Some(puller),
                            tx,
                            paused: false,
                            finished: false,
                            ephemeral: true,
                            last_used: Instant::now(),
                            gen_started: Instant::now(),
                            last_gen_spec: gen_spec,
                        },
                    );
                }
            }
        }
    }
}

fn dispatch_cmd(
    cmd: ControllerCmd,
    engine: &mut Engine,
    chats: &mut HashMap<String, ChatStream>,
    max_active: usize,
    caps: &mut crate::gen2::backend::caps::BackendCaps,
) -> ControlFlow {
    match cmd {
        ControllerCmd::LoadModel {
            model_path,
            mmproj_path,
            settings,
            api_key,
            api_format,
            resp,
        } => {
            let r = (|| -> Result<(), String> {
                let mut load_req = build_load_request(model_path, mmproj_path, &settings);
                load_req.api_key = api_key;
                load_req.api_format = api_format;
                engine
                    .upload_settings(settings)
                    .map_err(|e| format!("{:?}", e))?;
                engine
                    .load_model(load_req)
                    .map_err(|e| format!("{:?}", e))?;
                Ok(())
            })();
            if r.is_ok() {
                for (_, chat) in chats.drain() {
                    stop_notify_and_end(engine, chat);
                }
                engine.hooks().clear();
                // Refresh cached capabilities for the newly loaded backend.
                *caps = engine.backend_caps();
            }
            let _ = resp.send(r);
            ControlFlow::Continue
        }
        ControllerCmd::ApplySettings { settings, resp } => {
            let active = chats.values().filter(|c| !c.finished && !c.paused).count();
            if active > 0 {
                tracing::info!(
                    active_chats = active,
                    "settings updated with active chats; changes take effect on next generation"
                );
            }
            let res = engine
                .upload_settings(settings)
                .map_err(|e| format!("{:?}", e));
            let _ = resp.send(res);
            ControlFlow::Continue
        }
        ControllerCmd::LoadEmbedder { model_path, resp } => {
            let res = engine
                .load_embedder(EmbedLoadRequest { model_path })
                .map_err(|e| format!("{:?}", e));
            let _ = resp.send(res);
            ControlFlow::Continue
        }
        ControllerCmd::GenerateEmbeddings { inputs, resp } => {
            let res = engine
                .generate_embeddings(&inputs)
                .map_err(|e| format!("{:?}", e));
            let _ = resp.send(res);
            ControlFlow::Continue
        }
        ControllerCmd::IsModelLoaded { resp } => {
            let _ = resp.send(engine.is_model_loaded());
            ControlFlow::Continue
        }
        ControllerCmd::IsEmbedderLoaded { resp } => {
            let _ = resp.send(engine.is_embedder_loaded());
            ControlFlow::Continue
        }
        ControllerCmd::IsMmprojLoaded { resp } => {
            let supports_images = engine.does_model_support_images();
            let _ = resp.send(supports_images);
            ControlFlow::Continue
        }
        ControllerCmd::StartChat {
            chat_id,
            messages,
            gen_spec,
            tx,
        } => {
            if let Some(chat) = chats.get_mut(&chat_id) {
                chat.last_used = Instant::now();
                chat.tx = tx.clone();
                if chat.puller.is_some() && !chat.finished {
                    try_emit(
                        &tx,
                        ControllerEvent::Error {
                            code: "generation_error".into(),
                            message: "chat already generating; pause/stop first".into(),
                        },
                    );
                    return ControlFlow::Continue;
                }

                let last_vec: Vec<_> = messages.last().into_iter().cloned().collect();

                match chat.session.append_messages(last_vec) {
                    Err(e) => {
                        try_emit(
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                        return ControlFlow::Continue;
                    }
                    Ok(dropped) if dropped > 0 => {
                        try_emit(&tx, ControllerEvent::ContextTruncated(dropped));
                    }
                    _ => {}
                }

                let pull_spec = apply_generation_defaults(engine, gen_spec.clone());
                match chat.session.pull(pull_spec.clone()) {
                    Ok(puller) => {
                        chat.puller = Some(puller);
                        chat.paused = false;
                        chat.finished = false;
                        chat.gen_started = Instant::now();
                        chat.last_gen_spec = pull_spec;
                    }
                    Err(e) => {
                        try_emit(
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                    }
                }
                return ControlFlow::Continue;
            } else {
                // chat not loaded loading new chat
                if chats.len() >= max_active
                    && let Some((_k, victim)) = evict_lru(chats)
                {
                    stop_notify_and_end(engine, victim);
                }
                match engine.start_session(SessionSpec {
                    messages,
                    ..Default::default()
                }) {
                    Err(e) => {
                        try_emit(
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                    }
                    Ok(session) => {
                        if caps.context_truncation_tracking {
                            let dropped = session.initial_messages_dropped();
                            if dropped > 0 {
                                try_emit(&tx, ControllerEvent::ContextTruncated(dropped));
                            }
                        }
                        let sid = session.id();
                        engine.hooks().register_with_id(
                            sid,
                            Arc::new(Forwarder {
                                sid,
                                tx: tx.clone(),
                            }),
                        );
                        let pull_spec = apply_generation_defaults(engine, gen_spec);
                        match session.pull(pull_spec.clone()) {
                            Err(e) => {
                                engine.hooks().deregister(sid);
                                let _ = engine.end_session(sid);
                                try_emit(
                                    &tx,
                                    ControllerEvent::Error {
                                        code: "generation_error".into(),
                                        message: format!("{:?}", e),
                                    },
                                );
                            }
                            Ok(puller) => {
                                let evicted = chats.insert(
                                    chat_id,
                                    ChatStream {
                                        session, // <-- keep it here
                                        puller: Some(puller),
                                        tx,
                                        paused: false,
                                        finished: false,
                                        ephemeral: false, // <-- key bit
                                        last_used: Instant::now(),
                                        gen_started: Instant::now(),
                                        last_gen_spec: pull_spec,
                                    },
                                );
                                if let Some(evicted_chat) = evicted {
                                    evicted_chat.session.stop();
                                    try_emit(&evicted_chat.tx, ControllerEvent::Stopped);
                                }
                            }
                        }
                    }
                }
            }
            ControlFlow::Continue
        }
        ControllerCmd::ContinueChat {
            chat_id,
            new_messages,
            gen_spec,
            tx,
        } => {
            // Resume from last point on the same Session
            if let Some(chat) = chats.get_mut(&chat_id) {
                chat.tx = tx.clone();
                if chat.puller.is_some() && !chat.finished {
                    try_emit(
                        &tx,
                        ControllerEvent::Error {
                            code: "generation_error".into(),
                            message: "chat already generating; pause/stop first".into(),
                        },
                    );
                    return ControlFlow::Continue;
                }
                // If you add Session::append_messages(Vec<Message>), call it here with `new_messages`.
                match chat.session.append_messages(new_messages) {
                    Err(e) => {
                        try_emit(
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                        return ControlFlow::Continue;
                    }
                    Ok(dropped) if dropped > 0 => {
                        try_emit(&tx, ControllerEvent::ContextTruncated(dropped));
                    }
                    _ => {}
                }

                let pull_spec = apply_generation_defaults(engine, gen_spec);
                match chat.session.pull(pull_spec.clone()) {
                    Ok(puller) => {
                        chat.puller = Some(puller);
                        chat.paused = false;
                        chat.finished = false;
                        chat.gen_started = Instant::now();
                        chat.last_gen_spec = pull_spec;
                    }
                    Err(e) => {
                        try_emit(
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                    }
                }
            } else {
                try_emit(
                    &tx,
                    ControllerEvent::Error {
                        code: "not_found".into(),
                        message: format!("chat_id '{}' not found", chat_id),
                    },
                );
            }
            ControlFlow::Continue
        }
        ControllerCmd::SystemInfer {
            task,
            chat_id,
            messages,
            gen_spec,
            tx,
        } => {
            start_ephemeral(
                engine,
                chats,
                chat_id,
                task.suffix(),
                messages,
                gen_spec,
                tx,
            );
            ControlFlow::Continue
        }
        ControllerCmd::StopChat { chat_id } => {
            if let Some(chat) = chats.remove(&chat_id) {
                stop_notify_and_end(engine, chat);
            }
            ControlFlow::Continue
        }
        ControllerCmd::PauseChat { chat_id } => {
            if let Some(chat) = chats.get_mut(&chat_id) {
                chat.puller = None;
                chat.paused = true;
                chat.session.pause(); // or chat.puller.cancel();
            }
            ControlFlow::Continue
        }
        ControllerCmd::ResumeChat { chat_id } => {
            if let Some(chat) = chats.get_mut(&chat_id) {
                chat.paused = false;
                chat.gen_started = Instant::now();
                chat.session.resume();
                // Recreate the puller that was dropped during pause
                if chat.puller.is_none() {
                    match chat.session.pull(chat.last_gen_spec.clone()) {
                        Ok(puller) => {
                            chat.puller = Some(puller);
                            chat.finished = false;
                        }
                        Err(e) => {
                            try_emit(
                                &chat.tx,
                                ControllerEvent::Error {
                                    code: "generation_error".into(),
                                    message: format!("failed to resume generation: {:?}", e),
                                },
                            );
                        }
                    }
                }
            }
            ControlFlow::Continue
        }
        ControllerCmd::Shutdown => ControlFlow::Break,
        ControllerCmd::IsChatLoaded { chat_id, resp } => {
            let loaded = chats.get(&chat_id).is_some();
            let _ = resp.send(loaded);
            ControlFlow::Continue
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
        let result = try_emit(&tx, ControllerEvent::Stopped);
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

        let result = try_emit(&tx, ControllerEvent::Stopped);
        assert!(
            matches!(result, EmitResult::Disconnected),
            "expected EmitResult::Disconnected when receiver is dropped"
        );
    }

    #[test]
    fn try_emit_sent() {
        let (tx, rx) = sync_channel::<ControllerEvent>(EVENT_CHANNEL_CAPACITY);

        let result = try_emit(&tx, ControllerEvent::Stopped);
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
