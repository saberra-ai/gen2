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
use crate::generation::model_runner::types::Message;

/// Commands accepted by the inference controller thread.
///
/// Grouped into four categories:
/// - **Model lifecycle** — load/reload models and apply settings
/// - **Status queries** — synchronous checks on loaded state
/// - **Chat operations** — start, continue, pause, stop generation
/// - **Utility** — embeddings, shutdown
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
    IsModelLoaded {
        resp: Sender<bool>,
    },
    /// Check whether the embedding model is loaded.
    IsEmbedderLoaded {
        resp: Sender<bool>,
    },
    /// Check whether the multimodal projector is loaded (image support).
    IsMmprojLoaded {
        resp: Sender<bool>,
    },
    /// Check whether a chat session is active for the given chat_id.
    IsChatLoaded {
        chat_id: String,
        resp: Sender<bool>,
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
    /// Generate a title for a chat (ephemeral session, auto-cleaned).
    CreateTitle {
        chat_id: String,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        tx: SyncSender<ControllerEvent>,
    },
    /// Generate follow-up suggestions (ephemeral-style generation).
    CreateSuggestions {
        chat_id: String,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        tx: SyncSender<ControllerEvent>,
    },
    /// Abort and remove a chat session.
    StopChat {
        chat_id: String,
    },
    /// Pause token generation for a chat (session stays in memory).
    PauseChat {
        chat_id: String,
    },
    /// Resume a paused chat session.
    ResumeChat {
        chat_id: String,
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
    /// An error occurred during generation.
    Error(String),
    /// Final execution statistics for the completed generation.
    FinalStats(ExecutionStats),
    /// Context was truncated — N old messages dropped to fit context window.
    ContextTruncated(usize),
}

#[derive(Clone)]
pub struct ControllerHandle {
    tx: Sender<ControllerCmd>,
}

impl ControllerHandle {
    pub fn send(&self, cmd: ControllerCmd) -> Result<(), String> {
        self.tx.send(cmd).map_err(|e| e.to_string())
    }
}

/// Unified handle that dispatches to either a local or remote controller.
///
/// The API layer uses this everywhere — it doesn't need to know whether
/// inference is running locally or on a remote peer.
#[derive(Clone)]
pub enum InferenceHandle {
    Local(ControllerHandle),
    #[cfg(feature = "p2p-client")]
    Remote(crate::p2p::client::RemoteControllerHandle),
}

impl InferenceHandle {
    pub fn send(&self, cmd: ControllerCmd) -> Result<(), String> {
        match self {
            Self::Local(h) => h.send(cmd),
            #[cfg(feature = "p2p-client")]
            Self::Remote(h) => h.send(cmd),
        }
    }
}

pub fn start_controller_with_limit(max_active: usize) -> ControllerHandle {
    let (tx, rx): (Sender<ControllerCmd>, Receiver<ControllerCmd>) = channel();
    thread::spawn(move || run_loop(rx, max_active));
    ControllerHandle { tx }
}

// keep the old helper as an unlimited default (or pick a sensible default)
pub fn start_controller() -> ControllerHandle {
    let default_max_active = 3;
    start_controller_with_limit(default_max_active) // or e.g., 8
}

/// Capacity for bounded event channels to callers.
/// Large enough to absorb burst without blocking the controller,
/// small enough to bound memory if the receiver stalls.
pub const EVENT_CHANNEL_CAPACITY: usize = 512;

struct ChatStream {
    session: Arc<crate::gen2::session_rt::Session>,
    puller: Option<crate::gen2::generation::TokenPuller>,
    tx: SyncSender<ControllerEvent>,
    paused: bool,
    finished: bool,
    ephemeral: bool,
    last_used: Instant, // ⬅ NEW: track most recent access
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
        if let HookEvent::FinalStats { session_id, stats } = ev {
            if *session_id == self.sid {
                try_emit(&self.tx, ControllerEvent::FinalStats(stats.clone()));
            }
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
    let _ = chat.session.stop();
    try_emit(&chat.tx, ControllerEvent::Stopped);
    engine.hooks().deregister(sid);
    let _ = engine.end_session(sid);
}

/// Evict the least recently used ChatStream (smallest `last_used`)
fn evict_lru(chats: &mut HashMap<String, ChatStream>) -> Option<(String, ChatStream)> {
    chats
        .iter()
        .min_by_key(|(_k, c)| c.last_used) // oldest access wins
        .map(|(k, _)| k.clone())
        .and_then(|k| chats.remove_entry(&k))
}

fn run_loop(rx: Receiver<ControllerCmd>, max_active: usize) {
    let mut engine = Engine::new();
    let mut chats: HashMap<String, ChatStream> = HashMap::new();
    let tick_idle = Duration::from_millis(2);
    let tick_busy = Duration::from_millis(0);

    'outer: loop {
        // If there are no active chats, block waiting for a command.
        if chats.is_empty() {
            match rx.recv() {
                Err(_) => break, // controller dropped
                Ok(ControllerCmd::Shutdown) => break,
                Ok(cmd) => {
                    if let ControlFlow::Break = dispatch_cmd(cmd, &mut engine, &mut chats, max_active) {
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
                if let ControlFlow::Break = dispatch_cmd(cmd, &mut engine, &mut chats, max_active) {
                    break 'outer;
                }
            }
            Err(_timeout) => {
                // no command; proceed to tick chats
            }
        }

        // Priority scheduling: primary chats tick before ephemeral.
        // This ensures the active user chat always gets compute first;
        // title gen and suggestions don't steal ticks from the live chat.
        let mut keys: Vec<String> = Vec::new();
        let mut ephemeral_start = 0usize;
        for (id, chat) in chats.iter() {
            if chat.paused || chat.finished || chat.puller.is_none() {
                continue;
            }
            if !chat.ephemeral {
                // Insert primary chats at the front
                keys.insert(ephemeral_start, id.clone());
                ephemeral_start += 1;
            } else {
                keys.push(id.clone());
            }
        }
        let mut to_remove: Vec<String> = Vec::new();

        for id in keys {
            if let Some(chat) = chats.get_mut(&id) {
                if chat.paused || chat.finished {
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
                        emit(ControllerEvent::Error(
                            "inference panic: session state lost; restart chat".into(),
                        ));
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
                        let msg = if chat.session.is_poisoned() {
                            format!("session state lost (possible FFI crash): {:?}", e)
                        } else {
                            format!("{:?}", e)
                        };
                        emit(ControllerEvent::Error(msg));
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
                // NEW: schedule ephemeral cleanup after finishing
                if chat.finished && chat.ephemeral {
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

fn dispatch_cmd(
    cmd: ControllerCmd,
    engine: &mut Engine,
    chats: &mut HashMap<String, ChatStream>,
    max_active: usize,
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
                    try_emit(&tx, ControllerEvent::Error(
                        "chat already generating; pause/stop first".into(),
                    ));
                    return ControlFlow::Continue;
                }

                let last_vec: Vec<_> = messages.last().into_iter().cloned().collect();

                match chat.session.append_messages(last_vec) {
                    Err(e) => {
                        try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
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
                        chat.last_gen_spec = pull_spec;
                    }
                    Err(e) => {
                        try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
                    }
                }
                return ControlFlow::Continue;
            } else {
                // chat not loaded loading new chat
                if chats.len() >= max_active {
                    if let Some((_k, victim)) = evict_lru(chats) {
                        stop_notify_and_end(engine, victim);
                    }
                }
                match engine.start_session(SessionSpec {
                    messages,
                    ..Default::default()
                }) {
                    Err(e) => {
                        try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
                    }
                    Ok(session) => {
                        let dropped = session.initial_messages_dropped();
                        if dropped > 0 {
                            try_emit(&tx, ControllerEvent::ContextTruncated(dropped));
                        }
                        let sid = session.id();
                        engine.hooks().register_with_id(sid, Arc::new(Forwarder {
                            sid,
                            tx: tx.clone(),
                        }));
                        let pull_spec = apply_generation_defaults(engine, gen_spec);
                        match session.pull(pull_spec.clone()) {
                            Err(e) => {
                                engine.hooks().deregister(sid);
                                let _ = engine.end_session(sid);
                                try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
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
                    try_emit(&tx, ControllerEvent::Error(
                        "chat already generating; pause/stop first".into(),
                    ));
                    return ControlFlow::Continue;
                }
                // If you add Session::append_messages(Vec<Message>), call it here with `new_messages`.
                match chat.session.append_messages(new_messages) {
                    Err(e) => {
                        try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
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
                        chat.last_gen_spec = pull_spec;
                    }
                    Err(e) => {
                        try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
                    }
                }
            } else {
                try_emit(&tx, ControllerEvent::Error(format!(
                    "chat_id '{}' not found",
                    chat_id
                )));
            }
            ControlFlow::Continue
        }
        ControllerCmd::CreateSuggestions {
            chat_id,
            messages,
            gen_spec,
            tx,
        } => {
            if let Some(chat) = chats.get_mut(&chat_id) {
                chat.last_used = Instant::now();
                if chat.puller.is_some() && !chat.finished {
                    try_emit(&tx, ControllerEvent::Error(
                        "chat already generating; pause/stop first".into(),
                    ));
                    return ControlFlow::Continue;
                }
                match chat.session.append_messages(messages) {
                    Err(e) => {
                        try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
                        return ControlFlow::Continue;
                    }
                    Ok(_) => {}
                }

                let gen_spec = apply_generation_defaults(engine, gen_spec);
                chat.tx = tx.clone();
                match chat.session.pull(gen_spec.clone()) {
                    Ok(puller) => {
                        chat.puller = Some(puller);
                        chat.paused = false;
                        chat.finished = false;
                        chat.last_gen_spec = gen_spec;
                    }
                    Err(e) => {
                        try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
                    }
                }
                return ControlFlow::Continue;
            }

            try_emit(&tx, ControllerEvent::Error(format!(
                "chat_id '{}' is not loaded",
                chat_id
            )));
            ControlFlow::Continue
        }
        ControllerCmd::CreateTitle {
            mut chat_id,
            messages,
            gen_spec,
            tx,
        } => {
            // ensure uniqueness (don’t collide with live chats)
            let base = format!("{chat_id}::title");
            if chats.contains_key(&base) {
                use std::time::{SystemTime, UNIX_EPOCH};
                let ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                chat_id = format!("{base}:{ts}");
            } else {
                chat_id = base;
            }

            let gen_spec = apply_generation_defaults(engine, gen_spec);

            match engine.start_session(SessionSpec {
                messages,
                ..Default::default()
            }) {
                Err(e) => {
                    try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
                }
                Ok(session) => {
                    let sid = session.id();
                    engine.hooks().register_with_id(sid, Arc::new(Forwarder {
                        sid,
                        tx: tx.clone(),
                    }));
                    match session.pull(gen_spec.clone()) {
                        Err(e) => {
                            try_emit(&tx, ControllerEvent::Error(format!("{:?}", e)));
                        }
                        Ok(puller) => {
                            // Insert as ephemeral so it auto-cleans after EOS/Stopped
                            let _ = chats.insert(
                                chat_id,
                                ChatStream {
                                    session,
                                    puller: Some(puller),
                                    tx,
                                    paused: false,
                                    finished: false,
                                    ephemeral: true, // <-- key bit
                                    last_used: Instant::now(),
                                    last_gen_spec: gen_spec,
                                },
                            );
                        }
                    }
                }
            }
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
                let _ = chat.session.pause(); // or chat.puller.cancel();
            }
            ControlFlow::Continue
        }
        ControllerCmd::ResumeChat { chat_id } => {
            if let Some(chat) = chats.get_mut(&chat_id) {
                chat.paused = false;
                let _ = chat.session.resume();
                // Recreate the puller that was dropped during pause
                if chat.puller.is_none() {
                    match chat.session.pull(chat.last_gen_spec.clone()) {
                        Ok(puller) => {
                            chat.puller = Some(puller);
                            chat.finished = false;
                        }
                        Err(e) => {
                            try_emit(&chat.tx, ControllerEvent::Error(format!(
                                "failed to resume generation: {:?}", e
                            )));
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
