use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
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
        tx: Sender<ControllerEvent>,
    },
    /// Continue an existing chat session with newly appended messages.
    ContinueChat {
        chat_id: String,
        new_messages: Vec<Message>,
        gen_spec: GenSpec,
        tx: Sender<ControllerEvent>,
    },
    /// Generate a title for a chat (ephemeral session, auto-cleaned).
    CreateTitle {
        chat_id: String,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        tx: Sender<ControllerEvent>,
    },
    /// Generate follow-up suggestions (ephemeral-style generation).
    CreateSuggestions {
        chat_id: String,
        messages: Vec<Message>,
        gen_spec: GenSpec,
        tx: Sender<ControllerEvent>,
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

struct ChatStream {
    session: Arc<crate::gen2::session_rt::Session>,
    puller: Option<crate::gen2::generation::TokenPuller>,
    tx: Sender<ControllerEvent>,
    paused: bool,
    finished: bool,
    ephemeral: bool,
    last_used: Instant, // ⬅ NEW: track most recent access
}

/// Hook listener forwarding final stats for a specific session id back to the UI channel.
struct Forwarder {
    sid: u64,
    tx: Sender<ControllerEvent>,
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
                let _ = self.tx.send(ControllerEvent::FinalStats(stats.clone()));
            }
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
    let _ = chat.tx.send(ControllerEvent::Stopped);
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

        // Round-robin one step per active chat
        let keys: Vec<String> = chats.keys().cloned().collect();
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
                match puller.next() {
                    Some(Ok(TokenEvent::Token(tok))) => {
                        if !tok.text.is_empty() {
                            let _ = chat.tx.send(ControllerEvent::Token(tok.text));
                            chat.last_used = Instant::now(); // keep LRU fresh while active
                        }
                    }
                    Some(Ok(TokenEvent::Eos)) => {
                        let _ = chat.tx.send(ControllerEvent::Eos);
                        chat.finished = true;
                        chat.puller = None;
                    }
                    Some(Ok(TokenEvent::Stopped)) => {
                        let _ = chat.tx.send(ControllerEvent::Stopped);
                        chat.finished = true;
                        chat.puller = None;
                    }
                    Some(Ok(TokenEvent::MediaBoundary(boundary))) => {
                        let _ = chat.tx.send(ControllerEvent::MediaBoundary(boundary));
                    }
                    Some(Ok(TokenEvent::Paused)) => { /* handled by paused flag */ }
                    Some(Ok(TokenEvent::Special(_))) => { /* no-op for now */ }
                    Some(Ok(_)) => { /* ignore other events */ }
                    Some(Err(e)) => {
                        let _ = chat.tx.send(ControllerEvent::Error(format!("{:?}", e)));
                        chat.finished = true;
                        chat.puller = None;
                    }
                    None => {
                        // Iterator ended unexpectedly; treat as stopped
                        let _ = chat.tx.send(ControllerEvent::Stopped);
                        chat.finished = true;
                        chat.puller = None;
                    }
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
            resp,
        } => {
            let r = (|| -> Result<(), String> {
                let load_req = build_load_request(model_path, mmproj_path, &settings);
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
                    let _ = tx.send(ControllerEvent::Error(
                        "chat already generating; pause/stop first".into(),
                    ));
                    return ControlFlow::Continue;
                }

                let last_vec: Vec<_> = messages.last().into_iter().cloned().collect();

                match chat.session.append_messages(last_vec) {
                    Err(e) => {
                        let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
                    }
                    _ => {}
                }

                let pull_spec = apply_generation_defaults(engine, gen_spec.clone());
                match chat.session.pull(pull_spec) {
                    Ok(puller) => {
                        chat.puller = Some(puller);
                        chat.paused = false;
                        chat.finished = false;
                    }
                    Err(e) => {
                        let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
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
                        let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
                    }
                    Ok(session) => {
                        let sid = session.id();
                        engine.hooks().register_with_id(sid, Arc::new(Forwarder {
                            sid,
                            tx: tx.clone(),
                        }));
                        let pull_spec = apply_generation_defaults(engine, gen_spec);
                        match session.pull(pull_spec) {
                            Err(e) => {
                                engine.hooks().deregister(sid);
                                let _ = engine.end_session(sid);
                                let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
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
                                    },
                                );
                                if let Some(evicted_chat) = evicted {
                                    evicted_chat.session.stop();
                                    let _ = evicted_chat.tx.send(ControllerEvent::Stopped);
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
                    let _ = tx.send(ControllerEvent::Error(
                        "chat already generating; pause/stop first".into(),
                    ));
                    return ControlFlow::Continue;
                }
                // If you add Session::append_messages(Vec<Message>), call it here with `new_messages`.
                match chat.session.append_messages(new_messages) {
                    Err(e) => {
                        let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
                    }
                    _ => {}
                }

                let pull_spec = apply_generation_defaults(engine, gen_spec);
                match chat.session.pull(pull_spec) {
                    Ok(puller) => {
                        chat.puller = Some(puller);
                        chat.paused = false;
                        chat.finished = false;
                    }
                    Err(e) => {
                        let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
                    }
                }
            } else {
                let _ = tx.send(ControllerEvent::Error(format!(
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
                    let _ = tx.send(ControllerEvent::Error(
                        "chat already generating; pause/stop first".into(),
                    ));
                    return ControlFlow::Continue;
                }
                match chat.session.append_messages(messages) {
                    Err(e) => {
                        let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
                        return ControlFlow::Continue;
                    }
                    Ok(_) => {}
                }

                let gen_spec = apply_generation_defaults(engine, gen_spec);
                chat.tx = tx.clone();
                match chat.session.pull(gen_spec) {
                    Ok(puller) => {
                        chat.puller = Some(puller);
                        chat.paused = false;
                        chat.finished = false;
                    }
                    Err(e) => {
                        let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
                    }
                }
                return ControlFlow::Continue;
            }

            let _ = tx.send(ControllerEvent::Error(format!(
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
                    let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
                }
                Ok(session) => {
                    let sid = session.id();
                    engine.hooks().register_with_id(sid, Arc::new(Forwarder {
                        sid,
                        tx: tx.clone(),
                    }));
                    match session.pull(gen_spec) {
                        Err(e) => {
                            let _ = tx.send(ControllerEvent::Error(format!("{:?}", e)));
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
                let _ = chat.session.resume(); // or chat.puller.cancel();
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
