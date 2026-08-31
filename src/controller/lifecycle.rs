//! Controller lifecycle helpers — runtime outcome, session teardown,
//! puller attachment, eviction, load-request shaping, ephemeral session
//! setup. Moved out of `mod.rs` in Phase 6 for readability; no behavior
//! change, same `pub(super)` visibility.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::SyncSender;
use std::time::Instant;

use super::{
    ChatRunState, ChatRuntime, CompletionReason, ControllerEvent, FailureReason, SystemTask,
    WorkloadKind, metrics, scheduler,
};
use crate::engine::LoadRequest;
use crate::generation::GenSpec;
use crate::session_rt::SessionSpec;
use crate::types::message::Message;
use crate::{Engine, Settings};

/// Terminal outcome for a chat runtime.
pub(super) enum RuntimeOutcome {
    Completed(CompletionReason),
    Failed(FailureReason),
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

/// Enter [`ChatRunState::Generating`] with a fresh puller (Start / Continue / Resume).
pub(super) fn attach_generating_puller(
    chat: &mut ChatRuntime,
    puller: crate::generation::TokenPuller,
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
    thinking: crate::generation::ThinkingMode,
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
        thinking,
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
                Arc::new(super::observability::Forwarder {
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
                        health: Default::default(),
                    };
                    attach_generating_puller(&mut runtime, puller, pull_spec);
                    let _ = chats.insert(chat_id, runtime);
                }
            }
        }
    }
}
