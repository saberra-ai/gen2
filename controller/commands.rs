//! Command dispatch and generation tick — split by domain (PR4).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::diagnostics::current_memory_governor;
use crate::gen2::engine::EmbedLoadRequest;
use crate::gen2::generation::TokenEvent;
use crate::gen2::residency::{ResidentRuntime, RuntimeKind};
use crate::gen2::residency_policy::{
    estimate_resident_mb_for_path, estimate_resident_mb_for_path_offloaded,
};
use crate::gen2::session_rt::SessionSpec;

use super::lifecycle::{
    RuntimeOutcome, apply_generation_defaults, attach_generating_puller, build_load_request,
    deregister_and_end_session, evict_lru, runtime_outcome_from_state, start_ephemeral,
    terminate_runtime, terminate_runtime_owned,
};
use super::metrics::{emit_best_effort, emit_must_deliver};
use super::observability::{EmitResult, Forwarder};
use super::state::ControllerState;
use super::{
    ChatRunState, ChatRuntime, CompletionReason, ControlFlow, ControllerCmd, ControllerEvent,
    FailureReason, WorkloadKind, scheduler, state_transitions,
};

fn handle_model_command(state: &mut ControllerState, cmd: ControllerCmd) -> ControlFlow {
    match cmd {
        ControllerCmd::LoadModel {
            model_path,
            mmproj_path,
            settings,
            api_key,
            api_format,
            resp,
        } => {
            // Capture the loaded LLM's whole-model byte size on success so the
            // flock fit gate can source a real footprint (Part A of VRAM-aware
            // routing). Computed from the same path the engine stats at load
            // (`metadata().len()` in `engine::validate_model_file`); replaced on
            // every successful reload, cleared on failure.
            let mut loaded_file_bytes: Option<u64> = None;
            let r = (|| -> Result<(), String> {
                let mut load_req = build_load_request(model_path, mmproj_path, &settings);
                load_req.api_key = api_key;
                load_req.api_format = api_format;
                loaded_file_bytes = ControllerState::model_file_bytes_of(&load_req.model_path);
                let runtime_name = load_req.model_path.display().to_string();
                // Offloaded weights live in VRAM, not host RAM — don't deny a
                // GPU-bound model on a RAM-tight host (residency_policy.rs).
                let estimated_mb = estimate_resident_mb_for_path_offloaded(
                    &load_req.model_path,
                    load_req.model_params.gpu_layers,
                );
                let governor = current_memory_governor();
                if state.residency_policy.llm_swap_requires_unload && state.residency.llm.is_some()
                {
                    state.engine.unload_model();
                    let _ = state.residency.unload(RuntimeKind::Llm);
                }
                if !state
                    .residency
                    .can_admit(RuntimeKind::Llm, estimated_mb, &governor)
                {
                    return Err("llm admission denied by residency policy".into());
                }
                state
                    .engine
                    .upload_settings(settings)
                    .map_err(|e| format!("{:?}", e))?;
                state
                    .engine
                    .load_model(load_req)
                    .map_err(|e| format!("{:?}", e))?;
                let admitted = state.residency.admit(
                    ResidentRuntime::new(
                        RuntimeKind::Llm,
                        runtime_name,
                        estimated_mb,
                        chrono::Utc::now().timestamp(),
                    ),
                    &governor,
                );
                if !admitted {
                    state.engine.unload_model();
                    return Err("llm residency registration denied after load".into());
                }
                Ok(())
            })();
            if r.is_ok() {
                for (_, chat) in state.chats.drain() {
                    terminate_runtime_owned(
                        &state.engine,
                        chat,
                        RuntimeOutcome::Completed(CompletionReason::ModelReloaded),
                        state.metrics.as_ref(),
                    );
                }
                state.engine.hooks().clear();
                state.caps = state.engine.backend_caps();
                // Record the freshly-loaded model's file size (replaces any
                // prior model's size on reload).
                state.loaded_model_file_bytes = loaded_file_bytes;
            }
            let _ = resp.send(r);
            ControlFlow::Continue
        }
        ControllerCmd::ApplySettings { settings, resp } => {
            let active = state
                .chats
                .values()
                .filter(|c| c.state.is_generating())
                .count();
            if active > 0 {
                tracing::info!(
                    active_chats = active,
                    "settings updated with active chats; changes take effect on next generation"
                );
            }
            let res = state
                .engine
                .upload_settings(settings)
                .map_err(|e| format!("{:?}", e));
            let _ = resp.send(res);
            ControlFlow::Continue
        }
        ControllerCmd::LoadEmbedder {
            model_path,
            kind,
            resp,
        } => {
            let estimated_mb = estimate_resident_mb_for_path(&model_path);
            let name = model_path.display().to_string();
            let governor = current_memory_governor();
            if !state
                .residency
                .can_admit(RuntimeKind::Embedder, estimated_mb, &governor)
            {
                let _ = resp.send(Err("embedder admission denied by residency policy".into()));
                return ControlFlow::Continue;
            }
            let res = state
                .engine
                .load_embedder(EmbedLoadRequest { model_path, kind })
                .map_err(|e| format!("{:?}", e));
            if res.is_ok() {
                let admitted = state.residency.admit(
                    ResidentRuntime::new(
                        RuntimeKind::Embedder,
                        name,
                        estimated_mb,
                        chrono::Utc::now().timestamp(),
                    ),
                    &governor,
                );
                if !admitted {
                    state.engine.unload_embedder();
                    let _ = resp.send(Err(
                        "embedder residency registration denied after load".into()
                    ));
                    return ControlFlow::Continue;
                }
            }
            let _ = resp.send(res);
            ControlFlow::Continue
        }
        _ => unreachable!("handle_model_command: non-model cmd"),
    }
}

fn handle_status_command(state: &mut ControllerState, cmd: ControllerCmd) -> ControlFlow {
    match cmd {
        ControllerCmd::IsModelLoaded { resp } => {
            let _ = resp.send(state.engine.is_model_loaded());
            ControlFlow::Continue
        }
        ControllerCmd::GetActiveBackendName { resp } => {
            let _ = resp.send(state.engine.active_backend_name());
            ControlFlow::Continue
        }
        ControllerCmd::IsEmbedderLoaded { resp } => {
            let _ = resp.send(state.engine.is_embedder_loaded());
            ControlFlow::Continue
        }
        ControllerCmd::IsMmprojLoaded { resp } => {
            let supports_images = state.engine.does_model_support_images();
            let _ = resp.send(supports_images);
            ControlFlow::Continue
        }
        ControllerCmd::IsChatLoaded { chat_id, resp } => {
            let loaded = state.chats.contains_key(&chat_id);
            let _ = resp.send(loaded);
            ControlFlow::Continue
        }
        ControllerCmd::GetControllerMetrics { resp } => {
            let _ = resp.send(state.metrics.snapshot());
            ControlFlow::Continue
        }
        ControllerCmd::GetControllerRuntimeSnapshot { resp } => {
            let _ = resp.send(super::runtime_snapshot::build_runtime_snapshot(state));
            ControlFlow::Continue
        }
        ControllerCmd::GetControllerObservabilitySnapshot { resp } => {
            let _ = resp.send(super::observability_snapshot::build_observability_snapshot(
                state,
            ));
            ControlFlow::Continue
        }
        _ => unreachable!("handle_status_command: non-status cmd"),
    }
}

fn handle_chat_command(state: &mut ControllerState, cmd: ControllerCmd) -> ControlFlow {
    match cmd {
        ControllerCmd::StartChat {
            chat_id,
            messages,
            gen_spec,
            thinking,
            // The model id is a routing-only fence consumed upstream at the
            // flock dispatch seam; the controller serves with its loaded model.
            model_id: _,
            // Routing-only fit-gate size consumed at the flock seam.
            model_size_bytes: _,
            tx,
        } => {
            if let Some(chat) = state.chats.get_mut(&chat_id) {
                chat.last_used = std::time::Instant::now();
                chat.tx = tx.clone();
                if chat.state.is_generating() {
                    emit_must_deliver(
                        state.metrics.as_ref(),
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
                        emit_must_deliver(
                            state.metrics.as_ref(),
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                        return ControlFlow::Continue;
                    }
                    Ok(dropped) if dropped > 0 => {
                        emit_must_deliver(
                            state.metrics.as_ref(),
                            &tx,
                            ControllerEvent::ContextTruncated(dropped),
                        );
                    }
                    _ => {}
                }

                let pull_spec = apply_generation_defaults(&state.engine, gen_spec.clone());
                match chat.session.pull(pull_spec.clone()) {
                    Ok(puller) => {
                        attach_generating_puller(chat, puller, pull_spec);
                    }
                    Err(e) => {
                        emit_must_deliver(
                            state.metrics.as_ref(),
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                    }
                }
                return ControlFlow::Continue;
            }
            let max_active = state.max_active_chats();
            if state.chats.len() >= max_active
                && let Some((_k, victim)) = evict_lru(&mut state.chats)
            {
                terminate_runtime_owned(
                    &state.engine,
                    victim,
                    RuntimeOutcome::Completed(CompletionReason::Evicted),
                    state.metrics.as_ref(),
                );
            }
            match state.engine.start_session(SessionSpec {
                messages,
                thinking,
                ..Default::default()
            }) {
                Err(e) => {
                    emit_must_deliver(
                        state.metrics.as_ref(),
                        &tx,
                        ControllerEvent::Error {
                            code: "generation_error".into(),
                            message: format!("{:?}", e),
                        },
                    );
                }
                Ok(session) => {
                    // Every backend exposes initial_messages_dropped() via trait default 0 —
                    // no need to gate on backend-specific caps flag.
                    let dropped = session.initial_messages_dropped();
                    if dropped > 0 {
                        emit_must_deliver(
                            state.metrics.as_ref(),
                            &tx,
                            ControllerEvent::ContextTruncated(dropped),
                        );
                    }
                    let sid = session.id();
                    state.engine.hooks().register_with_id(
                        sid,
                        Arc::new(Forwarder {
                            sid,
                            tx: tx.clone(),
                            metrics: state.metrics.clone(),
                        }),
                    );
                    let pull_spec = apply_generation_defaults(&state.engine, gen_spec);
                    match session.pull(pull_spec.clone()) {
                        Err(e) => {
                            deregister_and_end_session(&state.engine, sid);
                            emit_must_deliver(
                                state.metrics.as_ref(),
                                &tx,
                                ControllerEvent::Error {
                                    code: "generation_error".into(),
                                    message: format!("{:?}", e),
                                },
                            );
                        }
                        Ok(puller) => {
                            let mut runtime = ChatRuntime {
                                session,
                                tx,
                                workload: WorkloadKind::PrimaryChat,
                                last_used: std::time::Instant::now(),
                                last_gen_spec: pull_spec.clone(),
                                state: ChatRunState::Idle,
                                health: Default::default(),
                            };
                            attach_generating_puller(&mut runtime, puller, pull_spec);
                            let evicted = state.chats.insert(chat_id, runtime);
                            if let Some(evicted_chat) = evicted {
                                terminate_runtime_owned(
                                    &state.engine,
                                    evicted_chat,
                                    RuntimeOutcome::Completed(CompletionReason::Evicted),
                                    state.metrics.as_ref(),
                                );
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
            // Routing-only fence consumed at the flock seam (see StartChat).
            model_id: _,
            // Routing-only fit-gate size consumed at the flock seam.
            model_size_bytes: _,
            tx,
        } => {
            if let Some(chat) = state.chats.get_mut(&chat_id) {
                chat.tx = tx.clone();
                if chat.state.is_generating() {
                    emit_must_deliver(
                        state.metrics.as_ref(),
                        &tx,
                        ControllerEvent::Error {
                            code: "generation_error".into(),
                            message: "chat already generating; pause/stop first".into(),
                        },
                    );
                    return ControlFlow::Continue;
                }
                match chat.session.append_messages(new_messages) {
                    Err(e) => {
                        emit_must_deliver(
                            state.metrics.as_ref(),
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                        return ControlFlow::Continue;
                    }
                    Ok(dropped) if dropped > 0 => {
                        emit_must_deliver(
                            state.metrics.as_ref(),
                            &tx,
                            ControllerEvent::ContextTruncated(dropped),
                        );
                    }
                    _ => {}
                }

                let pull_spec = apply_generation_defaults(&state.engine, gen_spec);
                match chat.session.pull(pull_spec.clone()) {
                    Ok(puller) => {
                        attach_generating_puller(chat, puller, pull_spec);
                    }
                    Err(e) => {
                        emit_must_deliver(
                            state.metrics.as_ref(),
                            &tx,
                            ControllerEvent::Error {
                                code: "generation_error".into(),
                                message: format!("{:?}", e),
                            },
                        );
                    }
                }
            } else {
                emit_must_deliver(
                    state.metrics.as_ref(),
                    &tx,
                    ControllerEvent::Error {
                        code: "not_found".into(),
                        message: format!("chat_id '{}' not found", chat_id),
                    },
                );
            }
            ControlFlow::Continue
        }
        ControllerCmd::StopChat { chat_id } => {
            terminate_runtime(
                &state.engine,
                &mut state.chats,
                &chat_id,
                RuntimeOutcome::Completed(CompletionReason::StoppedByUser),
                state.metrics.as_ref(),
            );
            ControlFlow::Continue
        }
        ControllerCmd::PauseChat { chat_id } => {
            if let Some(chat) = state.chats.get_mut(&chat_id) {
                chat.session.pause();
                chat.state = match std::mem::replace(&mut chat.state, ChatRunState::Idle) {
                    ChatRunState::Generating {
                        puller,
                        gen_started: _,
                        last_gen_spec,
                    } => {
                        drop(puller);
                        chat.last_gen_spec = last_gen_spec.clone();
                        ChatRunState::Paused { last_gen_spec }
                    }
                    other => other,
                };
            }
            ControlFlow::Continue
        }
        ControllerCmd::ResumeChat { chat_id } => {
            if let Some(chat) = state.chats.get_mut(&chat_id) {
                chat.session.resume();
                if !chat.state.is_generating() {
                    let spec = match &chat.state {
                        ChatRunState::Paused { last_gen_spec } => last_gen_spec.clone(),
                        _ => chat.last_gen_spec.clone(),
                    };
                    match chat.session.pull(spec.clone()) {
                        Ok(puller) => {
                            attach_generating_puller(chat, puller, spec);
                        }
                        Err(e) => {
                            emit_must_deliver(
                                state.metrics.as_ref(),
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
        _ => unreachable!("handle_chat_command: non-chat cmd"),
    }
}

fn handle_system_command(state: &mut ControllerState, cmd: ControllerCmd) -> ControlFlow {
    match cmd {
        ControllerCmd::SystemInfer {
            task,
            chat_id,
            messages,
            gen_spec,
            thinking,
            // Routing-only fence consumed at the flock seam (see StartChat).
            model_id: _,
            // Routing-only fit-gate size consumed at the flock seam.
            model_size_bytes: _,
            tx,
        } => {
            start_ephemeral(
                &mut state.engine,
                &mut state.chats,
                chat_id,
                task,
                task.suffix(),
                messages,
                gen_spec,
                thinking,
                state.metrics.clone(),
                tx,
            );
            ControlFlow::Continue
        }
        _ => unreachable!("handle_system_command: non-system cmd"),
    }
}

fn handle_utility_command(state: &mut ControllerState, cmd: ControllerCmd) -> ControlFlow {
    match cmd {
        ControllerCmd::GenerateEmbeddings { inputs, resp } => {
            let res = state
                .engine
                .generate_embeddings(&inputs)
                .map_err(|e| format!("{:?}", e));
            if res.is_ok() {
                state
                    .residency
                    .touch(RuntimeKind::Embedder, chrono::Utc::now().timestamp());
            }
            let _ = resp.send(res);
            ControlFlow::Continue
        }
        ControllerCmd::WarmModel { model_dir } => {
            state.engine.warm_model(model_dir);
            ControlFlow::Continue
        }
        ControllerCmd::Shutdown => ControlFlow::Break,
        _ => unreachable!("handle_utility_command: non-utility cmd"),
    }
}

pub(super) fn dispatch_cmd(cmd: ControllerCmd, state: &mut ControllerState) -> ControlFlow {
    match cmd {
        ControllerCmd::LoadModel { .. }
        | ControllerCmd::ApplySettings { .. }
        | ControllerCmd::LoadEmbedder { .. } => handle_model_command(state, cmd),
        ControllerCmd::IsModelLoaded { .. }
        | ControllerCmd::GetActiveBackendName { .. }
        | ControllerCmd::IsEmbedderLoaded { .. }
        | ControllerCmd::IsMmprojLoaded { .. }
        | ControllerCmd::IsChatLoaded { .. }
        | ControllerCmd::GetControllerMetrics { .. }
        | ControllerCmd::GetControllerRuntimeSnapshot { .. }
        | ControllerCmd::GetControllerObservabilitySnapshot { .. } => {
            handle_status_command(state, cmd)
        }
        ControllerCmd::StartChat { .. }
        | ControllerCmd::ContinueChat { .. }
        | ControllerCmd::StopChat { .. }
        | ControllerCmd::PauseChat { .. }
        | ControllerCmd::ResumeChat { .. } => handle_chat_command(state, cmd),
        ControllerCmd::SystemInfer { .. } => handle_system_command(state, cmd),
        ControllerCmd::GenerateEmbeddings { .. }
        | ControllerCmd::WarmModel { .. }
        | ControllerCmd::Shutdown => handle_utility_command(state, cmd),
    }
}

/// One scheduler pass over actively generating runtimes.
pub(super) fn tick_active_chats(state: &mut ControllerState, tick_busy: Duration) {
    let active_foreground = if state.chats.is_empty() {
        None
    } else {
        Some(RuntimeKind::Llm)
    };
    let evicted_for_pressure = state
        .residency
        .unload_for_pressure(&current_memory_governor(), active_foreground);
    for runtime in evicted_for_pressure {
        if matches!(runtime.kind, RuntimeKind::Embedder) {
            state.engine.unload_embedder();
        }
    }
    let evicted_idle = state
        .residency
        .evict_idle_helpers(chrono::Utc::now().timestamp(), &state.residency_policy);
    for runtime in evicted_idle {
        if matches!(runtime.kind, RuntimeKind::Embedder) {
            state.engine.unload_embedder();
        }
    }

    let generation_timeout = state.generation_timeout();
    let metrics = state.metrics.clone();
    let metrics_ref = metrics.as_ref();
    let sched_views = scheduler::views(&state.chats);
    let keys = scheduler::tick_order(&sched_views);
    let mut to_remove: Vec<String> = Vec::new();

    for id in keys {
        if let Some(chat) = state.chats.get_mut(&id) {
            if !chat.should_tick() {
                continue;
            }
            let timed_out = chat
                .state
                .generation_started_at()
                .is_some_and(|t| t.elapsed() > generation_timeout);
            if timed_out {
                let elapsed_secs = chat
                    .state
                    .generation_started_at()
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);
                tracing::warn!(
                    chat_id = %id,
                    elapsed_secs,
                    "generation timed out"
                );
                emit_must_deliver(
                    metrics_ref,
                    &chat.tx,
                    ControllerEvent::Error {
                        code: "timeout".into(),
                        message: format!(
                            "generation timed out after {}s",
                            generation_timeout.as_secs()
                        ),
                    },
                );
                metrics_ref
                    .generation_timeouts
                    .fetch_add(1, Ordering::Relaxed);
                chat.state = ChatRunState::Failed(FailureReason::Timeout);
                if chat.workload.is_system_task() {
                    to_remove.push(id.clone());
                }
                continue;
            }
            let step = {
                let puller = match &mut chat.state {
                    ChatRunState::Generating { puller, .. } => puller,
                    _ => continue,
                };
                catch_unwind(AssertUnwindSafe(|| puller.next()))
            };
            let mut receiver_dead = false;
            let mut emit = |event: ControllerEvent| {
                let r = match event {
                    ControllerEvent::Token(t) => {
                        emit_best_effort(metrics_ref, &chat.tx, ControllerEvent::Token(t))
                    }
                    ControllerEvent::MediaBoundary(b) => {
                        emit_best_effort(metrics_ref, &chat.tx, ControllerEvent::MediaBoundary(b))
                    }
                    other => emit_must_deliver(metrics_ref, &chat.tx, other),
                };
                if let EmitResult::Disconnected = r {
                    receiver_dead = true;
                }
            };

            match step {
                Err(_panic) => {
                    chat.health.decode_panics = chat.health.decode_panics.saturating_add(1);
                    emit(ControllerEvent::Error {
                        code: "session_poisoned".into(),
                        message: "inference panic: session state lost; restart chat".into(),
                    });
                    chat.state = ChatRunState::Failed(FailureReason::SessionPoisoned);
                    metrics_ref
                        .session_poisonings
                        .fetch_add(1, Ordering::Relaxed);
                }
                Ok(Some(Ok(TokenEvent::Token(tok)))) => {
                    // Successful token — reset consecutive-error streak.
                    chat.health.consecutive_errors = 0;
                    if !tok.text.is_empty() {
                        emit(ControllerEvent::Token(tok.text));
                        chat.last_used = std::time::Instant::now();
                    }
                }
                Ok(Some(Ok(TokenEvent::Eos))) => {
                    emit(ControllerEvent::Eos);
                    chat.state = ChatRunState::Completed(CompletionReason::Eos);
                }
                Ok(Some(Ok(TokenEvent::Stopped))) => {
                    emit(ControllerEvent::Stopped);
                    chat.state = ChatRunState::Completed(CompletionReason::StoppedByUser);
                }
                Ok(Some(Ok(TokenEvent::MediaBoundary(boundary)))) => {
                    emit(ControllerEvent::MediaBoundary(boundary));
                }
                Ok(Some(Ok(TokenEvent::ToolCall(tc)))) => {
                    emit(ControllerEvent::ToolCall(tc));
                    chat.last_used = std::time::Instant::now();
                }
                Ok(Some(Ok(TokenEvent::Paused))) => {}
                Ok(Some(Ok(TokenEvent::Special(_)))) => {}
                Ok(Some(Err(e))) => {
                    // Update generic session health: backend poison signal +
                    // error counter. Any backend whose is_poisoned() defaults
                    // to false (non-llama) surfaces only the error counter.
                    chat.health.poisoned_by_backend = chat.session.is_poisoned();
                    chat.health.consecutive_errors =
                        chat.health.consecutive_errors.saturating_add(1);
                    let (code, msg, failure) = if chat.health.is_unhealthy() {
                        (
                            "session_poisoned",
                            format!("session state lost: {:?}", e),
                            FailureReason::SessionPoisoned,
                        )
                    } else {
                        (
                            "generation_error",
                            format!("{:?}", e),
                            FailureReason::GenerationError,
                        )
                    };
                    emit(ControllerEvent::Error {
                        code: code.into(),
                        message: msg,
                    });
                    chat.state = ChatRunState::Failed(failure);
                    if matches!(failure, FailureReason::SessionPoisoned) {
                        metrics_ref
                            .session_poisonings
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                Ok(None) => {
                    emit(ControllerEvent::Stopped);
                    chat.state = ChatRunState::Completed(CompletionReason::StoppedByUser);
                }
            }
            let st = std::mem::replace(&mut chat.state, ChatRunState::Idle);
            chat.state = state_transitions::apply_receiver_disconnect_override(receiver_dead, st);
            if scheduler::should_cleanup(&chat.schedule_view()) {
                to_remove.push(id.clone());
            }
        }
    }

    for id in to_remove.drain(..) {
        if let Some(chat) = state.chats.remove(&id) {
            let outcome = runtime_outcome_from_state(&chat);
            terminate_runtime_owned(&state.engine, chat, outcome, metrics_ref);
        }
    }

    if !state.chats.is_empty() {
        std::thread::sleep(tick_busy);
    }
}
