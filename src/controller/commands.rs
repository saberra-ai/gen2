//! Command dispatch and generation tick — split by domain (PR4).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::engine::EmbedLoadRequest;
use crate::generation::TokenEvent;
use crate::memory::current_memory_governor;
use crate::residency::{ResidentRuntime, RuntimeKind};
use crate::residency_policy::{
    estimate_resident_mb_for_path, estimate_resident_mb_for_path_offloaded,
};
use crate::session_rt::SessionSpec;

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

/// One rung of the model-load fallback ladder.
struct LoadRung {
    label: &'static str,
    drop_mmproj: bool,
    cpu_only: bool,
}

impl LoadRung {
    fn apply(&self, settings: &crate::engine::Settings) -> crate::engine::Settings {
        let mut s = settings.clone();
        if self.cpu_only {
            s.system.gpu_layers = Some(0);
        }
        s
    }
}

/// The ladder for a load request: as-requested, then (when a projector is
/// present) without the mmproj, then CPU-only weights (both degradations
/// combined on the last rung — each rung is strictly safer than the one
/// before). CPU-only is skipped when the caller already requested it.
fn load_fallback_rungs(settings: &crate::engine::Settings, has_mmproj: bool) -> Vec<LoadRung> {
    let mut rungs = vec![LoadRung {
        label: "requested",
        drop_mmproj: false,
        cpu_only: false,
    }];
    if has_mmproj {
        rungs.push(LoadRung {
            label: "no-mmproj",
            drop_mmproj: true,
            cpu_only: false,
        });
    }
    if settings.system.gpu_layers != Some(0) {
        rungs.push(LoadRung {
            label: "cpu-only",
            drop_mmproj: has_mmproj,
            cpu_only: true,
        });
    }
    rungs
}

struct LoadAttemptError {
    message: String,
    /// Fatal errors abort the ladder: retrying a corrupt file, unsupported
    /// architecture, or a host-level admission denial with a safer config
    /// cannot succeed and just burns a full model load.
    fatal: bool,
}

fn load_error_is_fatal(e: &crate::engine::ExecError) -> bool {
    use crate::engine::ExecError as E;
    matches!(
        e,
        E::InvalidModelFile(_)
            | E::UnsupportedArchitecture(_)
            | E::SettingsError(_)
            | E::InvalidArg(_)
            | E::FeatureUnsupported(_)
            | E::ModelNotLoaded
    )
}

/// One load attempt with a concrete config — the pre-ladder body of
/// `ControllerCmd::LoadModel`, unchanged in behavior: residency estimate →
/// admission → settings upload (deferred for lazily-created backends) →
/// engine load → residency registration.
#[allow(clippy::too_many_arguments)]
fn attempt_load(
    state: &mut ControllerState,
    model_path: std::path::PathBuf,
    mmproj_path: Option<std::path::PathBuf>,
    settings: crate::engine::Settings,
    api_key: Option<String>,
    api_format: Option<String>,
    loaded_file_bytes: &mut Option<u64>,
) -> Result<(), LoadAttemptError> {
    let fatal_str = |message: String| LoadAttemptError {
        message,
        fatal: true,
    };
    let from_exec = |e: crate::engine::ExecError| LoadAttemptError {
        message: format!("{e:?}"),
        fatal: load_error_is_fatal(&e),
    };

    let mut load_req = build_load_request(model_path, mmproj_path, &settings);
    load_req.api_key = api_key;
    load_req.api_format = api_format;
    *loaded_file_bytes = ControllerState::model_file_bytes_of(&load_req.model_path);
    let runtime_name = load_req.model_path.display().to_string();
    // Offloaded weights live in VRAM, not host RAM — don't deny a
    // GPU-bound model on a RAM-tight host (residency_policy.rs).
    let estimated_mb = estimate_resident_mb_for_path_offloaded(
        &load_req.model_path,
        load_req.model_params.gpu_layers,
    );
    let governor = current_memory_governor();
    if state.residency_policy.llm_swap_requires_unload && state.residency.llm.is_some() {
        state.engine.unload_model();
        let _ = state.residency.unload(RuntimeKind::Llm);
    }
    if !state
        .residency
        .can_admit(RuntimeKind::Llm, estimated_mb, &governor)
    {
        return Err(fatal_str("llm admission denied by residency policy".into()));
    }
    // Lazily-created backends (external-api, or a no-eager-init
    // build) have no backend to accept settings before the first
    // load — `upload_settings` returns ModelNotLoaded. Defer and
    // re-apply after `load_model` instantiates the backend, so
    // the caller's full settings (sampling etc. — the LoadRequest
    // only carries ctx/threads/gpu_layers) are never dropped.
    let deferred_settings = match state.engine.upload_settings(settings.clone()) {
        Ok(()) => None,
        Err(crate::engine::ExecError::ModelNotLoaded) => Some(settings),
        Err(e) => return Err(from_exec(e)),
    };
    state.engine.load_model(load_req).map_err(from_exec)?;
    if let Some(settings) = deferred_settings {
        state.engine.upload_settings(settings).map_err(from_exec)?;
    }
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
        return Err(fatal_str(
            "llm residency registration denied after load".into(),
        ));
    }
    Ok(())
}

/// Keepwarm: persist every quiesced chat's KV to the store. Generating
/// chats are skipped — mid-decode the transcript lacks the partial
/// reply, so a restore would diverge. Budget enforced after.
fn save_all_chat_kv(state: &ControllerState) {
    use crate::kv::store;
    if !store::keepwarm_enabled() || state.chats.is_empty() {
        return;
    }
    let dir = store::kv_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    for (chat_id, chat) in state.chats.iter() {
        if matches!(chat.state, ChatRunState::Generating { .. }) {
            continue;
        }
        let path = store::path_for_chat(&dir, chat_id);
        match chat
            .session
            .save_cache(crate::kv::KvSaveSpec::ToPath(path.clone()))
        {
            Ok(snap) => tracing::info!(
                target: "gen2::kv::keepwarm",
                chat_id = %chat_id,
                tokens = snap.tokens_covered,
                path = %path.display(),
                "saved chat KV for keepwarm"
            ),
            Err(e) => tracing::warn!(
                target: "gen2::kv::keepwarm",
                chat_id = %chat_id,
                error = ?e,
                "failed to save chat KV"
            ),
        }
    }
    store::enforce_budget(&dir);
}

/// Keepwarm idle-unload: when enabled (`PIO_LLM_IDLE_UNLOAD_SECS`), a
/// quiesced LLM past the idle budget saves its chats' KV and unloads —
/// freeing the memory while the saved state makes the next request a
/// sub-second resume instead of a cold prefill.
/// Memory-pressure eviction and idle unloading. See [`MAINTENANCE_INTERVAL`]
/// for why this is not on the per-token path.
fn run_residency_maintenance(state: &mut ControllerState) {
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
            state.utilities.unload_embedder();
        }
    }
    let evicted_idle = state
        .residency
        .evict_idle_helpers(chrono::Utc::now().timestamp(), &state.residency_policy);
    maybe_idle_unload_llm(state);
    for runtime in evicted_idle {
        if matches!(runtime.kind, RuntimeKind::Embedder) {
            state.utilities.unload_embedder();
        }
    }
}

/// Keepwarm idle-unload: when enabled (`PIO_LLM_IDLE_UNLOAD_SECS`), a
/// quiesced LLM past the idle budget saves its chats' KV and unloads —
/// freeing the memory while the saved state makes the next request a
/// sub-second resume instead of a cold prefill.
pub(super) fn maybe_idle_unload_llm(state: &mut ControllerState) {
    use crate::kv::store;
    if !store::keepwarm_enabled() || state.residency.llm.is_none() {
        return;
    }
    let Some(timeout) = std::env::var("PIO_LLM_IDLE_UNLOAD_SECS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|t| *t > 0)
    else {
        return;
    };
    let now = chrono::Utc::now().timestamp();
    let any_active = state
        .chats
        .values()
        .any(|c| matches!(c.state, ChatRunState::Generating { .. }));
    if any_active {
        state.last_llm_activity_unix = now;
        return;
    }
    if now - state.last_llm_activity_unix < timeout {
        return;
    }
    tracing::info!(
        target: "gen2::kv::keepwarm",
        idle_secs = now - state.last_llm_activity_unix,
        "idle-unloading LLM (keepwarm)"
    );
    save_all_chat_kv(state);
    for (_, chat) in state.chats.drain() {
        terminate_runtime_owned(
            &state.engine,
            chat,
            RuntimeOutcome::Completed(CompletionReason::Evicted),
            state.metrics.as_ref(),
        );
    }
    state.engine.unload_model();
    if let Some(rt) = state.residency.unload(RuntimeKind::Llm) {
        state.idle_unloaded_llm = Some((rt.name, rt.estimated_resident_mb));
    }
    state.loaded_model_file_bytes = None;
}

/// Keepwarm wake: an idle-unloaded model reloads on demand when a chat
/// arrives. The backend retains its last LoadRequest across unload, and
/// the saved residency identity re-admits the same runtime.
fn maybe_wake_llm(state: &mut ControllerState) {
    if state.engine.is_model_loaded() {
        return;
    }
    let Some((name, mb)) = state.idle_unloaded_llm.clone() else {
        return;
    };
    match state.engine.reload_model() {
        Ok(()) => {
            let governor = current_memory_governor();
            let now = chrono::Utc::now().timestamp();
            state.residency.admit(
                ResidentRuntime::new(RuntimeKind::Llm, name, mb, now),
                &governor,
            );
            state.idle_unloaded_llm = None;
            state.last_llm_activity_unix = now;
            tracing::info!(
                target: "gen2::kv::keepwarm",
                "woke idle-unloaded LLM on demand"
            );
        }
        Err(e) => tracing::warn!(
            target: "gen2::kv::keepwarm",
            error = ?e,
            "failed to wake idle-unloaded LLM"
        ),
    }
}

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
            // remote fit gate can source a real footprint (Part A of VRAM-aware
            // routing). Computed from the same path the engine stats at load
            // (`metadata().len()` in `engine::validate_model_file`); replaced on
            // every successful reload, cleared on failure.
            let mut loaded_file_bytes: Option<u64> = None;
            // Keepwarm: the incoming load will unload the current model
            // (swap policy) — persist quiesced chats' KV first so their
            // conversations resume warm if this model comes back.
            save_all_chat_kv(state);
            // Fallback ladder: retry a failed load with progressively safer
            // configs before surfacing an error. Rung 1 drops the mmproj (a
            // broken vision projector shouldn't brick text chat); rung 2 moves
            // weights to CPU (rescues VRAM/Metal buffer-alloc failures).
            // Fatal classes (corrupt file, unsupported arch, bad settings,
            // admission denials) never retry — see `load_error_is_fatal`.
            let rungs = load_fallback_rungs(&settings, mmproj_path.is_some());
            let mut rung_history: Vec<String> = Vec::new();
            let mut succeeded_as: Option<crate::engine::LoadOutcome> = None;
            let mut r: Result<(), String> = Err("load ladder never ran".into());
            for (rung_idx, rung) in rungs.iter().enumerate() {
                let attempt_settings = rung.apply(&settings);
                let attempt_mmproj = if rung.drop_mmproj {
                    None
                } else {
                    mmproj_path.clone()
                };
                let attempt = attempt_load(
                    state,
                    model_path.clone(),
                    attempt_mmproj,
                    attempt_settings,
                    api_key.clone(),
                    api_format.clone(),
                    &mut loaded_file_bytes,
                );
                match attempt {
                    Ok(()) => {
                        let mut outcome = crate::engine::LoadOutcome::default();
                        if rung.drop_mmproj {
                            outcome
                                .degraded
                                .push(crate::engine::Degraded::VisionProjector);
                        }
                        if rung.cpu_only {
                            outcome.degraded.push(crate::engine::Degraded::GpuOffload);
                        }
                        succeeded_as = Some(outcome);
                        if rung_idx > 0 {
                            tracing::warn!(
                                target: "gen2::load_ladder",
                                rung = rung.label,
                                history = ?rung_history,
                                "model load rescued by fallback rung"
                            );
                        }
                        r = Ok(());
                        break;
                    }
                    Err(LoadAttemptError { message, fatal }) => {
                        rung_history.push(format!("{}: {}", rung.label, message));
                        tracing::warn!(
                            target: "gen2::load_ladder",
                            rung = rung.label,
                            fatal,
                            error = %message,
                            "model load attempt failed"
                        );
                        r = Err(if rung_history.len() > 1 || rung_idx + 1 == rungs.len() {
                            format!(
                                "model load failed after {} attempt(s): [{}]",
                                rung_history.len(),
                                rung_history.join(" | ")
                            )
                        } else {
                            message
                        });
                        if fatal {
                            break;
                        }
                    }
                }
            }
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
            // Report what the load actually did. A ladder that rescues a
            // failing load is the right instinct, and `Ok(())` alone cannot
            // say whether the caller got the projector and the offload it
            // asked for.
            let _ = resp.send(r.map(|()| succeeded_as.unwrap_or_default()));
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
            // The worker owns it, so the embedder no longer has to be
            // implemented by whichever backend holds the chat model.
            let res = state
                .utilities
                .load_embedder(EmbedLoadRequest { model_path, kind }, estimated_mb);
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
                    state.utilities.unload_embedder();
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
        ControllerCmd::GetCapabilities { resp } => {
            let _ = resp.send(state.engine.capabilities());
            ControlFlow::Continue
        }
        ControllerCmd::UnloadModel { resp } => {
            state.engine.unload_model();
            let _ = resp.send(());
            ControlFlow::Continue
        }
        ControllerCmd::ReloadModel { resp } => {
            let _ = resp.send(state.engine.reload_model().map_err(|e| e.to_string()));
            ControlFlow::Continue
        }
        ControllerCmd::GetActiveBackendName { resp } => {
            let _ = resp.send(state.engine.active_backend_name());
            ControlFlow::Continue
        }
        ControllerCmd::IsEmbedderLoaded { resp } => {
            // The worker is the authority now: asking the generation backend
            // would report on a helper it no longer owns.
            let _ = resp.send(state.utilities.status().embedder.is_some());
            ControlFlow::Continue
        }
        ControllerCmd::GetUtilityStatus { resp } => {
            let _ = resp.send(state.utilities.status());
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
            // remote dispatch seam; the controller serves with its loaded model.
            model_id: _,
            // Routing-only fit-gate size consumed at the remote seam.
            model_size_bytes: _,
            tools,
            tx,
        } => {
            if let Some(chat) = state.chats.get_mut(&chat_id) {
                if chat.state.is_generating() {
                    chat.tx = tx.clone();
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

                // `StartChat` means "this is the conversation", not "add to
                // whatever you have". The caller only sends it when its own
                // state says the conversation must be built afresh — after an
                // edit, a clear, a model swap, or a tool-set change — and the
                // messages it carries are the authoritative transcript.
                //
                // Appending the last of them to a runtime that still holds the
                // old history, which is what this used to do, leaves the model
                // working from messages the caller deleted, and silently drops
                // the tools and thinking mode the new prefix was supposed to
                // carry. Retire the runtime and build it again below.
                if let Some(stale) = state.chats.remove(&chat_id) {
                    terminate_runtime_owned(
                        &state.engine,
                        stale,
                        RuntimeOutcome::Completed(CompletionReason::Evicted),
                        state.metrics.as_ref(),
                    );
                }
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
            // Keepwarm: wake an idle-unloaded model, then a saved KV blob
            // for this chat resumes without re-prefilling the whole
            // history. Lenient — any identity or transcript mismatch
            // falls back to the cold path inside the backend.
            maybe_wake_llm(state);
            let cache = if crate::kv::store::keepwarm_enabled() {
                let dir = crate::kv::store::kv_dir();
                let cand = crate::kv::store::candidate_for_chat(&dir, &chat_id);
                if cand.is_none() {
                    tracing::info!(
                        target: "gen2::kv::keepwarm",
                        chat_id = %chat_id,
                        "no KV candidate for chat — cold start"
                    );
                }
                cand.map(crate::kv::KvLoadSpec::Lenient)
            } else {
                None
            };
            state.last_llm_activity_unix = chrono::Utc::now().timestamp();
            // How much of the caller's conversation this runtime will hold, so
            // the acknowledgement can correct a stale delivered-count rather
            // than leaving the next turn to send a suffix the backend lacks.
            let delivered_count = messages.len();
            match state.engine.start_session(SessionSpec {
                messages,
                thinking,
                cache,
                tools,
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
                            // Everything that could fail has succeeded, so the
                            // caller may now record what the engine holds.
                            emit_must_deliver(
                                state.metrics.as_ref(),
                                &runtime.tx,
                                ControllerEvent::Accepted {
                                    delivered: delivered_count,
                                },
                            );
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
            transcript,
            gen_spec,
            // Routing hints for a remote dispatch. The local loop already
            // knows which model it has, but they are carried through when a
            // missing runtime forces a rebuild below.
            model_id,
            model_size_bytes,
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
                let appended = new_messages.len();
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
                        emit_must_deliver(
                            state.metrics.as_ref(),
                            &chat.tx,
                            ControllerEvent::Accepted {
                                // A continuation adds to what the runtime
                                // already had; the facade tracks the total, so
                                // it adds this to its own count.
                                delivered: appended,
                            },
                        );
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
                // The runtime is gone — evicted for capacity, unloaded when
                // idle, or lost to a reload. None of that is the caller's
                // doing, and a conversation it still holds must not stop
                // working because of it. Rebuild from the transcript it sent
                // and carry on; the cost is one prefill.
                let mut full = transcript;
                full.extend(new_messages);
                tracing::debug!(
                    chat_id = %chat_id,
                    messages = full.len(),
                    "continue found no runtime — rebuilding from the caller's transcript"
                );
                return dispatch_cmd(
                    ControllerCmd::StartChat {
                        chat_id,
                        messages: full,
                        gen_spec,
                        thinking: crate::generation::ThinkingMode::default(),
                        model_id,
                        model_size_bytes,
                        tools: None,
                        tx,
                    },
                    state,
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
            // Routing hints for a remote dispatch; the local loop already
            // knows which model it has (see StartChat).
            model_id: _,
            model_size_bytes: _,
            required_node: _,
            tx,
        } => {
            let label = task.label().to_string();
            start_ephemeral(
                &mut state.engine,
                &mut state.chats,
                chat_id,
                task,
                &label,
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
            // Handed to the utility worker, which answers `resp` itself. This
            // thread does not wait: embedding a long batch here would stop
            // chat token scheduling for the whole call, and the helpers still
            // to come — transcription, OCR — take seconds rather than
            // milliseconds.
            match state.utilities.embed_forwarding(inputs, resp.clone()) {
                Ok(()) => {
                    // Touched on acceptance rather than completion. The reply
                    // never comes back through here, so this is the last
                    // moment the controller knows the helper was used — and a
                    // request in flight is exactly when eviction would hurt.
                    state
                        .residency
                        .touch(RuntimeKind::Embedder, chrono::Utc::now().timestamp());
                }
                Err(e) => {
                    let _ = resp.send(Err(e));
                }
            }
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
        | ControllerCmd::GetCapabilities { .. }
        | ControllerCmd::UnloadModel { .. }
        | ControllerCmd::ReloadModel { .. }
        | ControllerCmd::GetActiveBackendName { .. }
        | ControllerCmd::IsEmbedderLoaded { .. }
        | ControllerCmd::GetUtilityStatus { .. }
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
/// How often residency maintenance runs.
///
/// It probes memory pressure and evicts idle runtimes — decisions measured in
/// seconds. The tick it used to live in pulls one token per call, so it ran at
/// token frequency, and a `getrusage`, two eviction scans and a clock read on
/// every token cost roughly a sixth of decode throughput. Frequent enough that
/// pressure is still noticed promptly, rare enough to be free.
const MAINTENANCE_INTERVAL: Duration = Duration::from_millis(250);

pub(super) fn tick_active_chats(state: &mut ControllerState, tick_busy: Duration) {
    let due = state
        .last_maintenance
        .is_none_or(|at| at.elapsed() >= MAINTENANCE_INTERVAL);
    if due {
        state.last_maintenance = Some(Instant::now());
        run_residency_maintenance(state);
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

    // Zero is the default and means "do not pace"; sleeping for it is not
    // free. `thread::sleep(Duration::ZERO)` still enters the kernel and yields
    // the thread, and being rescheduled can cost a scheduler quantum — paid
    // once per token, because the tick pulls one.
    if !tick_busy.is_zero() && !state.chats.is_empty() {
        std::thread::sleep(tick_busy);
    }
}

#[cfg(test)]
mod ladder_tests {
    use super::*;
    use crate::engine::{ExecError, Settings};

    #[test]
    fn rungs_full_ladder_with_mmproj_and_gpu() {
        let rungs = load_fallback_rungs(&Settings::default(), true);
        let labels: Vec<_> = rungs.iter().map(|r| r.label).collect();
        assert_eq!(labels, vec!["requested", "no-mmproj", "cpu-only"]);
        // Last rung is strictly safest: both degradations.
        assert!(rungs[2].drop_mmproj && rungs[2].cpu_only);
    }

    #[test]
    fn rungs_no_mmproj_skips_projector_rung() {
        let rungs = load_fallback_rungs(&Settings::default(), false);
        let labels: Vec<_> = rungs.iter().map(|r| r.label).collect();
        assert_eq!(labels, vec!["requested", "cpu-only"]);
    }

    #[test]
    fn rungs_cpu_request_has_no_cpu_rung() {
        let mut s = Settings::default();
        s.system.gpu_layers = Some(0);
        let rungs = load_fallback_rungs(&s, false);
        let labels: Vec<_> = rungs.iter().map(|r| r.label).collect();
        assert_eq!(labels, vec!["requested"]);
    }

    #[test]
    fn cpu_rung_zeroes_gpu_layers_only() {
        let mut s = Settings::default();
        s.system.gpu_layers = Some(99);
        s.system.ctx_size = Some(4096);
        let rung = LoadRung {
            label: "cpu-only",
            drop_mmproj: false,
            cpu_only: true,
        };
        let out = rung.apply(&s);
        assert_eq!(out.system.gpu_layers, Some(0));
        assert_eq!(out.system.ctx_size, Some(4096), "other settings untouched");
    }

    #[test]
    fn fatal_classes_never_retry() {
        for e in [
            ExecError::InvalidModelFile("bad magic".into()),
            ExecError::UnsupportedArchitecture("qwen35".into()),
            ExecError::SettingsError("nope".into()),
            ExecError::ModelNotLoaded,
        ] {
            assert!(load_error_is_fatal(&e), "{e:?} must be fatal");
        }
    }

    #[test]
    fn oom_class_retries() {
        for e in [
            ExecError::OutOfMemory("metal buffer".into()),
            ExecError::Io("mmap failed".into()),
            ExecError::MmprojIncompatible("clip mismatch"),
        ] {
            assert!(!load_error_is_fatal(&e), "{e:?} must be retryable");
        }
    }
}
