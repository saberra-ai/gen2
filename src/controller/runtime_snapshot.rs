//! Serializable view of active controller runtimes (PR7 — introspection / debug).
//!
//! Omits pullers, channels, and message payloads; stable sort by `chat_id`.

use serde::{Deserialize, Serialize};

use super::state::ControllerState;
use super::{ChatRunState, CompletionReason, FailureReason, WorkloadKind};

/// One row per loaded chat runtime, sorted by `chat_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ControllerRuntimeSnapshot {
    pub chats: Vec<ActiveChatSnapshot>,
    /// Lowercase GGUF `general.architecture` (or HF `model_type`) of the
    /// currently-loaded model, if any. Used by callers that need to
    /// route per-architecture chat-stream behavior — e.g. the daemon's
    /// turn streamer picks `ChannelMarkers::gemma4()` when this reads
    /// `gemma4`, so the live wire splits visible vs reasoning content.
    /// `None` when no model is loaded.
    #[serde(default)]
    pub loaded_model_architecture: Option<String>,
    /// Whole-model on-disk byte size of the currently-loaded primary LLM,
    /// captured at `LoadModel` time from `metadata().len()`. Sources the
    /// flock VRAM/RAM fit gate (Part A of VRAM-aware routing) via
    /// `FlockHandle::resolve_route_model_size` — a peer can advertise the
    /// real footprint of what it has loaded so a request that names that
    /// model is gated against peers that can't hold it. `None` when no model
    /// is loaded, or when the model is a directory bundle (MLX/ONNX) whose
    /// size isn't a single `metadata().len()`.
    #[serde(default)]
    pub loaded_model_file_bytes: Option<u64>,
}

/// Observable fields for a single [`super::ChatRuntime`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ActiveChatSnapshot {
    pub chat_id: String,
    pub session_id: u64,
    pub workload: WorkloadKind,
    pub lifecycle: RuntimeLifecycleSnapshot,
}

/// Lifecycle without engine internals (no puller / gen spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum RuntimeLifecycleSnapshot {
    Idle,
    Generating { elapsed_ms: u64 },
    Paused,
    Completed(CompletionReason),
    Failed(FailureReason),
}

fn lifecycle_from_state(state: &ChatRunState) -> RuntimeLifecycleSnapshot {
    match state {
        ChatRunState::Idle => RuntimeLifecycleSnapshot::Idle,
        ChatRunState::Generating { gen_started, .. } => RuntimeLifecycleSnapshot::Generating {
            elapsed_ms: gen_started.elapsed().as_millis() as u64,
        },
        ChatRunState::Paused { .. } => RuntimeLifecycleSnapshot::Paused,
        ChatRunState::Completed(r) => RuntimeLifecycleSnapshot::Completed(*r),
        ChatRunState::Failed(f) => RuntimeLifecycleSnapshot::Failed(*f),
    }
}

pub(super) fn build_runtime_snapshot(state: &ControllerState) -> ControllerRuntimeSnapshot {
    let mut chats: Vec<ActiveChatSnapshot> = state
        .chats
        .iter()
        .map(|(chat_id, chat)| ActiveChatSnapshot {
            chat_id: chat_id.clone(),
            session_id: chat.session.id(),
            workload: chat.workload,
            lifecycle: lifecycle_from_state(&chat.state),
        })
        .collect();
    chats.sort_by(|a, b| a.chat_id.cmp(&b.chat_id));
    let loaded_model_architecture = state.engine.bundle_architecture();
    ControllerRuntimeSnapshot {
        chats,
        loaded_model_architecture,
        loaded_model_file_bytes: state.loaded_model_file_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{CompletionReason, SystemTask, WorkloadKind};

    #[test]
    fn empty_controller_runtime_snapshot_roundtrips_json() {
        let snap = ControllerRuntimeSnapshot::default();
        let json = serde_json::to_string(&snap).expect("serialize");
        let back: ControllerRuntimeSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(snap, back);
    }

    #[test]
    fn controller_runtime_snapshot_row_roundtrips_json() {
        let snap = ControllerRuntimeSnapshot {
            chats: vec![
                ActiveChatSnapshot {
                    chat_id: "b".into(),
                    session_id: 2,
                    workload: WorkloadKind::PrimaryChat,
                    lifecycle: RuntimeLifecycleSnapshot::Generating { elapsed_ms: 7 },
                },
                ActiveChatSnapshot {
                    chat_id: "a".into(),
                    session_id: 1,
                    workload: WorkloadKind::SystemTask(SystemTask::Title),
                    lifecycle: RuntimeLifecycleSnapshot::Completed(CompletionReason::Eos),
                },
            ],
            loaded_model_architecture: Some("gemma4".into()),
            loaded_model_file_bytes: Some(8 << 30),
        };
        let json = serde_json::to_string(&snap).expect("serialize");
        let back: ControllerRuntimeSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(snap, back);
        // Wire / TS bindings must match serde's externally-tagged enum encoding.
        assert_eq!(
            serde_json::to_string(&WorkloadKind::PrimaryChat).unwrap(),
            "\"PrimaryChat\""
        );
        assert_eq!(
            serde_json::to_string(&WorkloadKind::SystemTask(SystemTask::Title)).unwrap(),
            "{\"SystemTask\":\"Title\"}"
        );
        assert_eq!(
            serde_json::to_string(&RuntimeLifecycleSnapshot::Completed(CompletionReason::Eos))
                .unwrap(),
            "{\"Completed\":\"Eos\"}"
        );
    }
}
