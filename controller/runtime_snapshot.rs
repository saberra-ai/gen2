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
    ControllerRuntimeSnapshot { chats }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen2::controller::{CompletionReason, SystemTask, WorkloadKind};

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
