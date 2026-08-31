//! Small, testable pieces of the chat run state machine (PR3).
//!
//! The controller `run_loop` owns timing and I/O; this module holds pure
//! transitions that must stay aligned with observable outcomes.

use super::{ChatRunState, CompletionReason};

/// After a tick step, force `ReceiverDropped` when the UI channel is gone.
#[must_use]
pub(super) fn apply_receiver_disconnect_override(
    receiver_dead: bool,
    state: ChatRunState,
) -> ChatRunState {
    if receiver_dead {
        ChatRunState::Completed(CompletionReason::ReceiverDropped)
    } else {
        state
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ChatRunState, CompletionReason, FailureReason};
    use super::apply_receiver_disconnect_override;
    use crate::generation::GenSpec;

    #[test]
    fn allows_user_pull_start_permits_idle_paused_and_terminal() {
        let spec = GenSpec::default();
        assert!(ChatRunState::Idle.allows_user_pull_start());
        assert!(
            ChatRunState::Paused {
                last_gen_spec: spec.clone()
            }
            .allows_user_pull_start()
        );
        assert!(ChatRunState::Completed(CompletionReason::Eos).allows_user_pull_start());
        assert!(ChatRunState::Failed(FailureReason::Timeout).allows_user_pull_start());
    }

    #[test]
    fn receiver_disconnect_overrides_completed_eos() {
        let s = apply_receiver_disconnect_override(
            true,
            ChatRunState::Completed(CompletionReason::Eos),
        );
        assert!(matches!(
            s,
            ChatRunState::Completed(CompletionReason::ReceiverDropped)
        ));
    }

    #[test]
    fn receiver_disconnect_no_override_when_false() {
        let s = apply_receiver_disconnect_override(
            false,
            ChatRunState::Completed(CompletionReason::Eos),
        );
        assert!(matches!(s, ChatRunState::Completed(CompletionReason::Eos)));
    }

    #[test]
    fn generation_started_at_is_none_unless_generating() {
        assert!(ChatRunState::Idle.generation_started_at().is_none());
        assert!(
            ChatRunState::Paused {
                last_gen_spec: GenSpec::default()
            }
            .generation_started_at()
            .is_none()
        );
        assert!(
            ChatRunState::Completed(CompletionReason::StoppedByUser)
                .generation_started_at()
                .is_none()
        );
    }
}
