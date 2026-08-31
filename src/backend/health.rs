//! Generic session health observable by the controller.
//!
//! Replaces backend-specific poison detection (llama's `is_poisoned()`) with a
//! generic signal every backend feeds. Backend-specific evidence (FFI-panic
//! detection, network failures, etc.) becomes inputs rather than the whole
//! signal.

/// Per-session health aggregate tracked on `ChatRuntime` in the controller.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SessionHealth {
    /// Number of caught panics inside the decode loop for this session.
    pub decode_panics: u32,
    /// Consecutive token/pull errors — reset on any successful token.
    pub consecutive_errors: u32,
    /// Latest signal from the backend itself (e.g. llama `is_poisoned()`).
    pub poisoned_by_backend: bool,
}

impl SessionHealth {
    /// Returns `true` when the session should be treated as unhealthy and the
    /// controller should surface `FailureReason::SessionPoisoned` rather than
    /// `GenerationError`.
    pub fn is_unhealthy(&self) -> bool {
        self.poisoned_by_backend || self.decode_panics > 0 || self.consecutive_errors >= 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_healthy() {
        let h = SessionHealth::default();
        assert!(!h.is_unhealthy());
    }

    #[test]
    fn backend_poison_marks_unhealthy() {
        let h = SessionHealth {
            poisoned_by_backend: true,
            ..Default::default()
        };
        assert!(h.is_unhealthy());
    }

    #[test]
    fn panic_marks_unhealthy() {
        let h = SessionHealth {
            decode_panics: 1,
            ..Default::default()
        };
        assert!(h.is_unhealthy());
    }

    #[test]
    fn three_consecutive_errors_mark_unhealthy() {
        let h = SessionHealth {
            consecutive_errors: 3,
            ..Default::default()
        };
        assert!(h.is_unhealthy());
    }

    #[test]
    fn two_consecutive_errors_still_healthy() {
        let h = SessionHealth {
            consecutive_errors: 2,
            ..Default::default()
        };
        assert!(!h.is_unhealthy());
    }
}
