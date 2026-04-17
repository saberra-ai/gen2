//! Controller-side delivery policy and counters (PR5).
//!
//! **Best-effort:** token streaming (and media boundaries) may be dropped when the UI channel is full.
//! **Must-deliver:** lifecycle and routing events use bounded spin + `try_send` before giving up.
//!
//! Full policy and invariants: `docs/gen2-controller-runtime-contract.md` (repository root).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};

use serde::{Deserialize, Serialize};

use super::ControllerEvent;
use super::observability::EmitResult;

/// Monotonic counters for controller-visible backpressure and termination causes.
#[derive(Debug, Default)]
pub struct ControllerMetrics {
    pub dropped_token_events: AtomicU64,
    /// Must-deliver events that could not be queued after bounded retries (channel saturated).
    pub dropped_must_deliver_events: AtomicU64,
    pub receiver_disconnects: AtomicU64,
    pub generation_timeouts: AtomicU64,
    pub session_poisonings: AtomicU64,
    pub evictions: AtomicU64,
    pub model_reload_terminations: AtomicU64,
}

impl ControllerMetrics {
    /// Monotonic counter snapshot for observability (`ControllerCmd::GetControllerMetrics`).
    pub fn snapshot(&self) -> ControllerMetricsSnapshot {
        ControllerMetricsSnapshot {
            dropped_token_events: self.dropped_token_events.load(Ordering::Relaxed),
            dropped_must_deliver_events: self.dropped_must_deliver_events.load(Ordering::Relaxed),
            receiver_disconnects: self.receiver_disconnects.load(Ordering::Relaxed),
            generation_timeouts: self.generation_timeouts.load(Ordering::Relaxed),
            session_poisonings: self.session_poisonings.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            model_reload_terminations: self.model_reload_terminations.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ControllerMetricsSnapshot {
    pub dropped_token_events: u64,
    pub dropped_must_deliver_events: u64,
    pub receiver_disconnects: u64,
    pub generation_timeouts: u64,
    pub session_poisonings: u64,
    pub evictions: u64,
    pub model_reload_terminations: u64,
}

/// `Token` / `MediaBoundary` — may drop on full channel; counts streaming drops.
pub(super) fn emit_best_effort(
    metrics: &ControllerMetrics,
    tx: &SyncSender<ControllerEvent>,
    event: ControllerEvent,
) -> EmitResult {
    let count_drop = matches!(
        &event,
        ControllerEvent::Token(_) | ControllerEvent::MediaBoundary(_)
    );
    let r = try_send_event(tx, event);
    if count_drop && matches!(r, EmitResult::Full) {
        metrics.dropped_token_events.fetch_add(1, Ordering::Relaxed);
    }
    record_disconnect(metrics, &r);
    r
}

/// Errors, EOS, stop signals, stats — spin briefly before dropping.
pub(super) fn emit_must_deliver(
    metrics: &ControllerMetrics,
    tx: &SyncSender<ControllerEvent>,
    event: ControllerEvent,
) -> EmitResult {
    const SPINS: u32 = 64;
    let mut ev = event;
    for _ in 0..SPINS {
        match tx.try_send(ev) {
            Ok(()) => return EmitResult::Sent,
            Err(TrySendError::Disconnected(_)) => {
                metrics.receiver_disconnects.fetch_add(1, Ordering::Relaxed);
                return EmitResult::Disconnected;
            }
            Err(TrySendError::Full(v)) => {
                ev = v;
                std::thread::yield_now();
            }
        }
    }
    metrics
        .dropped_must_deliver_events
        .fetch_add(1, Ordering::Relaxed);
    EmitResult::Full
}

fn try_send_event(tx: &SyncSender<ControllerEvent>, event: ControllerEvent) -> EmitResult {
    match tx.try_send(event) {
        Ok(()) => EmitResult::Sent,
        Err(TrySendError::Full(_)) => EmitResult::Full,
        Err(TrySendError::Disconnected(_)) => EmitResult::Disconnected,
    }
}

fn record_disconnect(metrics: &ControllerMetrics, r: &EmitResult) {
    if matches!(r, EmitResult::Disconnected) {
        metrics.receiver_disconnects.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    use crate::gen2::controller::EVENT_CHANNEL_CAPACITY;

    #[test]
    fn best_effort_full_channel_increments_token_drop_metric() {
        let metrics = ControllerMetrics::default();
        let (tx, _rx) = sync_channel::<ControllerEvent>(1);
        assert!(tx.try_send(ControllerEvent::Token("x".into())).is_ok());
        let r = emit_best_effort(&metrics, &tx, ControllerEvent::Token("y".into()));
        assert!(matches!(r, EmitResult::Full));
        assert_eq!(metrics.dropped_token_events.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn must_deliver_empty_channel_sends() {
        let metrics = ControllerMetrics::default();
        let (tx, rx) = sync_channel::<ControllerEvent>(EVENT_CHANNEL_CAPACITY);
        let r = emit_must_deliver(&metrics, &tx, ControllerEvent::Eos);
        assert!(matches!(r, EmitResult::Sent));
        assert_eq!(
            metrics.dropped_must_deliver_events.load(Ordering::Relaxed),
            0
        );
        assert!(matches!(rx.try_recv(), Ok(ControllerEvent::Eos)));
    }

    #[test]
    fn must_deliver_disconnected_increments_receiver_metric() {
        let metrics = ControllerMetrics::default();
        let (tx, rx) = sync_channel::<ControllerEvent>(2);
        drop(rx);
        let r = emit_must_deliver(&metrics, &tx, ControllerEvent::Eos);
        assert!(matches!(r, EmitResult::Disconnected));
        assert_eq!(metrics.receiver_disconnects.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn terminal_outcome_state_survives_best_effort_token_drop() {
        // Metric + state semantics: a full channel drops a token but EOS still queues after drain.
        let metrics = ControllerMetrics::default();
        let (tx, rx) = sync_channel::<ControllerEvent>(1);
        assert!(tx.try_send(ControllerEvent::Token("fill".into())).is_ok());
        let _ = emit_best_effort(&metrics, &tx, ControllerEvent::Token("drop".into()));
        assert_eq!(metrics.dropped_token_events.load(Ordering::Relaxed), 1);
        // Drain one token so must-deliver can succeed.
        assert!(matches!(rx.try_recv(), Ok(ControllerEvent::Token(_))));
        let r = emit_must_deliver(&metrics, &tx, ControllerEvent::Eos);
        assert!(matches!(r, EmitResult::Sent));
        assert!(matches!(rx.try_recv(), Ok(ControllerEvent::Eos)));
    }
}
