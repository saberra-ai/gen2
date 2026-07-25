//! Controller observability helpers — hook forwarder and bounded-emit result.
//! Moved out of `mod.rs` in Phase 6; no behavior change.

use std::sync::Arc;
use std::sync::mpsc::SyncSender;

use super::{ControllerEvent, metrics};
use crate::gen2::engine::{HookEvent, HookListener};

/// Hook listener forwarding final stats for a specific session id back to the UI channel.
pub(super) struct Forwarder {
    pub(super) sid: u64,
    pub(super) tx: SyncSender<ControllerEvent>,
    pub(super) metrics: Arc<metrics::ControllerMetrics>,
}

impl std::fmt::Debug for Forwarder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Forwarder")
    }
}

impl HookListener for Forwarder {
    fn on_event(&self, ev: &HookEvent) {
        if let HookEvent::FinalStats { session_id, stats } = ev
            && *session_id == self.sid
        {
            metrics::emit_must_deliver(
                self.metrics.as_ref(),
                &self.tx,
                ControllerEvent::FinalStats(stats.clone()),
            );
        }
        // Healing telemetry (unsloth-adoption 12): per-generation
        // tool-call outcome counts, INFO-logged so field logs show a
        // model whose calls keep falling through before users report
        // "the tool did nothing". Arch rides the surrounding load spans.
        if let HookEvent::ToolCallOutcomes { session_id, tally } = ev
            && *session_id == self.sid
        {
            tracing::info!(
                target: "pio::gen2::tool_healing",
                session_id,
                clean = tally.clean,
                gemma_dialect = tally.gemma_dialect,
                commaless_array = tally.commaless_array,
                fell_through = tally.fell_through,
                "tool-call outcomes for generation"
            );
        }
    }
}

/// Result of bounded emit — tells the caller whether the receiver is still alive.
pub(super) enum EmitResult {
    Sent,
    Full,         // channel full, event dropped (receiver alive but slow)
    Disconnected, // receiver gone — stop wasting compute
}
