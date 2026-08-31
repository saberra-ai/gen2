//! Where a generation actually ran — the receipt the engine emits.
//!
//! Moved from `pio-core::compute::escalation` during the gen2 crate split: the
//! handle that dispatched the work is the only thing that knows which brain
//! executed it, so the record is produced here. The *policy* that decides what
//! is allowed to escalate (and the signed evidence chain this gets sealed into)
//! stayed in the host — it consumes this, not the other way around.

use serde::{Deserialize, Serialize};

/// What actually ran — sealed into the signed evidence chain so a receipt shows
/// exactly which brain executed a step and, if remote, what left the device.
/// This is the "visible audit of every byte that left your machine" the product
/// promise rests on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ComputeProvenance {
    /// Brain id that ran the step (`"local:gemma-e2b"`, `"nest:…"`, `"byo:…"`).
    pub brain_id: String,
    /// Did the work run somewhere other than the local brain?
    pub remote: bool,
    /// Did data leave the user's hardware entirely (cloud)? `false` for local
    /// and own-device (flock/Nest) — those stay on hardware the user owns.
    pub off_user_hardware: bool,
    /// If remote: a short, human-readable description of what was sent (for the
    /// audit line). `None` for local.
    pub sent_summary: Option<String>,
}

impl ComputeProvenance {
    /// Ran on the local brain. Nothing left the machine.
    pub fn local(brain_id: impl Into<String>) -> Self {
        Self {
            brain_id: brain_id.into(),
            remote: false,
            off_user_hardware: false,
            sent_summary: None,
        }
    }

    /// Ran on another of the user's own devices (flock / Nest). Left this
    /// machine but stayed on the user's hardware.
    pub fn own_device(brain_id: impl Into<String>, sent: impl Into<String>) -> Self {
        Self {
            brain_id: brain_id.into(),
            remote: true,
            off_user_hardware: false,
            sent_summary: Some(sent.into()),
        }
    }

    /// Ran on a bring-your-own cloud brain. Data left the user's hardware.
    pub fn cloud(brain_id: impl Into<String>, sent: impl Into<String>) -> Self {
        Self {
            brain_id: brain_id.into(),
            remote: true,
            off_user_hardware: true,
            sent_summary: Some(sent.into()),
        }
    }

    /// The receipt line. Loudest when data left the user's hardware.
    pub fn audit_line(&self) -> String {
        if !self.remote {
            return format!("ran on your device ({})", self.brain_id);
        }
        let sent = self.sent_summary.as_deref().unwrap_or("this task");
        if self.off_user_hardware {
            format!("used {} — left your machine; sent: {sent}", self.brain_id)
        } else {
            format!(
                "ran on your {} (your hardware); sent: {sent}",
                self.brain_id
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_receipt_says_nothing_left() {
        let p = ComputeProvenance::local("local:gemma-e2b");
        assert!(!p.remote);
        assert!(!p.off_user_hardware);
        assert_eq!(p.sent_summary, None);
        assert!(p.audit_line().contains("ran on your device"));
    }

    #[test]
    fn own_device_is_remote_but_stays_on_user_hardware() {
        let p = ComputeProvenance::own_device("flock", "the goal text");
        assert!(p.remote);
        assert!(
            !p.off_user_hardware,
            "a user's own peer is still their hardware"
        );
        assert!(p.audit_line().contains("your hardware"));
        assert!(p.audit_line().contains("the goal text"));
    }

    #[test]
    fn cloud_audit_line_says_it_left_the_machine() {
        let p = ComputeProvenance::cloud("byo:claude-opus", "the prompt");
        assert!(p.off_user_hardware);
        assert!(p.audit_line().contains("left your machine"));
    }
}
