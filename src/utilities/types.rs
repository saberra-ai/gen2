//! What the utility worker holds, as the outside world sees it.

use serde::{Deserialize, Serialize};

/// One loaded helper runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct LoadedUtility {
    /// What was loaded — a file name, usually.
    pub name: String,
    /// Roughly how much memory it is holding.
    pub estimated_resident_mb: u64,
}

/// Which auxiliary runtimes are loaded right now.
///
/// Deliberately separate from [`Capabilities`](crate::Capabilities), which
/// says what the *generative* model can accept. The two answer different
/// questions, and overloading `AUDIO` to mean "a transcription helper happens
/// to be installed" would make both answers useless: a caller checking whether
/// they can attach a sound file to a prompt would get `true` because something
/// unrelated could transcribe one.
///
/// `#[non_exhaustive]` because helpers are still being added.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[non_exhaustive]
pub struct UtilityStatus {
    /// The embedding model, if one is loaded.
    pub embedder: Option<LoadedUtility>,
}

impl UtilityStatus {
    /// Whether any helper is loaded at all.
    pub fn is_empty(&self) -> bool {
        self.embedder.is_none()
    }
}
