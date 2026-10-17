//! Counts of owned grammar objects, not allocation bytes or RSS.
use serde::{Deserialize, Serialize};

/// Grammar vocabulary/parser ownership for one logical model backend engine.
/// Counters survive eviction and restore. Unsupported backends return None.
/// A vocabulary release means its last owner was destroyed, not that malloc
/// returned pages to the operating system. These counts do not change admission.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct GrammarOwnership {
    /// Successfully prepared immutable vocabularies since engine creation.
    pub vocabulary_builds: u64,
    /// Prepared vocabularies destroyed after their last owner releases them.
    pub vocabulary_releases: u64,
    /// Prepared vocabularies with at least one live owner.
    pub live_vocabularies: u64,
    /// Constructed per-request matchers not yet destroyed.
    pub active_matchers: u64,
}
