//! Persona — the system-prompt identity pinned at session start.
//!
//! Extracted from `pio-core`'s `types` module during the gen2 crate split.
//! The engine only ever reads `instructions`; the id/name/timestamps travel
//! with it so a host app can round-trip its own catalog record unchanged.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Persona {
    pub id: String,
    pub name: String,
    pub instructions: String,
    pub is_selected: bool,
    pub created_at: i64,
    pub updated_at: i64,
}
