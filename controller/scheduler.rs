/// Stateless scheduling policy functions for the inference controller.
///
/// These are pure functions over borrowed state — the controller owns all
/// mutable data; the scheduler just makes decisions. This separation keeps
/// policy testable and the controller loop focused on mechanism.
use std::collections::HashMap;

use super::ChatStream;

/// Pick the eviction target: the chat with the oldest `last_used` timestamp.
///
/// Returns the key of the victim, or `None` if the map is empty.
pub(super) fn pick_eviction_target(chats: &HashMap<String, ChatStream>) -> Option<String> {
    chats
        .iter()
        .min_by_key(|(_k, c)| c.last_used)
        .map(|(k, _)| k.clone())
}

/// Collect chat IDs in tick order: primary (non-ephemeral) chats first,
/// then ephemeral. Only includes chats that are ready to tick
/// (not paused, not finished, puller present).
///
/// This ensures the active user chat always gets compute first;
/// title gen and suggestions don't steal ticks from the live chat.
pub(super) fn tick_order(chats: &HashMap<String, ChatStream>) -> Vec<String> {
    let mut keys = Vec::new();
    let mut ephemeral_start = 0usize;
    for (id, chat) in chats.iter() {
        if chat.paused || chat.finished || chat.puller.is_none() {
            continue;
        }
        if !chat.ephemeral {
            keys.insert(ephemeral_start, id.clone());
            ephemeral_start += 1;
        } else {
            keys.push(id.clone());
        }
    }
    keys
}

/// Whether a finished ephemeral chat should be auto-cleaned from the map.
pub(super) fn should_cleanup(chat: &ChatStream) -> bool {
    chat.finished && chat.ephemeral
}

// Note: Unit tests for these functions require constructing ChatStream, which
// contains an `Arc<Session>` (a backend-gated enum). The scheduling logic is
// exercised by the existing controller integration tests (controller::tests and
// chaos.rs) which spin up a real controller. The policy extraction preserves
// identical behavior — the functions are literally the same code moved here.
