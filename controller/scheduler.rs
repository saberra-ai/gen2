/// Stateless scheduling policy functions for the inference controller.
///
/// These are pure functions over [`ScheduleView`] — a backend-independent
/// snapshot of the scheduling-relevant fields. The controller owns all mutable
/// data; the scheduler just makes decisions. This separation keeps policy
/// testable and the controller loop focused on mechanism.
use std::collections::HashMap;
use std::time::Instant;

/// Backend-independent snapshot of the fields a scheduling decision needs.
///
/// Created from `ChatStream` in the controller; testable without any backend.
#[derive(Debug)]
pub(super) struct ScheduleView {
    pub ephemeral: bool,
    pub paused: bool,
    pub finished: bool,
    pub has_puller: bool,
    pub last_used: Instant,
}

impl super::ChatStream {
    /// Snapshot the scheduling-relevant fields.
    pub(super) fn schedule_view(&self) -> ScheduleView {
        ScheduleView {
            ephemeral: self.ephemeral,
            paused: self.paused,
            finished: self.finished,
            has_puller: self.puller.is_some(),
            last_used: self.last_used,
        }
    }
}

/// Build a schedule-view map from the live chats. Cheap — copies a few bools
/// and an `Instant` per chat, no Arc clones.
pub(super) fn views(chats: &HashMap<String, super::ChatStream>) -> HashMap<String, ScheduleView> {
    chats
        .iter()
        .map(|(k, c)| (k.clone(), c.schedule_view()))
        .collect()
}

/// Pick the eviction target: the chat with the oldest `last_used` timestamp.
///
/// Returns the key of the victim, or `None` if the map is empty.
pub(super) fn pick_eviction_target(views: &HashMap<String, ScheduleView>) -> Option<String> {
    views
        .iter()
        .min_by_key(|(_k, v)| v.last_used)
        .map(|(k, _)| k.clone())
}

/// Collect chat IDs in tick order: primary (non-ephemeral) chats first,
/// then ephemeral. Only includes chats that are ready to tick
/// (not paused, not finished, puller present).
///
/// This ensures the active user chat always gets compute first;
/// title gen and suggestions don't steal ticks from the live chat.
pub(super) fn tick_order(views: &HashMap<String, ScheduleView>) -> Vec<String> {
    let mut keys = Vec::new();
    let mut ephemeral_start = 0usize;
    for (id, v) in views.iter() {
        if v.paused || v.finished || !v.has_puller {
            continue;
        }
        if !v.ephemeral {
            keys.insert(ephemeral_start, id.clone());
            ephemeral_start += 1;
        } else {
            keys.push(id.clone());
        }
    }
    keys
}

/// Whether a chat should be auto-cleaned from the map.
pub(super) fn should_cleanup(view: &ScheduleView) -> bool {
    view.finished && view.ephemeral
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn view(
        ephemeral: bool,
        paused: bool,
        finished: bool,
        has_puller: bool,
        age_ms: u64,
    ) -> ScheduleView {
        ScheduleView {
            ephemeral,
            paused,
            finished,
            has_puller,
            last_used: Instant::now() - Duration::from_millis(age_ms),
        }
    }

    // ── Eviction ───────────────────────────────────────────────────────

    #[test]
    fn eviction_empty_map() {
        let m: HashMap<String, ScheduleView> = HashMap::new();
        assert!(pick_eviction_target(&m).is_none());
    }

    #[test]
    fn eviction_single_entry() {
        let mut m = HashMap::new();
        m.insert("a".into(), view(false, false, false, true, 100));
        assert_eq!(pick_eviction_target(&m), Some("a".into()));
    }

    #[test]
    fn eviction_picks_oldest() {
        let mut m = HashMap::new();
        m.insert("new".into(), view(false, false, false, true, 10));
        m.insert("old".into(), view(false, false, false, true, 1000));
        m.insert("mid".into(), view(false, false, false, true, 500));
        assert_eq!(pick_eviction_target(&m), Some("old".into()));
    }

    // ── Tick order ─────────────────────────────────────────────────────

    #[test]
    fn tick_order_empty() {
        let m: HashMap<String, ScheduleView> = HashMap::new();
        assert!(tick_order(&m).is_empty());
    }

    #[test]
    fn tick_order_filters_paused_finished_no_puller() {
        let mut m = HashMap::new();
        m.insert("paused".into(), view(false, true, false, true, 0));
        m.insert("finished".into(), view(false, false, true, true, 0));
        m.insert("no_puller".into(), view(false, false, false, false, 0));
        assert!(tick_order(&m).is_empty());
    }

    #[test]
    fn tick_order_primary_before_ephemeral() {
        let mut m = HashMap::new();
        m.insert("eph1".into(), view(true, false, false, true, 0));
        m.insert("primary".into(), view(false, false, false, true, 0));
        m.insert("eph2".into(), view(true, false, false, true, 0));
        let order = tick_order(&m);
        // Primary must come first
        assert_eq!(order[0], "primary");
        // Ephemerals after
        assert_eq!(order.len(), 3);
        assert!(order[1..].iter().all(|k| k.starts_with("eph")));
    }

    #[test]
    fn tick_order_multiple_primaries_before_ephemerals() {
        let mut m = HashMap::new();
        m.insert("e1".into(), view(true, false, false, true, 0));
        m.insert("p1".into(), view(false, false, false, true, 0));
        m.insert("p2".into(), view(false, false, false, true, 0));
        m.insert("e2".into(), view(true, false, false, true, 0));
        let order = tick_order(&m);
        assert_eq!(order.len(), 4);
        // First two should be primaries (order among them is HashMap-order, unspecified)
        let primaries: Vec<_> = order.iter().take(2).collect();
        assert!(primaries.iter().all(|k| k.starts_with("p")));
        let ephemerals: Vec<_> = order.iter().skip(2).collect();
        assert!(ephemerals.iter().all(|k| k.starts_with("e")));
    }

    #[test]
    fn tick_order_all_ephemeral() {
        let mut m = HashMap::new();
        m.insert("e1".into(), view(true, false, false, true, 0));
        m.insert("e2".into(), view(true, false, false, true, 0));
        let order = tick_order(&m);
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn tick_order_all_primary() {
        let mut m = HashMap::new();
        m.insert("p1".into(), view(false, false, false, true, 0));
        m.insert("p2".into(), view(false, false, false, true, 0));
        let order = tick_order(&m);
        assert_eq!(order.len(), 2);
    }

    // ── Cleanup ────────────────────────────────────────────────────────

    #[test]
    fn cleanup_ephemeral_finished() {
        assert!(should_cleanup(&view(true, false, true, false, 0)));
    }

    #[test]
    fn cleanup_primary_finished_no_cleanup() {
        assert!(!should_cleanup(&view(false, false, true, false, 0)));
    }

    #[test]
    fn cleanup_ephemeral_running_no_cleanup() {
        assert!(!should_cleanup(&view(true, false, false, true, 0)));
    }

    #[test]
    fn cleanup_primary_running_no_cleanup() {
        assert!(!should_cleanup(&view(false, false, false, true, 0)));
    }
}
