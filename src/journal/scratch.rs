//! What an agent remembers on purpose.
//!
//! The transcript is continuity for the *conversation*. This is continuity for
//! *conclusions* — the things worth keeping after the context window has
//! forgotten how they were reached. An agent that has to re-derive what it
//! already worked out is not continuous in any useful sense.
//!
//! # Why this is not the filesystem
//!
//! It is tempting to make scratch a directory: models understand files, and a
//! `plan.md` reads better than a key. But scratch is a *materialised view of
//! the journal* — that is what makes crash recovery a replay rather than a
//! reconciliation — and a directory the agent can also reach through an
//! ordinary `run_command` or `edit_file` tool would diverge from the log the
//! moment it did so. Two sources of truth, no way to tell which is right.
//!
//! So keys look like paths (`plan.md`, `notes/observations.md`) because that
//! is what a model reads well, and they are not files. The filesystem remains
//! perfectly reachable through ordinary tools; it is simply not scratch, and
//! not replayed.

use std::collections::BTreeMap;

use super::entry::{JournalEntry, Record};

/// The agent's durable notes, rebuilt from the journal.
///
/// Every mutation is a journal entry, so this type never holds anything the
/// log does not. It is a cache of a fold, and [`Scratch::replay`] is that fold.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Scratch {
    /// Ordered so listing is stable — an agent shown its own notes in a
    /// different order each wake would have no way to tell what changed.
    entries: BTreeMap<String, String>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild scratch from a journal.
    ///
    /// The only constructor that matters. A crash loses nothing that was
    /// written down, because this is what "written down" means.
    pub fn replay(records: &[Record]) -> Self {
        let mut scratch = Self::new();
        for record in records {
            scratch.apply(&record.entry);
        }
        scratch
    }

    /// Fold one entry in. Anything that is not a scratch mutation is ignored.
    pub fn apply(&mut self, entry: &JournalEntry) {
        match entry {
            JournalEntry::ScratchSet { key, value } => {
                self.entries.insert(key.clone(), value.clone());
            }
            JournalEntry::ScratchDelete { key } => {
                self.entries.remove(key);
            }
            _ => {}
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(String::as_str)
    }

    /// Every key, in a stable order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Render for a model to read.
    ///
    /// Included in the preamble on every wake, which is the point of scratch
    /// existing: it survives compaction, so the conclusions outlive the
    /// conversation that produced them.
    pub fn to_prompt(&self) -> String {
        if self.entries.is_empty() {
            return String::new();
        }
        let mut out = String::from("Your notes:\n");
        for (key, value) in &self.entries {
            out.push_str(&format!("\n## {key}\n{value}\n"));
        }
        out
    }
}

/// The mutations a scratch write implies.
///
/// Returned rather than applied, so the caller appends to the journal and
/// folds the result in — keeping "the journal is the source of truth" a
/// property of the code rather than a comment.
pub fn set(key: impl Into<String>, value: impl Into<String>) -> JournalEntry {
    JournalEntry::ScratchSet {
        key: key.into(),
        value: value.into(),
    }
}

pub fn delete(key: impl Into<String>) -> JournalEntry {
    JournalEntry::ScratchDelete { key: key.into() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::entry::{EntryId, InputSource, WakeReason};
    use crate::journal::{Journal, JsonlJournal, MemoryJournal};

    fn record(id: u64, entry: JournalEntry) -> Record {
        Record {
            id: EntryId(id),
            at: 0,
            entry,
        }
    }

    #[test]
    fn a_write_then_a_read_returns_what_was_written() {
        let records = vec![record(0, set("plan", "step one"))];
        assert_eq!(Scratch::replay(&records).get("plan"), Some("step one"));
    }

    #[test]
    fn the_last_write_wins() {
        let records = vec![record(0, set("plan", "old")), record(1, set("plan", "new"))];
        assert_eq!(Scratch::replay(&records).get("plan"), Some("new"));
    }

    #[test]
    fn a_delete_removes_it() {
        let records = vec![record(0, set("plan", "x")), record(1, delete("plan"))];
        assert!(Scratch::replay(&records).get("plan").is_none());
    }

    /// Deleting something that was never there is not an error.
    #[test]
    fn deleting_a_key_that_does_not_exist_is_harmless() {
        let scratch = Scratch::replay(&[record(0, delete("never-set"))]);
        assert!(scratch.is_empty());
    }

    /// The property the design rests on: scratch *is* the journal.
    ///
    /// Not "is kept in sync with" — replaying the log reconstructs it exactly,
    /// which is why a crash costs nothing that was written down.
    #[test]
    fn scratch_is_exactly_what_replaying_the_journal_produces() {
        let journal = MemoryJournal::new();
        journal.append(set("plan", "one")).unwrap();
        journal
            .append(JournalEntry::Wake {
                reason: WakeReason::Heartbeat,
            })
            .unwrap();
        journal.append(set("notes/a.md", "seen this")).unwrap();
        journal.append(set("plan", "two")).unwrap();
        journal.append(delete("notes/a.md")).unwrap();
        journal
            .append(JournalEntry::Input {
                source: InputSource::User,
                content: crate::types::message::Message::user("hi"),
            })
            .unwrap();

        // What a live agent would hold, folded as it went.
        let mut live = Scratch::new();
        for r in journal.replay().unwrap() {
            live.apply(&r.entry);
        }
        // What a restart would rebuild.
        let recovered = Scratch::replay(&journal.replay().unwrap());

        assert_eq!(live, recovered);
        assert_eq!(recovered.get("plan"), Some("two"));
        assert!(recovered.get("notes/a.md").is_none());
        assert_eq!(recovered.len(), 1);
    }

    /// And across an actual process boundary, on disk.
    #[test]
    fn scratch_survives_a_restart() {
        let path = std::env::temp_dir().join(format!(
            "gen2-scratch-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        {
            let journal = JsonlJournal::open(&path).unwrap();
            journal.append(set("plan", "survive this")).unwrap();
        }
        let reopened = JsonlJournal::open(&path).unwrap();
        let scratch = Scratch::replay(&reopened.replay().unwrap());
        assert_eq!(scratch.get("plan"), Some("survive this"));
    }

    #[test]
    fn keys_are_listed_in_a_stable_order() {
        let records = vec![
            record(0, set("zebra", "z")),
            record(1, set("alpha", "a")),
            record(2, set("middle", "m")),
        ];
        let scratch = Scratch::replay(&records);
        assert_eq!(
            scratch.keys().collect::<Vec<_>>(),
            vec!["alpha", "middle", "zebra"]
        );
    }

    #[test]
    fn empty_scratch_renders_to_nothing_rather_than_an_empty_heading() {
        assert_eq!(Scratch::new().to_prompt(), "");
    }

    #[test]
    fn notes_render_with_their_keys_so_the_model_can_address_them() {
        let scratch = Scratch::replay(&[record(0, set("plan.md", "step one"))]);
        let prompt = scratch.to_prompt();
        assert!(prompt.contains("plan.md"), "{prompt}");
        assert!(prompt.contains("step one"), "{prompt}");
    }
}
