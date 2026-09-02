//! Where the journal lives.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::entry::{EntryId, JournalEntry, Record};

/// An append-only log of what happened.
///
/// Append-only is the whole point. A journal that can be rewritten is a
/// journal that can disagree with itself, and every useful property below —
/// replay, recovery, projecting the same history two different ways — rests on
/// entries never changing after they land.
pub trait Journal: Send + Sync {
    /// Write one entry down. Returns where it landed.
    fn append(&self, entry: JournalEntry) -> Result<EntryId, JournalError>;

    /// Everything, oldest first.
    fn replay(&self) -> Result<Vec<Record>, JournalError>;

    /// How many entries there are.
    fn len(&self) -> Result<usize, JournalError> {
        Ok(self.replay()?.len())
    }

    fn is_empty(&self) -> Result<bool, JournalError> {
        Ok(self.len()? == 0)
    }
}

/// Something went wrong reaching the journal.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JournalError {
    #[error("journal io: {0}")]
    Io(String),
    /// A line could not be read back.
    ///
    /// Carries the position so a damaged log can be inspected rather than
    /// simply declared unreadable.
    #[error("journal entry {line} is unreadable: {reason}")]
    Corrupt { line: usize, reason: String },
}

/// A journal in memory. Nothing survives the process.
///
/// For tests, and for an agent that genuinely has no durable identity.
#[derive(Default)]
pub struct MemoryJournal {
    records: Mutex<Vec<Record>>,
}

impl MemoryJournal {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Journal for MemoryJournal {
    fn append(&self, entry: JournalEntry) -> Result<EntryId, JournalError> {
        let mut records = self
            .records
            .lock()
            .map_err(|_| JournalError::Io("journal lock poisoned".into()))?;
        let id = EntryId(records.len() as u64);
        records.push(Record {
            id,
            at: now(),
            entry,
        });
        Ok(id)
    }

    fn replay(&self) -> Result<Vec<Record>, JournalError> {
        self.records
            .lock()
            .map(|r| r.clone())
            .map_err(|_| JournalError::Io("journal lock poisoned".into()))
    }
}

/// A journal on disk, one JSON object per line.
///
/// JSONL rather than a database, for now, because the access pattern is
/// exactly append-and-replay and because a text log can be read with `tail`
/// when something has gone wrong at three in the morning. When replay cost
/// starts to matter — and it will, at a few hundred thousand entries — the
/// answer is a snapshot beside the log, not a different file format.
pub struct JsonlJournal {
    path: PathBuf,
    /// Serialises appends and guards the counter. Two writers interleaving
    /// half-lines would corrupt the log in a way replay could not repair.
    next: Mutex<u64>,
}

impl JsonlJournal {
    /// Open a journal, creating it if it does not exist.
    ///
    /// Reads what is already there to find the next id, so reopening continues
    /// the sequence rather than restarting it and producing two entries that
    /// claim the same position.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, JournalError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| JournalError::Io(e.to_string()))?;
        }
        let existing = read_all(&path)?;
        let next = existing.last().map(|r| r.id.0 + 1).unwrap_or(0);
        Ok(Self {
            path,
            next: Mutex::new(next),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Journal for JsonlJournal {
    fn append(&self, entry: JournalEntry) -> Result<EntryId, JournalError> {
        let mut next = self
            .next
            .lock()
            .map_err(|_| JournalError::Io("journal lock poisoned".into()))?;
        let record = Record {
            id: EntryId(*next),
            at: now(),
            entry,
        };
        let mut line =
            serde_json::to_string(&record).map_err(|e| JournalError::Io(e.to_string()))?;
        line.push('\n');

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| JournalError::Io(e.to_string()))?;
        file.write_all(line.as_bytes())
            .map_err(|e| JournalError::Io(e.to_string()))?;
        // Durability is the point of writing this down at all. A journal that
        // loses the last few entries to the page cache when the machine dies
        // is exactly the case it exists to survive.
        file.sync_data()
            .map_err(|e| JournalError::Io(e.to_string()))?;

        *next += 1;
        Ok(record.id)
    }

    fn replay(&self) -> Result<Vec<Record>, JournalError> {
        read_all(&self.path)
    }
}

fn read_all(path: &Path) -> Result<Vec<Record>, JournalError> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        // No journal yet is not an error; it is an empty history.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(JournalError::Io(e.to_string())),
    };

    let mut records = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| JournalError::Io(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Record>(&line) {
            Ok(record) => records.push(record),
            // A half-written final line is what a crash mid-append leaves.
            // Everything before it is intact and is the history; refusing to
            // load any of it would turn a torn write into total loss. A bad
            // line anywhere else is genuine corruption and is reported, since
            // silently skipping it would hand back a history missing a middle.
            Err(e) => {
                return if is_last_line(path, index)? {
                    Ok(records)
                } else {
                    Err(JournalError::Corrupt {
                        line: index + 1,
                        reason: e.to_string(),
                    })
                };
            }
        }
    }
    Ok(records)
}

/// Whether `index` is the final line in the file.
fn is_last_line(path: &Path, index: usize) -> Result<bool, JournalError> {
    let file = std::fs::File::open(path).map_err(|e| JournalError::Io(e.to_string()))?;
    let total = BufReader::new(file).lines().count();
    Ok(index + 1 == total)
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::entry::{InputSource, WakeReason};
    use crate::types::message::Message;

    fn input(text: &str) -> JournalEntry {
        JournalEntry::Input {
            source: InputSource::User,
            content: Message::user(text),
        }
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gen2-journal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join(name)
    }

    #[test]
    fn ids_are_assigned_in_order() {
        let journal = MemoryJournal::new();
        assert_eq!(journal.append(input("a")).unwrap(), EntryId(0));
        assert_eq!(journal.append(input("b")).unwrap(), EntryId(1));
        assert_eq!(journal.replay().unwrap().len(), 2);
    }

    #[test]
    fn a_journal_on_disk_survives_being_reopened() {
        let path = temp("agent.jsonl");
        {
            let journal = JsonlJournal::open(&path).unwrap();
            journal.append(input("before the crash")).unwrap();
            journal
                .append(JournalEntry::Wake {
                    reason: WakeReason::Heartbeat,
                })
                .unwrap();
        }
        let reopened = JsonlJournal::open(&path).unwrap();
        let records = reopened.replay().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, EntryId(0));
    }

    /// Reopening must continue the sequence, not restart it.
    ///
    /// Two entries claiming position 0 would make ordering — the one thing
    /// `EntryId` is for — a lie.
    #[test]
    fn reopening_continues_the_id_sequence() {
        let path = temp("continue.jsonl");
        {
            let journal = JsonlJournal::open(&path).unwrap();
            journal.append(input("one")).unwrap();
            journal.append(input("two")).unwrap();
        }
        let reopened = JsonlJournal::open(&path).unwrap();
        assert_eq!(reopened.append(input("three")).unwrap(), EntryId(2));

        let ids: Vec<u64> = reopened.replay().unwrap().iter().map(|r| r.id.0).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    /// A crash mid-append leaves a torn final line. Everything before it is
    /// still the history and must still load.
    #[test]
    fn a_half_written_final_line_does_not_destroy_the_history() {
        let path = temp("torn.jsonl");
        {
            let journal = JsonlJournal::open(&path).unwrap();
            journal.append(input("intact")).unwrap();
            journal.append(input("also intact")).unwrap();
        }
        // The machine died here.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"{\"id\":{\"0\":2},\"at\":0,\"ki").unwrap();
        drop(file);

        let records = JsonlJournal::open(&path).unwrap().replay().unwrap();
        assert_eq!(
            records.len(),
            2,
            "the two complete entries are still the history"
        );
    }

    #[test]
    fn a_missing_journal_reads_as_an_empty_history() {
        let journal = JsonlJournal::open(temp("never-written.jsonl")).unwrap();
        assert!(journal.is_empty().unwrap());
    }

    /// Every entry variant has to survive a round trip, or the log is only
    /// readable by the version that wrote it.
    #[test]
    fn every_entry_kind_round_trips_through_the_log() {
        let path = temp("kinds.jsonl");
        let journal = JsonlJournal::open(&path).unwrap();
        let entries = vec![
            input("hello"),
            JournalEntry::Assistant {
                message: Message::assistant_structured("hi", None),
            },
            JournalEntry::ToolCall {
                id: "c1".into(),
                tool: "read".into(),
                args: serde_json::json!({"path": "x"}),
            },
            JournalEntry::ToolResult {
                id: "c1".into(),
                tool: "read".into(),
                output: "contents".into(),
                ok: true,
            },
            JournalEntry::ScratchSet {
                key: "plan".into(),
                value: "step one".into(),
            },
            JournalEntry::ScratchDelete { key: "plan".into() },
            JournalEntry::Wake {
                reason: WakeReason::External("filesystem".into()),
            },
        ];
        for entry in &entries {
            journal.append(entry.clone()).unwrap();
        }

        let read_back: Vec<JournalEntry> = JsonlJournal::open(&path)
            .unwrap()
            .replay()
            .unwrap()
            .into_iter()
            .map(|r| r.entry)
            .collect();
        assert_eq!(read_back, entries);
    }

    /// Concurrent appends must not interleave half-lines.
    #[test]
    fn parallel_appends_all_land_and_stay_readable() {
        let path = temp("parallel.jsonl");
        let journal = std::sync::Arc::new(JsonlJournal::open(&path).unwrap());
        let mut handles = Vec::new();
        for w in 0..8 {
            let journal = std::sync::Arc::clone(&journal);
            handles.push(std::thread::spawn(move || {
                for i in 0..25 {
                    journal.append(input(&format!("{w}-{i}"))).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let records = JsonlJournal::open(&path).unwrap().replay().unwrap();
        assert_eq!(records.len(), 200, "every append should be readable back");

        let mut ids: Vec<u64> = records.iter().map(|r| r.id.0).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 200, "no two entries may claim the same position");
    }
}
