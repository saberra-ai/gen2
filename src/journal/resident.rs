//! An agent that outlives any one run.
//!
//! [`Resident`] is the durable half of the split this module exists for: it
//! holds identity, journal, scratch and heartbeat policy, and a *run* is one
//! period of activity it goes through. That is the difference between
//! `agent.goal("do X").run()` — where the agent is synonymous with the work —
//! and something that is still there when the work is done.
//!
//! # What this deliberately is not
//!
//! There are no hooks, no middleware, no plugin registry, no interception
//! points. Extensibility here is exactly five verbs — give it tools, send it
//! input, observe it, read and write its scratch, wake it — and that is enough
//! to build a great deal on. A seam gets added when three real consumers need
//! the same one, not in anticipation.

use std::sync::Arc;
use std::time::Instant;

use super::entry::{InputSource, JournalEntry, WakeReason};
use super::projection::{Projection, RecentTurns, WithPreamble};
use super::scratch::Scratch;
use super::store::{Journal, JournalError, MemoryJournal};
use super::wake::{Declined, Heartbeat, WakeScheduler};
use crate::types::message::Message;

/// What a wake tells the model, when the wake was the agent's own timer.
///
/// Ephemeral by construction — it is built here and prepended to one
/// inference, never appended to the journal as conversation. That is the whole
/// reason a heartbeat does not pollute the transcript.
const SELF_PROMPT: &str = "You have been woken by your periodic heartbeat. Review your notes and \
     recent activity. If there is useful work you can do now, do it. If there \
     is not, say so briefly and stop.";

/// A persistent agent: journal, scratch, tools, and a policy for waking.
///
/// Holds no engine and runs nothing. Driving it is the caller's — or a future
/// handle's — job; what lives here is the state that survives a run, and the
/// decisions about what the model should see.
pub struct Resident {
    journal: Arc<dyn Journal>,
    scratch: Scratch,
    schedule: WakeScheduler,
    /// Standing instruction. Configuration, not history: an agent that
    /// re-derived it from its own log would lose it the first time the context
    /// budget bit.
    system: Option<String>,
    /// How much transcript to show the model.
    context_budget: usize,
}

impl Resident {
    /// Open an agent over a journal, recovering whatever it already knows.
    ///
    /// Recovery is a replay, not a reconciliation — which is the property that
    /// made scratch journal-backed in the first place.
    pub fn open(journal: Arc<dyn Journal>, heartbeat: Heartbeat) -> Result<Self, JournalError> {
        let scratch = Scratch::replay(&journal.replay()?);
        Ok(Self {
            journal,
            scratch,
            schedule: WakeScheduler::new(heartbeat),
            system: None,
            context_budget: 4096,
        })
    }

    /// An agent that remembers nothing after this process ends.
    pub fn ephemeral(heartbeat: Heartbeat) -> Self {
        Self::open(Arc::new(MemoryJournal::new()), heartbeat)
            .expect("an in-memory journal cannot fail to replay")
    }

    /// Set the standing instruction.
    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.system = Some(text.into());
        self
    }

    /// How many tokens of transcript to show the model.
    pub fn context_budget(mut self, tokens: usize) -> Self {
        self.context_budget = tokens;
        self
    }

    /// The notes, as they stand.
    pub fn scratch(&self) -> &Scratch {
        &self.scratch
    }

    /// Write a note, durably.
    ///
    /// Journal first, then fold — so the in-memory view can never hold
    /// something the log does not. If the append fails, nothing changed.
    pub fn remember(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), JournalError> {
        let entry = super::scratch::set(key, value);
        self.journal.append(entry.clone())?;
        self.scratch.apply(&entry);
        Ok(())
    }

    /// Forget a note.
    pub fn forget(&mut self, key: impl Into<String>) -> Result<(), JournalError> {
        let entry = super::scratch::delete(key);
        self.journal.append(entry.clone())?;
        self.scratch.apply(&entry);
        Ok(())
    }

    /// Record something the agent was told.
    pub fn receive(&mut self, source: InputSource, content: Message) -> Result<(), JournalError> {
        let reason = source_reason(&source).unwrap_or(WakeReason::User);
        self.journal
            .append(JournalEntry::Input { source, content })?;
        self.schedule.request(reason);
        Ok(())
    }

    /// Ask whether to run now, and on what.
    ///
    /// Returns the transcript to run with, already projected — so a caller
    /// never has to decide what the model sees, and cannot get the pairing
    /// wrong by deciding it differently.
    pub fn poll(&mut self, now: Instant) -> Result<Wake, Declined> {
        let reason = self.schedule.poll(now)?;
        let records = self.journal.replay().map_err(|_| Declined::Busy)?;

        let mut preamble = Vec::new();
        if let Some(system) = &self.system {
            preamble.push(Message::system(system.clone()));
        }
        if !self.scratch.is_empty() {
            // Notes ride in the preamble rather than the history, so they
            // survive the context budget. Conclusions outliving the
            // conversation that produced them is the entire point of scratch.
            preamble.push(Message::system(self.scratch.to_prompt()));
        }
        if matches!(reason, WakeReason::Heartbeat) {
            // The self-prompt, ephemeral. It is a stimulus for this one
            // inference and is never written down as a conversation turn.
            preamble.push(Message::system(SELF_PROMPT));
        }

        let messages = WithPreamble {
            preamble,
            inner: RecentTurns::new(self.context_budget),
        }
        .project(&records);

        self.journal
            .append(JournalEntry::Wake {
                reason: reason.clone(),
            })
            .map_err(|_| Declined::Busy)?;
        self.schedule.began_running();

        Ok(Wake { reason, messages })
    }

    /// The run is over.
    ///
    /// `entries` is everything it produced — the assistant's replies, the
    /// tools it called and their results. Appending them here rather than as
    /// the run goes keeps "what happened" a single decision, and lets
    /// `did_something` be answered by the only honest measure: whether
    /// anything was written down.
    pub fn finished(&mut self, entries: Vec<JournalEntry>) -> Result<(), JournalError> {
        let did_something = !entries.is_empty();
        for entry in entries {
            self.journal.append(entry.clone())?;
            self.scratch.apply(&entry);
        }
        self.schedule.finished_running(did_something);
        Ok(())
    }

    /// Whether heartbeats have stopped for want of anything to do.
    pub fn is_bored(&self) -> bool {
        self.schedule.is_bored()
    }

    /// The journal, for a caller that wants to read the history directly.
    pub fn journal(&self) -> &Arc<dyn Journal> {
        &self.journal
    }
}

fn source_reason(source: &InputSource) -> Option<WakeReason> {
    match source {
        InputSource::User => Some(WakeReason::User),
        InputSource::External(what) => Some(WakeReason::External(what.clone())),
        InputSource::SelfPrompt => None,
    }
}

/// A run the agent should perform.
#[derive(Debug, Clone)]
pub struct Wake {
    /// Why it was woken.
    pub reason: WakeReason,
    /// What to show the model — projected, with the preamble already on it.
    pub messages: Vec<Message>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::MIN_INTERVAL;
    use crate::journal::store::JsonlJournal;

    fn heartbeat() -> Heartbeat {
        Heartbeat::every(MIN_INTERVAL)
    }

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "gen2-resident-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn a_heartbeat_wake_carries_a_self_prompt_that_is_never_written_down() {
        let mut agent = Resident::ephemeral(heartbeat()).system("be useful");
        let wake = agent.poll(Instant::now()).expect("first poll wakes");

        assert!(
            wake.messages.iter().any(|m| m.text().contains("heartbeat")),
            "the model should be told why it woke: {:?}",
            wake.messages
        );

        // ...and the journal holds the fact, not the prompt.
        let records = agent.journal().replay().unwrap();
        assert!(matches!(
            records[0].entry,
            JournalEntry::Wake {
                reason: WakeReason::Heartbeat
            }
        ));
        assert_eq!(records.len(), 1, "one fact, not a conversation turn");
    }

    /// The thing that makes a persistent agent survivable.
    #[test]
    fn ten_thousand_heartbeats_do_not_fill_the_context() {
        let mut agent = Resident::ephemeral(heartbeat().never_bored());
        let start = Instant::now();
        for minute in 0..200 {
            if let Ok(_wake) = agent.poll(start + std::time::Duration::from_secs(minute * 60)) {
                agent.finished(Vec::new()).unwrap();
            }
        }
        let wake = agent
            .poll(start + std::time::Duration::from_secs(200 * 60))
            .expect("still waking");

        // Only the preamble: 200 wakes contributed no conversation at all.
        assert!(
            wake.messages.len() <= 2,
            "heartbeats leaked into the transcript: {} messages",
            wake.messages.len()
        );
        assert!(
            agent.journal().len().unwrap() >= 200,
            "but all were recorded"
        );
    }

    #[test]
    fn notes_survive_the_context_budget() {
        let mut agent = Resident::ephemeral(heartbeat()).context_budget(1);
        agent.remember("plan.md", "finish the migration").unwrap();
        for i in 0..40 {
            agent
                .receive(InputSource::User, Message::user(format!("chatter {i}")))
                .unwrap();
        }
        let wake = agent.poll(Instant::now()).unwrap();
        let all: String = wake.messages.iter().map(|m| m.text()).collect();
        assert!(
            all.contains("finish the migration"),
            "notes must outlive the conversation that produced them: {all}"
        );
    }

    #[test]
    fn an_agent_reopens_with_everything_it_knew() {
        let path = temp("agent.jsonl");
        {
            let journal = Arc::new(JsonlJournal::open(&path).unwrap());
            let mut agent = Resident::open(journal, heartbeat()).unwrap();
            agent.remember("plan.md", "step one").unwrap();
            agent
                .receive(InputSource::User, Message::user("hello"))
                .unwrap();
        }
        let journal = Arc::new(JsonlJournal::open(&path).unwrap());
        let agent = Resident::open(journal, heartbeat()).unwrap();
        assert_eq!(agent.scratch().get("plan.md"), Some("step one"));
    }

    /// A note is durable before it is visible.
    #[test]
    fn a_note_is_in_the_journal_before_it_is_in_scratch() {
        let mut agent = Resident::ephemeral(heartbeat());
        agent.remember("k", "v").unwrap();
        let records = agent.journal().replay().unwrap();
        assert!(matches!(records[0].entry, JournalEntry::ScratchSet { .. }));
        assert_eq!(agent.scratch().get("k"), Some("v"));
    }

    #[test]
    fn user_input_wakes_the_agent_ahead_of_the_timer() {
        let start = Instant::now();
        let mut agent = Resident::ephemeral(heartbeat());
        agent.poll(start).unwrap();
        agent.finished(Vec::new()).unwrap();

        agent
            .receive(InputSource::User, Message::user("do a thing"))
            .unwrap();
        let Ok(wake) = agent.poll(start + std::time::Duration::from_secs(1)) else {
            panic!("a person should not have to wait out the heartbeat interval");
        };
        assert_eq!(wake.reason, WakeReason::User);
    }

    /// What the agent produced becomes history; what it saw does not.
    #[test]
    fn a_finished_run_records_what_it_did() {
        let mut agent = Resident::ephemeral(heartbeat());
        agent.poll(Instant::now()).unwrap();
        agent
            .finished(vec![JournalEntry::Assistant {
                message: Message::assistant_structured("I looked and found nothing", None),
            }])
            .unwrap();

        let records = agent.journal().replay().unwrap();
        assert_eq!(records.len(), 2, "the wake and the reply");
        assert!(matches!(records[1].entry, JournalEntry::Assistant { .. }));
    }

    /// A run that wrote a note counts as having done something.
    #[test]
    fn a_run_that_only_took_a_note_still_counts_as_work() {
        let start = Instant::now();
        let mut agent = Resident::ephemeral(Heartbeat::every(MIN_INTERVAL).idle_after(1));
        agent.poll(start).unwrap();
        agent
            .finished(vec![super::super::scratch::set("found", "something")])
            .unwrap();
        assert!(!agent.is_bored());
        assert_eq!(agent.scratch().get("found"), Some("something"));
    }

    /// And one that did nothing, repeatedly, stops.
    #[test]
    fn an_agent_with_nothing_to_do_eventually_stops_waking() {
        let start = Instant::now();
        let mut agent = Resident::ephemeral(Heartbeat::every(MIN_INTERVAL).idle_after(2));
        for minute in 0..2 {
            agent
                .poll(start + std::time::Duration::from_secs(minute * 60))
                .expect("wakes while it still has patience");
            agent.finished(Vec::new()).unwrap();
        }
        assert!(
            matches!(
                agent.poll(start + std::time::Duration::from_secs(180)),
                Err(Declined::Bored)
            ),
            "an agent with nothing to do must stop waking"
        );
    }
}
