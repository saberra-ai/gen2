//! An append-only record of what an agent did, and the view of it a model sees.
//!
//! # Two different things
//!
//! A conversation and a history are not the same object, and gen2 has been
//! treating them as one. [`Session`](crate::Session) holds a flat list of
//! messages that *is* both: trimming it to fit a context window destroys the
//! history, and keeping the history means the context window eventually
//! refuses the conversation.
//!
//! ```text
//!   append-only journal          everything that happened
//!            │
//!            ▼
//!       Projection               what fits, chosen per inference
//!            │
//!            ▼
//!    Vec<Message>                what the model is shown
//! ```
//!
//! An agent woken every minute accumulates ten thousand facts and should show
//! the model none of them. That is only expressible when the log and the view
//! are separate — which is also what lets the same history be projected two
//! different ways, and lets a projection be *changed* without rewriting
//! anything that already happened.
//!
//! # The invariant this module exists to hold
//!
//! A projection selects [`Turn`]s, never individual records, and a `Turn`
//! keeps a tool call with its results. So no projection — including one
//! written later by someone who has not read this — can show a model a tool
//! result with nothing that asked for it, or a call that never resolves.
//!
//! That failure is not hypothetical: `session_rt::truncate` drops
//! messages oldest-first by count today, with no idea what it is splitting.
//! Making it impossible was the reason to build this before anything that
//! depends on it.

mod entry;
mod projection;
mod resident;
mod scratch;
mod store;
mod turn;
mod wake;

pub use entry::{EntryId, InputSource, JournalEntry, Record, WakeReason};
pub use projection::{Everything, Projection, RecentTurns, WithPreamble};
pub use resident::{Resident, Wake};
pub use scratch::Scratch;
pub use store::{Journal, JournalError, JsonlJournal, MemoryJournal};
pub use turn::{Turn, round_len};
pub use wake::{Declined, Heartbeat, MIN_INTERVAL, WakeScheduler};
