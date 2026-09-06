//! [`Session`] — model-agnostic conversational state you own.
//!
//! A session holds four things (api_spec.md §7):
//!
//! - the **system prompt** — first-order state, not a message;
//! - the **tool set** — first-order state, not a message;
//! - the **active projection**: the messages the next turn is built from;
//! - the **append-only event log** that every one of those is derived from.
//!
//! ```text
//!  append-only SessionEvent history      session.events()
//!               │
//!               ▼
//!      all immutable message records     session.all_messages()
//!               │
//!               ▼
//!         active projection              session.messages()
//!               │
//!               ▼
//!    runtime / model context window      (the engine's business)
//! ```
//!
//! No layer destroys the layer above it. Editing, removing, compacting and
//! restoring change what `messages()` returns; they never rewrite what
//! `events()` says happened, and every message version that ever existed is
//! still in `all_messages()`.
//!
//! # The tool-round invariant
//!
//! A tool call and its results are one unit. [`crate::journal::Turn`] holds
//! that for the agent journal by making it impossible to address half a
//! round; a session cannot borrow the type (its unit is a message, addressed
//! by id), so it holds the same rule at the mutation boundary instead:
//! [`Session::remove_message`], [`Session::replace_messages`],
//! [`Session::restore_context`] and [`Session::fork_at`] refuse a projection
//! that would show a result whose call is gone, or a call whose results are.
//! [`Session::edit`], which cannot fail, trims to a round boundary instead.
//!
//! # `Message`
//!
//! The message type is still the wire [`Message`] — role string, body — this
//! release. The spec's `enum Message { User, Assistant, Tool }` (§9) lands
//! with the root-namespace rewrite (roadmap S2.6); the operations here are
//! written so that swap is a type change, not a behaviour change.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::tool_defs::{ToolDefinition, ToolSet};
use crate::types::message::{Message, MessageBody};

// ── Identity ────────────────────────────────────────────────────────────────

/// Identifies a conversation.
///
/// Stable for the session's life, and the key the engine files its cached
/// state under. A [`Session::fork`] gets a new one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    fn random() -> Self {
        Self(format!("session-{}", uuid::Uuid::new_v4()))
    }

    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for SessionId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&SessionId> for String {
    fn from(id: &SessionId) -> Self {
        id.0.clone()
    }
}

impl PartialEq<str> for SessionId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for SessionId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

/// How many context-affecting mutations a session has seen.
///
/// Strictly increases on every change to the system prompt, the tool set, or
/// the active projection, and never on a read. A cheap "has anything changed"
/// signal for a runtime deciding whether cached model state is still valid.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct SessionRevision(u64);

impl SessionRevision {
    /// The revision as a number.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SessionRevision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// Identifies one immutable message record within a session.
///
/// Minted on append, never reused, and carried unchanged into forks — so an
/// id taken from a parent still names the same record in a branch. Ids are
/// scoped to a session lineage; one from an unrelated session names nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MessageId(u64);

impl MessageId {
    /// The id as a number.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "m{}", self.0)
    }
}

// ── History ─────────────────────────────────────────────────────────────────

/// One thing that happened to a session (api_spec.md §8).
///
/// The log is append-only and is the authoritative history: everything else
/// on a [`Session`] — the system prompt, the tools, the records, the active
/// projection — is a replay of it. `#[non_exhaustive]` because variants may
/// join; the invariant is stronger than the encoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SessionEvent {
    /// The system prompt was set — or cleared, when `system` is `None`.
    SystemSet {
        /// The whole new prompt, not a delta.
        system: Option<String>,
    },
    /// The tool set was replaced. Additions and removals record the whole
    /// resulting set, so a reader never has to fold deltas.
    ToolsSet {
        /// The whole new set.
        tools: ToolSet,
    },
    /// A message was appended to the active projection.
    MessageAdded {
        /// The new record.
        id: MessageId,
        /// Its content.
        message: Message,
    },
    /// A message was replaced by a new version in the same position.
    MessageReplaced {
        /// The record that was replaced. It still exists.
        old: MessageId,
        /// The record that took its place.
        new: MessageId,
        /// The new content.
        message: Message,
    },
    /// A message left the active projection. Its record still exists.
    MessageRemoved {
        /// The record.
        id: MessageId,
    },
    /// A record was created for the context replacement that follows it —
    /// a compaction summary, a rewritten message — and is not active until
    /// that replacement names it.
    ///
    /// Bookkeeping rather than a mutation in its own right: it does not move
    /// the [`SessionRevision`]; the `ContextReplaced` it belongs to does.
    MessageRecorded {
        /// The new record.
        id: MessageId,
        /// Its content.
        message: Message,
    },
    /// The active projection was replaced wholesale — compaction, `clear`,
    /// `edit`, `restore_context`.
    ContextReplaced {
        /// The projection afterwards.
        active: Vec<MessageId>,
        /// Which operation did it, when known.
        reason: Option<String>,
    },
    /// This session began as a branch of another.
    ///
    /// The parent's events precede this one verbatim — that is what
    /// "historical provenance" means here: a fork carries its parent's whole
    /// log, records and ids included, and diverges from this point.
    Forked {
        /// The session this one was forked from.
        parent: SessionId,
        /// The last active message kept, for [`Session::fork_at`]; `None`
        /// for a plain [`Session::fork`].
        at: Option<MessageId>,
    },
}

/// One immutable message version, with its standing in the projection
/// (api_spec.md §8.1).
#[derive(Debug, Clone, PartialEq)]
pub struct MessageRecord<'a> {
    /// The record's id.
    pub id: MessageId,
    /// The message.
    pub message: &'a Message,
    /// Whether it is in the active projection right now.
    pub active: bool,
    /// The record that replaced it, if [`Session::replace_message`] did.
    pub replaced_by: Option<MessageId>,
}

/// What a tool produced, fed back to the model (api_spec.md §9.4).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ToolResult {
    /// Plain text, shown verbatim.
    Text(String),
    /// Structured data, serialized before it reaches the model.
    Json(serde_json::Value),
}

impl ToolResult {
    /// The text the model is shown.
    pub fn to_model_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Json(v) => v.to_string(),
        }
    }
}

impl From<String> for ToolResult {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for ToolResult {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<serde_json::Value> for ToolResult {
    fn from(v: serde_json::Value) -> Self {
        Self::Json(v)
    }
}

impl From<super::tools::ToolOutput> for ToolResult {
    fn from(out: super::tools::ToolOutput) -> Self {
        match out {
            super::tools::ToolOutput::Text(s) => Self::Text(s),
            super::tools::ToolOutput::Json(v) => Self::Json(v),
        }
    }
}

/// Why a session mutation was refused. The session is unchanged.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SessionError {
    /// No record has this id in this session.
    #[error("no message {0} in this session")]
    UnknownMessage(MessageId),
    /// The record exists but is not in the active projection.
    #[error("message {0} is not in the active projection")]
    NotActive(MessageId),
    /// The same id was named twice in one projection.
    #[error("message {0} named twice in one projection")]
    Duplicate(MessageId),
    /// The projection would show a tool result whose call is not shown.
    #[error("would show the result of tool call `{call_id}` without the call")]
    OrphanResult {
        /// The call the result answers.
        call_id: String,
    },
    /// The projection would show a tool call whose result exists but is not
    /// shown.
    #[error("would show tool call `{call_id}` without its result")]
    UnansweredCall {
        /// The call left hanging.
        call_id: String,
    },
}

/// The crate's [`Result`](super::Result), for the operations below.
type Result<T> = std::result::Result<T, super::Error>;

/// A stored message version.
#[derive(Debug, Clone)]
struct StoredRecord {
    id: MessageId,
    message: Message,
    replaced_by: Option<MessageId>,
}

// ── Session ─────────────────────────────────────────────────────────────────

/// A conversation: its instruction state, its active messages, and the
/// lossless history they are derived from.
///
/// You hold this, not the engine. Read it, render it, edit it, persist it —
/// none of which an engine-owned conversation behind a string id would allow.
/// The engine keys its warm KV cache by [`Session::id`], so owning the history
/// costs nothing in speed: a follow-up turn still reuses the last prefill.
///
/// ```no_run
/// # use gen2::{Engine, Session};
/// # let engine = Engine::load("m.gguf")?;
/// let mut session = Session::new().with_system("You are terse.");
///
/// engine.chat(&mut session).user("Name two colours.").send()?;
/// println!("{}", session.latest_text().unwrap_or_default());
///
/// // A follow-up. The history is already here; you don't resend it.
/// engine.chat(&mut session).user("Now one more.").send()?;
/// # Ok::<(), gen2::Error>(())
/// ```
///
/// # Instruction state is not a message
///
/// The system prompt ([`Session::set_system`]) and tools
/// ([`Session::set_tools`]) are state the next turn is rendered with, not
/// entries in [`Session::messages`]. Changing either bumps the
/// [`Session::revision`], is recorded in [`Session::events`], and makes the
/// next turn rebuild the model's prefix — the messages are untouched.
///
/// # Edits are lossless
///
/// [`Session::replace_message`], [`Session::remove_message`],
/// [`Session::replace_messages`] and [`Session::restore_context`] change the
/// active projection. The records they displace stay in
/// [`Session::all_messages`], and [`Session::events`] says what was done.
///
/// # The session outlives the model's view of it
///
/// When a conversation outgrows the context window, the engine sheds its
/// oldest messages from *its* working set and carries on. The session is not
/// rewritten; [`Session::shed`] counts what fell out of the model's view.
///
/// # Not `Clone`
///
/// Two copies would share one engine conversation and overwrite each other's
/// cached prefill. [`Session::fork`] is the independent copy.
///
/// # Persistence
///
/// `Serialize`/`Deserialize` write the id and the event log; everything else
/// is replayed on read. Engine bookkeeping (`opened`, `shed`) is process-local
/// and deliberately not persisted, so a restored session is resent whole.
#[derive(Debug)]
pub struct Session {
    id: SessionId,
    events: Vec<SessionEvent>,
    // ── Derived from `events` ───────────────────────────────────────────
    /// Context-affecting events so far — every event but `MessageRecorded`.
    revision: u64,
    system: Option<String>,
    tools: ToolSet,
    records: Vec<StoredRecord>,
    index: HashMap<MessageId, usize>,
    active: Vec<MessageId>,
    /// The projection, rebuilt on every mutation so reads are borrows.
    projection: Vec<Message>,
    next_id: u64,
    // ── Engine bookkeeping, process-local ───────────────────────────────
    /// Whether the engine has opened this conversation yet. Lives here rather
    /// than in the engine so it's dropped with the session; not persisted,
    /// because it describes *this process's* engine state.
    pub(crate) opened: bool,
    /// Messages the engine has shed from its working set to fit the context
    /// window, across the whole conversation. Not persisted: a restored
    /// session is resent whole.
    pub(crate) shed: usize,
    /// Fingerprint of the executable tool prefix this conversation was opened
    /// with (the agent's registry). A change reopens the conversation so the
    /// new definitions actually reach the model.
    pub(crate) tools_fingerprint: Option<u64>,
    /// Which model generation this conversation was opened against. A cached
    /// prefill belongs to the model that produced it.
    pub(crate) model_generation: Option<u64>,
}

impl Session {
    /// A new, empty conversation.
    pub fn new() -> Self {
        Self::blank(SessionId::random())
    }

    fn blank(id: SessionId) -> Self {
        Self {
            id,
            events: Vec::new(),
            revision: 0,
            system: None,
            tools: ToolSet::new(),
            records: Vec::new(),
            index: HashMap::new(),
            active: Vec::new(),
            projection: Vec::new(),
            next_id: 0,
            opened: false,
            shed: 0,
            tools_fingerprint: None,
            model_generation: None,
        }
    }

    /// Rebuild a conversation from stored messages — after an app restart, say.
    ///
    /// A leading `system`-role message becomes the session's system prompt
    /// rather than a message, so a transcript saved when the prompt still
    /// travelled as a message renders exactly as it did. Everything else is
    /// appended in order.
    ///
    /// The engine has no cached state for it, so the next turn re-reads this
    /// history once and is warm from then on.
    pub fn from_messages(messages: impl IntoIterator<Item = Message>) -> Self {
        let mut session = Self::new();
        let mut messages = messages.into_iter().peekable();
        if messages.peek().is_some_and(|m| m.role == "system") {
            let system = messages.next().expect("peeked").text();
            session.set_system(system);
        }
        for m in messages {
            session.push(m);
        }
        session
    }

    // ── Builders ────────────────────────────────────────────────────────────

    /// Set the system prompt, builder-style.
    #[must_use]
    pub fn with_system(mut self, text: impl Into<String>) -> Self {
        self.set_system(text);
        self
    }

    /// Set the tool set, builder-style.
    #[must_use]
    pub fn with_tools(mut self, tools: impl Into<ToolSet>) -> Self {
        self.set_tools(tools);
        self
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// The conversation's id — the key the engine files its cached state
    /// under. Stable for the session's life.
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// How many context-affecting mutations this session has seen.
    pub fn revision(&self) -> SessionRevision {
        SessionRevision(self.revision)
    }

    /// The system prompt, if one is set.
    pub fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// The tools the model is offered.
    pub fn tools(&self) -> &ToolSet {
        &self.tools
    }

    /// The active projection: the messages the next turn is built from,
    /// oldest first. The system prompt is not among them — see
    /// [`Session::system`].
    pub fn messages(&self) -> &[Message] {
        &self.projection
    }

    /// The ids of the active projection, in order. What
    /// [`Session::restore_context`] takes.
    pub fn active_ids(&self) -> &[MessageId] {
        &self.active
    }

    /// Every message version that has ever existed here, oldest first, with
    /// whether each is currently active and what replaced it.
    pub fn all_messages(&self) -> Vec<MessageRecord<'_>> {
        self.records
            .iter()
            .map(|r| MessageRecord {
                id: r.id,
                message: &r.message,
                active: self.active.contains(&r.id),
                replaced_by: r.replaced_by,
            })
            .collect()
    }

    /// One record by id, active or not.
    pub fn message(&self, id: MessageId) -> Option<&Message> {
        self.index.get(&id).map(|&i| &self.records[i].message)
    }

    /// Everything that has happened to this session, oldest first.
    pub fn events(&self) -> &[SessionEvent] {
        &self.events
    }

    /// The most recent active message — after a turn, the assistant's reply.
    pub fn latest(&self) -> Option<&Message> {
        self.projection.last()
    }

    /// The most recent active message's text.
    pub fn latest_text(&self) -> Option<String> {
        self.latest().map(Message::text)
    }

    /// How many messages the active projection holds.
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// Whether the active projection is empty.
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// How many messages have fallen out of the model's context window.
    ///
    /// Grows when a turn overflows the window and the engine sheds history to
    /// make room. This counts what left the *model's* view; the session's
    /// messages are all still here.
    pub fn shed(&self) -> usize {
        self.shed
    }

    /// Whether the model can still see the whole conversation.
    pub fn fully_in_context(&self) -> bool {
        self.shed == 0
    }

    // ── System prompt (§7.3) ────────────────────────────────────────────────

    /// Set the system prompt. Setting the text already in place is not a
    /// change and records nothing.
    pub fn set_system(&mut self, text: impl Into<String>) {
        self.put_system(Some(text.into()));
    }

    /// Append to the system prompt (or set it, if there is none).
    pub fn append_system(&mut self, text: impl AsRef<str>) {
        let mut system = self.system.clone().unwrap_or_default();
        system.push_str(text.as_ref());
        self.put_system(Some(system));
    }

    /// Remove the system prompt.
    pub fn clear_system(&mut self) {
        self.put_system(None);
    }

    fn put_system(&mut self, system: Option<String>) {
        if self.system == system {
            return;
        }
        self.system = system.clone();
        self.log(SessionEvent::SystemSet { system });
        // The prefix the model was opened with is gone.
        self.opened = false;
    }

    // ── Tools (§7.4) ────────────────────────────────────────────────────────

    /// Replace the tool set. The same set again is not a change.
    pub fn set_tools(&mut self, tools: impl Into<ToolSet>) {
        self.put_tools(tools.into());
    }

    /// Add one tool (replacing a same-named one in place).
    pub fn add_tool(&mut self, tool: impl Into<ToolDefinition>) {
        let mut tools = self.tools.clone();
        tools.add(tool);
        self.put_tools(tools);
    }

    /// Remove a tool by name. Returns whether it was there.
    pub fn remove_tool(&mut self, name: &str) -> bool {
        let mut tools = self.tools.clone();
        let removed = tools.remove(name).is_some();
        self.put_tools(tools);
        removed
    }

    fn put_tools(&mut self, tools: ToolSet) {
        if self.tools == tools {
            return;
        }
        self.tools = tools.clone();
        self.log(SessionEvent::ToolsSet { tools });
        self.opened = false;
    }

    // ── Appending (§7.5, §9) ────────────────────────────────────────────────

    /// Append a message to the active projection.
    pub fn push(&mut self, message: Message) -> MessageId {
        let id = self.mint(message.clone());
        self.active.push(id);
        self.projection.push(message.clone());
        self.log(SessionEvent::MessageAdded { id, message });
        id
    }

    /// Append a user message.
    pub fn push_user(&mut self, text: impl Into<String>) -> MessageId {
        self.push(Message::user(text))
    }

    /// Append a user message carrying images.
    ///
    /// Paths become `file://` URLs; already-formed `http(s)://` or `file://`
    /// URLs pass through. With no images this is exactly [`Session::push_user`],
    /// so a caller can pass an empty slice unconditionally.
    ///
    /// The model must be multimodal and loaded with a projector — see
    /// [`EngineBuilder::mmproj`](super::EngineBuilder::mmproj).
    pub fn push_user_with_images<I, P>(&mut self, text: impl Into<String>, images: I) -> MessageId
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        self.push(Message::user_with_images(
            text,
            images
                .into_iter()
                .map(|p| crate::types::message::to_file_url(p.as_ref())),
        ))
    }

    /// Append a tool's result, tied to the call it answers (§9.4).
    ///
    /// `call_id` is the id from the call the model made — a
    /// [`ToolCallId`](crate::output::ToolCallId), or its text — so parallel
    /// calls stay distinguishable all the way back to the model.
    pub fn push_tool_result(
        &mut self,
        call_id: impl AsRef<str>,
        result: impl Into<ToolResult>,
    ) -> MessageId {
        let result = result.into();
        self.push(Message::tool_result_for(
            call_id.as_ref(),
            result.to_model_text(),
        ))
    }

    // ── Editing (§7.6–§7.8) ─────────────────────────────────────────────────

    /// Replace an active message with a new version in the same position.
    ///
    /// The old record stays in [`Session::all_messages`], marked with what
    /// replaced it, and the event log says so. Returns the new record's id.
    ///
    /// Refused if `old` is unknown or not active, or if the new projection
    /// would split a tool round (a call's results answering a message that no
    /// longer makes those calls).
    pub fn replace_message(&mut self, old: MessageId, message: Message) -> Result<MessageId> {
        let slot = self.active_slot(old)?;
        let new = MessageId(self.next_id);
        let mut next = self.active.clone();
        next[slot] = new;
        self.check_rounds(&next, Some((new, &message)))?;

        let new = self.mint(message.clone());
        self.records[self.index[&old]].replaced_by = Some(new);
        self.active = next;
        self.log(SessionEvent::MessageReplaced { old, new, message });
        self.after_projection_change();
        Ok(new)
    }

    /// Take a message out of the active projection.
    ///
    /// The record stays in [`Session::all_messages`]. Refused if the id is
    /// unknown or not active, or if removing it would leave a tool result
    /// without its call or a call without its result.
    pub fn remove_message(&mut self, id: MessageId) -> Result<()> {
        let slot = self.active_slot(id)?;
        let mut next = self.active.clone();
        next.remove(slot);
        self.check_rounds(&next, None)?;

        self.active = next;
        self.log(SessionEvent::MessageRemoved { id });
        self.after_projection_change();
        Ok(())
    }

    /// Replace the whole active projection — compaction, say (§7.8, §20).
    ///
    /// Every message given becomes a new record; the old ones stay in
    /// [`Session::all_messages`] and the log records a context replacement,
    /// not a deletion. Refused, with nothing changed, if the new projection
    /// would show half a tool round.
    pub fn replace_messages(&mut self, messages: impl IntoIterator<Item = Message>) -> Result<()> {
        let messages: Vec<Message> = messages.into_iter().collect();
        // Check before minting anything, so a refusal leaves no trace.
        let ids: Vec<MessageId> = (0..messages.len() as u64)
            .map(|i| MessageId(self.next_id + i))
            .collect();
        {
            let staged: Vec<(MessageId, &Message)> = ids.iter().copied().zip(&messages).collect();
            check_rounds_over(&self.lookup_with(&staged), &ids, &self.records)?;
        }
        let ids: Vec<MessageId> = messages.into_iter().map(|m| self.record(m)).collect();
        self.set_active(ids, "replace_messages", true);
        Ok(())
    }

    /// Restore an earlier projection by id — undo, branch navigation,
    /// retry-from-here (§7.8).
    ///
    /// Every id must name a record in this session; records need not be
    /// active now. Refused if any id is unknown or repeated, or if the
    /// projection would show half a tool round.
    pub fn restore_context(&mut self, ids: impl IntoIterator<Item = MessageId>) -> Result<()> {
        let ids: Vec<MessageId> = ids.into_iter().collect();
        let mut seen = std::collections::HashSet::new();
        for id in &ids {
            if !self.index.contains_key(id) {
                return Err(SessionError::UnknownMessage(*id).into());
            }
            if !seen.insert(*id) {
                return Err(SessionError::Duplicate(*id).into());
            }
        }
        self.check_rounds(&ids, None)?;
        self.set_active(ids, "restore_context", true);
        Ok(())
    }

    /// Edit the active projection in place — trim it, delete a message,
    /// rewrite one.
    ///
    /// Lossless like every other mutation: the result is recorded as a
    /// context replacement over new records where a message changed, and the
    /// previous records stay in [`Session::all_messages`]. Messages the
    /// closure leaves unchanged keep their ids.
    ///
    /// This cannot refuse, so where the closure leaves half a tool round it
    /// trims to the round boundary: a result whose call was removed goes too,
    /// and so does a call whose results were. Prefer
    /// [`Session::replace_messages`] or [`Session::remove_message`] when you
    /// want a refusal instead.
    ///
    /// Marks the conversation unopened either way: the engine's cached prefill
    /// describes the old history.
    pub fn edit(&mut self, f: impl FnOnce(&mut Vec<Message>)) {
        let mut messages = self.projection.clone();
        f(&mut messages);
        let ids = self.ids_for_edited(messages);
        let ids = self.trim_to_round_boundary(ids);
        if ids != self.active {
            self.set_active(ids, "edit", true);
        } else {
            self.opened = false;
        }
    }

    /// Empty the active projection and start over, keeping the same id, the
    /// system prompt, and the tools.
    ///
    /// The old messages stay in [`Session::all_messages`]; this is a context
    /// replacement, not a deletion. Marks the conversation unopened, so the
    /// next turn rebuilds the engine's state.
    pub fn clear(&mut self) {
        if !self.active.is_empty() {
            self.set_active(Vec::new(), "clear", true);
        }
        self.opened = false;
        self.shed = 0;
        self.tools_fingerprint = None;
        self.model_generation = None;
    }

    // ── Forking (§7.9) ──────────────────────────────────────────────────────

    /// Branch this conversation into an independent copy.
    ///
    /// The fork has a new id, the same system prompt, tools and active
    /// projection, and the parent's whole event log and records — its
    /// provenance — followed by a [`SessionEvent::Forked`]. From there the
    /// two diverge: independent future events, independent engine state.
    ///
    /// The fork starts unopened: the engine has nothing cached for it, so its
    /// first turn resends the history once and is warm from then on.
    pub fn fork(&self) -> Self {
        self.fork_with(self.active.clone(), None)
    }

    /// Branch, keeping the active projection only up to and including `at`.
    ///
    /// Refused if `at` is unknown or not active, or if cutting there would
    /// split a tool round.
    pub fn fork_at(&self, at: MessageId) -> Result<Self> {
        let slot = self.active_slot(at)?;
        let active: Vec<MessageId> = self.active[..=slot].to_vec();
        self.check_rounds(&active, None)?;
        Ok(self.fork_with(active, Some(at)))
    }

    fn fork_with(&self, active: Vec<MessageId>, at: Option<MessageId>) -> Self {
        let mut fork = Self {
            id: SessionId::random(),
            events: self.events.clone(),
            system: self.system.clone(),
            tools: self.tools.clone(),
            records: self.records.clone(),
            index: self.index.clone(),
            active,
            projection: Vec::new(),
            next_id: self.next_id,
            revision: self.revision,
            opened: false,
            shed: 0,
            tools_fingerprint: None,
            model_generation: None,
        };
        fork.log(SessionEvent::Forked {
            parent: self.id.clone(),
            at,
        });
        fork.rebuild_projection();
        fork
    }

    // ── Engine-facing ───────────────────────────────────────────────────────

    /// The messages the model is rendered from: the system prompt, if any,
    /// followed by the active projection.
    ///
    /// The one place the system prompt rejoins the message list — it is what
    /// the backends' templates, truncation and compaction expect at index 0.
    pub(crate) fn transcript(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.projection.len() + 1);
        if let Some(system) = &self.system {
            out.push(Message::system(system.clone()));
        }
        out.extend(self.projection.iter().cloned());
        out
    }

    /// Messages the engine still needs for the next turn.
    ///
    /// A conversation the engine already has open only needs what's new since;
    /// one it doesn't needs the lot. Counts are over [`Session::transcript`],
    /// system prompt included, because that is what the engine was handed.
    pub(crate) fn pending(&self, sent_through: usize) -> Vec<Message> {
        let mut transcript = self.transcript();
        if self.opened {
            transcript.drain(..sent_through.min(transcript.len()));
        }
        transcript
    }

    /// Undo messages appended since the projection had `len` of them.
    ///
    /// For a turn rejected before anything was sent: the builder appends as it
    /// is configured, so a turn that never runs would otherwise leave its
    /// messages behind. Lossless — the withdrawn records stay in the log.
    /// Does not close the conversation: nothing withdrawn ever reached the
    /// engine.
    pub(crate) fn rollback_to(&mut self, len: usize) {
        if self.active.len() <= len {
            return;
        }
        let ids = self.trim_to_round_boundary(self.active[..len].to_vec());
        self.set_active(ids, "rollback", false);
    }

    /// Record that the engine shed `n` messages this turn.
    pub(crate) fn note_shed(&mut self, n: usize) {
        self.shed = self.shed.saturating_add(n);
    }

    /// Declare which model generation a turn is about to run against.
    ///
    /// A different one reopens the conversation, because the engine's cached
    /// prefill was produced by weights that are no longer loaded.
    ///
    /// Returns whether the conversation was reopened.
    pub(crate) fn note_model(&mut self, generation: u64) -> bool {
        match self.model_generation {
            Some(current) if current == generation => false,
            None if !self.opened => {
                self.model_generation = Some(generation);
                false
            }
            _ => {
                self.model_generation = Some(generation);
                self.opened = false;
                true
            }
        }
    }

    /// Declare the executable tool prefix a run is about to use.
    ///
    /// A set different from the one this conversation was opened with reopens
    /// it, so the new definitions actually reach the model.
    ///
    /// Returns whether the conversation was reopened.
    pub(crate) fn note_tools(&mut self, fingerprint: u64) -> bool {
        match self.tools_fingerprint {
            Some(current) if current == fingerprint => false,
            None if !self.opened => {
                self.tools_fingerprint = Some(fingerprint);
                false
            }
            _ => {
                self.tools_fingerprint = Some(fingerprint);
                self.opened = false;
                true
            }
        }
    }

    // ── Internals ───────────────────────────────────────────────────────────

    /// Append an event, moving the revision when it is context-affecting.
    fn log(&mut self, event: SessionEvent) {
        if !matches!(event, SessionEvent::MessageRecorded { .. }) {
            self.revision += 1;
        }
        self.events.push(event);
    }

    /// Store a new record for a context replacement to name, and log it so
    /// a replay can rebuild it.
    fn record(&mut self, message: Message) -> MessageId {
        let id = self.mint(message.clone());
        self.log(SessionEvent::MessageRecorded { id, message });
        id
    }

    /// Store a new record and return its id. The caller logs it.
    fn mint(&mut self, message: Message) -> MessageId {
        let id = MessageId(self.next_id);
        self.next_id += 1;
        self.index.insert(id, self.records.len());
        self.records.push(StoredRecord {
            id,
            message,
            replaced_by: None,
        });
        id
    }

    /// Where an active id sits in the projection.
    fn active_slot(&self, id: MessageId) -> std::result::Result<usize, SessionError> {
        if !self.index.contains_key(&id) {
            return Err(SessionError::UnknownMessage(id));
        }
        self.active
            .iter()
            .position(|a| *a == id)
            .ok_or(SessionError::NotActive(id))
    }

    /// Commit a new projection as a `ContextReplaced` event.
    fn set_active(&mut self, active: Vec<MessageId>, reason: &str, close: bool) {
        if active == self.active {
            // The same projection again is not a change.
            return;
        }
        self.active = active.clone();
        self.log(SessionEvent::ContextReplaced {
            active,
            reason: Some(reason.to_string()),
        });
        if close {
            self.after_projection_change();
        } else {
            self.rebuild_projection();
        }
    }

    fn after_projection_change(&mut self) {
        self.rebuild_projection();
        // The engine's cached prefill describes the old projection.
        self.opened = false;
    }

    fn rebuild_projection(&mut self) {
        self.projection = self
            .active
            .iter()
            .map(|id| self.records[self.index[id]].message.clone())
            .collect();
    }

    /// Ids for an edited projection: an unchanged message keeps its record,
    /// a changed one gets a new record. Matching is in order, so a message
    /// moved earlier than one it used to follow is treated as new.
    fn ids_for_edited(&mut self, messages: Vec<Message>) -> Vec<MessageId> {
        let mut cursor = 0;
        let mut ids = Vec::with_capacity(messages.len());
        for message in messages {
            let found = self.active[cursor..]
                .iter()
                .position(|id| self.records[self.index[id]].message == message);
            match found {
                Some(offset) => {
                    let id = self.active[cursor + offset];
                    cursor += offset + 1;
                    ids.push(id);
                }
                None => ids.push(self.record(message)),
            }
        }
        ids
    }

    /// Drop whatever half-rounds a projection contains, to a fixpoint.
    fn trim_to_round_boundary(&self, mut ids: Vec<MessageId>) -> Vec<MessageId> {
        loop {
            let lookup = self.lookup_with(&[]);
            let drop: Vec<MessageId> = match check_rounds_over(&lookup, &ids, &self.records) {
                Ok(()) => return ids,
                // A result whose call is not shown.
                Err(SessionError::OrphanResult { call_id }) if call_id.is_empty() => {
                    // Id-less: the first one the walk could not attribute.
                    attribute_legacy(&lookup, &ids)
                        .into_iter()
                        .find(|(_, attributed)| attributed.is_empty())
                        .map(|(m, _)| m)
                        .and_then(|orphan| {
                            ids.iter()
                                .copied()
                                .find(|id| std::ptr::eq(self.message_by(*id), orphan))
                        })
                        .into_iter()
                        .collect()
                }
                Err(SessionError::OrphanResult { call_id }) => ids
                    .iter()
                    .copied()
                    .filter(|id| {
                        let m = self.message_by(*id);
                        is_result(m) && m.tool_call_id.as_deref() == Some(call_id.as_str())
                    })
                    .collect(),
                // A call whose results are not shown.
                Err(SessionError::UnansweredCall { call_id }) => ids
                    .iter()
                    .copied()
                    .filter(|id| makes_call(self.message_by(*id), &call_id))
                    .collect(),
                Err(_) => return ids,
            };
            if drop.is_empty() {
                // Nothing addressable to drop; better a stale round than a
                // loop that never ends.
                return ids;
            }
            ids.retain(|id| !drop.contains(id));
        }
    }

    fn message_by(&self, id: MessageId) -> &Message {
        &self.records[self.index[&id]].message
    }

    /// A lookup over stored records plus not-yet-minted staged messages.
    fn lookup_with<'a>(
        &'a self,
        staged: &'a [(MessageId, &'a Message)],
    ) -> impl Fn(MessageId) -> Option<&'a Message> + 'a {
        move |id| {
            staged
                .iter()
                .find(|(s, _)| *s == id)
                .map(|(_, m)| *m)
                .or_else(|| self.index.get(&id).map(|&i| &self.records[i].message))
        }
    }

    /// Check a candidate projection, with an optional message that is about
    /// to be minted under a given id.
    fn check_rounds(
        &self,
        ids: &[MessageId],
        staged: Option<(MessageId, &Message)>,
    ) -> std::result::Result<(), SessionError> {
        let staged: Vec<(MessageId, &Message)> = staged.into_iter().collect();
        check_rounds_over(&self.lookup_with(&staged), ids, &self.records)
    }
}

// ── The tool-round invariant ────────────────────────────────────────────────

/// The call ids an assistant tool-call message makes.
fn calls_of(message: &Message) -> Vec<&str> {
    match &message.body {
        MessageBody::Tool { tool_calls } => tool_calls.iter().map(|c| c.id.as_str()).collect(),
        _ => Vec::new(),
    }
}

fn makes_call(message: &Message, call_id: &str) -> bool {
    calls_of(message).contains(&call_id)
}

fn is_result(message: &Message) -> bool {
    message.role == "tool"
}

/// Attribute id-less results to calls the way a backend would: each one
/// answers the first unanswered call of the nearest preceding call message.
/// Returns `(message, call id)` pairs; an unattributable result gets `""`.
fn attribute_legacy<'a>(
    lookup: &impl Fn(MessageId) -> Option<&'a Message>,
    ids: &[MessageId],
) -> Vec<(&'a Message, String)> {
    let mut out = Vec::new();
    let mut open: Vec<String> = Vec::new();
    for id in ids {
        let Some(m) = lookup(*id) else { continue };
        let calls = calls_of(m);
        if !calls.is_empty() {
            open = calls.iter().map(|c| c.to_string()).collect();
            continue;
        }
        if is_result(m) {
            match m.tool_call_id.as_deref() {
                Some(cid) => open.retain(|o| o != cid),
                None => {
                    let attributed = if open.is_empty() {
                        String::new()
                    } else {
                        open.remove(0)
                    };
                    out.push((m, attributed));
                }
            }
        }
    }
    out
}

/// The invariant `journal::Turn` holds by construction, checked over a
/// projection: no result without its call, no call without a result that
/// exists.
///
/// A call nothing has answered *anywhere* in the records is not a violation
/// — the round is in progress, or was interrupted — which is what lets a
/// harness heal a transcript by removing the dangling call alone.
fn check_rounds_over<'a>(
    lookup: &impl Fn(MessageId) -> Option<&'a Message>,
    ids: &[MessageId],
    records: &[StoredRecord],
) -> std::result::Result<(), SessionError> {
    // Walk forward: every result must answer a call already shown.
    let mut declared: Vec<String> = Vec::new();
    let mut answered: Vec<String> = Vec::new();
    let mut legacy_open: Vec<String> = Vec::new();
    for id in ids {
        let Some(m) = lookup(*id) else { continue };
        let calls = calls_of(m);
        if !calls.is_empty() {
            declared.extend(calls.iter().map(|c| c.to_string()));
            legacy_open = calls.iter().map(|c| c.to_string()).collect();
            continue;
        }
        if is_result(m) {
            match m.tool_call_id.as_deref() {
                Some(cid) => {
                    if !declared.iter().any(|d| d == cid) {
                        return Err(SessionError::OrphanResult {
                            call_id: cid.to_string(),
                        });
                    }
                    answered.push(cid.to_string());
                    legacy_open.retain(|o| o != cid);
                }
                None => {
                    if legacy_open.is_empty() {
                        return Err(SessionError::OrphanResult {
                            call_id: String::new(),
                        });
                    }
                    answered.push(legacy_open.remove(0));
                }
            }
        }
    }

    // Every shown call must be answered if an answer exists at all.
    let mut answered_somewhere: Vec<&str> = Vec::new();
    for r in records {
        if is_result(&r.message)
            && let Some(cid) = r.message.tool_call_id.as_deref()
        {
            answered_somewhere.push(cid);
        }
    }
    for call in &declared {
        if !answered.iter().any(|a| a == call) && answered_somewhere.contains(&call.as_str()) {
            return Err(SessionError::UnansweredCall {
                call_id: call.clone(),
            });
        }
    }
    Ok(())
}

// ── Serde: the event log is the truth ───────────────────────────────────────

#[derive(Serialize)]
struct SessionWireRef<'a> {
    id: &'a SessionId,
    events: &'a [SessionEvent],
}

#[derive(Deserialize)]
struct SessionWire {
    id: SessionId,
    #[serde(default)]
    events: Vec<SessionEvent>,
    /// The pre-event format: a flat transcript. Read as
    /// [`Session::from_messages`] would, so stored conversations still open.
    #[serde(default)]
    messages: Vec<Message>,
}

impl Serialize for Session {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        SessionWireRef {
            id: &self.id,
            events: &self.events,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Session {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let wire = SessionWire::deserialize(deserializer)?;
        if wire.events.is_empty() && !wire.messages.is_empty() {
            let mut session = Session::from_messages(wire.messages);
            session.id = wire.id;
            return Ok(session);
        }
        Ok(Session::replay(wire.id, wire.events))
    }
}

impl Session {
    /// Rebuild every derived field from an event log.
    fn replay(id: SessionId, events: Vec<SessionEvent>) -> Self {
        let mut s = Self::blank(id);
        for event in &events {
            match event {
                SessionEvent::SystemSet { system } => s.system = system.clone(),
                SessionEvent::ToolsSet { tools } => s.tools = tools.clone(),
                SessionEvent::MessageAdded { id, message } => {
                    s.store(*id, message.clone());
                    s.active.push(*id);
                }
                SessionEvent::MessageReplaced { old, new, message } => {
                    s.store(*new, message.clone());
                    if let Some(&i) = s.index.get(old) {
                        s.records[i].replaced_by = Some(*new);
                    }
                    match s.active.iter().position(|a| a == old) {
                        Some(slot) => s.active[slot] = *new,
                        None => s.active.push(*new),
                    }
                }
                SessionEvent::MessageRemoved { id } => s.active.retain(|a| a != id),
                SessionEvent::MessageRecorded { id, message } => s.store(*id, message.clone()),
                SessionEvent::ContextReplaced { active, .. } => {
                    s.active = active
                        .iter()
                        .copied()
                        .filter(|id| s.index.contains_key(id))
                        .collect();
                }
                SessionEvent::Forked { at: Some(at), .. } => {
                    if let Some(slot) = s.active.iter().position(|a| a == at) {
                        s.active.truncate(slot + 1);
                    }
                }
                SessionEvent::Forked { at: None, .. } => {}
            }
        }
        s.revision = events
            .iter()
            .filter(|e| !matches!(e, SessionEvent::MessageRecorded { .. }))
            .count() as u64;
        s.events = events;
        s.rebuild_projection();
        s
    }

    /// Store a record under a given id (replay only).
    fn store(&mut self, id: MessageId, message: Message) {
        self.next_id = self.next_id.max(id.0 + 1);
        self.index.insert(id, self.records.len());
        self.records.push(StoredRecord {
            id,
            message,
            replaced_by: None,
        });
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::{FunctionDefinition, ToolCall};

    fn call_msg(ids: &[&str]) -> Message {
        Message::assistant_tool_calls(
            ids.iter()
                .map(|id| ToolCall {
                    id: id.to_string(),
                    r#type: "function".into(),
                    function: FunctionDefinition {
                        description: None,
                        name: "t".into(),
                        arguments: serde_json::json!({}),
                    },
                })
                .collect(),
        )
    }

    fn texts(s: &Session) -> Vec<String> {
        s.messages().iter().map(Message::text).collect()
    }

    /// The §29 invariants every mutation test asserts.
    ///
    /// 5: edits are lossless — nothing an earlier snapshot recorded is gone.
    /// 6: active and record are separate — the projection is a subset of the
    /// records, and the records are never fewer.
    fn assert_invariants(before: &[SessionEvent], before_records: usize, s: &Session) {
        assert!(
            s.events().starts_with(before),
            "earlier events changed or vanished"
        );
        assert!(
            s.all_messages().len() >= before_records,
            "a record was lost"
        );
        for m in s.messages() {
            assert!(
                s.all_messages().iter().any(|r| r.message == m && r.active),
                "an active message is not among the records"
            );
        }
        assert_eq!(
            s.all_messages().iter().filter(|r| r.active).count(),
            s.len(),
            "active flags disagree with the projection"
        );
        let context_events = s
            .events()
            .iter()
            .filter(|e| !matches!(e, SessionEvent::MessageRecorded { .. }))
            .count() as u64;
        assert_eq!(s.revision().as_u64(), context_events);
    }

    // ── Identity and basics ─────────────────────────────────────────────

    #[test]
    fn sessions_get_distinct_ids() {
        assert_ne!(Session::new().id(), Session::new().id());
    }

    #[test]
    fn latest_is_the_most_recent_message() {
        let mut s = Session::new();
        s.push_user("first");
        s.push_user("second");
        assert_eq!(s.latest_text().as_deref(), Some("second"));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn a_new_session_has_no_history_and_revision_zero() {
        let s = Session::new();
        assert_eq!(s.revision().as_u64(), 0);
        assert!(s.events().is_empty());
        assert!(s.all_messages().is_empty());
        assert_eq!(s.system(), None);
        assert!(s.tools().is_empty());
    }

    #[test]
    fn reads_do_not_change_the_revision() {
        let mut s = Session::new().with_system("sys");
        s.push_user("hi");
        let r = s.revision();
        let _ = (
            s.messages(),
            s.all_messages(),
            s.events(),
            s.system(),
            s.tools(),
            s.latest(),
            s.latest_text(),
            s.transcript(),
            s.pending(0),
            s.len(),
        );
        assert_eq!(s.revision(), r);
    }

    // ── §7.3 system prompt ──────────────────────────────────────────────

    #[test]
    fn the_system_prompt_is_state_not_a_message() {
        let s = Session::new().with_system("Be terse.");
        assert_eq!(s.system(), Some("Be terse."));
        assert!(s.messages().is_empty(), "not in the projection");
        assert!(s.all_messages().is_empty(), "not a record either");
        assert_eq!(s.revision().as_u64(), 1);
        assert!(matches!(
            s.events(),
            [SessionEvent::SystemSet { system: Some(t) }] if t == "Be terse."
        ));
        // But it is what the model is rendered from, at index 0.
        let transcript = s.transcript();
        assert_eq!(transcript.len(), 1);
        assert_eq!(transcript[0].role, "system");
        assert_eq!(transcript[0].text(), "Be terse.");
    }

    #[test]
    fn set_and_append_system_bump_the_revision_and_close_the_conversation() {
        let mut s = Session::new().with_system("a");
        s.push_user("hi");
        s.opened = true;
        let before = s.events().to_vec();
        let r = s.revision();

        s.append_system("b");
        assert_eq!(s.system(), Some("ab"));
        assert!(s.revision() > r);
        assert!(!s.opened, "the prefix changed");
        assert_eq!(s.len(), 1, "messages untouched");
        assert_invariants(&before, 1, &s);

        let r = s.revision();
        s.set_system("ab");
        assert_eq!(s.revision(), r, "the same text again is not a change");

        s.clear_system();
        assert_eq!(s.system(), None);
        assert!(s.transcript().iter().all(|m| m.role != "system"));
    }

    // ── §7.4 tools ──────────────────────────────────────────────────────

    #[test]
    fn tools_are_state_with_deterministic_order() {
        let mut s = Session::new();
        s.opened = true;
        let r = s.revision();
        s.add_tool(ToolDefinition::new("read"));
        s.add_tool(ToolDefinition::new("write"));
        assert_eq!(s.tools().names(), ["read", "write"]);
        assert!(s.revision() > r);
        assert!(!s.opened);
        assert!(s.messages().is_empty(), "tools are not messages");

        assert!(s.remove_tool("read"));
        assert!(!s.remove_tool("read"), "already gone");
        assert_eq!(s.tools().names(), ["write"]);

        let r = s.revision();
        s.set_tools(s.tools().clone());
        assert_eq!(s.revision(), r, "the same set again is not a change");

        let set_events = s
            .events()
            .iter()
            .filter(|e| matches!(e, SessionEvent::ToolsSet { .. }))
            .count();
        assert_eq!(set_events, 3, "add, add, remove — each recorded whole");
    }

    // ── §7.5 identity ───────────────────────────────────────────────────

    #[test]
    fn every_push_gets_a_fresh_id_and_a_record() {
        let mut s = Session::new();
        let a = s.push_user("a");
        let b = s.push(Message::user("b"));
        assert_ne!(a, b);
        assert_eq!(s.active_ids(), [a, b]);
        assert_eq!(s.message(a).map(Message::text).as_deref(), Some("a"));
        let records = s.all_messages();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|r| r.active && r.replaced_by.is_none()));
        assert_eq!(s.revision().as_u64(), 2);
    }

    // ── §7.6 / §28.8 replace ────────────────────────────────────────────

    #[test]
    fn walkthrough_28_8_edit_a_user_message_without_losing_history() {
        let mut s = Session::new();
        let id = s.push_user("helo");
        let before = s.events().to_vec();

        let new = s
            .replace_message(id, Message::user("hello"))
            .expect("a plain edit");

        assert_eq!(s.messages().last().unwrap().text(), "hello");
        assert_eq!(s.len(), 1);

        // Both revisions still exist.
        let records = s.all_messages();
        assert_eq!(records.len(), 2);
        let old = records.iter().find(|r| r.id == id).unwrap();
        let replacement = records.iter().find(|r| r.id == new).unwrap();
        assert_eq!(old.message.text(), "helo");
        assert!(!old.active);
        assert_eq!(old.replaced_by, Some(new));
        assert_eq!(replacement.message.text(), "hello");
        assert!(replacement.active);
        assert_eq!(replacement.replaced_by, None);

        // And the log explains it.
        assert!(matches!(
            s.events().last(),
            Some(SessionEvent::MessageReplaced { old, new: n, .. }) if *old == id && *n == new
        ));
        assert_invariants(&before, 1, &s);
    }

    #[test]
    fn replacing_keeps_the_position() {
        let mut s = Session::new();
        s.push_user("a");
        let b = s.push_user("b");
        s.push_user("c");
        s.replace_message(b, Message::user("B")).unwrap();
        assert_eq!(texts(&s), ["a", "B", "c"]);
    }

    #[test]
    fn replacing_an_unknown_or_inactive_message_is_refused() {
        let mut s = Session::new();
        let a = s.push_user("a");
        s.remove_message(a).unwrap();
        let events = s.events().to_vec();
        let err = s.replace_message(a, Message::user("x")).unwrap_err();
        assert!(matches!(
            err,
            super::super::Error::Session(SessionError::NotActive(id)) if id == a
        ));
        let err = s
            .replace_message(MessageId(99), Message::user("x"))
            .unwrap_err();
        assert!(matches!(
            err,
            super::super::Error::Session(SessionError::UnknownMessage(_))
        ));
        assert_eq!(s.events(), &events[..], "a refusal leaves no trace");
    }

    // ── §7.7 remove ─────────────────────────────────────────────────────

    #[test]
    fn removing_hides_a_message_without_erasing_it() {
        let mut s = Session::new();
        let a = s.push_user("a");
        let b = s.push_user("b");
        let before = s.events().to_vec();
        s.remove_message(a).unwrap();
        assert_eq!(s.active_ids(), [b]);
        assert_eq!(texts(&s), ["b"]);
        let records = s.all_messages();
        assert_eq!(records.len(), 2);
        assert!(!records[0].active);
        assert_eq!(records[0].message.text(), "a");
        assert!(matches!(
            s.events().last(),
            Some(SessionEvent::MessageRemoved { id }) if *id == a
        ));
        assert_invariants(&before, 2, &s);
        assert!(s.remove_message(a).is_err(), "not active any more");
    }

    // ── §7.8 / §28.9 replace_messages, restore_context ─────────────────

    #[test]
    fn walkthrough_28_9_compact_without_erasing_old_history() {
        let mut s = Session::new().with_system("sys");
        s.push_user("one");
        s.push(Message::assistant_structured("two", None));
        s.push_user("three");
        let latest = s.latest().unwrap().clone();
        let before = s.events().to_vec();
        let before_ids = s.active_ids().to_vec();

        s.replace_messages([Message::user("Summary of prior context: ..."), latest])
            .expect("compaction");

        assert_eq!(texts(&s), ["Summary of prior context: ...", "three"]);
        assert_eq!(s.system(), Some("sys"), "instruction state untouched");
        // `all_messages()` and `events()` retain pre-compaction history.
        let records = s.all_messages();
        assert_eq!(records.len(), 5);
        assert!(
            records
                .iter()
                .filter(|r| !r.active)
                .map(|r| r.message.text())
                .eq(["one", "two", "three"]),
            "the three originals are all still here, inactive"
        );
        assert!(matches!(
            s.events().last(),
            Some(SessionEvent::ContextReplaced { active, .. }) if active.len() == 2
        ));
        assert_invariants(&before, 3, &s);

        // And undo is a restore of the old ids.
        s.restore_context(before_ids.clone()).expect("undo");
        assert_eq!(texts(&s), ["one", "two", "three"]);
        assert_eq!(s.active_ids(), &before_ids[..]);
        assert_eq!(s.all_messages().len(), 5, "restore mints nothing");
    }

    #[test]
    fn restore_context_refuses_unknown_and_duplicate_ids() {
        let mut s = Session::new();
        let a = s.push_user("a");
        let events = s.events().to_vec();
        assert!(matches!(
            s.restore_context([a, MessageId(7)]).unwrap_err(),
            super::super::Error::Session(SessionError::UnknownMessage(_))
        ));
        assert!(matches!(
            s.restore_context([a, a]).unwrap_err(),
            super::super::Error::Session(SessionError::Duplicate(_))
        ));
        assert_eq!(s.events(), &events[..]);
    }

    #[test]
    fn restore_context_can_reorder_and_revive() {
        let mut s = Session::new();
        let a = s.push_user("a");
        let b = s.push_user("b");
        s.remove_message(a).unwrap();
        s.restore_context([b, a]).unwrap();
        assert_eq!(texts(&s), ["b", "a"]);
    }

    // ── §7.9 fork ───────────────────────────────────────────────────────

    #[test]
    fn a_fork_shares_state_and_provenance_but_not_identity() {
        let mut a = Session::new().with_system("sys");
        a.add_tool(ToolDefinition::new("read"));
        let first = a.push_user("hello");
        a.opened = true;
        a.note_shed(2);

        let b = a.fork();
        assert_ne!(
            b.id(),
            a.id(),
            "a fork must not share the engine's cache key"
        );
        assert_eq!(b.messages(), a.messages());
        assert_eq!(b.system(), a.system());
        assert_eq!(b.tools(), a.tools());
        assert!(!b.opened, "the engine has nothing cached for a fork");
        assert_eq!(b.shed(), 0, "a fork is resent whole, so nothing is missing");

        // Provenance: the parent's log verbatim, then the fork event.
        assert!(b.events().starts_with(a.events()));
        assert!(matches!(
            b.events().last(),
            Some(SessionEvent::Forked { parent, at: None }) if parent == a.id()
        ));
        assert_eq!(b.revision().as_u64(), a.revision().as_u64() + 1);
        // Ids survive, so a parent's id still names the same record.
        assert_eq!(b.message(first).map(Message::text), Some("hello".into()));
    }

    #[test]
    fn the_two_branches_of_a_fork_diverge_independently() {
        let mut a = Session::new();
        a.push_user("shared");
        let mut b = a.fork();
        a.push_user("only in a");
        b.push_user("only in b");

        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(a.latest_text().as_deref(), Some("only in a"));
        assert_eq!(b.latest_text().as_deref(), Some("only in b"));
    }

    #[test]
    fn fork_at_cuts_the_projection_after_the_named_message() {
        let mut a = Session::new();
        let one = a.push_user("one");
        a.push_user("two");
        a.push_user("three");
        let b = a.fork_at(one).unwrap();
        assert_eq!(texts(&b), ["one"]);
        assert_eq!(b.all_messages().len(), 3, "the records all came along");
        assert!(matches!(
            b.events().last(),
            Some(SessionEvent::Forked { at: Some(id), .. }) if *id == one
        ));
        assert_eq!(
            texts(&a),
            ["one", "two", "three"],
            "the parent is untouched"
        );
        assert!(a.fork_at(MessageId(42)).is_err());
    }

    // ── §7.10 revision ──────────────────────────────────────────────────

    #[test]
    fn every_context_affecting_mutation_bumps_the_revision_exactly_once() {
        let mut s = Session::new();
        let mut last = s.revision();
        let mut bumped = |s: &Session, what: &str| {
            assert_eq!(
                s.revision().as_u64(),
                last.as_u64() + 1,
                "{what} should bump exactly once"
            );
            last = s.revision();
        };
        s.set_system("x");
        bumped(&s, "set_system");
        s.set_tools(ToolSet::new().with(ToolDefinition::new("t")));
        bumped(&s, "set_tools");
        let a = s.push_user("a");
        bumped(&s, "push");
        let b = s.replace_message(a, Message::user("b")).unwrap();
        bumped(&s, "replace_message");
        s.remove_message(b).unwrap();
        bumped(&s, "remove_message");
        s.replace_messages([Message::user("c")]).unwrap();
        bumped(&s, "replace_messages");
        s.restore_context([a]).unwrap();
        bumped(&s, "restore_context");
        s.edit(|m| m.clear());
        bumped(&s, "edit");
        s.push_user("d");
        bumped(&s, "push");
        s.clear();
        bumped(&s, "clear");
    }

    // ── edit / clear / rollback keep working, losslessly ────────────────

    #[test]
    fn edit_is_a_lossless_context_replacement() {
        let mut s = Session::new();
        let a = s.push_user("a");
        let b = s.push_user("b");
        s.push_user("c");
        s.opened = true;
        let before = s.events().to_vec();

        s.edit(|m| m.truncate(2));
        assert_eq!(texts(&s), ["a", "b"]);
        assert_eq!(s.active_ids(), [a, b], "unchanged messages keep their ids");
        assert!(!s.opened);
        assert_eq!(
            s.all_messages().len(),
            3,
            "the trimmed message is still recorded"
        );
        assert_invariants(&before, 3, &s);

        s.edit(|m| m[0] = Message::user("A"));
        assert_eq!(texts(&s), ["A", "b"]);
        assert_ne!(s.active_ids()[0], a, "a rewritten message is a new record");
        assert_eq!(s.active_ids()[1], b);
        assert_eq!(s.all_messages().len(), 4);
    }

    #[test]
    fn editing_history_invalidates_the_engines_cached_prefill() {
        let mut s = Session::new();
        s.push_user("secret");
        s.opened = true;
        s.edit(|m| m.clear());
        assert!(!s.opened);
        assert_eq!(s.pending(1).len(), 0);
    }

    #[test]
    fn clearing_resets_the_conversation_but_keeps_the_id_and_instructions() {
        let mut s = Session::new().with_system("sys");
        let id = s.id().clone();
        s.push_user("hi");
        s.opened = true;
        s.clear();
        assert!(s.is_empty());
        assert!(!s.opened);
        assert_eq!(s.id(), &id);
        assert_eq!(s.system(), Some("sys"), "instruction state is not history");
        assert_eq!(s.all_messages().len(), 1, "cleared, not erased");
    }

    #[test]
    fn rollback_withdraws_appended_messages_without_closing() {
        let mut s = Session::new();
        s.push_user("sent");
        s.opened = true;
        s.push_user("staged");
        s.rollback_to(1);
        assert_eq!(texts(&s), ["sent"]);
        assert!(s.opened, "nothing withdrawn ever reached the engine");
        assert_eq!(s.all_messages().len(), 2, "still lossless");
        let r = s.revision();
        s.rollback_to(5);
        assert_eq!(s.revision(), r, "nothing to withdraw is not a change");
    }

    // ── Engine-facing: the rendered prompt is unchanged ─────────────────

    #[test]
    fn an_unopened_session_sends_its_whole_history_system_first() {
        let mut s = Session::new().with_system("sys");
        s.push_user("hi");
        let pending = s.pending(0);
        assert_eq!(pending.len(), 2, "engine has nothing yet");
        assert_eq!(pending[0].role, "system");
        assert_eq!(pending[1].role, "user");
    }

    #[test]
    fn an_opened_session_sends_only_what_is_new() {
        let mut s = Session::new().with_system("sys");
        s.push_user("hi");
        s.opened = true;
        s.push_user("more");
        assert_eq!(s.pending(2).len(), 1, "only the message the engine lacks");
        assert_eq!(s.pending(2)[0].text(), "more");
    }

    #[test]
    fn from_messages_lifts_a_leading_system_message_into_state() {
        let restored = Session::from_messages([
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant_structured("yo", None),
        ]);
        assert_eq!(restored.system(), Some("sys"));
        assert_eq!(restored.len(), 2);
        // What the model sees is exactly what the flat transcript was.
        let roles: Vec<String> = restored
            .transcript()
            .iter()
            .map(|m| m.role.clone())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant"]);
        // A system message that is not leading is left alone.
        let odd = Session::from_messages([Message::user("hi"), Message::system("late")]);
        assert_eq!(odd.system(), None);
        assert_eq!(odd.len(), 2);
    }

    // ── Images ──────────────────────────────────────────────────────────

    #[test]
    fn images_become_chunks_on_a_user_message() {
        use crate::types::message::{MessageChunk, MessageContent};
        let mut s = Session::new();
        s.push_user_with_images("what is this?", ["/tmp/a.png"]);

        let MessageBody::Content {
            content: MessageContent::MultipleChunks(chunks),
        } = &s.latest().unwrap().body
        else {
            panic!("expected a multi-chunk message");
        };
        assert_eq!(chunks.len(), 2, "text plus one image");
        assert!(matches!(chunks[0], MessageChunk::Text { .. }));
        match &chunks[1] {
            MessageChunk::ImageUrl { image_url } => {
                assert!(
                    image_url.url.starts_with("file://"),
                    "got {}",
                    image_url.url
                );
            }
            other => panic!("expected an image chunk, got {other:?}"),
        }
    }

    #[test]
    fn an_http_image_url_is_left_alone() {
        use crate::types::message::{MessageChunk, MessageContent};
        let mut s = Session::new();
        s.push_user_with_images("look", ["https://example.com/a.png"]);
        let MessageBody::Content {
            content: MessageContent::MultipleChunks(chunks),
        } = &s.latest().unwrap().body
        else {
            panic!("expected chunks");
        };
        match &chunks[1] {
            MessageChunk::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/a.png");
            }
            other => panic!("expected an image chunk, got {other:?}"),
        }
    }

    #[test]
    fn no_images_is_a_plain_user_message() {
        let mut s = Session::new();
        s.push_user_with_images("hello", Vec::<String>::new());
        assert_eq!(s.latest_text().as_deref(), Some("hello"));
    }

    // ── §9.4 tool results ───────────────────────────────────────────────

    #[test]
    fn a_tool_result_is_tied_to_its_call() {
        let mut s = Session::new();
        s.push(call_msg(&["c1"]));
        s.push_tool_result("c1", "18C, clear");
        let last = s.latest().unwrap();
        assert_eq!(last.role, "tool");
        assert_eq!(last.tool_call_id.as_deref(), Some("c1"));
        assert_eq!(last.text(), "18C, clear");

        s.push(call_msg(&["c2"]));
        s.push_tool_result(
            crate::output::ToolCallId("c2".into()),
            serde_json::json!({"temp": 18}),
        );
        assert_eq!(s.latest().unwrap().text(), r#"{"temp":18}"#);
        let out: ToolResult = super::super::tools::ToolOutput::from("x").into();
        assert_eq!(out, ToolResult::Text("x".into()));
    }

    // ── The tool-round invariant ────────────────────────────────────────

    #[test]
    fn removing_a_call_whose_result_stays_is_refused() {
        let mut s = Session::new();
        s.push_user("go");
        let call = s.push(call_msg(&["c1"]));
        s.push_tool_result("c1", "done");
        let events = s.events().to_vec();
        let err = s.remove_message(call).unwrap_err();
        assert!(
            matches!(
                err,
                super::super::Error::Session(SessionError::OrphanResult { ref call_id }) if call_id == "c1"
            ),
            "got {err:?}"
        );
        assert_eq!(s.events(), &events[..], "refused means untouched");
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn removing_a_result_whose_call_stays_is_refused() {
        let mut s = Session::new();
        let call = s.push(call_msg(&["c1", "c2"]));
        s.push_tool_result("c1", "one");
        let two = s.push_tool_result("c2", "two");
        let err = s.remove_message(two).unwrap_err();
        assert!(
            matches!(
                err,
                super::super::Error::Session(SessionError::UnansweredCall { ref call_id }) if call_id == "c2"
            ),
            "got {err:?}"
        );
        // The whole round can go together, though.
        s.restore_context([]).unwrap();
        assert!(s.is_empty());
        // And come back together.
        s.restore_context([call, two]).unwrap_err();
        let all: Vec<MessageId> = s.all_messages().iter().map(|r| r.id).collect();
        s.restore_context(all).unwrap();
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn a_call_nothing_has_answered_may_be_removed_alone() {
        // The interrupted-process case: the model asked, the process died.
        // Healing the transcript means dropping the dangling call, and that
        // must not be refused as "a call without its result".
        let mut s = Session::new();
        s.push_user("go");
        let call = s.push(call_msg(&["c1"]));
        s.remove_message(call)
            .expect("an unanswered call stands alone");
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn replace_messages_refuses_half_a_round_and_accepts_a_whole_one() {
        let mut s = Session::new();
        s.push(call_msg(&["c1"]));
        s.push_tool_result("c1", "r");
        let events = s.events().to_vec();
        let records = s.all_messages().len();

        assert!(
            s.replace_messages([Message::tool_result_for("c1", "r")])
                .is_err(),
            "a result with no call"
        );
        assert!(
            s.replace_messages([call_msg(&["c1"])]).is_err(),
            "a call whose answer exists but is not shown"
        );
        assert_eq!(s.events(), &events[..]);
        assert_eq!(s.all_messages().len(), records, "a refusal mints nothing");

        s.replace_messages([
            Message::user("summary"),
            call_msg(&["c1"]),
            Message::tool_result_for("c1", "r"),
        ])
        .expect("a whole round is fine");
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn restore_context_and_fork_at_refuse_to_split_a_round() {
        let mut s = Session::new();
        let call = s.push(call_msg(&["c1"]));
        let result = s.push_tool_result("c1", "r");
        assert!(s.restore_context([result]).is_err(), "result without call");
        assert!(
            s.restore_context([call]).is_err(),
            "call without its result"
        );
        assert!(s.fork_at(call).is_err(), "cutting between call and result");
        assert!(s.fork_at(result).is_ok(), "cutting after the round");
    }

    #[test]
    fn edit_trims_to_a_round_boundary_instead_of_refusing() {
        let mut s = Session::new();
        s.push_user("go");
        s.push(call_msg(&["c1", "c2"]));
        s.push_tool_result("c1", "one");
        s.push_tool_result("c2", "two");
        s.push_user("next");

        // Truncating between the two results would leave `c2` unanswered:
        // the whole round goes.
        s.edit(|m| m.truncate(3));
        assert_eq!(texts(&s), ["go"]);

        // Dropping only the call would orphan the results: they go too.
        let ids: Vec<MessageId> = s.all_messages().iter().map(|r| r.id).collect();
        s.restore_context(ids[..4].to_vec()).unwrap();
        s.edit(|m| {
            m.remove(1);
        });
        assert_eq!(texts(&s), ["go"]);
    }

    #[test]
    fn a_legacy_result_without_an_id_still_needs_a_preceding_call() {
        let mut s = Session::new();
        let call = s.push(call_msg(&["c1"]));
        let result = s.push(Message::tool_result("legacy"));
        assert!(s.remove_message(call).is_err(), "would orphan the result");
        // An id-less result cannot be proven to answer anything, so removing
        // it alone is allowed: the call is then merely unanswered.
        s.remove_message(result).unwrap();
        assert_eq!(s.len(), 1);
        assert!(
            s.restore_context([result]).is_err(),
            "a result first is an orphan"
        );
    }

    // ── Persistence ─────────────────────────────────────────────────────

    #[test]
    fn a_restored_session_never_claims_engine_state_it_lacks() {
        let mut a = Session::new();
        a.push_user("hello");
        a.opened = true;
        a.note_shed(3);

        let json = serde_json::to_string(&a).unwrap();
        let restored: Session = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.messages().len(), 1, "the transcript survives");
        assert!(!restored.opened, "but the engine's cache does not");
        assert_eq!(restored.shed(), 0);
    }

    #[test]
    fn the_event_log_round_trips_and_the_rest_is_replayed() {
        let mut a = Session::new().with_system("sys");
        a.add_tool(ToolDefinition::new("read"));
        let x = a.push_user("helo");
        let y = a.replace_message(x, Message::user("hello")).unwrap();
        a.push(call_msg(&["c1"]));
        a.push_tool_result("c1", "r");
        a.push_user("gone");
        a.edit(|m| m.truncate(3));
        let b = a.fork();

        for s in [&a, &b] {
            let json = serde_json::to_string(s).unwrap();
            let r: Session = serde_json::from_str(&json).unwrap();
            assert_eq!(r.id(), s.id());
            assert_eq!(r.events(), s.events());
            assert_eq!(r.revision(), s.revision());
            assert_eq!(r.system(), s.system());
            assert_eq!(r.tools(), s.tools());
            assert_eq!(r.messages(), s.messages());
            assert_eq!(r.active_ids(), s.active_ids());
            assert_eq!(r.all_messages(), s.all_messages());
            assert_eq!(
                r.all_messages()
                    .iter()
                    .find(|r| r.id == x)
                    .unwrap()
                    .replaced_by,
                Some(y)
            );
            // Fresh ids after a restore never collide with restored ones.
            let mut r = r;
            let fresh = r.push_user("after");
            assert!(s.all_messages().iter().all(|rec| rec.id != fresh));
        }
        assert!(
            serde_json::to_value(&a).unwrap()["messages"].is_null(),
            "the projection is derived, not stored"
        );
    }

    #[test]
    fn the_old_flat_transcript_format_still_opens() {
        let json = serde_json::json!({
            "id": "session-old",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]
        });
        let s: Session = serde_json::from_value(json).unwrap();
        assert_eq!(s.id().as_str(), "session-old");
        assert_eq!(s.system(), Some("sys"));
        assert_eq!(texts(&s), ["hi"]);
    }

    // ── Engine bookkeeping (unchanged behaviour) ────────────────────────

    #[test]
    fn swapping_the_model_reopens_a_conversation() {
        let mut s = Session::new();
        assert!(!s.note_model(0), "first use has nothing to invalidate");
        s.opened = true;
        assert!(!s.note_model(0), "the same model keeps the warm prefill");
        assert!(s.note_model(1), "a swap must reopen");
        assert!(!s.opened);
    }

    #[test]
    fn the_same_tool_set_does_not_reopen_a_conversation() {
        let mut s = Session::new();
        assert!(!s.note_tools(7), "first use has nothing to invalidate");
        s.opened = true;
        assert!(!s.note_tools(7), "an unchanged set keeps the warm prefill");
        assert!(s.opened);
    }

    #[test]
    fn a_changed_tool_set_reopens_the_conversation() {
        let mut s = Session::new();
        s.note_tools(7);
        s.opened = true;
        assert!(s.note_tools(9), "a different set must reopen");
        assert!(!s.opened, "so the new definitions actually get sent");
    }
}
