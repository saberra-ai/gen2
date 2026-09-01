//! [`Session`] — a conversation you own.

use crate::types::message::Message;

/// A conversation: its messages, and the engine state that belongs to it.
///
/// You hold this, not the engine. You can read the transcript, render it, edit
/// it, and persist it, none of which an engine-owned conversation behind a
/// string id would allow.
///
/// Dropping a session does not reach into the engine. The controller keeps its
/// own runtime until it is evicted (bounded by
/// [`ControllerConfig::max_active_chats`](crate::ControllerConfig)), and the
/// engine's per-session bookkeeping is cleared by
/// [`Engine::forget`](crate::Engine::forget). Call it when a conversation is
/// finished if you create many; nothing breaks without it, since a missing
/// entry just means the next turn resends.
///
/// Not `Clone`, deliberately. Two copies would share one engine conversation
/// and overwrite each other's cached prefill. [`Session::fork`] is the
/// independent copy.
///
/// It carries an opaque id that the engine uses to key its warm KV cache, so
/// owning the history costs you nothing in speed: a follow-up turn still reuses
/// the prefill from the last one.
///
/// # The session outlives the model's view of it
///
/// A session keeps every message. The engine's working set does not: when a
/// conversation outgrows the context window, the engine sheds its oldest
/// messages — compacting them into a summary where it can, dropping them
/// outright where it can't — and carries on. Nothing fails, and the session is
/// not rewritten.
///
/// So after a long conversation the transcript you hold is a superset of what
/// the model can still see. That is the right split — your record shouldn't be
/// destroyed to fit someone's context window — but it is worth knowing when you
/// render it. [`Session::shed`] counts how many messages have fallen out of the
/// model's view, and per turn [`Completion`](super::Completion) reports
/// `dropped` and `compacted`.
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
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Session {
    id: String,
    messages: Vec<Message>,
    /// Whether the engine has opened this conversation yet. Lives here rather
    /// than in the engine so it's dropped with the session.
    ///
    /// Deliberately not serialized. It describes *this process's* engine state,
    /// and a restored session that claimed the engine held its prefill would
    /// continue against a cache that does not exist.
    #[serde(skip)]
    pub(crate) opened: bool,
    /// Messages the engine has shed from its working set to fit the context
    /// window, across the whole conversation.
    ///
    /// Also not serialized: a restored session is resent whole, so the model
    /// starts with all of it in view again.
    #[serde(skip)]
    pub(crate) shed: usize,
    /// Fingerprint of the tool set this conversation was opened with.
    ///
    /// Tool definitions are rendered into the prompt prefix and only sent when
    /// a conversation is opened, so a later run registering a different set
    /// would otherwise be silently ignored — the model would keep seeing the
    /// original tools. Recording the set lets a change reopen the conversation
    /// instead.
    #[serde(skip)]
    pub(crate) tools_fingerprint: Option<u64>,
    /// Which model generation this conversation was opened against.
    ///
    /// A cached prefill belongs to the model that produced it. Swapping models
    /// leaves the engine holding a cache for a conversation the new weights
    /// never saw, so the swap has to reopen every live session.
    #[serde(skip)]
    pub(crate) model_generation: Option<u64>,
}

impl Session {
    /// A new, empty conversation.
    pub fn new() -> Self {
        Self {
            id: format!("session-{}", uuid::Uuid::new_v4()),
            messages: Vec::new(),
            opened: false,
            shed: 0,
            tools_fingerprint: None,
            model_generation: None,
        }
    }

    /// Rebuild a conversation from stored messages — after an app restart, say.
    ///
    /// The engine has no cached state for it, so the next turn re-reads this
    /// history once and is warm from then on.
    pub fn from_messages(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            ..Self::new()
        }
    }

    /// Branch this conversation into an independent copy.
    ///
    /// This is the reason `Session` is not `Clone`: a clone would share the
    /// engine identity and the two would fight over one cached prefill.
    ///
    /// The fork carries the same messages and a new engine identity, so the two
    /// run independently. Retry from a point, or compare two directions,
    /// without either overwriting the other's cached prefill.
    ///
    /// The fork starts unopened: the engine has nothing cached for it, so its
    /// first turn resends the history once and is warm from then on.
    pub fn fork(&self) -> Self {
        Self {
            id: format!("session-{}", uuid::Uuid::new_v4()),
            messages: self.messages.clone(),
            opened: false,
            shed: 0,
            tools_fingerprint: None,
            model_generation: None,
        }
    }

    /// Set the system prompt, builder-style.
    pub fn with_system(mut self, text: impl Into<String>) -> Self {
        self.push(Message::system(text));
        self
    }

    /// The opaque id keying the engine's cached state for this conversation.
    ///
    /// Stable for the session's life. Persist it if you want to correlate a
    /// stored transcript with engine telemetry; it is not needed to restore
    /// one.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Every message so far, oldest first.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// The most recent message — after a turn, the assistant's reply.
    pub fn latest(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// The most recent message's text.
    pub fn latest_text(&self) -> Option<String> {
        self.latest().map(text_of)
    }

    /// How many messages have fallen out of the model's context window.
    ///
    /// Grows when a turn overflows the window and the engine sheds history to
    /// make room. Non-zero means the transcript you hold is longer than what
    /// the model can still see — worth surfacing if a user is about to ask
    /// about something early in a long conversation.
    ///
    /// This counts what left the *model's* view. Your messages are all still
    /// here; nothing rewrites them.
    pub fn shed(&self) -> usize {
        self.shed
    }

    /// Whether the model can still see the whole conversation.
    pub fn fully_in_context(&self) -> bool {
        self.shed == 0
    }

    /// Undo messages added since the conversation had `len` of them.
    ///
    /// For a turn rejected before anything was sent: the builder appends as it
    /// is configured, so a turn that never runs would otherwise leave its
    /// messages behind and the next turn would inherit them.
    pub(crate) fn rollback_to(&mut self, len: usize) {
        self.messages.truncate(len);
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

    /// Declare the tool set a run is about to use.
    ///
    /// A set different from the one this conversation was opened with reopens
    /// it, so the new definitions actually reach the model. Costs one
    /// re-prefill; the alternative is a run whose tools are silently ignored.
    ///
    /// Returns whether the conversation was reopened.
    pub(crate) fn note_tools(&mut self, fingerprint: u64) -> bool {
        match self.tools_fingerprint {
            Some(current) if current == fingerprint => false,
            None if !self.opened => {
                // First use: nothing to invalidate.
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

    /// Append a message.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Append a user message.
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.push(Message::user(text));
    }

    /// Append a user message carrying images.
    ///
    /// Paths become `file://` URLs; already-formed `http(s)://` or `file://`
    /// URLs pass through. With no images this is exactly [`Session::push_user`],
    /// so a caller can pass an empty slice unconditionally.
    ///
    /// The model must be multimodal and loaded with a projector — see
    /// [`EngineBuilder::mmproj`](super::EngineBuilder::mmproj).
    pub fn push_user_with_images<I, P>(&mut self, text: impl Into<String>, images: I)
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        self.push(Message::user_with_images(
            text,
            images
                .into_iter()
                .map(|p| crate::types::message::to_file_url(p.as_ref())),
        ));
    }

    /// How many messages the conversation holds.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Drop every message and start over, keeping the same id.
    ///
    /// Marks the conversation unopened, so the next turn rebuilds the engine's
    /// state instead of continuing from a history that no longer exists.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.opened = false;
        self.shed = 0;
        self.tools_fingerprint = None;
        self.model_generation = None;
    }

    /// Edit the transcript in place — trim it, delete a message, rewrite one.
    ///
    /// Marks the conversation unopened: the engine's cached prefill describes
    /// the old history, so after an edit it has to be rebuilt. That costs one
    /// re-read; leaving a stale cache in place would silently answer from
    /// messages you deleted.
    pub fn edit(&mut self, f: impl FnOnce(&mut Vec<Message>)) {
        f(&mut self.messages);
        self.opened = false;
    }

    /// Messages the engine still needs for the next turn.
    ///
    /// A conversation the engine already has open only needs what's new since;
    /// one it doesn't needs the lot.
    pub(crate) fn pending(&self, sent_through: usize) -> Vec<Message> {
        if self.opened {
            self.messages[sent_through.min(self.messages.len())..].to_vec()
        } else {
            self.messages.clone()
        }
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// The plain text of a message, ignoring images and tool payloads.
fn text_of(msg: &Message) -> String {
    use crate::types::message::{MessageBody, MessageChunk, MessageContent};
    match &msg.body {
        MessageBody::Content { content } => match content {
            MessageContent::SingleText(s) => s.clone(),
            MessageContent::StructuredAssistant { content, .. } => content.clone(),
            MessageContent::MultipleChunks(chunks) => chunks
                .iter()
                .filter_map(|c| match c {
                    MessageChunk::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        },
        MessageBody::Tool { .. } => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn an_unopened_session_sends_its_whole_history() {
        let mut s = Session::new().with_system("sys");
        s.push_user("hi");
        assert_eq!(s.pending(0).len(), 2, "engine has nothing yet");
    }

    #[test]
    fn an_opened_session_sends_only_what_is_new() {
        let mut s = Session::new().with_system("sys");
        s.push_user("hi");
        s.opened = true;
        s.push_user("more");
        assert_eq!(s.pending(2).len(), 1, "only the message the engine lacks");
    }

    #[test]
    fn editing_history_invalidates_the_engines_cached_prefill() {
        // The cache describes the old messages. Continuing against it would
        // answer from text the caller just deleted.
        let mut s = Session::new();
        s.push_user("secret");
        s.opened = true;
        s.edit(|m| m.clear());
        assert!(!s.opened);
        assert_eq!(s.pending(1).len(), 0);
    }

    #[test]
    fn images_become_chunks_on_a_user_message() {
        use crate::types::message::{MessageBody, MessageChunk, MessageContent};
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
            // A bare path has to become a file:// URL — the backends resolve
            // URLs, not paths.
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
        use crate::types::message::{MessageBody, MessageChunk, MessageContent};
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
        // Lets a caller pass its image list unconditionally.
        let mut s = Session::new();
        s.push_user_with_images("hello", Vec::<String>::new());
        assert_eq!(s.latest_text().as_deref(), Some("hello"));
    }

    #[test]
    fn swapping_the_model_reopens_a_conversation() {
        // The engine's cached prefill was produced by weights that are no
        // longer loaded; continuing against it would answer from the wrong
        // model's state.
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
        // Tool definitions live in the prompt prefix and are only sent when a
        // conversation opens. Without this, a run registering different tools
        // is silently ignored and the model keeps seeing the old set.
        let mut s = Session::new();
        s.note_tools(7);
        s.opened = true;
        assert!(s.note_tools(9), "a different set must reopen");
        assert!(!s.opened, "so the new definitions actually get sent");
    }

    #[test]
    fn a_fork_shares_the_history_but_not_the_identity() {
        let mut a = Session::new().with_system("sys");
        a.push_user("hello");
        a.opened = true;
        a.note_shed(2);

        let b = a.fork();
        assert_eq!(b.messages().len(), a.messages().len());
        assert_ne!(
            b.id(),
            a.id(),
            "a fork must not share the engine's cache key"
        );
        assert!(!b.opened, "the engine has nothing cached for a fork");
        assert_eq!(b.shed(), 0, "a fork is resent whole, so nothing is missing");
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
    fn a_restored_session_never_claims_engine_state_it_lacks() {
        // The whole risk of persistence: `opened` says the engine holds a
        // prefill for this conversation. After a restart it holds nothing.
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
    fn clearing_resets_the_conversation_but_keeps_the_id() {
        let mut s = Session::new();
        let id = s.id().to_string();
        s.push_user("hi");
        s.opened = true;
        s.clear();
        assert!(s.is_empty());
        assert!(!s.opened);
        assert_eq!(s.id(), id);
    }
}
