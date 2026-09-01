//! [`Session`] — a conversation you own.

use crate::types::message::Message;

/// A conversation: its messages, and the engine state that belongs to it.
///
/// You hold this, not the engine. That means you can read the transcript,
/// render it, edit it, persist it, and drop it — and when you drop it, the
/// engine's cached state for it goes too. An engine that owned conversations
/// behind string ids could do none of that, and would leak one entry per
/// conversation for the life of the process.
///
/// It carries an opaque id that the engine uses to key its warm KV cache, so
/// owning the history costs you nothing in speed: a follow-up turn still reuses
/// the prefill from the last one.
///
/// ```no_run
/// # use pio_gen2::{Engine, Session};
/// # let engine = Engine::load("m.gguf")?;
/// let mut session = Session::new().with_system("You are terse.");
///
/// engine.chat(&mut session).user("Name two colours.").send()?;
/// println!("{}", session.latest_text().unwrap_or_default());
///
/// // A follow-up. The history is already here; you don't resend it.
/// engine.chat(&mut session).user("Now one more.").send()?;
/// # Ok::<(), pio_gen2::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Session {
    id: String,
    messages: Vec<Message>,
    /// Whether the engine has opened this conversation yet. Lives here rather
    /// than in the engine so it's dropped with the session.
    pub(crate) opened: bool,
}

impl Session {
    /// A new, empty conversation.
    pub fn new() -> Self {
        Self {
            id: format!("session-{}", uuid::Uuid::new_v4()),
            messages: Vec::new(),
            opened: false,
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

    /// Append a message.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Append a user message.
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.push(Message::user(text));
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
