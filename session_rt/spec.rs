use crate::gen2::engine::Settings;
use crate::types::Persona;
use crate::types::message::Message;

#[derive(Debug, Clone, Default)]
pub struct SessionSpec {
    pub attachments: Vec<crate::gen2::media::Attachment>,
    pub messages: Vec<Message>,
    pub cache: Option<crate::gen2::kv::KvLoadSpec>,
    pub overrides: Option<Settings>,
    /// Persona to inject into the system prompt. Pinned at session start.
    pub persona: Option<Persona>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Step 3d: SessionSpec tests ──────────────────────────────────

    /// SessionSpec::default() has empty messages and None fields.
    #[test]
    fn session_spec_default_is_empty() {
        let spec = SessionSpec::default();
        assert!(spec.messages.is_empty(), "messages should be empty");
        assert!(spec.attachments.is_empty(), "attachments should be empty");
        assert!(spec.cache.is_none(), "cache should be None");
        assert!(spec.overrides.is_none(), "overrides should be None");
        assert!(spec.persona.is_none(), "persona should be None");
    }

    /// Helper to build a simple text Message for testing.
    fn text_msg(role: &str, text: &str) -> Message {
        use crate::types::message::{MessageBody, MessageContent};
        Message {
            role: role.to_string(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.to_string()),
            },
            name: None,
        }
    }

    /// SessionSpec can be constructed with messages.
    #[test]
    fn session_spec_with_messages() {
        let msg = text_msg("user", "hello");
        let spec = SessionSpec {
            messages: vec![msg],
            ..Default::default()
        };
        assert_eq!(spec.messages.len(), 1);
        assert_eq!(spec.messages[0].role, "user");
        assert!(spec.overrides.is_none());
        assert!(spec.persona.is_none());
    }

    /// SessionSpec is Clone (required for controller usage).
    #[test]
    fn session_spec_clone() {
        let spec = SessionSpec {
            messages: vec![text_msg("assistant", "world")],
            ..Default::default()
        };
        let cloned = spec.clone();
        assert_eq!(cloned.messages.len(), 1);
        assert_eq!(cloned.messages[0].role, "assistant");
    }
}
