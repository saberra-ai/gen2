use crate::gen2::engine::Settings;
use crate::generation::model_runner::types::Message;
use crate::types::Persona;

#[derive(Debug, Clone, Default)]
pub struct SessionSpec {
    pub attachments: Vec<crate::gen2::media::Attachment>,
    pub messages: Vec<Message>,
    pub cache: Option<crate::gen2::kv::KvLoadSpec>,
    pub overrides: Option<Settings>,
    /// Persona to inject into the system prompt. Pinned at session start.
    pub persona: Option<Persona>,
}
