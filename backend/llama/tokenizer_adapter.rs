//! `SessionTokenizer` adapter for llama-cpp.
//!
//! Built per-session because `ChatTemplate` is session-scoped (persona +
//! settings influence template selection). Wraps the bundle's model and the
//! session's chat template so the generic truncation driver in
//! [`crate::gen2::session_rt::truncate`] can count tokens identically to the
//! pre-refactor inline code.

use std::sync::Arc;

use super::bundle::ModelBundle;
use super::session::tokenize_chat_prompt;
use crate::gen2::backend::common::chat_template::ChatTemplate;
use crate::gen2::backend::traits::SessionTokenizer;
use crate::gen2::engine::ExecError;
use crate::types::message::Message;

pub(crate) struct LlamaSessionTokenizer {
    pub(crate) bundle: Arc<ModelBundle>,
    pub(crate) chat_template: ChatTemplate,
}

impl SessionTokenizer for LlamaSessionTokenizer {
    fn count_tokens(&self, messages: &[Message]) -> Result<usize, ExecError> {
        let prompt = self
            .chat_template
            .apply(messages.to_vec(), None, None)
            .map_err(ExecError::Other)?;
        Ok(tokenize_chat_prompt(&self.bundle.model, &prompt)?.len())
    }
}
