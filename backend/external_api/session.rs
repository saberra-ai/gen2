//! External API session — holds message history and constructs
//! OpenAI-compatible chat completion requests.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use parking_lot::RwLock;

use crate::gen2::engine::{ExecError, HookBus, Settings};
use crate::gen2::generation::GenSpec;
use crate::generation::model_runner::types::{Message, MessageBody, MessageContent};
use super::puller::TokenPuller;

pub type SessionId = u64;

#[derive(Debug)]
pub struct Session {
    pub id: SessionId,
    server_url: String,
    model_id: String,
    client: reqwest::blocking::Client,
    hooks: Arc<HookBus>,
    settings: Settings,
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    messages: RwLock<Vec<Message>>,
}

impl Session {
    pub(crate) fn new(
        id: SessionId,
        server_url: String,
        model_id: String,
        client: reqwest::blocking::Client,
        hooks: Arc<HookBus>,
        settings: Settings,
        messages: Vec<Message>,
    ) -> Self {
        Self {
            id,
            server_url,
            model_id,
            client,
            hooks,
            settings,
            paused: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
            messages: RwLock::new(messages),
        }
    }

    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    pub fn pull(&self, mut gen_spec: GenSpec) -> Result<TokenPuller, ExecError> {
        if gen_spec.max_tokens.is_none() {
            gen_spec.max_tokens = self.settings.stopping.max_tokens;
        }

        let messages = self.messages.read().clone();

        // Build OpenAI-compatible messages array
        let api_messages = Self::build_api_messages(&messages, &self.settings);

        // Determine model id: use session model_id if set, else "default"
        let model = if self.model_id.is_empty() {
            "default".to_string()
        } else {
            self.model_id.clone()
        };

        // Build request body
        let mut body = serde_json::json!({
            "model": model,
            "messages": api_messages,
            "stream": true,
        });

        // Apply generation parameters
        if let Some(max_tokens) = gen_spec.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temp) = gen_spec.temperature.or(self.settings.sampling.temperature) {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = self.settings.sampling.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(seed) = gen_spec.seed.or(self.settings.sampling.seed.map(|s| s as u64)) {
            body["seed"] = serde_json::json!(seed);
        }
        if !self.settings.stopping.stopwords.is_empty() {
            body["stop"] = serde_json::json!(self.settings.stopping.stopwords);
        }

        let url = format!("{}/chat/completions", self.server_url);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .map_err(|e| {
                ExecError::Other(anyhow::anyhow!(
                    "failed to connect to external server: {}",
                    e
                ))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().unwrap_or_default();
            return Err(ExecError::Other(anyhow::anyhow!(
                "external server returned {}: {}",
                status,
                body_text
            )));
        }

        Ok(TokenPuller::new(
            self.id,
            self.hooks.clone(),
            response,
            gen_spec,
            self.paused.clone(),
            self.stopped.clone(),
        ))
    }

    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<(), ExecError> {
        if new_messages.is_empty() {
            return Ok(());
        }
        let mut msgs = self.messages.write();
        msgs.extend(new_messages);
        Ok(())
    }

    /// Convert internal Message types to OpenAI-compatible JSON messages.
    fn build_api_messages(
        messages: &[Message],
        settings: &Settings,
    ) -> Vec<serde_json::Value> {
        let mut api_msgs = Vec::with_capacity(messages.len() + 1);

        // Prepend system prompt if configured and no system message exists
        let has_system = messages.iter().any(|m| m.role == "system");
        if !has_system {
            if let Some(ref prompt) = settings.prompt.system_prompt {
                if !prompt.is_empty() {
                    api_msgs.push(serde_json::json!({
                        "role": "system",
                        "content": prompt,
                    }));
                }
            }
        }

        for msg in messages {
            let content = match &msg.body {
                MessageBody::Content { content } => match content {
                    MessageContent::SingleText(text) => text.clone(),
                    MessageContent::MultipleChunks(chunks) => {
                        // Concatenate text chunks; skip image/audio for external API
                        chunks
                            .iter()
                            .filter_map(|c| {
                                if let crate::generation::model_runner::types::MessageChunk::Text {
                                    text,
                                } = c
                                {
                                    Some(text.as_str())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    }
                },
            };

            api_msgs.push(serde_json::json!({
                "role": msg.role,
                "content": content,
            }));
        }

        api_msgs
    }
}
