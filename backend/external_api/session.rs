//! External API session — holds message history and constructs
//! OpenAI-compatible or Anthropic chat completion requests.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use parking_lot::RwLock;

use crate::gen2::engine::{ExecError, HookBus, Settings};
use crate::gen2::generation::{GenSpec, TokenEvent};
use crate::generation::model_runner::types::{Message, MessageBody, MessageContent};
use super::anthropic_puller::AnthropicPuller;
use super::puller::TokenPuller;

/// Wrapper enum for remote API token pullers (OpenAI vs Anthropic format).
pub enum RemotePuller {
    OpenAi(TokenPuller),
    Anthropic(AnthropicPuller),
}

impl Iterator for RemotePuller {
    type Item = Result<TokenEvent, ExecError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::OpenAi(p) => p.next(),
            Self::Anthropic(p) => p.next(),
        }
    }
}

pub type SessionId = u64;

#[derive(Debug)]
pub struct Session {
    pub id: SessionId,
    server_url: String,
    model_id: String,
    api_key: Option<String>,
    api_format: String,
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
        api_key: Option<String>,
        api_format: String,
        client: reqwest::blocking::Client,
        hooks: Arc<HookBus>,
        settings: Settings,
        messages: Vec<Message>,
    ) -> Self {
        Self {
            id, server_url, model_id, api_key, api_format,
            client, hooks, settings,
            paused: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
            messages: RwLock::new(messages),
        }
    }

    pub fn pause(&self) { self.paused.store(true, Ordering::SeqCst); }
    pub fn resume(&self) { self.paused.store(false, Ordering::SeqCst); }
    pub fn stop(&self) { self.stopped.store(true, Ordering::SeqCst); }

    pub fn pull(&self, mut gen_spec: GenSpec) -> Result<RemotePuller, ExecError> {
        if gen_spec.max_tokens.is_none() {
            gen_spec.max_tokens = self.settings.stopping.max_tokens;
        }
        let messages = self.messages.read().clone();
        let model = if self.model_id.is_empty() { "default".to_string() } else { self.model_id.clone() };

        let (url, body) = if self.api_format == "anthropic" {
            self.build_anthropic_request(&messages, &model, &gen_spec)
        } else {
            self.build_openai_request(&messages, &model, &gen_spec)
        };

        let mut req = self.client.post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");

        if let Some(ref key) = self.api_key {
            if self.api_format == "anthropic" {
                req = req.header("x-api-key", key).header("anthropic-version", "2023-06-01");
            } else {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
        }

        let response = req.json(&body).send().map_err(|e| {
            ExecError::Other(anyhow::anyhow!("failed to connect to external server: {}", e))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().unwrap_or_default();
            return Err(ExecError::Other(anyhow::anyhow!("external server returned {}: {}", status, body_text)));
        }

        if self.api_format == "anthropic" {
            Ok(RemotePuller::Anthropic(AnthropicPuller::new(
                self.id, self.hooks.clone(), response, gen_spec, self.paused.clone(), self.stopped.clone(),
            )))
        } else {
            Ok(RemotePuller::OpenAi(TokenPuller::new(
                self.id, self.hooks.clone(), response, gen_spec, self.paused.clone(), self.stopped.clone(),
            )))
        }
    }

    fn build_openai_request(&self, messages: &[Message], model: &str, gen_spec: &GenSpec) -> (String, serde_json::Value) {
        let api_messages = Self::build_api_messages(messages, &self.settings);
        let mut body = serde_json::json!({"model": model, "messages": api_messages, "stream": true});
        if let Some(mt) = gen_spec.max_tokens { body["max_tokens"] = serde_json::json!(mt); }
        if let Some(t) = gen_spec.temperature.or(self.settings.sampling.temperature) { body["temperature"] = serde_json::json!(t); }
        if let Some(tp) = self.settings.sampling.top_p { body["top_p"] = serde_json::json!(tp); }
        if let Some(s) = gen_spec.seed.or(self.settings.sampling.seed.map(|s| s as u64)) { body["seed"] = serde_json::json!(s); }
        if !self.settings.stopping.stopwords.is_empty() { body["stop"] = serde_json::json!(self.settings.stopping.stopwords); }
        (format!("{}/chat/completions", self.server_url), body)
    }

    fn build_anthropic_request(&self, messages: &[Message], model: &str, gen_spec: &GenSpec) -> (String, serde_json::Value) {
        let mut system_text = String::new();
        let mut api_msgs = Vec::with_capacity(messages.len());
        if let Some(ref prompt) = self.settings.prompt.system_prompt {
            if !prompt.is_empty() { system_text = prompt.clone(); }
        }
        for msg in messages {
            let content = match &msg.body {
                MessageBody::Content { content } => match content {
                    MessageContent::SingleText(text) => text.clone(),
                    MessageContent::MultipleChunks(chunks) => chunks.iter().filter_map(|c| {
                        if let crate::generation::model_runner::types::MessageChunk::Text { text } = c { Some(text.as_str()) } else { None }
                    }).collect::<Vec<_>>().join("")
                },
                MessageBody::Tool { .. } => continue,
            };
            if msg.role == "system" { if system_text.is_empty() { system_text = content; } continue; }
            api_msgs.push(serde_json::json!({"role": msg.role, "content": content}));
        }
        let max_tokens = gen_spec.max_tokens.unwrap_or(4096);
        let mut body = serde_json::json!({"model": model, "max_tokens": max_tokens, "messages": api_msgs, "stream": true});
        if !system_text.is_empty() { body["system"] = serde_json::json!(system_text); }
        if let Some(t) = gen_spec.temperature.or(self.settings.sampling.temperature) { body["temperature"] = serde_json::json!(t); }
        if let Some(tp) = self.settings.sampling.top_p { body["top_p"] = serde_json::json!(tp); }
        if !self.settings.stopping.stopwords.is_empty() { body["stop_sequences"] = serde_json::json!(self.settings.stopping.stopwords); }
        (format!("{}/messages", self.server_url), body)
    }

    pub fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        if new_messages.is_empty() { return Ok(0); }
        let mut msgs = self.messages.write();
        msgs.extend(new_messages);
        Ok(0)
    }

    fn build_api_messages(messages: &[Message], settings: &Settings) -> Vec<serde_json::Value> {
        let mut api_msgs = Vec::with_capacity(messages.len() + 1);
        let has_system = messages.iter().any(|m| m.role == "system");
        if !has_system {
            if let Some(ref prompt) = settings.prompt.system_prompt {
                if !prompt.is_empty() {
                    api_msgs.push(serde_json::json!({"role": "system", "content": prompt}));
                }
            }
        }
        for msg in messages {
            let content = match &msg.body {
                MessageBody::Content { content } => match content {
                    MessageContent::SingleText(text) => text.clone(),
                    MessageContent::MultipleChunks(chunks) => chunks.iter().filter_map(|c| {
                        if let crate::generation::model_runner::types::MessageChunk::Text { text } = c { Some(text.as_str()) } else { None }
                    }).collect::<Vec<_>>().join("")
                },
                MessageBody::Tool { .. } => { continue; }
            };
            api_msgs.push(serde_json::json!({"role": msg.role, "content": content}));
        }
        api_msgs
    }
}
