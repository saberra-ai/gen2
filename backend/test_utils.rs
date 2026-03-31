#![cfg(test)]

use std::path::PathBuf;

use crate::gen2::engine::Settings;
use crate::types::message::{Message, MessageBody, MessageContent};

/// Read model path from env var, return None if not set or file doesn't exist
pub fn model_path_from_env(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from).filter(|p| p.exists())
}

/// Build a simple "Hello" user message for testing
pub fn hello_message() -> Message {
    Message {
        name: None,
        role: "user".into(),
        body: MessageBody::Content {
            content: MessageContent::SingleText("Hello".into()),
        },
    }
}

/// Default small settings for testing
pub fn test_settings() -> Settings {
    let mut s = Settings::default();
    s.system.ctx_size = Some(512);
    s.system.batch_size = Some(16);
    s.system.threads = Some(2);
    s.system.threads_batch = Some(2);
    s
}
