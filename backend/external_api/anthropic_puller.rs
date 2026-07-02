//! Anthropic Messages API token puller — streams SSE from Anthropic's endpoint.
//!
//! Parses Anthropic-specific SSE events:
//! - `content_block_delta` with `text_delta` → yields `TokenEvent::Token`
//! - `message_stop` → yields `TokenEvent::Eos`
//! - `error` → returns `ExecError`

use std::io::{BufRead, BufReader};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::gen2::engine::{ExecError, ExecutionStats, HookBus, HookEvent};
use crate::gen2::generation::{GenSpec, Token, TokenEvent};

pub struct AnthropicPuller {
    session_id: u64,
    hooks: Arc<HookBus>,
    reader: BufReader<reqwest::blocking::Response>,
    produced: usize,
    max_tokens: Option<usize>,
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    start_us: u64,
    first_token_us: Option<u64>,
    done: bool,
    line_buf: String,
    pending_event_type: Option<String>,
}

impl AnthropicPuller {
    pub(crate) fn new(
        session_id: u64,
        hooks: Arc<HookBus>,
        response: reqwest::blocking::Response,
        gen_spec: GenSpec,
        paused: Arc<AtomicBool>,
        stopped: Arc<AtomicBool>,
    ) -> Self {
        Self {
            session_id,
            hooks,
            reader: BufReader::new(response),
            produced: 0,
            max_tokens: gen_spec.max_tokens,
            paused,
            stopped,
            start_us: now_us(),
            first_token_us: None,
            done: false,
            line_buf: String::with_capacity(512),
            pending_event_type: None,
        }
    }

    fn stats_now(&self) -> ExecutionStats {
        let elapsed_us = now_us().saturating_sub(self.start_us);
        let elapsed_s = (elapsed_us as f64) / 1_000_000.0;
        let avg_tps = if elapsed_s > 0.0 {
            (self.produced as f64 / elapsed_s) as f32
        } else {
            0.0
        };
        ExecutionStats {
            prompt_tokens: 0,
            decode_tokens: self.produced as u32,
            first_token_us: self.first_token_us.unwrap_or(0),
            avg_tps,
            ..Default::default()
        }
    }

    fn finish(&mut self, event: TokenEvent) -> Option<Result<TokenEvent, ExecError>> {
        let stats = self.stats_now();
        self.hooks.emit(HookEvent::FinalStats {
            session_id: self.session_id,
            stats,
        });
        self.done = true;
        Some(Ok(event))
    }
}

impl Iterator for AnthropicPuller {
    type Item = Result<TokenEvent, ExecError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if self.stopped.load(Ordering::Acquire) {
            return self.finish(TokenEvent::Stopped);
        }
        if self.paused.load(Ordering::Acquire) {
            return Some(Ok(TokenEvent::Paused));
        }
        if let Some(limit) = self.max_tokens
            && self.produced >= limit
        {
            return self.finish(TokenEvent::Eos);
        }

        loop {
            if self.stopped.load(Ordering::Acquire) {
                return self.finish(TokenEvent::Stopped);
            }

            self.line_buf.clear();
            match self.reader.read_line(&mut self.line_buf) {
                Ok(0) => return self.finish(TokenEvent::Eos),
                Ok(_) => {
                    let line = self.line_buf.trim();
                    if line.is_empty() {
                        self.pending_event_type = None;
                        continue;
                    }

                    if let Some(event_type) = line
                        .strip_prefix("event: ")
                        .or_else(|| line.strip_prefix("event:"))
                    {
                        self.pending_event_type = Some(event_type.trim().to_string());
                        continue;
                    }

                    if !line.starts_with("data: ") && !line.starts_with("data:") {
                        continue;
                    }

                    let data = if let Some(rest) = line.strip_prefix("data: ") {
                        rest
                    } else {
                        &line[5..]
                    };

                    let json: serde_json::Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!("anthropic_puller: failed to parse SSE data: {}", e);
                            continue;
                        }
                    };

                    let event_type = self
                        .pending_event_type
                        .as_deref()
                        .or_else(|| json.get("type").and_then(|t| t.as_str()));

                    match event_type {
                        Some("content_block_delta") => {
                            let text = json
                                .get("delta")
                                .and_then(|d| d.get("text"))
                                .and_then(|t| t.as_str());
                            if let Some(text) = text
                                && !text.is_empty()
                            {
                                if self.first_token_us.is_none() {
                                    self.first_token_us =
                                        Some(now_us().saturating_sub(self.start_us));
                                }
                                self.produced += 1;
                                self.hooks.emit(HookEvent::DecodeStep {
                                    session_id: self.session_id,
                                    token_id: 0,
                                    text_len: text.len(),
                                });
                                return Some(Ok(TokenEvent::Token(Token {
                                    id: 0,
                                    text: text.to_string(),
                                    logprob: None,
                                })));
                            }
                        }
                        Some("message_delta") => {
                            if json
                                .get("delta")
                                .and_then(|d| d.get("stop_reason"))
                                .and_then(|s| s.as_str())
                                .is_some()
                            {
                                return self.finish(TokenEvent::Eos);
                            }
                        }
                        Some("message_stop") => return self.finish(TokenEvent::Eos),
                        Some("error") => {
                            let msg = json
                                .get("error")
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str())
                                .unwrap_or("unknown Anthropic API error");
                            self.done = true;
                            return Some(Err(ExecError::Other(anyhow::anyhow!(
                                "Anthropic API error: {}",
                                msg
                            ))));
                        }
                        _ => continue,
                    }
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(ExecError::Other(anyhow::anyhow!(
                        "error reading Anthropic SSE stream: {}",
                        e
                    ))));
                }
            }
        }
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}
