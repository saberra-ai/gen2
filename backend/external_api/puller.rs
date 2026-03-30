//! External API token puller — streams SSE from an OpenAI-compatible endpoint.
//!
//! Reads `data: {...}` lines from the HTTP response, extracts
//! `choices[0].delta.content` for each chunk, and yields `TokenEvent::Token`.

use std::io::{BufRead, BufReader};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::gen2::engine::{ExecError, ExecutionStats, HookBus, HookEvent};
use crate::gen2::generation::{GenSpec, Token, TokenEvent};

pub struct TokenPuller {
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
}

impl TokenPuller {
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
            prompt_tokens: 0, // external server handles prompt
            decode_tokens: self.produced as u32,
            first_token_us: self.first_token_us.unwrap_or(0),
            avg_tps,
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

impl Iterator for TokenPuller {
    type Item = Result<TokenEvent, ExecError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Check stop flag
        if self.stopped.load(Ordering::Acquire) {
            return self.finish(TokenEvent::Stopped);
        }

        // Check pause flag
        if self.paused.load(Ordering::Acquire) {
            return Some(Ok(TokenEvent::Paused));
        }

        // Check token limit
        if let Some(limit) = self.max_tokens {
            if self.produced >= limit {
                return self.finish(TokenEvent::Eos);
            }
        }

        // Read SSE lines until we get a data payload or EOF
        loop {
            // Re-check stop between reads
            if self.stopped.load(Ordering::Acquire) {
                return self.finish(TokenEvent::Stopped);
            }

            self.line_buf.clear();
            match self.reader.read_line(&mut self.line_buf) {
                Ok(0) => {
                    // EOF — stream ended
                    return self.finish(TokenEvent::Eos);
                }
                Ok(_) => {
                    let line = self.line_buf.trim();

                    // Skip empty lines (SSE separator)
                    if line.is_empty() {
                        continue;
                    }

                    // Skip non-data lines (e.g. "event:", "id:", "retry:")
                    if !line.starts_with("data: ") && !line.starts_with("data:") {
                        continue;
                    }

                    // Extract the data payload
                    let data = if line.starts_with("data: ") {
                        &line[6..]
                    } else {
                        &line[5..]
                    };

                    // Check for stream termination
                    if data.trim() == "[DONE]" {
                        return self.finish(TokenEvent::Eos);
                    }

                    // Parse JSON
                    let json: serde_json::Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!("external_api: failed to parse SSE data: {}", e);
                            continue;
                        }
                    };

                    // Extract content from choices[0].delta.content
                    let content = json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str());

                    // Check for finish_reason
                    let finish_reason = json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("finish_reason"))
                        .and_then(|f| f.as_str());

                    if let Some("stop" | "length") = finish_reason {
                        // If there's also content in this chunk, yield it first
                        if let Some(text) = content {
                            if !text.is_empty() {
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
                                // We'll get Eos on the next call
                                self.done = true;
                                let stats = self.stats_now();
                                self.hooks.emit(HookEvent::FinalStats {
                                    session_id: self.session_id,
                                    stats,
                                });
                                return Some(Ok(TokenEvent::Token(Token {
                                    id: 0,
                                    text: text.to_string(),
                                    logprob: None,
                                })));
                            }
                        }
                        return self.finish(TokenEvent::Eos);
                    }

                    // Yield token if we got content
                    if let Some(text) = content {
                        if !text.is_empty() {
                            if self.first_token_us.is_none() {
                                self.first_token_us = Some(now_us().saturating_sub(self.start_us));
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

                    // Empty content chunk (e.g. role-only delta) — continue reading
                    continue;
                }
                Err(e) => {
                    // Read timeout (from 2s read_timeout) is not fatal —
                    // loop back to recheck stopped/paused flags.
                    if e.kind() == std::io::ErrorKind::TimedOut
                        || e.kind() == std::io::ErrorKind::WouldBlock
                    {
                        continue;
                    }
                    self.done = true;
                    return Some(Err(ExecError::Other(anyhow::anyhow!(
                        "error reading SSE stream: {}",
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
