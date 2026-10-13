use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use crate::backend::TokenPullerDyn;
use crate::backend::common::{
    stop_matcher::{StopMatcher, StopState},
    tool_calls::{ParserOutput, ToolCallParser},
};
use crate::engine::telemetry::{HookBus, HookEvent};
use crate::engine::{ExecError, ExecutionStats};
use crate::generation::{Token, TokenEvent};

type TokenSource = Box<dyn FnMut() -> Result<Option<(u32, String)>, ExecError>>;

pub(super) struct OnnxPuller {
    id: u64,
    source: TokenSource,
    stopped: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    parser: ToolCallParser,
    stops: StopMatcher,
    stop_ids: Vec<u32>,
    emitted: usize,
    ready: VecDeque<TokenEvent>,
    ended: bool,
    stats: ExecutionStats,
    started: Instant,
    first_decode: Option<Instant>,
    hooks: Arc<HookBus>,
    engine_stats: Arc<RwLock<ExecutionStats>>,
}
impl OnnxPuller {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        id: u64,
        source: TokenSource,
        stopped: Arc<AtomicBool>,
        paused: Arc<AtomicBool>,
        parser: ToolCallParser,
        stops: StopMatcher,
        stop_ids: Vec<u32>,
        prompt_tokens: usize,
        started: Instant,
        hooks: Arc<HookBus>,
        engine_stats: Arc<RwLock<ExecutionStats>>,
    ) -> Self {
        Self {
            id,
            source,
            stopped,
            paused,
            parser,
            stops,
            stop_ids,
            emitted: 0,
            ready: VecDeque::new(),
            ended: false,
            stats: ExecutionStats {
                prompt_tokens: prompt_tokens as u32,
                ..Default::default()
            },
            started,
            first_decode: None,
            hooks,
            engine_stats,
        }
    }
    fn absorb(&mut self, outputs: Vec<ParserOutput>, token_id: u32) {
        for output in outputs {
            match output {
                ParserOutput::Text(text) if !text.is_empty() => {
                    self.ready.push_back(TokenEvent::Token(Token {
                        id: token_id,
                        text,
                        logprob: None,
                    }))
                }
                ParserOutput::ToolCall(call) => self.ready.push_back(TokenEvent::ToolCall(call)),
                _ => {}
            }
        }
    }
    fn emit_text(&mut self, text: &str, id: u32) {
        let output = self.parser.push(text);
        self.absorb(output, id);
    }
    fn finish(&mut self, terminal: Option<TokenEvent>) {
        if self.ended {
            return;
        }
        self.ended = true;
        let tail = self.parser.flush();
        self.absorb(tail, 0);
        if let Some(event) = terminal {
            self.ready.push_back(event);
        }
        let decode_secs = self
            .first_decode
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0);
        self.stats.avg_tps = if decode_secs > 0.0 {
            self.stats.decode_tokens.saturating_sub(1) as f32 / decode_secs
        } else {
            0.0
        };
        *self.engine_stats.write() = self.stats.clone();
        self.hooks.emit(HookEvent::FinalStats {
            session_id: self.id,
            stats: self.stats.clone(),
        });
    }
    fn flush_held_stop_prefix(&mut self) {
        if !self.stops.is_empty() {
            let rest = self.stops.buffer()[self.emitted..].to_owned();
            self.emitted = self.stops.buffer().len();
            self.emit_text(&rest, 0);
        }
    }
}
impl TokenPullerDyn for OnnxPuller {
    fn next_event(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        loop {
            if let Some(event) = self.ready.pop_front() {
                return Some(Ok(event));
            }
            if self.ended {
                return None;
            }
            if self.stopped.load(Ordering::Acquire) {
                self.flush_held_stop_prefix();
                self.finish(Some(TokenEvent::Stopped));
                continue;
            }
            if self.paused.load(Ordering::Acquire) {
                return Some(Ok(TokenEvent::Paused));
            }
            let (id, text) = match (self.source)() {
                Ok(Some(token)) => token,
                Ok(None) => {
                    self.flush_held_stop_prefix();
                    self.finish(Some(TokenEvent::Eos));
                    continue;
                }
                Err(error) => {
                    self.finish(None);
                    self.ready.clear();
                    return Some(Err(error));
                }
            };
            let now = Instant::now();
            if self.first_decode.is_none() {
                self.first_decode = Some(now);
                self.stats.first_token_us = self.started.elapsed().as_micros().max(1) as u64;
            }
            self.stats.decode_tokens = self.stats.decode_tokens.saturating_add(1);
            self.hooks.emit(HookEvent::DecodeStep {
                session_id: self.id,
                token_id: id,
                text_len: text.len(),
            });
            if self.stop_ids.contains(&id) {
                self.flush_held_stop_prefix();
                self.finish(Some(TokenEvent::Eos));
                continue;
            }
            if self.stops.is_empty() {
                self.emit_text(&text, id);
                continue;
            }
            let (end, hit) = match self.stops.push(&text) {
                StopState::Clean => (self.stops.buffer().len(), false),
                StopState::Partial { hold } => (self.stops.buffer().len() - hold, false),
                StopState::Full { emit_at, .. } => (emit_at, true),
            };
            let visible = self.stops.buffer()[self.emitted..end].to_owned();
            self.emitted = end;
            self.emit_text(&visible, id);
            if hit {
                self.finish(Some(TokenEvent::Eos));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn puller(chunks: &[&str], stops: &[&str]) -> OnnxPuller {
        let mut chunks = chunks
            .iter()
            .map(|s| (1, s.to_string()))
            .collect::<VecDeque<_>>();
        OnnxPuller::new(
            1,
            Box::new(move || Ok(chunks.pop_front())),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            ToolCallParser::new(Default::default()),
            StopMatcher::from_strings(stops.iter().map(|s| s.to_string()).collect()),
            vec![2],
            3,
            Instant::now(),
            Arc::new(HookBus::default()),
            Arc::new(RwLock::new(ExecutionStats::default())),
        )
    }
    fn text(puller: &mut OnnxPuller) -> String {
        let mut text = String::new();
        let mut terminals = 0;
        while let Some(event) = puller.next_event() {
            match event.unwrap() {
                TokenEvent::Token(token) => {
                    assert_eq!(terminals, 0);
                    text.push_str(&token.text);
                }
                TokenEvent::Eos | TokenEvent::Stopped => terminals += 1,
                _ => {}
            }
        }
        assert_eq!(terminals, 1);
        text
    }
    #[test]
    fn split_stop_is_withheld_and_unicode_survives() {
        assert_eq!(
            text(&mut puller(&["café ", "EN", "D trailing"], &["END"])),
            "café "
        );
        assert_eq!(text(&mut puller(&["café ", "EN"], &["END"])), "café EN");
    }
    #[test]
    fn cancellation_ends_once_without_another_native_step() {
        let mut p = puller(&["unread"], &[]);
        p.stopped.store(true, Ordering::Release);
        assert_eq!(text(&mut p), "");
        assert_eq!(p.stats.decode_tokens, 0);
    }
}
