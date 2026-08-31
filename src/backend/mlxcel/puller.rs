//! [`TokenPullerDyn`] for the mlxcel backend — the PULL side of the PUSH/PULL
//! bridge.
//!
//! The worker's `generate_streaming` callback PUSHes decoded `(id, text)` tuples
//! onto a bounded channel (see [`super::worker`]). This puller PULLs them: each
//! `next_event()` recvs one tuple → `TokenEvent::Token`. When the worker
//! finishes (loop ends / stop / EOS), it drops the sender, the channel closes,
//! `recv()` returns `Err`, and we emit a terminal `TokenEvent::Eos` once, then
//! `None` forever after — matching the AR-puller contract the controller drains.

use std::sync::mpsc::Receiver;

use crate::engine::ExecError;
use crate::generation::{Token, TokenEvent};

use super::worker::DecodedToken;

pub(crate) struct MlxcelTokenPuller {
    /// Decoded tokens pushed by the worker's `on_token` callback.
    rx: Receiver<DecodedToken>,
    /// True once the channel has closed and we've emitted the terminal `Eos`.
    done: bool,
}

impl MlxcelTokenPuller {
    pub(crate) fn new(rx: Receiver<DecodedToken>) -> Self {
        Self { rx, done: false }
    }

    fn next_inner(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        if self.done {
            return None;
        }
        match self.rx.recv() {
            Ok(tok) => Some(Ok(TokenEvent::Token(Token {
                id: tok.id,
                text: tok.text,
                // Logprobs deferred (tracer-bullet scope, S2).
                logprob: None,
            }))),
            // Sender dropped → generation finished. Emit exactly one `Eos`, then
            // terminate on the next call.
            Err(_) => {
                self.done = true;
                Some(Ok(TokenEvent::Eos))
            }
        }
    }
}

impl crate::backend::traits::TokenPullerDyn for MlxcelTokenPuller {
    fn next_event(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        self.next_inner()
    }
}
