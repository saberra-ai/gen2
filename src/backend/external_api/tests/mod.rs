//! Hermetic tests for the external-API backend.
//!
//! Split by the three things that can go wrong between this crate and a
//! provider: what gets serialized on the way out, how the stream is read on
//! the way back, and what a failure looks like to the caller. Nothing here
//! touches the network or a model — every test stands up a loopback server,
//! so the whole directory runs in well under a second.

mod anthropic_request;
mod anthropic_sse;
mod harness;
mod http_failures;
mod openai_request;
mod openai_sse;
