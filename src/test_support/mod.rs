//! Machinery that exists so tests can be fast and deterministic.
//!
//! Compiled only under `cfg(test)`: none of this is public surface, and none
//! of it is compiled into a release build.
//!
//! The centrepiece is [`FakeBackend`], which replaces the model with a script
//! so orchestration can be tested without a GGUF. See its module docs.

pub(crate) mod fake_backend;
pub(crate) mod harness;

#[allow(unused_imports)]
pub(crate) use fake_backend::{FakeBackend, Gate, Script, Step};
#[allow(unused_imports)]
pub(crate) use harness::{Harness, assert_valid_trace, text_of};
