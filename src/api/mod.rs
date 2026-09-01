//! The public API.
//!
//! Load a model, run turns, read tokens back:
//!
//! ```no_run
//! use pio_gen2::Engine;
//!
//! let engine = Engine::load("/models/model.gguf")?;
//! let reply = engine.prompt("Explain entropy in one sentence.")
//!     .max_tokens(256)
//!     .text()?;
//! # Ok::<(), pio_gen2::Error>(())
//! ```
//!
//! Everything underneath — backend dispatch, session runtime, KV cache, the
//! model zoo, placement routing, residency policy — is internal, so it can
//! change without breaking callers. [`Engine::controller`] is the escape hatch
//! for what this doesn't cover.

mod chat;
mod engine;
mod error;
mod stream;

pub use chat::Chat;
pub use engine::{Engine, EngineBuilder};
pub use error::{Error, Result};
pub use stream::{Event, Finish, TokenStream};
