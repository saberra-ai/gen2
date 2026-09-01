//! The public API.
//!
//! Load a model, hold a conversation, read tokens back:
//!
//! ```no_run
//! use pio_gen2::{Engine, Session};
//!
//! let engine = Engine::load("/models/model.gguf")?;
//!
//! // A conversation you own. The reply is appended to it.
//! let mut session = Session::new();
//! engine.chat(&mut session).user("Explain entropy.").send()?;
//! println!("{}", session.latest_text().unwrap_or_default());
//!
//! // A follow-up: the history is already here, so nothing is resent.
//! engine.chat(&mut session).user("Simpler?").send()?;
//!
//! // One-off, nothing kept.
//! let title = engine.infer("Title this in three words.").max_tokens(16).text()?;
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
mod inference;
mod session;
mod spawned;
mod stream;

pub use chat::Chat;
pub use engine::{Engine, EngineBuilder};
pub use error::{Error, Result};
pub use inference::Inference;
pub use session::Session;
pub use spawned::{Canceller, OwnedChat, Turn, Update};
pub use stream::{Completion, Event, Finish, TokenStream, Tokens};
