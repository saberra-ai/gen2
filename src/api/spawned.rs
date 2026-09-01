//! Running a turn on a worker thread.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::thread::JoinHandle;

use super::chat::Chat;
use super::engine::Engine;
use super::error::{Error, Result};
use super::stream::Completion;

/// An update from a turn running on a worker thread.
///
/// `#[non_exhaustive]`: match with a trailing `_ =>`.
#[derive(Debug)]
#[non_exhaustive]
pub enum Update {
    /// A fragment of text, as it was decoded.
    Delta(String),
    /// The turn finished. Carries the full text, stats, and finish reason.
    ///
    /// A cancelled turn lands here too, with `finish: Finish::Stopped` — a
    /// stopped reply is still a reply.
    Done(Completion),
    /// The turn failed. Nothing further arrives.
    Failed(Error),
}

/// A turn running on a worker thread.
///
/// Iterate it for [`Update`]s. The channel closes when the turn ends, so a
/// `for` loop over it terminates on its own.
pub struct Turn {
    rx: Receiver<Update>,
    engine: Arc<Engine>,
    chat_id: String,
    join: Option<JoinHandle<()>>,
}

impl Turn {
    /// Stop this turn. The stream ends with a [`Update::Done`] carrying
    /// whatever had been generated.
    ///
    /// Callable while another thread iterates the updates — that is the point,
    /// since the iterating thread is blocked.
    pub fn cancel(&self) -> Result<()> {
        self.engine.stop(self.chat_id.clone())
    }

    /// A cancel handle that can be moved to another thread.
    pub fn canceller(&self) -> Canceller {
        Canceller {
            engine: Arc::clone(&self.engine),
            chat_id: self.chat_id.clone(),
        }
    }

    /// The conversation this turn belongs to.
    pub fn chat_id(&self) -> &str {
        &self.chat_id
    }
}

impl Iterator for Turn {
    type Item = Update;

    fn next(&mut self) -> Option<Update> {
        self.rx.recv().ok()
    }
}

impl std::iter::FusedIterator for Turn {}

impl Drop for Turn {
    fn drop(&mut self) {
        // Join so the worker cannot outlive the handle and go on sending into
        // a dropped receiver. It exits once the generation ends; a caller that
        // wants out sooner calls `cancel` first.
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl std::fmt::Debug for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Turn")
            .field("chat_id", &self.chat_id)
            .finish_non_exhaustive()
    }
}

/// Cancels a running [`Turn`] from anywhere. Cheap to clone.
#[derive(Clone, Debug)]
pub struct Canceller {
    engine: Arc<Engine>,
    chat_id: String,
}

impl Canceller {
    /// Stop the turn.
    pub fn cancel(&self) -> Result<()> {
        self.engine.stop(self.chat_id.clone())
    }
}

impl Chat<Arc<Engine>> {
    /// Run this turn on a worker thread, streaming [`Update`]s back.
    ///
    /// This is the shape a UI needs: the caller never blocks, deltas arrive as
    /// they decode, and the final [`Completion`] comes through the same
    /// channel. Build the turn with [`Engine::chat_owned`] or
    /// [`Engine::prompt_owned`].
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use pio_gen2::{Engine, Update};
    /// # let engine = Arc::new(Engine::load("m.gguf")?);
    /// let turn = engine.chat_owned("general").user("Hello").spawn();
    ///
    /// for update in turn {
    ///     match update {
    ///         Update::Delta(t) => print!("{t}"),
    ///         Update::Done(done) => println!("\n[{} chars]", done.text.len()),
    ///         Update::Failed(e) => eprintln!("\n[{e}]"),
    ///         _ => {}
    ///     }
    /// }
    /// # Ok::<(), pio_gen2::Error>(())
    /// ```
    pub fn spawn(self) -> Turn {
        let (tx, rx) = channel();
        let engine = Arc::clone(self.engine_handle());
        let chat_id = self.chat_id().to_string();

        let join = std::thread::spawn(move || {
            let deltas = tx.clone();
            let result = self.complete_streaming(|fragment| {
                // A send failure means the receiver went away. Nothing to
                // do but keep draining so the engine's session still ends
                // cleanly.
                let _ = deltas.send(Update::Delta(fragment.to_string()));
            });
            let _ = match result {
                Ok(done) => tx.send(Update::Done(done)),
                Err(e) => tx.send(Update::Failed(e)),
            };
        });

        Turn {
            rx,
            engine,
            chat_id,
            join: Some(join),
        }
    }
}
