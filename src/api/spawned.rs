//! Running a turn on a worker thread.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::thread::JoinHandle;

use crate::backend::common::grammar::GrammarSpec;
use crate::generation::{GenSpec, ThinkingMode};
use crate::types::message::{Message, Tool};

use super::engine::Engine;
use super::error::{Error, Result};
use super::session::Session;
use super::stream::Completion;

/// An update from a turn running on a worker thread.
///
/// `#[non_exhaustive]`: match with a trailing `_ =>`.
#[derive(Debug)]
#[non_exhaustive]
pub enum Update {
    /// A fragment of text, as it was decoded.
    Delta(String),
    /// The turn finished. Carries the outcome and hands the session back with
    /// the reply already appended.
    ///
    /// A cancelled turn lands here too, with `finish: Finish::Stopped` — a
    /// stopped reply is still a reply.
    Done {
        completion: Completion,
        session: Session,
    },
    /// The turn failed. The session is returned unchanged apart from the
    /// messages you added, so it can be retried or shown.
    Failed { error: Error, session: Session },
}

/// A turn built against an owned [`Session`], ready to run on a worker thread.
///
/// Built by [`Engine::chat_owned`]. Same builder surface as
/// [`Chat`](super::Chat); the difference is that it owns its session, which is
/// what lets the work outlive the calling scope. The session comes back on
/// [`Update::Done`].
#[must_use = "an OwnedChat does nothing until .spawn() is called"]
pub struct OwnedChat {
    engine: Arc<Engine>,
    session: Session,
    spec: GenSpec,
    thinking: ThinkingMode,
    tools: Option<(Vec<Tool>, String)>,
}

impl OwnedChat {
    pub(crate) fn new(engine: Arc<Engine>, session: Session) -> Self {
        Self {
            spec: engine.default_gen_spec(),
            engine,
            session,
            thinking: ThinkingMode::default(),
            tools: None,
        }
    }

    /// Append a user message.
    pub fn user(mut self, text: impl Into<String>) -> Self {
        self.session.push(Message::user(text));
        self
    }

    /// Append a system message.
    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.session.push(Message::system(text));
        self
    }

    /// Cap how many tokens this turn may generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.spec.max_tokens = Some(n);
        self
    }

    /// Sampling temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.spec.temperature = Some(t);
        self
    }

    /// Seed the sampler.
    pub fn seed(mut self, seed: u64) -> Self {
        self.spec.seed = Some(seed);
        self
    }

    /// Decode deterministically: temperature 0 with a fixed seed.
    pub fn greedy(mut self) -> Self {
        self.spec.temperature = Some(0.0);
        self.spec.seed = Some(self.spec.seed.unwrap_or(0));
        self
    }

    /// Constrain output to a grammar.
    pub fn grammar(mut self, grammar: GrammarSpec) -> Self {
        self.spec.grammar = Some(grammar);
        self
    }

    /// Drop any engine-level grammar for this turn.
    pub fn unconstrained(mut self) -> Self {
        self.spec.grammar = None;
        self
    }

    /// Use a fully-built [`GenSpec`], overriding everything above.
    pub fn gen_spec(mut self, spec: GenSpec) -> Self {
        self.spec = spec;
        self
    }

    /// Offer tools to the model.
    pub fn tools(mut self, tools: Vec<Tool>, prompt: impl Into<String>) -> Self {
        self.tools = Some((tools, prompt.into()));
        self
    }

    /// Force the reasoning channel on or off.
    pub fn thinking(mut self, mode: ThinkingMode) -> Self {
        self.thinking = mode;
        self
    }

    /// Run the turn on a worker thread, streaming [`Update`]s back.
    ///
    /// This is the shape a UI needs: the caller never blocks, deltas arrive as
    /// they decode, and the session comes back on [`Update::Done`] with the
    /// reply appended.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use pio_gen2::{Engine, Session, Update};
    /// # let engine = Arc::new(Engine::load("m.gguf")?);
    /// let turn = engine.chat_owned(Session::new()).user("Hello").spawn();
    ///
    /// for update in turn {
    ///     match update {
    ///         Update::Delta(t) => print!("{t}"),
    ///         Update::Done { session, .. } => {
    ///             println!("\n{}", session.latest_text().unwrap_or_default());
    ///         }
    ///         Update::Failed { error, .. } => eprintln!("\n{error}"),
    ///         _ => {}
    ///     }
    /// }
    /// # Ok::<(), pio_gen2::Error>(())
    /// ```
    pub fn spawn(self) -> Turn {
        let (tx, rx) = channel();
        let engine = Arc::clone(&self.engine);
        let session_id = self.session.id().to_string();

        let join = std::thread::spawn(move || {
            let OwnedChat {
                engine,
                mut session,
                spec,
                thinking,
                tools,
            } = self;

            let deltas = tx.clone();
            let mut chat = engine.chat(&mut session).gen_spec(spec).thinking(thinking);
            if let Some((tools, prompt)) = tools {
                chat = chat.tools(tools, prompt);
            }

            // A send failure means the receiver went away. Keep draining so the
            // engine's session still ends cleanly.
            let result = chat.send_streaming(|fragment| {
                let _ = deltas.send(Update::Delta(fragment.to_string()));
            });

            let _ = match result {
                Ok(completion) => tx.send(Update::Done {
                    completion,
                    session,
                }),
                Err(error) => tx.send(Update::Failed { error, session }),
            };
        });

        Turn {
            rx,
            engine,
            session_id,
            join: Some(join),
        }
    }
}

impl std::fmt::Debug for OwnedChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedChat")
            .field("session", &self.session.id())
            .finish_non_exhaustive()
    }
}

/// A turn running on a worker thread.
///
/// Iterate it for [`Update`]s. The channel closes when the turn ends, so a
/// `for` loop over it terminates on its own.
pub struct Turn {
    rx: Receiver<Update>,
    engine: Arc<Engine>,
    session_id: String,
    join: Option<JoinHandle<()>>,
}

impl Turn {
    /// Stop this turn. It ends with [`Update::Done`] carrying whatever had
    /// been generated.
    pub fn cancel(&self) -> Result<()> {
        self.engine.stop(self.session_id.clone())
    }

    /// A cancel handle that can be moved to another thread.
    ///
    /// Cancellation has to come from somewhere other than the thread iterating
    /// updates, because that one is blocked.
    pub fn canceller(&self) -> Canceller {
        Canceller {
            engine: Arc::clone(&self.engine),
            session_id: self.session_id.clone(),
        }
    }

    /// The conversation this turn belongs to.
    pub fn session_id(&self) -> &str {
        &self.session_id
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
        // Join so the worker cannot outlive the handle and go on sending into a
        // dropped receiver. It exits once the generation ends; a caller wanting
        // out sooner calls `cancel` first.
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl std::fmt::Debug for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Turn")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

/// Cancels a running [`Turn`] from anywhere. Cheap to clone.
#[derive(Clone, Debug)]
pub struct Canceller {
    engine: Arc<Engine>,
    session_id: String,
}

impl Canceller {
    /// Stop the turn.
    pub fn cancel(&self) -> Result<()> {
        self.engine.stop(self.session_id.clone())
    }
}
