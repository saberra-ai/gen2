//! Async support, behind the `tokio` feature.
//!
//! Decoding is a blocking loop over a native backend — that is what inference
//! is, and no amount of `async` changes it. What this layer does is stop that
//! loop from occupying a runtime worker: the drain runs on
//! [`tokio::task::spawn_blocking`], and results arrive over a Tokio channel as
//! a [`Stream`].
//!
//! Without it, an async caller writes that bridge themselves for every call —
//! the same boilerplate [`OwnedChat::spawn`] removed for sync callers.
//!
//! Every entry point here needs a Tokio runtime; `spawn_blocking` panics
//! outside one.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

use super::agent::Steering;
use super::agent_spawned::OwnedAgent;
use super::engine::Engine;
use super::error::{Error, Result};
use super::session::Session;
use super::spawned::{Canceller, OwnedChat, Update};
use super::stream::Completion;

impl OwnedChat {
    /// Run the turn on a blocking worker, streaming [`Update`]s back.
    ///
    /// The async counterpart of [`OwnedChat::spawn`]. The session comes back on
    /// [`Update::Done`], as it does there.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use futures::StreamExt;
    /// # use gen2::{Engine, Session, Update};
    /// # async fn demo() -> Result<(), gen2::Error> {
    /// # let engine = Arc::new(Engine::load("m.gguf")?);
    /// let mut turn = engine.chat_owned(Session::new()).user("Hello").spawn_async();
    ///
    /// while let Some(update) = turn.next().await {
    ///     match update {
    ///         Update::Delta(t) => print!("{t}"),
    ///         Update::Done { session, .. } => drop(session),
    ///         Update::Failed { error, .. } => eprintln!("{error}"),
    ///         _ => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn_async(self) -> AsyncTurn {
        let (tx, rx) = unbounded_channel();
        let engine = self.engine_handle();
        let session_id = self.session_id().to_string();

        // Unbounded on purpose: the decode loop is blocking, so a bounded
        // channel would need a blocking send, and a slow consumer would stall
        // the backend mid-generation rather than merely lag behind it.
        let join = tokio::task::spawn_blocking(move || {
            let deltas = tx.clone();
            let result = self.run_blocking(|fragment| {
                let _ = deltas.send(Update::Delta(fragment.to_string()));
            });
            let _ = match result {
                Ok((completion, session)) => tx.send(Update::Done {
                    completion,
                    session,
                }),
                Err((error, session)) => tx.send(Update::Failed { error, session }),
            };
        });

        AsyncTurn {
            rx,
            engine,
            session_id,
            join: Some(join),
        }
    }

    /// Run the turn to completion, returning the outcome and the session.
    ///
    /// The async counterpart of [`Chat::send`](super::Chat::send). Use
    /// [`OwnedChat::spawn_async`] when you want the fragments as they arrive.
    pub async fn send_async(self) -> Result<(Completion, Session)> {
        tokio::task::spawn_blocking(move || self.run_blocking(|_| {}))
            .await
            .map_err(|_| Error::Generation {
                code: "task_panicked".into(),
                message: "the generation task panicked".into(),
            })?
            .map_err(|(error, _session)| error)
    }

    /// Run the turn to completion, streaming fragments to `on_token`.
    ///
    /// `on_token` runs on a blocking worker, so it must be `Send`. Keep it
    /// short — anything slow belongs behind a channel.
    pub async fn send_streaming_async(
        self,
        on_token: impl FnMut(&str) + Send + 'static,
    ) -> Result<(Completion, Session)> {
        tokio::task::spawn_blocking(move || self.run_blocking(on_token))
            .await
            .map_err(|_| Error::Generation {
                code: "task_panicked".into(),
                message: "the generation task panicked".into(),
            })?
            .map_err(|(error, _session)| error)
    }
}

impl OwnedAgent {
    /// Run the agent on a blocking worker, streaming [`Update`]s back.
    ///
    /// The async counterpart of [`OwnedAgent::spawn`]. Steering — including
    /// cutting a generation short — works the same, since the run owns an
    /// engine either way.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use futures::StreamExt;
    /// # use gen2::{Engine, Session, Update};
    /// # async fn demo() -> Result<(), gen2::Error> {
    /// # let engine = Arc::new(Engine::load("m.gguf")?);
    /// let mut run = engine.agent_owned(Session::new())
    ///     .goal("Summarise the repository")
    ///     .spawn_async();
    ///
    /// let steering = run.steering();
    /// while let Some(update) = run.next().await {
    ///     if let Update::Delta(t) = update { print!("{t}"); }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn_async(self) -> AsyncAgentRun {
        let (tx, rx) = unbounded_channel();

        // The synchronous run goes on a blocking worker; forwarding its updates
        // over a Tokio channel is the whole bridge.
        let run = self.spawn();
        let steering = run.steering();
        let session_id = run.session_id().to_string();

        let join = tokio::task::spawn_blocking(move || {
            for update in run {
                if tx.send(update).is_err() {
                    // Receiver gone. Keep draining so the worker still finishes
                    // and the engine's session ends cleanly.
                    continue;
                }
            }
        });

        AsyncAgentRun {
            rx,
            steering,
            session_id,
            join: Some(join),
        }
    }
}

/// An agent running on a blocking worker, yielding [`Update`]s as a [`Stream`].
pub struct AsyncAgentRun {
    rx: UnboundedReceiver<Update>,
    steering: Steering,
    session_id: String,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl AsyncAgentRun {
    /// A handle for injecting messages while this runs.
    pub fn steering(&self) -> Steering {
        self.steering.clone()
    }

    /// The conversation this run belongs to.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl Stream for AsyncAgentRun {
    type Item = Update;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Update>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for AsyncAgentRun {
    fn drop(&mut self) {
        // Abort rather than join: a drop can happen inside async context, where
        // blocking is forbidden. The worker finishes on its own.
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

impl std::fmt::Debug for AsyncAgentRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncAgentRun")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

/// A turn running on a blocking worker, yielding [`Update`]s.
///
/// Poll it as a [`Stream`]. It ends when the generation does.
pub struct AsyncTurn {
    rx: UnboundedReceiver<Update>,
    engine: Arc<Engine>,
    session_id: String,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl AsyncTurn {
    /// Stop this turn. It ends with [`Update::Done`] carrying whatever had
    /// been generated.
    pub fn cancel(&self) -> Result<()> {
        self.engine.stop(self.session_id.clone())
    }

    /// A cancel handle that can be moved elsewhere.
    pub fn canceller(&self) -> Canceller {
        Canceller::new(Arc::clone(&self.engine), self.session_id.clone())
    }

    /// The conversation this turn belongs to.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl Stream for AsyncTurn {
    type Item = Update;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Update>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for AsyncTurn {
    fn drop(&mut self) {
        // Unlike the sync `Turn`, this cannot join: dropping may happen inside
        // async context where blocking is forbidden. Aborting the handle is a
        // no-op for a blocking task already running, so the worker finishes on
        // its own and its sends land in a closed channel — harmless. The
        // engine's own Drop still joins the controller loop, which is what
        // guards the backend teardown.
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

impl std::fmt::Debug for AsyncTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncTurn")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}
