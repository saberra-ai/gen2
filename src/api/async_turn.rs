//! The async surface (api_spec.md §22), behind the `tokio` feature.
//!
//! One builder vocabulary, two ways to run it. A [`Turn`] configured the same
//! way runs blocking or async; the async methods are the only additions, and
//! they return the same [`Response`], [`Event`]s, and [`Canceller`]:
//!
//! ```no_run
//! # async fn demo() -> gen2::Result<()> {
//! # let model = gen2::load("m.gguf")?;
//! # let mut session = gen2::Session::new();
//! use futures::StreamExt;
//! use gen2::event::Event;
//!
//! // Blocking.
//! let response = model.turn(&mut session).user("hello").run()?;
//!
//! // Async: the same turn, the same response.
//! let response = model.turn(&mut session).user("hello").run_async().await?;
//!
//! // Streaming, blocking.
//! let mut stream = model.turn(&mut session).user("hello").stream()?;
//! while let Some(event) = stream.next() {
//!     if let Event::TextDelta(text) = event? {
//!         print!("{text}");
//!     }
//! }
//! let response = stream.finish()?;
//!
//! // Streaming, async: the same events, the same `finish()`.
//! let mut stream = model.turn(&mut session).user("hello").stream_async().await?;
//! while let Some(event) = stream.next().await {
//!     if let Event::TextDelta(text) = event? {
//!         print!("{text}");
//!     }
//! }
//! let response = stream.finish().await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! # How it runs
//!
//! Decoding is a blocking loop over a native backend; `async` does not
//! change that. What this layer does is keep that loop off the runtime's
//! workers. A turn's engine half — the token stream and the event assembly,
//! which is `Send` and borrows nothing — goes to
//! [`tokio::task::spawn_blocking`]; the caller's `&mut Session` never leaves
//! the future that borrows it. Events cross from the worker over a bounded
//! [`tokio::sync::mpsc`] channel, so a consumer that stops polling holds the
//! worker rather than accumulating events. When the worker is done it hands
//! the stream's state back, and the outcome is written to the session on the
//! caller's side — the same code path `Turn::run` and `Turn::stream` use, so
//! the four cannot disagree about what was said or recorded.
//!
//! Dropping an [`AsyncEventStream`] before it ends cancels the generation:
//! the backend stops, and the session is marked to rebuild on its next turn,
//! since the partial reply was never recorded. Use
//! [`AsyncEventStream::finish`] to keep a partial reply after a cancel (§16.1).
//!
//! Every entry point here needs a Tokio runtime; `spawn_blocking` panics
//! outside one.
//!
//! [`Turn`]: super::turn::Turn

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt};
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use super::error::{Error, Result};
use super::event::{Canceller, Event, SessionSlot, StreamCore};
use super::generation::Generation;
use super::model::Model;
use super::response::Response;
use super::turn::Turn;

/// How many events the bridge holds between the worker and the consumer.
///
/// Small on purpose: it is a hand-off, not a buffer. A consumer that stops
/// polling holds the worker here, with this much memory and no more; the
/// slack that matters is the engine's own event channel behind it, which
/// gives an idle async reader exactly what it gives an idle blocking one —
/// and beyond which the controller sheds tokens, for both alike.
pub(crate) const EVENT_BUFFER: usize = 64;

impl<'a> Turn<'a> {
    /// Run the turn on a blocking worker and return the [`Response`].
    ///
    /// The async counterpart of [`Turn::run`]: the same response, the same
    /// assistant message appended to the session. The session stays borrowed
    /// until the future resolves.
    ///
    /// ```no_run
    /// # async fn demo() -> gen2::Result<()> {
    /// # let model = gen2::load("m.gguf")?;
    /// # let mut session = gen2::Session::new();
    /// let response = model
    ///     .turn(&mut session)
    ///     .user("hello")
    ///     .run_async()
    ///     .await?;
    /// println!("{}", response.text());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run_async(self) -> Result<Response> {
        self.stream_async().await?.finish().await
    }

    /// Run the turn as an async stream of semantic [`Event`]s.
    ///
    /// The async counterpart of [`Turn::stream`]: the same events, in the
    /// same order, ending with [`Event::Finished`]; then
    /// [`AsyncEventStream::finish`] for the [`Response`].
    ///
    /// ```no_run
    /// # async fn demo() -> gen2::Result<()> {
    /// # let model = gen2::load("m.gguf")?;
    /// # let mut session = gen2::Session::new();
    /// use futures::StreamExt;
    /// use gen2::event::Event;
    ///
    /// let mut stream = model.turn(&mut session).user("hello").stream_async().await?;
    /// let cancel = stream.canceller();
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         Event::TextDelta(text) => print!("{text}"),
    ///         Event::ToolCallStart { name, .. } => eprintln!("calling {name}"),
    ///         _ => {}
    ///     }
    /// }
    /// let response = stream.finish().await?;
    /// # let _ = (cancel, response);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream_async(self) -> Result<AsyncEventStream<'a>> {
        // Validation, commit, and dispatch are not blocking: the command
        // goes on the controller's channel and returns. Only the pull is.
        let (core, session) = self.begin()?;
        Ok(AsyncEventStream::start(
            core,
            SessionSlot::Borrowed(session),
        ))
    }
}

impl Generation<'_> {
    /// Run it on a blocking worker and return the full [`Response`].
    ///
    /// The async counterpart of [`Generation::run`].
    pub async fn run_async(self) -> Result<Response> {
        self.stream_async().await?.finish().await
    }

    /// Run it on a blocking worker and return the reply text.
    ///
    /// The async counterpart of [`Generation::text`].
    ///
    /// ```no_run
    /// # async fn demo() -> gen2::Result<()> {
    /// # let model = gen2::load("m.gguf")?;
    /// let text = model.generate("Why is the sky blue?").text_async().await?;
    /// # let _ = text;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn text_async(self) -> Result<String> {
        Ok(self.run_async().await?.text())
    }

    /// Run it as an async stream of semantic [`Event`]s.
    ///
    /// The async counterpart of [`Generation::stream`]. The stream owns the
    /// ephemeral session, so it borrows nothing and can be moved to another
    /// task.
    pub async fn stream_async(self) -> Result<AsyncEventStream<'static>> {
        let (model, session, staged) = self.prepare();
        let (core, session) = staged.begin(model, session)?;
        Ok(AsyncEventStream::start(
            core,
            SessionSlot::Owned(Box::new(session)),
        ))
    }
}

/// A turn running on a blocking worker, as an async [`Stream`] of [`Event`]s.
///
/// Built by [`Turn::stream_async`] or [`Generation::stream_async`]. Poll it
/// to the end, then [`AsyncEventStream::finish`] for the [`Response`] — the
/// same one [`Turn::run_async`] would have returned, with the assistant
/// message already appended to the session.
///
/// Dropping it before the end cancels the generation; the session's next
/// turn rebuilds from its messages, which do not include the abandoned reply.
#[must_use = "streams do nothing unless polled"]
pub struct AsyncEventStream<'a> {
    rx: Receiver<Result<Event>>,
    /// The worker pulling events; `None` once joined, or never spawned.
    worker: Option<JoinHandle<StreamCore>>,
    /// The stream's state, back from the worker and settled into the
    /// session: what is left to yield, and the response.
    tail: Option<StreamCore>,
    session: SessionSlot<'a>,
    model: Model,
    canceller: Canceller,
}

impl<'a> AsyncEventStream<'a> {
    /// Send the core to a blocking worker and bridge its events back.
    fn start(mut core: StreamCore, session: SessionSlot<'a>) -> Self {
        let model = core.model().clone();
        let canceller = core.canceller(session.id().to_string());
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);

        if core.is_done() {
            // Cancelled before it began: nothing to pull, nothing to spawn.
            drop(tx);
            return Self {
                rx,
                worker: None,
                tail: Some(core),
                session,
                model,
                canceller,
            };
        }

        let worker = tokio::task::spawn_blocking(move || {
            while let Some(event) = core.pull() {
                if tx.blocking_send(event).is_err() {
                    // The receiver is gone, and its Drop stopped the
                    // generation. Keep draining so the backend's turn ends
                    // cleanly rather than with a stranded puller.
                    continue;
                }
            }
            core
        });
        Self {
            rx,
            worker: Some(worker),
            tail: None,
            session,
            model,
            canceller,
        }
    }

    /// A handle that stops this turn from any task or thread. The same
    /// [`Canceller`] the blocking stream hands out.
    pub fn canceller(&self) -> Canceller {
        self.canceller.clone()
    }

    /// Drain whatever is left and return the outcome.
    ///
    /// The same [`Response`] shape as [`Turn::run`]: text, reasoning, tool
    /// calls, finish reason, usage, and the id of the assistant message now
    /// in the session. After a cancel, the partial reply is kept (§16.1).
    pub async fn finish(mut self) -> Result<Response> {
        while let Some(event) = self.next().await {
            event?;
        }
        let response = match self.tail.as_mut() {
            Some(core) => core.take_response()?,
            None => {
                return Err(Error::Generation {
                    code: "stream_failed".into(),
                    message: "the stream ended without an outcome".into(),
                });
            }
        };
        Ok(if self.session.is_owned() {
            response.detached()
        } else {
            response
        })
    }

    /// The worker has ended: take its state back and settle the session.
    fn settle(&mut self, mut core: StreamCore) {
        if core.needs_completion() {
            core.complete(self.session.get());
        }
        self.tail = Some(core);
    }
}

impl Stream for AsyncEventStream<'_> {
    type Item = Result<Event>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(core) = this.tail.as_mut() {
                // The closing events — usage, the finish reason — are queued
                // by completion, after the session is written.
                return Poll::Ready(core.pop_queued().map(Ok));
            }
            match this.rx.poll_recv(cx) {
                Poll::Ready(Some(event)) => return Poll::Ready(Some(event)),
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // The worker dropped its sender: it is returning.
                    let Some(worker) = this.worker.as_mut() else {
                        return Poll::Ready(None);
                    };
                    match Pin::new(worker).poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Ok(core)) => {
                            this.worker = None;
                            this.settle(core);
                        }
                        Poll::Ready(Err(_)) => {
                            this.worker = None;
                            return Poll::Ready(Some(Err(Error::Generation {
                                code: "task_panicked".into(),
                                message: "the generation task panicked".into(),
                            })));
                        }
                    }
                }
            }
        }
    }
}

impl futures::stream::FusedStream for AsyncEventStream<'_> {
    fn is_terminated(&self) -> bool {
        self.worker.is_none()
            && match &self.tail {
                Some(core) => !core.has_queued(),
                None => true,
            }
    }
}

impl Drop for AsyncEventStream<'_> {
    fn drop(&mut self) {
        if self.tail.is_none() {
            // Abandoned mid-flight. Stop the backend — the worker's next
            // send fails and it drains to the end on its own; a drop can be
            // inside async context, so it is not joined. The reply was never
            // recorded and the backend's context is ahead of the session's
            // messages: the next turn rebuilds from what the session holds.
            self.canceller.cancel();
            self.session.get().opened = false;
            if let Some(worker) = self.worker.take() {
                worker.abort();
            }
        }
        // An ephemeral session is over with its stream, however it ended.
        if let SessionSlot::Owned(session) = &self.session {
            self.model.engine().forget(session);
        }
    }
}

impl std::fmt::Debug for AsyncEventStream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncEventStream")
            .field("session", &self.session.id())
            .field("running", &self.worker.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use futures::stream::FusedStream;

    use super::*;
    use crate::api::output::FinishReason;
    use crate::api::runtime::Runtime;
    use crate::api::session::Session;
    use crate::api::tool_defs::{ToolDefinition, ToolSet};
    use crate::test_support::{Gate, Script, Step};

    fn tools() -> ToolSet {
        ToolSet::new()
            .with(ToolDefinition::new("read").description("Read a file"))
            .with(ToolDefinition::new("write").description("Write a file"))
    }

    /// The program every "same as sync" test runs: reasoning, text, a tool
    /// call — one of every event kind.
    fn program() -> Script {
        Script::new().program([
            Step::token("<think>\n"),
            Step::token("plan"),
            Step::token("\n</think>\n\n"),
            Step::token("Done"),
            Step::tool_call("write", r#"{"path":"a"}"#),
            Step::eos(),
        ])
    }

    /// Wait for the scripted generation to reach a hold, off the runtime.
    async fn reached(gate: &Arc<Gate>) -> bool {
        let gate = Arc::clone(gate);
        tokio::task::spawn_blocking(move || gate.wait_until_reached())
            .await
            .unwrap()
    }

    async fn eventually(what: &str, mut ok: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ok() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn run_async_returns_the_same_response_as_run() {
        let model = Runtime::scripted(program());
        let mut sync_session = Session::new().with_tools(tools());
        let mut async_session = Session::new().with_tools(tools());

        let sync = model.turn(&mut sync_session).user("go").run().unwrap();
        let asynced = model
            .turn(&mut async_session)
            .user("go")
            .run_async()
            .await
            .unwrap();

        assert_eq!(asynced.text(), sync.text());
        assert_eq!(asynced.reasoning(), sync.reasoning());
        assert_eq!(asynced.finish_reason(), sync.finish_reason());
        assert_eq!(asynced.tool_calls(), sync.tool_calls());
        assert_eq!(asynced.message_id(), sync.message_id());
        assert_eq!(
            async_session.messages(),
            sync_session.messages(),
            "the same messages were recorded"
        );
        assert_eq!(
            async_session.revision(),
            sync_session.revision(),
            "and the session moved the same amount"
        );
        assert!(async_session.opened, "settled: the conversation is open");
    }

    #[tokio::test]
    async fn stream_async_yields_the_same_events_as_stream() {
        let model = Runtime::scripted(program());
        let mut sync_session = Session::new().with_tools(tools());
        let mut async_session = Session::new().with_tools(tools());

        let mut sync = model.turn(&mut sync_session).user("go").stream().unwrap();
        let sync_events: Vec<Event> = sync.by_ref().map(|e| e.unwrap()).collect();
        let sync_response = sync.finish().unwrap();

        let mut stream = model
            .turn(&mut async_session)
            .user("go")
            .stream_async()
            .await
            .unwrap();
        let mut async_events = Vec::new();
        while let Some(event) = stream.next().await {
            async_events.push(event.unwrap());
        }
        assert!(stream.is_terminated());
        assert!(
            stream.next().await.is_none(),
            "fused: nothing after the end"
        );
        let async_response = stream.finish().await.unwrap();

        assert_eq!(async_events, sync_events);
        assert_eq!(
            async_events.last(),
            Some(&Event::Finished(FinishReason::ToolCall))
        );
        assert_eq!(async_response.text(), sync_response.text());
        assert_eq!(async_response.tool_calls(), sync_response.tool_calls());
        assert_eq!(async_response.message_id(), sync_response.message_id());
        assert_eq!(async_session.messages(), sync_session.messages());
    }

    #[tokio::test]
    async fn one_shot_generation_runs_async_over_a_throwaway_session() {
        let model = Runtime::scripted(Script::new().say(["hi ", "there"]));
        let script = model.engine().script().clone();

        let text = model
            .generate("hello")
            .system("Be brief.")
            .text_async()
            .await
            .unwrap();
        assert_eq!(text, "hi there");

        let mut stream = model.generate("hello").stream_async().await.unwrap();
        let mut deltas = Vec::new();
        while let Some(event) = stream.next().await {
            deltas.push(event.unwrap());
        }
        let response = stream.finish().await.unwrap();
        assert_eq!(
            deltas,
            vec![
                Event::TextDelta("hi ".into()),
                Event::TextDelta("there".into()),
                Event::Finished(FinishReason::Stop),
            ]
        );
        assert_eq!(response.text(), "hi there");
        assert_eq!(response.message_id(), None, "there is no session to name");

        // The blocking stream on a generation is the same thing.
        let mut stream = model.generate("hello").stream().unwrap();
        let sync: Vec<Event> = stream.by_ref().map(|e| e.unwrap()).collect();
        assert_eq!(sync, deltas);
        assert_eq!(stream.finish().unwrap().message_id(), None);
        assert_eq!(script.seen().iter().filter(|m| *m == "hello").count(), 3);
    }

    #[tokio::test]
    async fn cancelling_from_another_task_ends_with_cancelled() {
        // §16 across tasks: the canceller goes to a spawned task.
        let gate = Gate::new();
        let model = Runtime::scripted(Script::new().program([
            Step::token("kept"),
            Step::Hold(Arc::clone(&gate)),
            Step::token(" never seen"),
            Step::eos(),
        ]));
        let mut session = Session::new();
        let mut stream = model
            .turn(&mut session)
            .user("Write a novel")
            .stream_async()
            .await
            .unwrap();
        let cancel = stream.canceller();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, Event::TextDelta("kept".into()));
        assert!(reached(&gate).await, "mid-flight");

        let stopper = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move {
                cancel.cancel();
                gate.open();
            }
        });
        let mut rest = Vec::new();
        while let Some(event) = stream.next().await {
            rest.push(event.unwrap());
        }
        stopper.await.unwrap();
        assert_eq!(
            rest.last(),
            Some(&Event::Finished(FinishReason::Cancelled)),
            "{rest:?}"
        );

        let response = stream.finish().await.unwrap();
        assert_eq!(*response.finish_reason(), FinishReason::Cancelled);
        assert!(response.text().starts_with("kept"), "{:?}", response.text());
        let id = response
            .message_id()
            .expect("the partial reply was recorded");
        assert_eq!(session.latest_text(), Some(response.text()));
        session.remove_message(id).unwrap();
        assert_eq!(session.len(), 1);
        assert!(!session.opened, "a stopped conversation rebuilds next turn");
    }

    #[tokio::test]
    async fn a_canceller_taken_before_run_makes_the_async_turn_not_run() {
        let model = Runtime::scripted(Script::new().say(["never"]));
        let script = model.engine().script().clone();
        let mut session = Session::new();
        let turn = model.turn(&mut session).user("go");
        turn.canceller().cancel();
        let response = turn.run_async().await.unwrap();
        assert_eq!(*response.finish_reason(), FinishReason::Cancelled);
        assert_eq!(response.message_id(), None);
        assert!(session.is_empty(), "nothing was staged or generated");
        assert_eq!(script.count("pull"), 0);
    }

    #[tokio::test]
    async fn dropping_the_stream_mid_way_stops_the_producer() {
        let first = Gate::new();
        let later = Gate::new();
        let model = Runtime::scripted(Script::new().turns([
            vec![Step::token("opening"), Step::eos()],
            vec![
                Step::token("a"),
                Step::Hold(Arc::clone(&first)),
                Step::token("b"),
                Step::Hold(Arc::clone(&later)),
                Step::token("c"),
                Step::eos(),
            ],
            vec![Step::token("fresh"), Step::eos()],
        ]));
        let script = model.engine().script().clone();
        let mut session = Session::new();
        // An open conversation, so the abandoned turn is a continuation.
        model.turn(&mut session).user("earlier").run().unwrap();
        assert!(session.opened);
        assert_eq!(script.count("start_session"), 1);
        let before = session.len();

        let mut stream = model
            .turn(&mut session)
            .user("go")
            .stream_async()
            .await
            .unwrap();
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            Event::TextDelta("a".into())
        );
        assert!(reached(&first).await, "mid-flight");
        assert_eq!(script.count("end_session"), 0);

        drop(stream);
        first.open();

        // The stop lands once the hold releases; the backend's session ends,
        // and the script never gets past the next token.
        eventually("the backend session to end", || {
            script.count("end_session") == 1
        })
        .await;
        assert!(
            !later.is_reached(),
            "the producer stopped before the second hold"
        );
        assert_eq!(
            session.len(),
            before + 1,
            "the user message was committed; the abandoned reply was not recorded"
        );
        assert!(!session.opened, "the next turn rebuilds from the session");

        // And the next turn works: a fresh start with the whole transcript.
        let response = model
            .turn(&mut session)
            .user("again")
            .run_async()
            .await
            .unwrap();
        assert_eq!(response.text(), "fresh");
        assert_eq!(script.count("start_session"), 2, "rebuilt, not continued");
        assert_eq!(
            script.seen().iter().filter(|m| *m == "go").count(),
            2,
            "the abandoned turn's user message was in the rebuilt transcript"
        );
    }

    #[tokio::test]
    async fn the_bridge_is_bounded_and_holds_the_worker_until_polled() {
        let capacity = EVENT_BUFFER;
        // More than the bridge holds, so the worker must park; fewer than
        // the engine's channel holds, so however the threads interleave
        // nothing is shed and every token arrives.
        let n = 300;
        let gate = Gate::new();
        let mut steps: Vec<Step> = (0..n).map(|i| Step::token(&format!("{i} "))).collect();
        steps.push(Step::Hold(Arc::clone(&gate)));
        steps.push(Step::eos());
        let model = Runtime::scripted(Script::new().program(steps));
        assert!(capacity < n && n < model.engine().event_channel_capacity());
        let mut session = Session::new();

        let mut stream = model
            .turn(&mut session)
            .user("count")
            .stream_async()
            .await
            .unwrap();

        // Unpolled: the backend runs through every token to the hold — the
        // engine's channel absorbs what the bridge will not — while the
        // bridge fills to its bound and the worker parks on it.
        assert!(reached(&gate).await, "the backend was not stalled");
        eventually("the bridge to fill", || stream.rx.len() == capacity).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(stream.rx.len(), capacity, "and never past its bound");

        // Polling releases it: everything arrives, in order, nothing lost.
        gate.open();
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.unwrap());
        }
        let deltas: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                Event::TextDelta(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), n);
        assert_eq!(deltas[0], "0 ");
        assert_eq!(deltas[n - 1], format!("{} ", n - 1));
        assert_eq!(events.last(), Some(&Event::Finished(FinishReason::Stop)));
        let response = stream.finish().await.unwrap();
        assert_eq!(response.text(), deltas.concat());
    }

    #[tokio::test]
    async fn a_refused_turn_leaves_the_session_untouched_async_too() {
        let model = Runtime::scripted(Script::new().say(["never"]));
        let mut session = Session::new();
        let revision = session.revision();
        let err = model
            .turn(&mut session)
            .user("look")
            .image("/tmp/photo.png")
            .run_async()
            .await
            .expect_err("a text-only model refuses images");
        assert!(matches!(err, Error::Unsupported(ref what) if what == "images"));
        assert!(session.is_empty());
        assert_eq!(session.revision(), revision);
    }

    #[test]
    fn the_async_surface_is_send() {
        fn assert_send<T: Send>() {}
        fn assert_send_future<F: std::future::Future + Send>(_: &F) {}
        fn assert_send_stream<S: Stream + Send>(_: &S) {}

        assert_send::<Model>();
        assert_send::<Response>();
        assert_send::<Event>();
        assert_send::<Canceller>();
        assert_send::<AsyncEventStream<'static>>();
        assert_send::<crate::api::event::EventStream<'static>>();

        // The futures themselves, over a real (scripted) model — built,
        // checked, and dropped unpolled: nothing runs.
        let model = Runtime::scripted(Script::new().say(["x"]));
        let mut session = Session::new();
        let run = model.turn(&mut session).user("hi").run_async();
        assert_send_future(&run);
        drop(run);
        let stream = model.turn(&mut session).user("hi").stream_async();
        assert_send_future(&stream);
        drop(stream);
        let text = model.generate("hi").text_async();
        assert_send_future(&text);
        drop(text);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let stream = model.generate("hi").stream_async().await.unwrap();
            assert_send_stream(&stream);
            let response = stream.finish().await.unwrap();
            assert_eq!(response.text(), "x");
        });
        assert!(session.is_empty(), "the dropped futures ran nothing");
    }
}
