//! The worker thread, and the handle the controller talks to it through.

use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;

use crate::engine::EmbedLoadRequest;

use super::embedding::{EmbedderFactory, EmbeddingRuntime, default_factory};
use super::types::{LoadedUtility, UtilityStatus};

/// What the controller can ask the worker to do.
///
/// Every payload is owned. Nothing borrowed and no FFI pointer crosses the
/// channel — the models live on the far side and stay there.
pub(crate) enum UtilityCmd {
    LoadEmbedder {
        req: Box<EmbedLoadRequest>,
        estimated_mb: u64,
        resp: Sender<Result<(), String>>,
    },
    UnloadEmbedder,
    /// Embed, answering `resp` directly.
    ///
    /// `resp` is the *caller's* channel, handed straight through by the
    /// controller. That is what keeps the controller free: it forwards and
    /// moves on, and the reply never travels back through it.
    Embed {
        inputs: Vec<String>,
        resp: Sender<Result<Vec<Vec<f32>>, String>>,
    },
    Status {
        resp: Sender<UtilityStatus>,
    },
    Shutdown,
}

/// A handle to the auxiliary-runtime thread.
///
/// Dropping it shuts the thread down and waits for it, so a controller that
/// panics cannot leave a native worker running.
pub(crate) struct UtilityWorker {
    tx: Sender<UtilityCmd>,
    join: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for UtilityWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UtilityWorker")
            .field("running", &self.join.is_some())
            .finish()
    }
}

impl UtilityWorker {
    /// Start a worker over the real runtimes.
    pub(crate) fn spawn() -> Self {
        Self::spawn_with(default_factory())
    }

    /// Start a worker whose embedder comes from `factory`.
    ///
    /// The factory runs on the worker's own thread, so what it builds never
    /// has to be `Send`.
    pub(crate) fn spawn_with(factory: EmbedderFactory) -> Self {
        let (tx, rx) = channel::<UtilityCmd>();
        let join = std::thread::Builder::new()
            .name("gen2-utilities".into())
            .spawn(move || run(rx, factory))
            .expect("the utility worker thread should start");
        Self {
            tx,
            join: Some(join),
        }
    }

    /// Load an embedding model, waiting for the answer.
    ///
    /// Synchronous on purpose. A load is an explicit lifecycle operation, like
    /// `LoadModel`, and making it implicit would mean a first embedding call
    /// silently paying for a multi-gigabyte read.
    pub(crate) fn load_embedder(
        &self,
        req: EmbedLoadRequest,
        estimated_mb: u64,
    ) -> Result<(), String> {
        let (resp, rx) = channel();
        self.send(UtilityCmd::LoadEmbedder {
            req: Box::new(req),
            estimated_mb,
            resp,
        })?;
        rx.recv().map_err(|_| gone())?
    }

    pub(crate) fn unload_embedder(&self) {
        let _ = self.send(UtilityCmd::UnloadEmbedder);
    }

    /// Hand an embedding request over, answering the caller directly.
    ///
    /// Returns as soon as the worker has taken it. The controller is free
    /// immediately, which is the whole point: a helper that takes five seconds
    /// must not stop chat tokens for five seconds.
    pub(crate) fn embed_forwarding(
        &self,
        inputs: Vec<String>,
        resp: Sender<Result<Vec<Vec<f32>>, String>>,
    ) -> Result<(), String> {
        self.send(UtilityCmd::Embed { inputs, resp })
    }

    /// Which helpers are loaded.
    pub(crate) fn status(&self) -> UtilityStatus {
        let (resp, rx) = channel();
        if self.send(UtilityCmd::Status { resp }).is_err() {
            return UtilityStatus::default();
        }
        rx.recv().unwrap_or_default()
    }

    fn send(&self, cmd: UtilityCmd) -> Result<(), String> {
        self.tx.send(cmd).map_err(|_| gone())
    }

    /// Stop the thread and wait for it.
    fn stop_and_join(&mut self) {
        let _ = self.tx.send(UtilityCmd::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for UtilityWorker {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

fn gone() -> String {
    "the utility worker is no longer running".to_string()
}

/// What the worker owns. Never leaves the thread.
#[derive(Default)]
struct Runtimes {
    embedder: Option<Loaded>,
}

struct Loaded {
    runtime: Box<dyn EmbeddingRuntime>,
    estimated_mb: u64,
}

fn run(rx: std::sync::mpsc::Receiver<UtilityCmd>, factory: EmbedderFactory) {
    let mut runtimes = Runtimes::default();

    while let Ok(cmd) = rx.recv() {
        match cmd {
            UtilityCmd::LoadEmbedder {
                req,
                estimated_mb,
                resp,
            } => {
                let outcome = factory(&req).map(|runtime| {
                    runtimes.embedder = Some(Loaded {
                        runtime,
                        estimated_mb,
                    });
                });
                let _ = resp.send(outcome);
            }
            UtilityCmd::UnloadEmbedder => {
                runtimes.embedder = None;
            }
            UtilityCmd::Embed { inputs, resp } => {
                let outcome = match runtimes.embedder.as_ref() {
                    Some(loaded) => loaded.runtime.embed(&inputs),
                    None => Err("no embedding model is loaded".to_string()),
                };
                // Straight to whoever asked. A closed channel means the caller
                // gave up, which is their right and not an error here.
                let _ = resp.send(outcome);
            }
            UtilityCmd::Status { resp } => {
                let _ = resp.send(UtilityStatus {
                    embedder: runtimes.embedder.as_ref().map(|l| LoadedUtility {
                        name: l.runtime.name(),
                        estimated_resident_mb: l.estimated_mb,
                    }),
                });
            }
            UtilityCmd::Shutdown => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::ScriptedEmbedder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    struct Handles {
        calls: Arc<AtomicUsize>,
    }

    fn worker_with_latency(latency: Duration) -> (UtilityWorker, Handles) {
        let busy = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let (b, c) = (busy, Arc::clone(&calls));
        let worker = UtilityWorker::spawn_with(Box::new(move |_req| {
            Ok(Box::new(ScriptedEmbedder {
                name: "scripted".into(),
                latency,
                busy: Arc::clone(&b),
                calls: Arc::clone(&c),
            }) as Box<dyn EmbeddingRuntime>)
        }));
        (worker, Handles { calls })
    }

    fn load_request() -> EmbedLoadRequest {
        EmbedLoadRequest {
            model_path: "/models/embedder.gguf".into(),
            kind: None,
        }
    }

    #[test]
    fn embedding_before_a_model_is_loaded_says_so() {
        let (worker, _) = worker_with_latency(Duration::ZERO);
        let (resp, rx) = channel();
        worker
            .embed_forwarding(vec!["hello".into()], resp)
            .expect("the worker should take the request");
        let outcome = rx.recv().expect("the worker answers the caller directly");
        assert!(
            matches!(outcome, Err(m) if m.contains("no embedding model")),
            "an unloaded helper has to say which model is missing"
        );
    }

    #[test]
    fn a_loaded_embedder_answers_the_caller_directly() {
        let (worker, handles) = worker_with_latency(Duration::ZERO);
        worker.load_embedder(load_request(), 64).expect("load");

        let (resp, rx) = channel();
        worker
            .embed_forwarding(vec!["hello".into(), "hi".into()], resp)
            .expect("forward");
        let vectors = rx.recv().expect("reply").expect("embeddings");

        assert_eq!(vectors.len(), 2, "one vector per input, in order");
        assert_eq!(vectors[0][0], 5.0);
        assert_eq!(vectors[1][0], 2.0);
        assert_eq!(handles.calls.load(Ordering::SeqCst), 1);
    }

    /// Forwarding must not wait for the answer.
    ///
    /// This is the property the whole module exists for. If
    /// `embed_forwarding` blocked until the helper finished, the controller
    /// calling it would stop scheduling chat tokens for exactly as long — and
    /// nothing else in the design would help.
    #[test]
    fn handing_over_a_slow_request_returns_before_it_finishes() {
        const LATENCY: Duration = Duration::from_millis(400);
        let (worker, _handles) = worker_with_latency(LATENCY);
        worker.load_embedder(load_request(), 64).expect("load");

        let (resp, rx) = channel();
        let handed_over = std::time::Instant::now();
        worker
            .embed_forwarding(vec!["slow".into()], resp)
            .expect("forward");
        let to_hand_over = handed_over.elapsed();

        // The reply cannot have been waited for, because it had not happened.
        // Deliberately not asserting on the runtime's `busy` flag: the
        // hand-off can return before the worker has even dequeued the command,
        // so `busy` is legitimately still false and a test reading it is
        // racing the thread it is trying to describe.
        assert!(
            to_hand_over < LATENCY / 2,
            "handing the request over took {to_hand_over:?} against a helper \
             latency of {LATENCY:?}; the caller waited for the helper instead \
             of forwarding it"
        );

        let vectors = rx.recv().expect("reply").expect("embeddings");
        assert_eq!(vectors.len(), 1);
        assert!(
            handed_over.elapsed() >= LATENCY,
            "the helper answered faster than it was told to work, so this test \
             was not measuring what it claims"
        );
    }

    #[test]
    fn status_reports_what_is_loaded_and_what_it_costs() {
        let (worker, _) = worker_with_latency(Duration::ZERO);
        assert!(worker.status().is_empty(), "nothing is loaded yet");

        worker.load_embedder(load_request(), 128).expect("load");
        let status = worker.status();
        let embedder = status.embedder.expect("the embedder should be reported");
        assert_eq!(embedder.name, "scripted");
        assert_eq!(embedder.estimated_resident_mb, 128);

        worker.unload_embedder();
        assert!(
            worker.status().is_empty(),
            "an unloaded helper must stop being reported as loaded"
        );
    }

    #[test]
    fn a_failing_load_leaves_nothing_loaded() {
        let worker = UtilityWorker::spawn_with(Box::new(|_req| Err("no such file".to_string())));
        let outcome = worker.load_embedder(load_request(), 64);
        assert!(matches!(outcome, Err(m) if m.contains("no such file")));
        assert!(
            worker.status().is_empty(),
            "a load that failed must not leave a helper reported as present"
        );
    }

    /// Dropping the handle must not leave the thread alive.
    #[test]
    fn dropping_the_worker_stops_its_thread() {
        let (worker, _) = worker_with_latency(Duration::ZERO);
        let (resp, rx) = channel();
        worker.embed_forwarding(vec!["x".into()], resp).unwrap();
        let _ = rx.recv();

        drop(worker);
        // If the thread had outlived the handle, the join in `Drop` would have
        // hung and this test would never have reached here.
    }

    /// A caller that gives up must not take the worker down with it.
    #[test]
    fn a_caller_that_stops_listening_does_not_break_the_worker() {
        let (worker, handles) = worker_with_latency(Duration::ZERO);
        worker.load_embedder(load_request(), 64).expect("load");

        let (resp, rx) = channel();
        drop(rx); // the caller walked away
        worker
            .embed_forwarding(vec!["orphan".into()], resp)
            .unwrap();

        // The worker is still answering afterwards.
        let (resp, rx) = channel();
        worker.embed_forwarding(vec!["after".into()], resp).unwrap();
        let vectors = rx.recv().expect("reply").expect("embeddings");
        assert_eq!(vectors.len(), 1);
        assert_eq!(handles.calls.load(Ordering::SeqCst), 2);
    }
}
