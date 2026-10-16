use super::*;
use std::{
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

/// Observable native lifecycle. Stopping retains admission until destruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ready,
    Running,
    Stopping,
    Unloaded,
}
struct State {
    status: Mutex<Status>,
    done: Condvar,
    stop: AtomicBool,
    busy: AtomicBool,
    active: Mutex<Option<Cancellation>>,
}
impl State {
    fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        let mut status = self.status.lock().unwrap_or_else(|e| e.into_inner());
        if *status != Status::Unloaded {
            *status = Status::Stopping;
        }
        if let Some(active) = self
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            active.cancel();
        }
    }
    fn wait(&self) {
        let mut status = self.status.lock().unwrap_or_else(|e| e.into_inner());
        while *status != Status::Unloaded {
            status = self.done.wait(status).unwrap_or_else(|e| e.into_inner());
        }
    }
}
#[cfg_attr(not(feature = "backend-diar-nemo"), allow(dead_code))]
enum Job {
    Recording {
        audio: Audio,
        options: DiarizeOptions,
        reply: mpsc::Sender<Result<Diarization>>,
    },
    Stream {
        input: mpsc::Receiver<Vec<f32>>,
        shared: Arc<super::stream::Shared>,
        ready: mpsc::Sender<Result<()>>,
    },
}
#[cfg(feature = "backend-diar-nemo")]
impl Job {
    fn options(&self) -> &DiarizeOptions {
        match self {
            Self::Recording { options, .. } => options,
            Self::Stream { shared, .. } => &shared.options,
        }
    }
}
struct Inner {
    tx: mpsc::SyncSender<Job>,
    state: Arc<State>,
    bundle: DiarizationBundle,
}
impl Drop for Inner {
    fn drop(&mut self) {
        self.state.stop();
    }
}
/// A cloneable admitted model. One recording owns its slot; competitors get Busy.
#[derive(Clone)]
pub struct Diarizer {
    inner: Arc<Inner>,
}
pub(crate) struct Monitor(Weak<Inner>);
impl Monitor {
    pub(crate) fn upgrade(&self) -> Option<Diarizer> {
        self.0.upgrade().map(|inner| Diarizer { inner })
    }
}
impl std::fmt::Debug for Diarizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Diarizer")
            .field("status", &self.status())
            .field("bundle", &self.inner.bundle)
            .finish()
    }
}
impl Diarizer {
    pub fn status(&self) -> Status {
        *self
            .inner
            .state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }
    pub fn bundle(&self) -> &DiarizationBundle {
        &self.inner.bundle
    }
    pub(crate) fn monitor(&self) -> Monitor {
        Monitor(Arc::downgrade(&self.inner))
    }
    /// Reject new work and request cooperative cancellation, without waiting for a kernel.
    pub fn unload(&self) {
        self.inner.state.stop();
    }
    /// Wait until native models and their memory reservation have been destroyed.
    pub fn shutdown(&self) {
        self.unload();
        self.inner.state.wait();
    }
    pub fn diarize(&self, audio: Audio, options: DiarizeOptions) -> Result<Diarization> {
        options.check()?;
        self.claim()?;
        let (reply, rx) = mpsc::channel();
        self.submit(Job::Recording {
            audio,
            options: options.clone(),
            reply,
        })?;
        loop {
            options.check()?;
            match rx.recv_timeout(Duration::from_millis(10)) {
                Ok(result) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => return Err(DiarizationError::Unloaded),
            }
        }
    }
    /// Start one incremental session, retaining its speaker cache until finish/cancel.
    pub fn start_stream(&self, options: DiarizeOptions) -> Result<DiarizationStream> {
        options.check()?;
        self.claim()?;
        let shared = super::stream::Shared::new(options);
        let (tx, input) = mpsc::sync_channel(2);
        let (ready, rx) = mpsc::channel();
        self.submit(Job::Stream {
            input,
            shared: Arc::clone(&shared),
            ready,
        })?;
        rx.recv()
            .map_err(|_| DiarizationError::Execution("stream failed to open".into()))??;
        Ok(DiarizationStream::new(self.clone(), tx, shared))
    }
    fn claim(&self) -> Result<()> {
        if self.inner.state.stop.load(Ordering::Acquire) {
            return Err(DiarizationError::Unloaded);
        }
        if self
            .inner
            .state
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(DiarizationError::Busy);
        }
        Ok(())
    }
    fn submit(&self, job: Job) -> Result<()> {
        if self.inner.tx.try_send(job).is_err() {
            self.inner.state.busy.store(false, Ordering::Release);
            return Err(DiarizationError::Unloaded);
        }
        Ok(())
    }
}

pub(crate) fn spawn(
    bundle: DiarizationBundle,
    options: LoadOptions,
    reservation: Box<dyn Send>,
) -> Result<Diarizer> {
    #[cfg(not(feature = "backend-diar-nemo"))]
    {
        let _ = (bundle, options, reservation);
        Err(DiarizationError::BackendUnavailable(
            "enable backend-diar-nemo".into(),
        ))
    }
    #[cfg(feature = "backend-diar-nemo")]
    {
        let state = Arc::new(State {
            status: Mutex::new(Status::Ready),
            done: Condvar::new(),
            stop: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            active: Mutex::new(None),
        });
        let worker_state = Arc::clone(&state);
        let worker_bundle = bundle.clone();
        let (tx, rx) = mpsc::sync_channel::<Job>(1);
        let (ready_tx, ready_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("gen2-nemotron-diar".into())
            .spawn(move || {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut model = match super::nemo::Native::load(&worker_bundle, options) {
                        Ok(model) => model,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    if ready_tx.send(Ok(())).is_err() {
                        return;
                    }
                    while !worker_state.stop.load(Ordering::Acquire) {
                        let job = match rx.recv_timeout(Duration::from_millis(20)) {
                            Ok(job) => job,
                            Err(mpsc::RecvTimeoutError::Timeout) => continue,
                            Err(_) => break,
                        };
                        let controls = job.options().clone();
                        *worker_state
                            .active
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) =
                            Some(controls.cancellation.clone());
                        if worker_state.stop.load(Ordering::Acquire) {
                            controls.cancellation.cancel();
                        }
                        {
                            let mut status = worker_state
                                .status
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            if *status != Status::Stopping {
                                *status = Status::Running;
                            }
                        }
                        let mut recording_reply = None;
                        let mut streaming_reply = None;
                        let result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job {
                                Job::Recording { audio, reply, .. } => {
                                    recording_reply = Some(reply);
                                    model.run(audio, &controls)
                                }
                                Job::Stream {
                                    input,
                                    shared,
                                    ready,
                                } => {
                                    streaming_reply = Some(Arc::clone(&shared));
                                    model.run_live(input, &shared, ready)
                                }
                            }));
                        let panicked = result.is_err();
                        let result = result
                            .unwrap_or_else(|_| {
                                Err(DiarizationError::Execution("native worker panicked".into()))
                            })
                            .and_then(|result| {
                                controls.check()?;
                                Ok(result)
                            });
                        *worker_state
                            .active
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = None;
                        worker_state.busy.store(false, Ordering::Release);
                        let mut status = worker_state
                            .status
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        if *status != Status::Stopping {
                            *status = Status::Ready;
                        }
                        drop(status);
                        if let Some(reply) = recording_reply {
                            let _ = reply.send(result);
                        } else if let Some(shared) = streaming_reply {
                            shared.complete(result);
                        }
                        if panicked {
                            break;
                        }
                    }
                }));
                // All model/session/library destructors run before releasing admission.
                drop(rx);
                drop(reservation);
                *worker_state
                    .status
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Status::Unloaded;
                worker_state.done.notify_all();
            })
            .map_err(|e| DiarizationError::Execution(e.to_string()))?;
        let loaded = ready_rx
            .recv()
            .map_err(|_| DiarizationError::Execution("diarizer worker failed during load".into()))
            .and_then(|x| x);
        match loaded {
            Ok(()) => Ok(Diarizer {
                inner: Arc::new(Inner { tx, state, bundle }),
            }),
            Err(e) => {
                state.wait();
                Err(e)
            }
        }
    }
}
