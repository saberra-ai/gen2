use super::*;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    time::{Duration, Instant},
};

/// Cooperative cancellation. Native work retains its resources until it returns.
#[derive(Debug, Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Required by `laya-dynamic`; supplied and packaged by the host.
    pub native_library: Option<PathBuf>,
    pub queue_capacity: usize,
    pub intra_threads: usize,
    /// Bound retained request bytes; increase explicitly for larger inputs.
    pub max_input_bytes: usize,
    pub execution: ExecutionOptions,
}
/// Explicit provider selection. Accelerator paths require a matching packaged
/// runtime and independent graph/device qualification; CPU is the default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    DirectMl {
        device_id: i32,
    },
    Cuda {
        device_id: i32,
    },
    CoreMl,
    Nnapi,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionOptions {
    pub provider: ExecutionProvider,
    /// Permit ONNX to partition unsupported accelerator nodes onto CPU.
    /// Registration failure always returns an error, even when this is true.
    pub allow_cpu_fallback: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionCapabilities {
    pub state_forms: Vec<String>,
    pub question_forms: Vec<String>,
    pub option_permutation: bool,
    pub multi_state_batch: bool,
    pub long_state_windows: bool,
    pub calibration_import_versions: Vec<u32>,
    pub envelope: SequenceEnvelope,
    pub execution: ExecutionOptions,
    pub queue_capacity: usize,
    pub max_input_bytes: usize,
    pub intra_threads: usize,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            native_library: None,
            queue_capacity: 8,
            intra_threads: 2,
            max_input_bytes: 16 * 1024 * 1024,
            execution: ExecutionOptions::default(),
        }
    }
}
impl LoadOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if matches!(self.execution.provider,
            ExecutionProvider::DirectMl { device_id } | ExecutionProvider::Cuda { device_id } if device_id < 0)
        {
            return Err(DecisionError::InvalidRequest(
                "device_id must be nonnegative".into(),
            ));
        }
        if self.queue_capacity == 0 || self.intra_threads == 0 || self.max_input_bytes == 0 {
            return Err(DecisionError::InvalidRequest(
                "load budgets must be nonzero".into(),
            ));
        }
        #[cfg(feature = "laya-dynamic")]
        if !self.native_library.as_ref().is_some_and(|p| p.is_file()) {
            return Err(DecisionError::BackendUnavailable(
                "native_library must name the host's ONNX Runtime 1.24 library".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Default)]
pub struct DecisionOptions {
    pub encode: Option<EncodeOptions>,
    pub language: Option<String>,
    pub deadline: Option<Instant>,
    pub cancellation: Cancellation,
}
impl DecisionOptions {
    pub(crate) fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            Err(DecisionError::Cancelled)
        } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
            Err(DecisionError::DeadlineExceeded)
        } else {
            Ok(())
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResult {
    pub answers: Vec<(String, Answer)>,
    pub inputs: Vec<EncodedQuestion>,
    pub manifest_sha256: String,
    pub checkpoint: Checkpoint,
    pub calibration_diagnostics: Vec<CalibrationDiagnostic>,
    pub execution: ExecutionOptions,
    /// Job-wide timings, repeated for each state in a batch. Inside a long-state
    /// result, queue time is zero and execution time covers this window only.
    pub queue_micros: u64,
    pub execution_micros: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionStatus {
    Ready,
    Running,
    Stopping,
    Unloaded,
}
struct State {
    status: Mutex<DecisionStatus>,
    done: Condvar,
    stop: AtomicBool,
    active: Mutex<Option<Cancellation>>,
}
impl State {
    #[cfg(any(feature = "backend-laya-onnx", test))]
    fn set(&self, v: DecisionStatus) {
        let mut status = self.status.lock().unwrap_or_else(|e| e.into_inner());
        *status = if self.stop.load(Ordering::Acquire) && v != DecisionStatus::Unloaded {
            DecisionStatus::Stopping
        } else {
            v
        };
        self.done.notify_all();
    }
    fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        if let Some(cancellation) = self
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            cancellation.cancel();
        }
    }
}
struct Inner {
    tx: SyncSender<Job>,
    state: Arc<State>,
    bundle: LayaBundle,
    max_input_bytes: usize,
    capabilities: DecisionCapabilities,
}
impl Drop for Inner {
    fn drop(&mut self) {
        self.state.stop();
    }
}
/// Cloneable handle to a single bounded worker. Cloning never copies weights.
#[derive(Clone)]
pub struct DecisionModel {
    inner: Arc<Inner>,
}
pub(crate) struct DecisionMonitor(std::sync::Weak<Inner>);
impl DecisionMonitor {
    pub(crate) fn upgrade(&self) -> Option<DecisionModel> {
        self.0.upgrade().map(|inner| DecisionModel { inner })
    }
}
impl std::fmt::Debug for DecisionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionModel")
            .field("checkpoint", &self.info().checkpoint)
            .field("status", &self.status())
            .finish()
    }
}
#[cfg_attr(not(any(feature = "backend-laya-onnx", test)), allow(dead_code))]
struct Job {
    requests: Vec<DecisionRequest>,
    scan: Option<LongStateOptions>,
    options: DecisionOptions,
    reply: mpsc::Sender<Result<WorkerResult>>,
    queued_at: Instant,
}
#[derive(Debug)]
#[cfg_attr(not(any(feature = "backend-laya-onnx", test)), allow(dead_code))]
enum WorkerResult {
    Batch(Vec<DecisionResult>),
    Long(LongDecisionResult),
}
#[cfg(feature = "tokio")]
struct CancelOnDrop(Option<Cancellation>);
#[cfg(feature = "tokio")]
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(token) = &self.0 {
            token.cancel();
        }
    }
}
struct InputBudget {
    remaining: usize,
    exceeded: bool,
}
impl std::io::Write for InputBudget {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            self.exceeded = true;
            return Err(std::io::Error::other("input byte budget exceeded"));
        }
        self.remaining -= bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[cfg(any(feature = "backend-laya-onnx", test))]
pub(crate) trait DecisionBackend: Send {
    fn run_batch(
        &mut self,
        requests: &[DecisionRequest],
        options: &DecisionOptions,
    ) -> Result<Vec<DecisionResult>>;
    fn run_long(
        &mut self,
        request: DecisionRequest,
        scan: LongStateOptions,
        options: DecisionOptions,
    ) -> Result<LongDecisionResult>;
}
impl DecisionModel {
    pub(crate) fn validate_admission(
        &self,
        requests: &[DecisionRequest],
        options: &DecisionOptions,
    ) -> Result<()> {
        options.check()?;
        if self.inner.state.stop.load(Ordering::Acquire) {
            return Err(DecisionError::Unloaded);
        }
        // Count serialized bytes without allocating a second copy of a possibly
        // oversized state. Apply this before validation can serialize the state.
        let mut budget = InputBudget {
            remaining: self.inner.max_input_bytes,
            exceeded: false,
        };
        let encoded = serde_json::to_writer(&mut budget, requests);
        if budget.exceeded {
            return Err(DecisionError::ResourceLimit(format!(
                "request exceeds configured input limit of {} bytes",
                self.inner.max_input_bytes
            )));
        }
        encoded.map_err(|e| DecisionError::InvalidRequest(e.to_string()))?;
        for request in requests {
            request.validate()?;
        }
        Ok(())
    }
    pub(crate) fn monitor(&self) -> DecisionMonitor {
        DecisionMonitor(Arc::downgrade(&self.inner))
    }
    pub(crate) fn bundle(&self) -> &LayaBundle {
        &self.inner.bundle
    }
    pub fn info(&self) -> &BundleManifest {
        self.inner.bundle.manifest()
    }
    /// The loaded bundle and host policy. Successful provider registration is
    /// not a claim that this device/provider has passed the parity release gate.
    pub fn capabilities(&self) -> &DecisionCapabilities {
        &self.inner.capabilities
    }
    pub fn status(&self) -> DecisionStatus {
        *self
            .inner
            .state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }
    /// Stop admission immediately. Safe in a mobile lifecycle callback; this
    /// does not wait for a native kernel or free buffers still in use.
    pub fn unload(&self) {
        tracing::debug!(
            manifest = self.inner.bundle.manifest_sha256(),
            "Laya unload requested"
        );
        self.inner.state.stop();
        let mut s = self
            .inner
            .state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *s != DecisionStatus::Unloaded {
            *s = DecisionStatus::Stopping;
        }
    }
    /// Wait for native work and resource release. Do not call on a UI thread.
    pub fn shutdown(&self) {
        self.unload();
        let mut s = self
            .inner
            .state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while *s != DecisionStatus::Unloaded {
            s = self
                .inner
                .state
                .done
                .wait(s)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
    pub fn decide(
        &self,
        request: DecisionRequest,
        options: DecisionOptions,
    ) -> Result<DecisionResult> {
        self.decide_batch(vec![request], options)?
            .pop()
            .ok_or_else(|| DecisionError::InvalidOutput("missing batch result".into()))
    }
    /// Submit multiple states as one bounded job. Question rows from different
    /// states share native microbatches; results retain state and question order.
    /// The input byte budget applies to the entire batch.
    pub fn decide_batch(
        &self,
        requests: Vec<DecisionRequest>,
        options: DecisionOptions,
    ) -> Result<Vec<DecisionResult>> {
        self.validate_admission(&requests, &options)?;
        if requests.is_empty() {
            return Ok(vec![]);
        }
        match self.submit(requests, None, options)? {
            WorkerResult::Batch(results) => Ok(results),
            _ => Err(DecisionError::InvalidOutput(
                "wrong worker result kind".into(),
            )),
        }
    }
    /// Run preprocessing and all windows as one admitted worker job. Selection
    /// follows Laya: highest P(true) for yes/no, highest answer confidence for
    /// choice/score, with the first window winning a tie.
    pub fn decide_long(
        &self,
        request: DecisionRequest,
        scan: LongStateOptions,
        options: DecisionOptions,
    ) -> Result<LongDecisionResult> {
        self.validate_admission(std::slice::from_ref(&request), &options)?;
        super::long::scan_encode(&self.info().envelope, &options)?;
        match self.submit(vec![request], Some(scan), options)? {
            WorkerResult::Long(result) => Ok(result),
            _ => Err(DecisionError::InvalidOutput(
                "wrong worker result kind".into(),
            )),
        }
    }
    fn submit(
        &self,
        requests: Vec<DecisionRequest>,
        scan: Option<LongStateOptions>,
        options: DecisionOptions,
    ) -> Result<WorkerResult> {
        let (tx, rx) = mpsc::channel();
        match self.inner.tx.try_send(Job {
            requests,
            scan,
            options: options.clone(),
            reply: tx,
            queued_at: Instant::now(),
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(DecisionError::Busy),
            Err(TrySendError::Disconnected(_)) => return Err(DecisionError::Unloaded),
        }
        loop {
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(result) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if self.inner.state.stop.load(Ordering::Acquire) {
                        return Err(DecisionError::Unloaded);
                    }
                    options.check()?;
                }
                Err(_) => return Err(DecisionError::Unloaded),
            }
        }
    }
    #[cfg(feature = "tokio")]
    pub async fn decide_async(
        &self,
        request: DecisionRequest,
        options: DecisionOptions,
    ) -> Result<DecisionResult> {
        self.decide_batch_async(vec![request], options)
            .await?
            .pop()
            .ok_or_else(|| DecisionError::InvalidOutput("missing batch result".into()))
    }
    #[cfg(feature = "tokio")]
    pub async fn decide_batch_async(
        &self,
        requests: Vec<DecisionRequest>,
        options: DecisionOptions,
    ) -> Result<Vec<DecisionResult>> {
        let mut guard = CancelOnDrop(Some(options.cancellation.clone()));
        let model = self.clone();
        let result = tokio::task::spawn_blocking(move || model.decide_batch(requests, options))
            .await
            .map_err(|e| DecisionError::Execution(e.to_string()))?;
        guard.0 = None;
        result
    }
    #[cfg(feature = "tokio")]
    pub async fn decide_long_async(
        &self,
        request: DecisionRequest,
        scan: LongStateOptions,
        options: DecisionOptions,
    ) -> Result<LongDecisionResult> {
        let mut guard = CancelOnDrop(Some(options.cancellation.clone()));
        let model = self.clone();
        let result = tokio::task::spawn_blocking(move || model.decide_long(request, scan, options))
            .await
            .map_err(|e| DecisionError::Execution(e.to_string()))?;
        guard.0 = None;
        result
    }
    #[cfg(feature = "tokio")]
    pub async fn shutdown_async(&self) -> Result<()> {
        let model = self.clone();
        tokio::task::spawn_blocking(move || model.shutdown())
            .await
            .map_err(|e| DecisionError::Execution(e.to_string()))
    }
}
pub(crate) fn spawn(
    bundle: LayaBundle,
    options: LoadOptions,
    reservation: Box<dyn Send>,
) -> Result<DecisionModel> {
    #[cfg(feature = "backend-laya-onnx")]
    {
        let b = bundle.clone();
        let o = options.clone();
        spawn_with(bundle, options, reservation, move || {
            Ok(Box::new(super::onnx::OnnxBackend::load(b, o)?))
        })
    }
    #[cfg(not(feature = "backend-laya-onnx"))]
    {
        let _ = (bundle, options, reservation);
        Err(DecisionError::BackendUnavailable(
            "enable backend-laya-onnx (app-linked) or laya-dynamic (desktop library)".into(),
        ))
    }
}
#[cfg(any(feature = "backend-laya-onnx", test))]
fn spawn_with<F>(
    bundle: LayaBundle,
    options: LoadOptions,
    reservation: Box<dyn Send>,
    factory: F,
) -> Result<DecisionModel>
where
    F: FnOnce() -> Result<Box<dyn DecisionBackend>> + Send + 'static,
{
    if options.queue_capacity == 0 || options.intra_threads == 0 || options.max_input_bytes == 0 {
        return Err(DecisionError::InvalidRequest(
            "load budgets must be nonzero".into(),
        ));
    }
    let (tx, rx) = mpsc::sync_channel::<Job>(options.queue_capacity);
    let (ready_tx, ready_rx) = mpsc::channel();
    let state = Arc::new(State {
        status: Mutex::new(DecisionStatus::Ready),
        done: Condvar::new(),
        stop: AtomicBool::new(false),
        active: Mutex::new(None),
    });
    let worker_state = state.clone();
    std::thread::Builder::new()
        .name("gen2-laya".into())
        .spawn(move || {
            // Drop order is deliberate: session, reservation, then Unloaded signal.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut backend = match factory() {
                    Ok(b) => b,
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
                        Ok(j) => j,
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(_) => break,
                    };
                    if worker_state.stop.load(Ordering::Acquire) {
                        let _ = job.reply.send(Err(DecisionError::Unloaded));
                        break;
                    }
                    worker_state.set(DecisionStatus::Running);
                    static NEXT_JOB: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(1);
                    let _span = tracing::info_span!(
                        "laya_decide",
                        job_id = NEXT_JOB.fetch_add(1, Ordering::Relaxed),
                        states = job.requests.len()
                    )
                    .entered();
                    *worker_state
                        .active
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) =
                        Some(job.options.cancellation.clone());
                    if worker_state.stop.load(Ordering::Acquire) {
                        job.options.cancellation.cancel();
                    }
                    let result = job
                        .options
                        .check()
                        .and_then(|_| {
                            let started = Instant::now();
                            let queue_micros = started
                                .duration_since(job.queued_at)
                                .as_micros()
                                .min(u64::MAX as u128)
                                as u64;
                            let mut response = if let Some(scan) = job.scan {
                                let request = job.requests.into_iter().next().ok_or_else(|| {
                                    DecisionError::InvalidRequest("missing scan state".into())
                                })?;
                                WorkerResult::Long(backend.run_long(
                                    request,
                                    scan,
                                    job.options.clone(),
                                )?)
                            } else {
                                let results = backend.run_batch(&job.requests, &job.options)?;
                                if results.len() != job.requests.len() {
                                    return Err(DecisionError::InvalidOutput(
                                        "batch result count mismatch".into(),
                                    ));
                                }
                                WorkerResult::Batch(results)
                            };
                            let execution_micros =
                                started.elapsed().as_micros().min(u64::MAX as u128) as u64;
                            match &mut response {
                                WorkerResult::Batch(results) => {
                                    for result in results {
                                        result.queue_micros = queue_micros;
                                        result.execution_micros = execution_micros;
                                    }
                                }
                                WorkerResult::Long(result) => {
                                    result.queue_micros = queue_micros;
                                    result.execution_micros = execution_micros;
                                }
                            }
                            tracing::debug!(queue_micros, execution_micros, "Laya job completed");
                            Ok(response)
                        })
                        .and_then(|r| {
                            job.options.check()?;
                            if worker_state.stop.load(Ordering::Acquire) {
                                Err(DecisionError::Unloaded)
                            } else {
                                Ok(r)
                            }
                        });
                    *worker_state
                        .active
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = None;
                    let result = if worker_state.stop.load(Ordering::Acquire) {
                        Err(DecisionError::Unloaded)
                    } else {
                        result
                    };
                    let _ = job.reply.send(result);
                    if !worker_state.stop.load(Ordering::Acquire) {
                        worker_state.set(DecisionStatus::Ready);
                    }
                }
            }));
            drop(reservation);
            worker_state.stop.store(true, Ordering::Release);
            worker_state.set(DecisionStatus::Unloaded);
            for job in rx.try_iter() {
                let _ = job.reply.send(Err(DecisionError::Unloaded));
            }
            if result.is_err() {
                tracing::error!("decision worker panicked; session and reservation released");
            }
        })
        .map_err(|e| DecisionError::Execution(e.to_string()))?;
    let ready = ready_rx.recv().unwrap_or_else(|_| {
        Err(DecisionError::Execution(
            "decision worker failed during load".into(),
        ))
    });
    if let Err(error) = ready {
        let mut status = state.status.lock().unwrap_or_else(|e| e.into_inner());
        while *status != DecisionStatus::Unloaded {
            status = state.done.wait(status).unwrap_or_else(|e| e.into_inner());
        }
        return Err(error);
    }
    Ok(DecisionModel {
        inner: Arc::new(Inner {
            tx,
            state,
            capabilities: DecisionCapabilities {
                state_forms: ["text", "ordered_json", "conversation"]
                    .map(String::from)
                    .to_vec(),
                question_forms: ["choice", "score", "yes_no"].map(String::from).to_vec(),
                option_permutation: true,
                multi_state_batch: true,
                long_state_windows: true,
                calibration_import_versions: vec![2],
                envelope: bundle.manifest().envelope.clone(),
                execution: options.execution,
                queue_capacity: options.queue_capacity,
                max_input_bytes: options.max_input_bytes,
                intra_threads: options.intra_threads,
            },
            bundle,
            max_input_bytes: options.max_input_bytes,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    fn bundle() -> LayaBundle {
        LayaBundle::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke"),
        )
        .unwrap()
    }
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct Fake {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Fake {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl DecisionBackend for Fake {
        fn run_long(
            &mut self,
            request: DecisionRequest,
            _: LongStateOptions,
            options: DecisionOptions,
        ) -> Result<LongDecisionResult> {
            let windows = self.run_batch(&[request], &options)?;
            Ok(LongDecisionResult {
                answers: vec![],
                windows,
                state_tokens: 0,
                window_tokens: 0,
                stride_tokens: 0,
                queue_micros: 0,
                execution_micros: 0,
            })
        }
        fn run_batch(
            &mut self,
            requests: &[DecisionRequest],
            _: &DecisionOptions,
        ) -> Result<Vec<DecisionResult>> {
            self.started.send(()).unwrap();
            self.release.recv().unwrap();
            Ok(requests
                .iter()
                .map(|_| DecisionResult {
                    answers: vec![],
                    inputs: vec![],
                    manifest_sha256: String::new(),
                    checkpoint: Checkpoint::English,
                    calibration_diagnostics: vec![],
                    execution: ExecutionOptions::default(),
                    queue_micros: 0,
                    execution_micros: 0,
                })
                .collect())
        }
    }
    #[test]
    fn bounded_queue_unload_and_reservations_survive_inflight_work() {
        check_bounded_queue_unload(false);
    }
    #[test]
    fn long_scan_queue_unload_retains_resources_until_worker_returns() {
        check_bounded_queue_unload(true);
    }
    fn check_bounded_queue_unload(long: bool) {
        let (started, rx) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let drops = Arc::new(AtomicUsize::new(0));
        let resources = drops.clone();
        let backend_drops = drops.clone();
        let model = spawn_with(
            bundle(),
            LoadOptions {
                queue_capacity: 1,
                ..Default::default()
            },
            Box::new(Guard(resources)),
            move || {
                Ok(Box::new(Fake {
                    started,
                    release: wait,
                    drops: backend_drops,
                }))
            },
        )
        .unwrap();
        let clone = model.clone();
        let call = std::thread::spawn(move || {
            if long {
                clone
                    .decide_long(
                        DecisionRequest::text(""),
                        LongStateOptions::default(),
                        DecisionOptions::default(),
                    )
                    .map(|_| ())
            } else {
                clone
                    .decide(DecisionRequest::text(""), DecisionOptions::default())
                    .map(|_| ())
            }
        });
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let (reply, pending) = mpsc::channel();
        model
            .inner
            .tx
            .try_send(Job {
                requests: vec![DecisionRequest::text("queued")],
                scan: long.then(LongStateOptions::default),
                options: DecisionOptions::default(),
                reply,
                queued_at: Instant::now(),
            })
            .unwrap();
        assert!(matches!(
            model.decide(DecisionRequest::text("full"), DecisionOptions::default()),
            Err(DecisionError::Busy)
        ));
        assert!(matches!(
            model.decide_long(
                DecisionRequest::text("full"),
                LongStateOptions::default(),
                DecisionOptions::default()
            ),
            Err(DecisionError::Busy)
        ));
        model.unload();
        assert_eq!(model.status(), DecisionStatus::Stopping);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        release.send(()).unwrap();
        model.shutdown();
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert!(matches!(call.join().unwrap(), Err(DecisionError::Unloaded)));
        assert!(matches!(
            pending.recv_timeout(Duration::from_secs(2)).unwrap(),
            Err(DecisionError::Unloaded)
        ));
        assert_eq!(model.status(), DecisionStatus::Unloaded);
    }
    #[test]
    fn failed_load_rolls_back_before_return() {
        let drops = Arc::new(AtomicUsize::new(0));
        let result = spawn_with(
            bundle(),
            LoadOptions::default(),
            Box::new(Guard(drops.clone())),
            || Err(DecisionError::Execution("load failed".into())),
        );
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn long_scan_checks_input_and_graph_budgets_before_preprocessing() {
        let (started, rx) = mpsc::channel();
        let (_release, wait) = mpsc::channel();
        let model = spawn_with(
            bundle(),
            LoadOptions {
                max_input_bytes: 300,
                ..Default::default()
            },
            Box::new(()),
            move || {
                Ok(Box::new(Fake {
                    started,
                    release: wait,
                    drops: Arc::new(AtomicUsize::new(0)),
                }))
            },
        )
        .unwrap();
        assert!(matches!(
            model.decide_long(
                DecisionRequest::text("a".repeat(1000)).question("q", Question::yes_no("x")),
                LongStateOptions::default(),
                DecisionOptions::default()
            ),
            Err(DecisionError::ResourceLimit(_))
        ));
        assert!(matches!(
            model.decide_long(
                DecisionRequest::text("a").question("q", Question::yes_no("x")),
                LongStateOptions::default(),
                DecisionOptions {
                    encode: Some(EncodeOptions {
                        max_len: usize::MAX,
                        head_max_len: usize::MAX - 1,
                        overflow: OverflowPolicy::Reject
                    }),
                    ..Default::default()
                }
            ),
            Err(DecisionError::ResourceLimit(_))
        ));
        assert!(rx.try_recv().is_err());
        model.shutdown();
        assert!(matches!(
            model.decide_long(
                DecisionRequest::text(""),
                LongStateOptions::default(),
                DecisionOptions::default()
            ),
            Err(DecisionError::Unloaded)
        ));
    }
    #[test]
    fn batch_is_one_job_and_uses_an_aggregate_input_budget() {
        let (started, rx) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let model = spawn_with(
            bundle(),
            LoadOptions {
                max_input_bytes: 200,
                ..Default::default()
            },
            Box::new(()),
            move || {
                Ok(Box::new(Fake {
                    started,
                    release: wait,
                    drops: Arc::new(AtomicUsize::new(0)),
                }))
            },
        )
        .unwrap();
        assert!(matches!(
            model.decide_batch(
                vec![DecisionRequest::text("x".repeat(80)); 2],
                DecisionOptions::default()
            ),
            Err(DecisionError::ResourceLimit(_))
        ));
        assert!(rx.try_recv().is_err());
        let clone = model.clone();
        let call = std::thread::spawn(move || {
            clone.decide_batch(
                vec![DecisionRequest::text("a"), DecisionRequest::text("b")],
                DecisionOptions::default(),
            )
        });
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        release.send(()).unwrap();
        assert_eq!(call.join().unwrap().unwrap().len(), 2);
        assert!(
            rx.try_recv().is_err(),
            "the batch must enter the backend once"
        );
        model.shutdown();
    }
    #[test]
    fn cancellation_and_deadline_do_not_run_backend() {
        let (started, rx) = mpsc::channel();
        let (_release, wait) = mpsc::channel();
        let drops = Arc::new(AtomicUsize::new(0));
        let model = spawn_with(bundle(), LoadOptions::default(), Box::new(()), move || {
            Ok(Box::new(Fake {
                started,
                release: wait,
                drops,
            }))
        })
        .unwrap();
        let options = DecisionOptions::default();
        options.cancellation.cancel();
        assert!(matches!(
            model.decide(DecisionRequest::text(""), options),
            Err(DecisionError::Cancelled)
        ));
        assert!(matches!(
            model.decide(
                DecisionRequest::text(""),
                DecisionOptions {
                    deadline: Some(Instant::now()),
                    ..Default::default()
                }
            ),
            Err(DecisionError::DeadlineExceeded)
        ));
        assert!(rx.try_recv().is_err());
        model.shutdown();
    }

    #[cfg(feature = "tokio")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_async_call_cancels_without_releasing_native_buffers_early() {
        check_async_drop(false).await;
    }
    #[cfg(feature = "tokio")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_async_long_scan_retains_worker_resources() {
        check_async_drop(true).await;
    }
    #[cfg(feature = "tokio")]
    async fn check_async_drop(long: bool) {
        let (started, rx) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let drops = Arc::new(AtomicUsize::new(0));
        let backend_drops = drops.clone();
        let model = spawn_with(
            bundle(),
            LoadOptions::default(),
            Box::new(Guard(drops.clone())),
            move || {
                Ok(Box::new(Fake {
                    started,
                    release: wait,
                    drops: backend_drops,
                }))
            },
        )
        .unwrap();
        let options = DecisionOptions::default();
        let cancellation = options.cancellation.clone();
        let clone = model.clone();
        let task = tokio::spawn(async move {
            if long {
                clone
                    .decide_long_async(
                        DecisionRequest::text(""),
                        LongStateOptions::default(),
                        options,
                    )
                    .await
                    .map(|_| ())
            } else {
                clone
                    .decide_async(DecisionRequest::text(""), options)
                    .await
                    .map(|_| ())
            }
        });
        tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_secs(2)).unwrap())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
        assert!(cancellation.is_cancelled());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        release.send(()).unwrap();
        model.shutdown_async().await.unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}
