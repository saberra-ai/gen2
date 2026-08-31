//! Streaming tool executor — concurrent execution with safety classification.
//!
//! Operations declare whether they are safe to run concurrently via
//! [`ConcurrencyGuard`]. The executor runs safe operations in parallel
//! (up to a configurable concurrency limit) and serializes unsafe ones,
//! guaranteeing that no unsafe operation overlaps with any other operation.
//!
//! Three concurrency tiers:
//! - **Safe** — parallel with everything except Exclusive.
//! - **GpuBound** — parallel with Safe, serialized against other GpuBound
//!   (prevents Metal/CUDA contention between embeddings and NER).
//! - **Exclusive** — runs alone, all other work drains first.
//!
//! Results are buffered and yielded in submission order regardless of
//! completion order, so callers see a deterministic stream.
//!
//! # Example
//!
//! ```ignore
//! let executor = StreamingToolExecutor::new(4);
//! executor.submit_fn(ConcurrencyGuard::Safe, || async { search_kg(query) });
//! executor.submit_fn(ConcurrencyGuard::GpuBound, || async { extract_entities(msg) });
//! executor.submit_fn(ConcurrencyGuard::Exclusive, || async { insert_triples(triples) });
//! executor.close();
//!
//! while let Some(result) = executor.next_result().await {
//!     handle(result);
//! }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{Mutex, Semaphore, mpsc};

// ── Concurrency classification ──────────────────────────────────────────

/// Whether an operation is safe to run concurrently with other operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyGuard {
    /// Pure reads, stateless transforms, or operations on disjoint resources.
    /// Multiple safe operations may execute simultaneously.
    Safe,
    /// GPU-bound work (embedding generation, ONNX NER inference).
    /// Concurrent with Safe ops, but serialized against other GpuBound ops
    /// to prevent Metal/CUDA contention with inference.
    GpuBound,
    /// Writes to shared mutable state (SQLite write lock, model load/unload,
    /// session cache mutation). The executor guarantees exclusive access:
    /// no other operation (safe or unsafe) runs while an exclusive one executes.
    Exclusive,
}

// ── Operation trait ─────────────────────────────────────────────────────

/// A single unit of work submitted to the executor.
///
/// Implementors declare their concurrency safety and provide an async
/// execution body. The executor uses `guard()` to decide scheduling.
pub trait Operation: Send + 'static {
    /// The result type produced by this operation.
    type Output: Send + 'static;

    /// Declare whether this operation is safe to run concurrently.
    fn guard(&self) -> ConcurrencyGuard;

    /// Execute the operation. Called exactly once by the executor.
    fn execute(self) -> Pin<Box<dyn Future<Output = Self::Output> + Send>>;
}

// ── Boxed operation (type-erased) ───────────────────────────────────────

type BoxedFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Type-erased wrapper so the executor can hold heterogeneous operations
/// with a common output type.
pub struct BoxedOperation<T: Send + 'static> {
    guard: ConcurrencyGuard,
    factory: Box<dyn FnOnce() -> BoxedFuture<T> + Send>,
}

impl<T: Send + 'static> BoxedOperation<T> {
    /// Create from any `Operation` with matching output type.
    pub fn new<O: Operation<Output = T>>(op: O) -> Self {
        let guard = op.guard();
        Self {
            guard,
            factory: Box::new(move || op.execute()),
        }
    }

    /// Create from a guard + async closure.
    pub fn from_fn<F, Fut>(guard: ConcurrencyGuard, f: F) -> Self
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        Self {
            guard,
            factory: Box::new(move || Box::pin(f())),
        }
    }

    /// Wrap this operation with a timeout. On timeout, `on_timeout` produces
    /// the result value. The executor stays generic — callers decide what a
    /// timeout looks like in their result type.
    ///
    /// ```ignore
    /// BoxedOperation::from_fn(ConcurrencyGuard::GpuBound, || async { embed(text) })
    ///     .with_timeout(Duration::from_secs(30), || EnrichResult::TimedOut)
    /// ```
    pub fn with_timeout<F>(self, duration: Duration, on_timeout: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
    {
        let guard = self.guard;
        let inner_factory = self.factory;
        Self {
            guard,
            factory: Box::new(move || {
                let fut = inner_factory();
                Box::pin(async move {
                    match tokio::time::timeout(duration, fut).await {
                        Ok(val) => val,
                        Err(_) => on_timeout(),
                    }
                })
            }),
        }
    }
}

// ── Executor ────────────────────────────────────────────────────────────

/// Internal envelope: pairs a sequence number with its operation.
struct Envelope<T: Send + 'static> {
    seq: usize,
    op: BoxedOperation<T>,
}

/// Streaming tool executor with safety-classified concurrency.
///
/// - Safe operations run in parallel (up to `max_concurrent`).
/// - GpuBound operations run concurrently with Safe but serialize against
///   each other (one GPU op at a time).
/// - Exclusive operations run alone — the executor drains all in-flight
///   operations before starting an exclusive one.
/// - Results are yielded in submission order via [`next_result`].
pub struct StreamingToolExecutor<T: Send + 'static> {
    /// Submit channel. Wrapped in std Mutex so `close()` can drop the sender
    /// to signal the executor loop. The lock is never held across an await
    /// point, so std::sync::Mutex is correct and avoids async overhead.
    submit_tx: std::sync::Mutex<Option<mpsc::UnboundedSender<Envelope<T>>>>,
    /// Cooperative cancellation flag.
    cancelled: Arc<AtomicBool>,
    /// Receive ordered results here.
    result_rx: Mutex<mpsc::UnboundedReceiver<T>>,
    /// Monotonic submission counter.
    next_seq: std::sync::atomic::AtomicUsize,
    /// Set when the executor loop has exited.
    done: Arc<AtomicBool>,
}

impl<T: Send + 'static> StreamingToolExecutor<T> {
    /// Create a new executor.
    ///
    /// `max_concurrent` caps how many safe operations run simultaneously.
    /// A value of 0 is treated as 1.
    pub fn new(max_concurrent: usize) -> Self {
        let max = max_concurrent.max(1);
        let (submit_tx, submit_rx) = mpsc::unbounded_channel::<Envelope<T>>();
        let (result_tx, result_rx) = mpsc::unbounded_channel::<T>();
        let done = Arc::new(AtomicBool::new(false));
        let cancelled = Arc::new(AtomicBool::new(false));

        let done_flag = done.clone();
        let cancel_flag = cancelled.clone();
        crate::task_util::spawn_logged("executor-loop", async move {
            run_executor_loop(submit_rx, result_tx, max, cancel_flag).await;
            done_flag.store(true, Ordering::Release);
        });

        Self {
            submit_tx: std::sync::Mutex::new(Some(submit_tx)),
            cancelled,
            result_rx: Mutex::new(result_rx),
            next_seq: std::sync::atomic::AtomicUsize::new(0),
            done,
        }
    }

    /// Submit an operation for execution.
    ///
    /// Returns `false` if the executor has been closed or cancelled.
    /// Synchronous — no async overhead on the submit path.
    pub fn submit(&self, op: BoxedOperation<T>) -> bool {
        if self.cancelled.load(Ordering::Acquire) {
            return false;
        }
        let guard = self.submit_tx.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(tx) => {
                let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
                tx.send(Envelope { seq, op }).is_ok()
            }
            None => false,
        }
    }

    /// Submit an operation built from a guard and an async closure.
    pub fn submit_fn<F, Fut>(&self, guard: ConcurrencyGuard, f: F) -> bool
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        self.submit(BoxedOperation::from_fn(guard, f))
    }

    /// Close the submission side. No more operations can be submitted.
    /// The executor will drain remaining work and then `next_result`
    /// will return `None`.
    pub fn close(&self) {
        let mut guard = self.submit_tx.lock().unwrap_or_else(|e| e.into_inner());
        *guard = None; // drops the sender, closing the channel
    }

    /// Cancel all pending work. In-flight operations complete naturally
    /// (cooperative, not preemptive), but no new operations will start.
    /// `next_result` will return `None` once in-flight work finishes.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.close();
    }

    /// Receive the next result in submission order.
    ///
    /// Returns `None` when all submitted operations have completed and
    /// the executor has been closed (or dropped).
    pub async fn next_result(&self) -> Option<T> {
        let mut rx = self.result_rx.lock().await;
        rx.recv().await
    }

    /// Check if the executor loop has finished.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

// ── Executor loop ───────────────────────────────────────────────────────

/// Flush sequential results from `buffer[cursor..]` into `result_tx`.
/// Returns `false` if the result channel is closed (receiver dropped).
fn flush_buffer<T>(
    buffer: &mut [Option<T>],
    cursor: &mut usize,
    result_tx: &mpsc::UnboundedSender<T>,
) -> bool {
    while *cursor < buffer.len() {
        if let Some(val) = buffer[*cursor].take() {
            if result_tx.send(val).is_err() {
                return false;
            }
            *cursor += 1;
        } else {
            break; // gap — wait for this seq to complete
        }
    }
    true
}

/// Spawn a Safe or GpuBound operation. Acquires the appropriate permits,
/// runs the future, and sends the result (or `None` if cancelled) back
/// through `done_tx`.
fn spawn_concurrent<T: Send + 'static>(
    guard: ConcurrencyGuard,
    seq: usize,
    factory: Box<dyn FnOnce() -> BoxedFuture<T> + Send>,
    semaphore: &Arc<Semaphore>,
    gpu_semaphore: &Arc<Semaphore>,
    done_tx: &mpsc::UnboundedSender<(usize, Option<T>)>,
    cancelled: &Arc<AtomicBool>,
) {
    let sem = semaphore.clone();
    let gpu_sem = gpu_semaphore.clone();
    let tx = done_tx.clone();
    let cancel = cancelled.clone();
    let needs_gpu = guard == ConcurrencyGuard::GpuBound;

    tokio::spawn(async move {
        let permit = sem
            .acquire_owned()
            .await
            .expect("executor semaphore closed");
        let gpu_permit = if needs_gpu {
            Some(gpu_sem.acquire_owned().await.expect("GPU semaphore closed"))
        } else {
            None
        };

        let result = if !cancel.load(Ordering::Acquire) {
            Some(factory().await)
        } else {
            None
        };

        let _ = tx.send((seq, result));
        drop(gpu_permit);
        drop(permit);
    });
}

/// The core scheduling loop. Runs on a spawned task.
///
/// Architecture: submissions arrive via `submit_rx`, spawned tasks report
/// completion via an internal `done_tx/done_rx` channel. The loop holds the
/// result buffer as a local `Vec` (no Arc, no Mutex) and flushes sequential
/// results to `result_tx` after each completion.
///
/// Exclusive operations drain all in-flight work inline (reading from
/// `done_rx` until `in_flight == 0`), then run the operation directly on
/// the executor task.
async fn run_executor_loop<T: Send + 'static>(
    mut submit_rx: mpsc::UnboundedReceiver<Envelope<T>>,
    result_tx: mpsc::UnboundedSender<T>,
    max_concurrent: usize,
    cancelled: Arc<AtomicBool>,
) {
    let semaphore = Arc::new(Semaphore::new(max_concurrent));
    let gpu_semaphore = Arc::new(Semaphore::new(1));

    // Spawned tasks send (seq, Some(output)) on success or (seq, None) on cancel.
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<(usize, Option<T>)>();

    let mut buffer: Vec<Option<T>> = Vec::new();
    let mut cursor: usize = 0;
    let mut in_flight: usize = 0;

    loop {
        tokio::select! {
            biased; // prefer completions — flush results before accepting new work

            Some((seq, result)) = done_rx.recv() => {
                in_flight -= 1;
                if let Some(output) = result {
                    buffer[seq] = Some(output);
                }
                if !flush_buffer(&mut buffer, &mut cursor, &result_tx) {
                    return; // receiver dropped
                }
            }

            recv = submit_rx.recv() => {
                let Some(envelope) = recv else { break }; // channel closed
                if cancelled.load(Ordering::Acquire) { break; }

                // Reserve slot in buffer.
                while buffer.len() <= envelope.seq {
                    buffer.push(None);
                }

                match envelope.op.guard {
                    ConcurrencyGuard::Safe | ConcurrencyGuard::GpuBound => {
                        in_flight += 1;
                        spawn_concurrent(
                            envelope.op.guard,
                            envelope.seq,
                            envelope.op.factory,
                            &semaphore,
                            &gpu_semaphore,
                            &done_tx,
                            &cancelled,
                        );
                    }
                    ConcurrencyGuard::Exclusive => {
                        // Drain all in-flight operations by reading completions.
                        while in_flight > 0 {
                            if let Some((seq, result)) = done_rx.recv().await {
                                in_flight -= 1;
                                if let Some(output) = result {
                                    buffer[seq] = Some(output);
                                }
                                flush_buffer(&mut buffer, &mut cursor, &result_tx);
                            } else {
                                break; // all senders dropped
                            }
                        }

                        // Run exclusive operation inline on this task.
                        let output = (envelope.op.factory)().await;
                        buffer[envelope.seq] = Some(output);
                        if !flush_buffer(&mut buffer, &mut cursor, &result_tx) {
                            return;
                        }
                    }
                }
            }
        }
    }

    // Submit channel closed or cancelled — drain remaining in-flight.
    drop(done_tx); // drop our sender; spawned tasks still hold clones
    while in_flight > 0 {
        if let Some((seq, result)) = done_rx.recv().await {
            in_flight -= 1;
            if let Some(output) = result {
                buffer[seq] = Some(output);
            }
            flush_buffer(&mut buffer, &mut cursor, &result_tx);
        } else {
            break; // all senders dropped (cancelled tasks exited)
        }
    }

    // Final flush.
    flush_buffer(&mut buffer, &mut cursor, &result_tx);
    // result_tx drops here, closing the result channel.
}

// ── Convenience: FnOperation ────────────────────────────────────────────

/// A simple operation built from a closure. Useful for one-off tasks
/// without defining a full struct.
pub struct FnOperation<T: Send + 'static> {
    guard: ConcurrencyGuard,
    f: Option<Box<dyn FnOnce() -> BoxedFuture<T> + Send>>,
}

impl<T: Send + 'static> FnOperation<T> {
    pub fn new<F, Fut>(guard: ConcurrencyGuard, f: F) -> Self
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        Self {
            guard,
            f: Some(Box::new(move || Box::pin(f()))),
        }
    }
}

impl<T: Send + 'static> Operation for FnOperation<T> {
    type Output = T;

    fn guard(&self) -> ConcurrencyGuard {
        self.guard
    }

    fn execute(mut self) -> Pin<Box<dyn Future<Output = T> + Send>> {
        (self.f.take().expect("execute called twice"))()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    /// Close and collect all results.
    async fn collect_all<T: Send + 'static>(executor: &StreamingToolExecutor<T>) -> Vec<T> {
        executor.close();
        let mut results = Vec::new();
        while let Some(r) = executor.next_result().await {
            results.push(r);
        }
        results
    }

    #[tokio::test]
    async fn safe_operations_run_concurrently() {
        let executor = StreamingToolExecutor::<u32>::new(4);
        let started = Arc::new(AtomicU32::new(0));
        let max_concurrent = Arc::new(AtomicU32::new(0));

        for i in 0..4u32 {
            let s = started.clone();
            let m = max_concurrent.clone();
            executor.submit_fn(ConcurrencyGuard::Safe, move || async move {
                let current = s.fetch_add(1, Ordering::SeqCst) + 1;
                m.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                s.fetch_sub(1, Ordering::SeqCst);
                i
            });
        }

        let results = collect_all(&executor).await;
        assert_eq!(results, vec![0, 1, 2, 3]);
        assert!(
            max_concurrent.load(Ordering::SeqCst) >= 2,
            "expected concurrency >= 2, got {}",
            max_concurrent.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn exclusive_operations_serialize() {
        let executor = StreamingToolExecutor::<String>::new(4);
        let active = Arc::new(AtomicU32::new(0));

        for i in 0..3 {
            let a = active.clone();
            executor.submit_fn(ConcurrencyGuard::Exclusive, move || async move {
                let concurrent = a.fetch_add(1, Ordering::SeqCst);
                assert_eq!(concurrent, 0, "exclusive op {i} ran concurrently!");
                tokio::time::sleep(Duration::from_millis(20)).await;
                a.fetch_sub(1, Ordering::SeqCst);
                format!("exclusive-{i}")
            });
        }

        let results = collect_all(&executor).await;
        assert_eq!(results, vec!["exclusive-0", "exclusive-1", "exclusive-2"]);
    }

    #[tokio::test]
    async fn mixed_safe_and_exclusive_ordering() {
        let executor = StreamingToolExecutor::<usize>::new(4);

        executor.submit_fn(ConcurrencyGuard::Safe, || async { 0 });
        executor.submit_fn(ConcurrencyGuard::Safe, || async { 1 });
        executor.submit_fn(ConcurrencyGuard::Exclusive, || async { 2 });
        executor.submit_fn(ConcurrencyGuard::Safe, || async { 3 });
        executor.submit_fn(ConcurrencyGuard::Safe, || async { 4 });

        let results = collect_all(&executor).await;
        assert_eq!(results, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn exclusive_waits_for_in_flight_safe() {
        let executor = StreamingToolExecutor::<&str>::new(4);
        let safe_finished = Arc::new(AtomicBool::new(false));

        let sf = safe_finished.clone();
        executor.submit_fn(ConcurrencyGuard::Safe, move || async move {
            tokio::time::sleep(Duration::from_millis(80)).await;
            sf.store(true, Ordering::SeqCst);
            "safe"
        });

        let sf2 = safe_finished.clone();
        executor.submit_fn(ConcurrencyGuard::Exclusive, move || async move {
            assert!(
                sf2.load(Ordering::SeqCst),
                "exclusive started before safe finished"
            );
            "exclusive"
        });

        let results = collect_all(&executor).await;
        assert_eq!(results, vec!["safe", "exclusive"]);
    }

    #[tokio::test]
    async fn respects_max_concurrent() {
        let executor = StreamingToolExecutor::<u32>::new(2);
        let peak = Arc::new(AtomicU32::new(0));
        let active = Arc::new(AtomicU32::new(0));

        for i in 0..6u32 {
            let p = peak.clone();
            let a = active.clone();
            executor.submit_fn(ConcurrencyGuard::Safe, move || async move {
                let current = a.fetch_add(1, Ordering::SeqCst) + 1;
                p.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                a.fetch_sub(1, Ordering::SeqCst);
                i
            });
        }

        let results = collect_all(&executor).await;
        assert_eq!(results, vec![0, 1, 2, 3, 4, 5]);
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "peak concurrency exceeded max_concurrent=2: {}",
            peak.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn fn_operation_trait() {
        let op = FnOperation::new(ConcurrencyGuard::Safe, || async { 42u32 });
        assert_eq!(op.guard(), ConcurrencyGuard::Safe);
        let result = op.execute().await;
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn empty_executor() {
        let executor = StreamingToolExecutor::<()>::new(4);
        executor.close();
        let result = executor.next_result().await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn submit_after_close_returns_false() {
        let executor = StreamingToolExecutor::<u32>::new(2);
        executor.close();
        let accepted = executor.submit_fn(ConcurrencyGuard::Safe, || async { 99 });
        assert!(!accepted);
    }

    #[tokio::test]
    async fn boxed_operation_from_trait() {
        let executor = StreamingToolExecutor::<u32>::new(2);
        let op = FnOperation::new(ConcurrencyGuard::Safe, || async { 7u32 });
        executor.submit(BoxedOperation::new(op));
        let results = collect_all(&executor).await;
        assert_eq!(results, vec![7]);
    }

    #[tokio::test]
    async fn drain_race_does_not_hang() {
        let executor = StreamingToolExecutor::<&str>::new(4);

        executor.submit_fn(ConcurrencyGuard::Safe, || async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            "safe"
        });
        executor.submit_fn(ConcurrencyGuard::Exclusive, || async { "exclusive" });

        let results = collect_all(&executor).await;
        assert_eq!(results, vec!["safe", "exclusive"]);
    }

    #[tokio::test]
    async fn cancel_stops_new_ops() {
        let executor = StreamingToolExecutor::<u32>::new(2);

        executor.submit_fn(ConcurrencyGuard::Safe, || async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        executor.cancel();

        let accepted = executor.submit_fn(ConcurrencyGuard::Safe, || async { 99 });
        assert!(!accepted);

        let mut results = Vec::new();
        while let Some(r) = executor.next_result().await {
            results.push(r);
        }
        assert!(results.len() <= 1);
    }

    #[tokio::test]
    async fn cancel_does_not_hang_with_in_flight() {
        // Regression: cancelled spawned tasks must not leave the executor
        // waiting for results that will never arrive.
        let executor = StreamingToolExecutor::<u32>::new(4);

        for i in 0..4u32 {
            executor.submit_fn(ConcurrencyGuard::Safe, move || async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                i
            });
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
        executor.cancel();

        // Must not hang — all in-flight ops complete and report back.
        let mut results = Vec::new();
        while let Some(r) = executor.next_result().await {
            results.push(r);
        }
        assert!(results.len() <= 4);
    }

    #[tokio::test]
    async fn timeout_via_with_timeout() {
        let executor = StreamingToolExecutor::<String>::new(2);

        let op = BoxedOperation::from_fn(ConcurrencyGuard::Safe, || async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            "never".to_string()
        })
        .with_timeout(Duration::from_millis(50), || "timed_out".to_string());

        executor.submit(op);
        let results = collect_all(&executor).await;
        assert_eq!(results, vec!["timed_out"]);
    }

    #[tokio::test]
    async fn timeout_does_not_fire_on_fast_op() {
        let executor = StreamingToolExecutor::<&str>::new(2);

        let op = BoxedOperation::from_fn(ConcurrencyGuard::Safe, || async { "fast" })
            .with_timeout(Duration::from_secs(10), || "timed_out");

        executor.submit(op);
        let results = collect_all(&executor).await;
        assert_eq!(results, vec!["fast"]);
    }

    #[tokio::test]
    async fn submit_is_sync() {
        let executor = StreamingToolExecutor::<u32>::new(2);
        let _: bool = executor.submit_fn(ConcurrencyGuard::Safe, || async { 1 });
        let _: bool = executor.submit(BoxedOperation::from_fn(ConcurrencyGuard::Safe, || async {
            2
        }));
        executor.close();
        while executor.next_result().await.is_some() {}
    }

    #[tokio::test]
    async fn gpu_bound_serializes_against_gpu_bound() {
        let executor = StreamingToolExecutor::<u32>::new(4);
        let active = Arc::new(AtomicU32::new(0));
        let peak_gpu = Arc::new(AtomicU32::new(0));

        for i in 0..4u32 {
            let a = active.clone();
            let p = peak_gpu.clone();
            executor.submit_fn(ConcurrencyGuard::GpuBound, move || async move {
                let current = a.fetch_add(1, Ordering::SeqCst) + 1;
                p.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                a.fetch_sub(1, Ordering::SeqCst);
                i
            });
        }

        let results = collect_all(&executor).await;
        assert_eq!(results, vec![0, 1, 2, 3]);
        assert_eq!(
            peak_gpu.load(Ordering::SeqCst),
            1,
            "expected peak GPU concurrency = 1, got {}",
            peak_gpu.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn gpu_bound_concurrent_with_safe() {
        let executor = StreamingToolExecutor::<&str>::new(4);
        let active = Arc::new(AtomicU32::new(0));
        let peak = Arc::new(AtomicU32::new(0));

        let a1 = active.clone();
        let p1 = peak.clone();
        executor.submit_fn(ConcurrencyGuard::Safe, move || async move {
            let current = a1.fetch_add(1, Ordering::SeqCst) + 1;
            p1.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(60)).await;
            a1.fetch_sub(1, Ordering::SeqCst);
            "safe"
        });

        let a2 = active.clone();
        let p2 = peak.clone();
        executor.submit_fn(ConcurrencyGuard::GpuBound, move || async move {
            let current = a2.fetch_add(1, Ordering::SeqCst) + 1;
            p2.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(60)).await;
            a2.fetch_sub(1, Ordering::SeqCst);
            "gpu"
        });

        let results = collect_all(&executor).await;
        assert_eq!(results, vec!["safe", "gpu"]);
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "expected Safe + GpuBound to overlap, peak was {}",
            peak.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn gpu_bound_blocked_by_exclusive() {
        let executor = StreamingToolExecutor::<&str>::new(4);
        let exclusive_done = Arc::new(AtomicBool::new(false));

        let ed = exclusive_done.clone();
        executor.submit_fn(ConcurrencyGuard::Exclusive, move || async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            ed.store(true, Ordering::SeqCst);
            "exclusive"
        });

        let ed2 = exclusive_done.clone();
        executor.submit_fn(ConcurrencyGuard::GpuBound, move || async move {
            assert!(
                ed2.load(Ordering::SeqCst),
                "GpuBound started before Exclusive finished"
            );
            "gpu"
        });

        let results = collect_all(&executor).await;
        assert_eq!(results, vec!["exclusive", "gpu"]);
    }
}
