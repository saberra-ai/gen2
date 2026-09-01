//! Panic-safe task spawning for internal background work.
//!
//! Vendored from `pio-core::tasks::spawn` during the gen2 crate split — the
//! executor loop needs it and it carries no host coupling. All long-lived
//! `tokio::spawn` calls go through [`spawn_logged`] so a panic in a background
//! task is caught and logged rather than silently swallowed. The returned
//! [`JoinHandle`] should be stored by the caller when graceful shutdown
//! matters.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::OnceLock;

use futures::FutureExt; // for .catch_unwind()
use tokio::runtime::{Builder, Runtime};
use tokio::task::JoinHandle;

/// Runtime used when `spawn_logged` is called from outside a Tokio context
/// (a synchronous host thread, or a unit test with no runtime).
fn fallback_runtime() -> &'static Runtime {
    static FALLBACK_RUNTIME: OnceLock<Runtime> = OnceLock::new();

    FALLBACK_RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("pio-bg")
            .enable_all()
            .build()
            .expect("fallback Tokio runtime should initialize")
    })
}

/// Run a future to completion from synchronous code.
///
/// Tools are async but the chat path is not, so the agent loop has to bridge.
/// Doing that with a bare `block_on` panics when the caller is already inside a
/// runtime — a real case, since a consumer may drive the sync API from a Tokio
/// thread. So work is always handed to the dedicated runtime and awaited over a
/// std channel: correct on a plain thread, and correct inside someone else's
/// runtime, at the cost of one hop.
pub(crate) fn block_on<T, F>(fut: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    fallback_runtime().spawn(async move {
        let _ = tx.send(fut.await);
    });
    rx.recv()
        .expect("the agent runtime dropped a task without returning a result")
}

/// Spawn a future on the Tokio runtime with panic logging.
///
/// If the future panics, the panic is caught and logged at `error!` level with
/// the task `name` for identification. The [`JoinHandle`] is returned so
/// callers can optionally `.await` it during shutdown.
pub fn spawn_logged(
    name: &'static str,
    fut: impl Future<Output = ()> + Send + 'static,
) -> JoinHandle<()> {
    let task = async move {
        let result = AssertUnwindSafe(fut).catch_unwind().await;
        if let Err(panic_payload) = result {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_string()
            };
            tracing::error!(task = name, panic = %msg, "background task panicked");
        }
    };

    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.spawn(task),
        Err(_) => fallback_runtime().spawn(task),
    }
}
