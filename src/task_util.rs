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
