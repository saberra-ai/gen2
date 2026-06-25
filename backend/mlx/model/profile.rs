//! MEASUREMENT-ONLY profiling harness for the MLX decode hot path.
//!
//! Gated entirely behind the `PIO_MLX_PROFILE` env var: when unset, every
//! entry point here is a cheap boolean check that returns immediately, so it
//! is safe to leave the instrumentation compiled into the hot path.
//!
//! # Why this exists / how to use it
//!
//! MLX builds a *lazy* op graph — wrapping an op in `Instant::now()` measures
//! only graph construction, not GPU compute. To attribute time to a component
//! we must force evaluation at the component boundary with
//! [`mlx_rs::transforms::eval`]. [`prof_eval`] does exactly that: it evals the
//! given array(s) and accumulates the elapsed wall time under a label.
//!
//! Inserting eval barriers destroys cross-op pipelining and slightly inflates
//! the *total* per-token time, so the breakdown is only valid for RELATIVE
//! localization. The un-instrumented baseline (one eval per token at the
//! sampler) is reported separately.
//!
//! Nothing here changes compute logic — it only forces evaluation earlier than
//! it would otherwise happen and records timings.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

thread_local! {
    /// label -> (total_nanos, hit_count)
    static ACCUM: RefCell<HashMap<&'static str, (u128, u64)>> = RefCell::new(HashMap::new());
    /// Insertion order of labels, so the report prints in hot-path order.
    static ORDER: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// Runtime override: 0 = unset (fall back to env), 1 = forced off, 2 = forced on.
/// Lets a single-threaded test toggle profiling without spawning a thread
/// (MLX `Array` is `!Send`, so we can't run the model on another thread).
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

fn env_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("PIO_MLX_PROFILE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("off") && !v.is_empty())
            .unwrap_or(false)
    })
}

/// Force profiling on/off at runtime, overriding the env var. TEST-ONLY.
pub fn set_override(on: Option<bool>) {
    OVERRIDE.store(
        match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

/// True when `PIO_MLX_PROFILE` is truthy or a runtime override forces it on.
#[inline]
pub fn enabled() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => env_enabled(),
    }
}

fn record(label: &'static str, nanos: u128) {
    ORDER.with(|o| {
        let mut o = o.borrow_mut();
        if !o.contains(&label) {
            o.push(label);
        }
    });
    ACCUM.with(|a| {
        let mut a = a.borrow_mut();
        let entry = a.entry(label).or_insert((0, 0));
        entry.0 += nanos;
        entry.1 += 1;
    });
}

/// Force-eval `arrays` and accumulate the elapsed time under `label`.
///
/// No-op (returns immediately, does NOT eval) when profiling is disabled, so
/// the lazy graph is preserved on the normal path. When enabled, this both
/// times AND forces evaluation of the boundary — which is the whole point.
#[inline]
pub fn prof_eval(label: &'static str, arrays: &[&mlx_rs::Array]) {
    if !enabled() {
        return;
    }
    let t = Instant::now();
    let _ = mlx_rs::transforms::eval(arrays.iter().copied());
    record(label, t.elapsed().as_nanos());
}

/// Reset all accumulators (call once after warmup / before the timed window).
pub fn reset() {
    ACCUM.with(|a| a.borrow_mut().clear());
    ORDER.with(|o| o.borrow_mut().clear());
}

/// Snapshot: `(label, total_nanos, hits)` in hot-path insertion order.
pub fn snapshot() -> Vec<(&'static str, u128, u64)> {
    ORDER.with(|o| {
        ACCUM.with(|a| {
            let a = a.borrow();
            o.borrow()
                .iter()
                .filter_map(|l| a.get(l).map(|(n, h)| (*l, *n, *h)))
                .collect()
        })
    })
}
