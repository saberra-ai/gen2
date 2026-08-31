//! MLX inference backend for Apple Silicon (macOS/iOS).

mod bundle;
pub(crate) mod eagle3_loader;
pub(crate) mod eagle_predictor;
mod engine;
#[cfg(test)]
mod golden;
#[cfg(test)]
mod kv_donation_probe;
mod loader;
pub(crate) mod model;
// ngram module replaced by cross-backend `common::speculative::*` —
// see that module for the trigram impl plus PLD / Lookahead / Eagle3
// alternatives. Kept as an alias for backward-compat in external code.
#[cfg(test)]
mod profile_decode;
mod puller;
mod sampler;
mod session;
mod tokenizer;
#[cfg(test)]
mod vision_parity;

pub use bundle::ModelBundle;
pub use engine::Engine;

use crate::engine::ExecError;

/// Extract a human-readable message from a caught panic payload.
///
/// The MLX FFI (`mlx-rs`) panics by `unwrap()`ing an internal `Exception`, so a
/// caught panic carries either a `&str` or a `String` with the underlying
/// `[metal::Device] …` / MLX message.
pub(crate) fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "MLX forward pass panicked".to_string()
    }
}

/// Classify a caught MLX forward-pass panic into the right [`ExecError`].
///
/// A caught panic from an MLX forward pass is NOT always OOM — the FFI panics
/// for several distinct failure classes and collapsing them all into
/// `OutOfMemory` misleads the user (e.g. telling them to free memory when the
/// real problem is a broken build). The important split:
///
/// - **Metal kernel library unavailable/broken** — an empty or
///   AIR-version-mismatched `mlx.metallib`, surfaced as
///   `"[metal::Device] Unable to load kernel …"`,
///   `"Unable to build metal library from source"`, or
///   `"Failed to load the default metallib"`. This is a build/toolchain
///   defect, not a runtime resource shortage. It's unrecoverable per-request,
///   so we surface it as a poisoned session with an honest, non-OOM message.
/// - **Everything else** — true allocation failures and unknown panics keep the
///   historical `OutOfMemory` mapping.
pub(crate) fn classify_forward_panic(e: Box<dyn std::any::Any + Send>) -> ExecError {
    let msg = panic_msg(e);
    let lower = msg.to_lowercase();
    let metal_backend_broken = lower.contains("unable to load kernel")
        || lower.contains("unable to build metal library")
        || lower.contains("failed to load the default metallib")
        || lower.contains("metal::device");
    if metal_backend_broken {
        ExecError::SessionPoisoned(format!(
            "MLX Metal kernel library unavailable — the backend could not load a \
             compute kernel. This is a build/toolchain issue (empty or \
             AIR-version-mismatched metallib), not out of memory. Detail: {msg}"
        ))
    } else {
        ExecError::OutOfMemory(msg)
    }
}

// Re-exported for the in-crate test modules (`golden.rs`, `vision_parity.rs`),
// which drive sessions/pullers directly. Gated so the non-test build (where no
// caller references them) stays warning-free.
#[cfg(test)]
pub use puller::TokenPuller;
#[cfg(test)]
pub use session::Session;

#[cfg(test)]
mod panic_classify_tests {
    use super::*;

    fn boxed(s: &str) -> Box<dyn std::any::Any + Send> {
        Box::new(s.to_string())
    }

    #[test]
    fn metal_kernel_load_failure_is_not_oom() {
        // The exact panic seen when the metallib is empty (AIR version mismatch
        // drops every kernel): a missing quant dequant kernel.
        let e =
            boxed("[metal::Device] Unable to load kernel affine_dequantize_float16_t_gs_64_b_8");
        match classify_forward_panic(e) {
            ExecError::SessionPoisoned(m) => {
                let lower = m.to_lowercase();
                assert!(lower.contains("not out of memory"), "msg: {m}");
                assert!(lower.contains("metal"), "msg: {m}");
            }
            other => panic!("expected SessionPoisoned, got {other:?}"),
        }
    }

    #[test]
    fn metal_jit_build_failure_is_not_oom() {
        let e = boxed(
            "[metal::Device] Unable to build metal library from source\nerror: invalid value 'metal4.0'",
        );
        assert!(matches!(
            classify_forward_panic(e),
            ExecError::SessionPoisoned(_)
        ));
    }

    #[test]
    fn genuine_oom_stays_oom() {
        let e = boxed("[METAL] Command buffer execution failed: Insufficient Memory");
        assert!(matches!(
            classify_forward_panic(e),
            ExecError::OutOfMemory(_)
        ));
    }

    #[test]
    fn unknown_panic_defaults_to_oom() {
        let e = boxed("some other panic");
        assert!(matches!(
            classify_forward_panic(e),
            ExecError::OutOfMemory(_)
        ));
    }
}
