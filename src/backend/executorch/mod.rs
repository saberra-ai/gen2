//! ExecuTorch backend — Phase D week 14 (scaffold).
//!
//! Mobile inference path. Meta's ExecuTorch runs exported PyTorch
//! models (`.pte`) with platform-specific delegates:
//! - **iOS**: CoreML delegate (Neural Engine + GPU) via the ExecuTorch
//!   CoreML backend
//! - **Android**: NNAPI + XNNPACK delegates
//! - **Desktop fallback**: XNNPACK (CPU)
//!
//! # Status — scaffold
//!
//! This module is a *scaffold*. It declares the backend module layout,
//! feature-gates it behind `backend-executorch`, and implements the
//! Backend trait surface with `ExecError::Unimplemented` placeholders so
//! the rest of the codebase can reference it without breaking. Real
//! inference is a multi-week milestone with its own work branch.
//!
//! What the scaffold reserves:
//! - Module tree shape (`bundle`, `engine`, `loader`, `puller`, `session`)
//!   matching the existing `llama` / `mlx` / `onnx` backends so future
//!   work plugs in with the same file layout the rest of gen2 already
//!   understands.
//! - Backend name `"executorch"` wired through the zoo (`zoo.json`
//!   points `ios` + `android` entries here).
//! - Trait impls that return Unimplemented so `gen2::router` can
//!   still *route* a request to this backend and fall back when the
//!   attempted load fails.
//!
//! # Real-impl roadmap
//!
//! 1. Add `executorch-rs` (or custom FFI to libexecutorch) as an
//!    optional dep gated on `backend-executorch`.
//! 2. Implement `loader::load_pte` → `bundle::ExecutorchModelBundle`.
//! 3. Wire `session::ExecutorchSession::pull` → streaming token generator
//!    via ExecuTorch's `Method::execute` loop.
//! 4. Tokenizer integration via the existing `tokenizers` crate.
//! 5. Delegate probe: try CoreML on iOS, NNAPI on Android, fall back to
//!    XNNPACK. Report the active delegate via `capabilities()`.
//! 6. KV cache interop: ExecuTorch's static-shape models mean KV is
//!    baked into the model graph; session append pattern differs from
//!    llama.cpp's incremental KV. Plumb via a new `StaticKv` capability.
//!
//! # Why not just use llama.cpp on mobile?
//!
//! llama.cpp works on iOS/Android but doesn't tap the Neural Engine.
//! ExecuTorch gives us ~3-5× speedup for the same quant on recent
//! Apple Silicon / modern Android SoCs. For the phone-runs-a-real-model
//! story (gemma-4-E2B on iPhone 15), ExecuTorch is the only acceptable
//! path. llama.cpp stays as the desktop+server fallback.

use std::sync::Arc;

use crate::backend::caps::LatencyTier;
use crate::backend::traits::{Backend, BackendSession, LocalBackend};
use crate::backend::{BackendCaps, SessionId};
use crate::engine::{Capabilities, ExecError, ExecutionStats, HookBus, LoadRequest, Settings};
use crate::session_rt::SessionSpec;

/// Scaffold Backend impl. Every method returns `ExecError::Unimplemented`
/// until the real integration lands. Kept visible so the router can
/// reference the backend name in zoo entries.
#[derive(Debug, Default)]
pub struct ExecutorchBackend {
    settings: Arc<Settings>,
    hooks: Arc<HookBus>,
}

impl ExecutorchBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Backend for ExecutorchBackend {
    fn backend_name(&self) -> &'static str {
        "executorch"
    }

    fn load_model(&self, _req: LoadRequest) -> Result<(), ExecError> {
        Err(ExecError::Unimplemented)
    }

    fn reload_model(&self) -> Result<(), ExecError> {
        Err(ExecError::Unimplemented)
    }

    fn unload_model(&self) {}

    fn is_model_loaded(&self) -> bool {
        false
    }

    fn upload_settings(&self, _settings: Settings) -> Result<(), ExecError> {
        Err(ExecError::Unimplemented)
    }

    fn settings(&self) -> Arc<Settings> {
        Arc::clone(&self.settings)
    }

    fn settings_version(&self) -> u64 {
        0
    }

    fn hooks(&self) -> Arc<HookBus> {
        Arc::clone(&self.hooks)
    }

    fn capabilities(&self) -> Capabilities {
        // Most conservative caps until real integration — we don't
        // overclaim features the scaffold can't deliver.
        Capabilities::default()
    }

    fn stats(&self) -> ExecutionStats {
        ExecutionStats::default()
    }

    fn first_token_tier(&self) -> LatencyTier {
        LatencyTier::Slow
    }

    fn start_session(&self, _spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        Err(ExecError::Unimplemented)
    }

    fn end_session(&self, _id: SessionId) -> Result<(), ExecError> {
        Err(ExecError::Unimplemented)
    }
}

impl LocalBackend for ExecutorchBackend {
    fn n_ctx(&self) -> usize {
        0
    }
}
