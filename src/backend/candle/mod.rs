//! Candle backend — Phase D week 14.
//!
//! HuggingFace's Rust-native inference library. Used as:
//! - **Linux CPU fallback** when `backend-llamacpp`'s C++ toolchain
//!   isn't viable (Alpine / minimal container images, no build-essential)
//! - **Portable server distro** — single static Rust binary
//!
//! Model zoo routes `linux_cpu` / gemma-4 safetensors through this
//! backend when the `backend-candle` feature is enabled.
//!
//! # What's wired today
//!
//! - Real dependency on `candle-core`, `candle-nn`, `candle-transformers`
//!   (feature-gated, so default builds are unaffected)
//! - Device probe: CUDA → Metal → CPU with graceful fallback
//! - Model directory validation (config.json, tokenizer.json, safetensors
//!   shards discovered, mmap-loaded into a `VarBuilder`)
//! - Backend + LocalBackend trait surface implemented, state plumbed,
//!   `load_model` does real work, subsequent `start_session` returns
//!   `Unimplemented` with a clear message so the router's fallback path
//!   triggers cleanly.
//!
//! # What's deferred (1–2 week follow-up milestone)
//!
//! The actual streaming generation loop needs deeper integration with
//! `gen2`'s `SessionSpec` (chat-template rendering from
//! `SessionSpec.messages`), `TokenEvent::Token { id, text, logprob }`
//! construction, pause/resume cancellation tokens, and Candle's
//! model-scoped KV state (`Model::forward(&mut self, ..., seqlen_offset)`).
//!
//! The scaffolding below loads weights and holds them ready; the
//! remaining work is:
//! 1. Implement chat-template rendering for Gemma (reuse the existing
//!    minijinja template path the llama backend uses).
//! 2. Wire `Model::forward` in a loop with `LogitsProcessor::sample`.
//! 3. Emit `TokenEvent::Token { id, text, logprob: None }` per step.
//! 4. Respect `spec.stop` / `spec.max_tokens` / `spec.temperature`.
//! 5. Handle the cancellation token for pause/resume.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::backend::SessionId;
use crate::backend::caps::LatencyTier;
use crate::backend::traits::{Backend, BackendSession, LocalBackend};
use crate::engine::{Capabilities, ExecError, ExecutionStats, HookBus, LoadRequest, Settings};
use crate::session_rt::SessionSpec;

/// Live state held once `load_model` succeeds. Weights are mmap'd so
/// swapping devices / models is cheap: drop + reload.
struct Loaded {
    /// VarBuilder is the weight handle; kept around so downstream code
    /// (deferred — Gemma `Model::new`) can rebuild the model without
    /// touching disk. `safetensors` files stay memory-mapped as long
    /// as this field is alive.
    _vb: VarBuilder<'static>,
    tokenizer: Tokenizer,
    device: Device,
    n_ctx: usize,
    model_dir: PathBuf,
}

impl std::fmt::Debug for Loaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded")
            .field("n_ctx", &self.n_ctx)
            .field("model_dir", &self.model_dir)
            .finish()
    }
}

#[derive(Debug)]
pub struct CandleBackend {
    loaded: Mutex<Option<Loaded>>,
    settings: Mutex<Arc<Settings>>,
    settings_version: std::sync::atomic::AtomicU64,
    hooks: Arc<HookBus>,
    stats: Mutex<ExecutionStats>,
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self {
            loaded: Mutex::new(None),
            settings: Mutex::new(Arc::new(Settings::default())),
            settings_version: std::sync::atomic::AtomicU64::new(0),
            hooks: Arc::new(HookBus::default()),
            stats: Mutex::new(ExecutionStats::default()),
        }
    }
}

impl CandleBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Best device available at build+run time. Candle's device enum
    /// accepts `Device::new_cuda(ord)` / `Device::new_metal(ord)`; those
    /// fail cleanly on hosts without the runtime, so we can probe
    /// unconditionally and fall through to CPU.
    fn pick_device() -> Device {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
        Device::Cpu
    }

    /// Parse `config.json` to discover `max_position_embeddings` (context
    /// window). Returns a conservative default when the field is absent.
    fn read_ctx_window(config_bytes: &[u8]) -> usize {
        let value: serde_json::Value = match serde_json::from_slice(config_bytes) {
            Ok(v) => v,
            Err(_) => return 2048,
        };
        value
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(2048)
    }
}

impl Backend for CandleBackend {
    fn backend_name(&self) -> &'static str {
        "candle"
    }

    fn load_model(&self, req: LoadRequest) -> Result<(), ExecError> {
        let model_path = &req.model_path;
        if !model_path.exists() {
            return Err(ExecError::InvalidModelFile(format!(
                "candle: model path does not exist: {}",
                model_path.display()
            )));
        }

        let model_dir: PathBuf = if model_path.is_file() {
            model_path
                .parent()
                .ok_or_else(|| {
                    ExecError::InvalidModelFile("candle: model file has no parent".into())
                })?
                .to_path_buf()
        } else {
            model_path.clone()
        };

        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_bytes = std::fs::read(&config_path).map_err(|e| {
            ExecError::InvalidModelFile(format!("candle: read {}: {e}", config_path.display()))
        })?;
        let n_ctx = Self::read_ctx_window(&config_bytes);

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            ExecError::InvalidModelFile(format!(
                "candle: tokenizer {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        let mut shards: Vec<PathBuf> = std::fs::read_dir(&model_dir)
            .map_err(|e| ExecError::Io(format!("candle: read dir: {e}")))?
            .filter_map(|entry| entry.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|ext| ext == "safetensors")
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(ExecError::InvalidModelFile(format!(
                "candle: no .safetensors in {}",
                model_dir.display()
            )));
        }

        let device = Self::pick_device();
        // SAFETY: VarBuilder::from_mmaped_safetensors is marked unsafe
        // because it relies on the file not being truncated while the
        // mapping is live. We hold the mapping for the lifetime of the
        // Loaded struct — dropping the field drops the maps. Not exposing
        // the VarBuilder to external callers keeps the contract intact.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device).map_err(|e| {
                ExecError::InvalidModelFile(format!("candle: safetensors mmap: {e}"))
            })?
        };

        *self.loaded.lock().expect("loaded lock") = Some(Loaded {
            _vb: vb,
            tokenizer,
            device,
            n_ctx,
            model_dir: model_dir.clone(),
        });
        tracing::info!(
            path = %model_dir.display(),
            n_ctx,
            "candle backend: weights mmap'd + tokenizer loaded"
        );
        Ok(())
    }

    fn reload_model(&self) -> Result<(), ExecError> {
        Err(ExecError::Unimplemented)
    }

    fn unload_model(&self) {
        *self.loaded.lock().expect("loaded lock") = None;
    }

    fn is_model_loaded(&self) -> bool {
        self.loaded.lock().expect("loaded lock").is_some()
    }

    fn upload_settings(&self, settings: Settings) -> Result<(), ExecError> {
        *self.settings.lock().expect("settings lock") = Arc::new(settings);
        self.settings_version
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn settings(&self) -> Arc<Settings> {
        Arc::clone(&*self.settings.lock().expect("settings lock"))
    }

    fn settings_version(&self) -> u64 {
        self.settings_version
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn hooks(&self) -> Arc<HookBus> {
        Arc::clone(&self.hooks)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn stats(&self) -> ExecutionStats {
        self.stats.lock().expect("stats lock").clone()
    }

    fn first_token_tier(&self) -> LatencyTier {
        // Candle on CPU is noticeably slower than llama.cpp CPU (no
        // ggml quant kernels). Router prefers other backends where
        // available.
        LatencyTier::Slow
    }

    fn start_session(&self, _spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        // Model is loaded, but the streaming generation loop (Gemma
        // forward + LogitsProcessor + KV state + TokenEvent emission)
        // is the next 1–2 week milestone — see module doc.
        Err(ExecError::Unimplemented)
    }

    fn end_session(&self, _id: SessionId) -> Result<(), ExecError> {
        Ok(())
    }
}

impl LocalBackend for CandleBackend {
    fn n_ctx(&self) -> usize {
        self.loaded
            .lock()
            .expect("loaded lock")
            .as_ref()
            .map(|l| l.n_ctx)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_probe_returns_something() {
        // Real integration smoke — proves candle-core links and can
        // instantiate a Device on this host. CPU fallback guarantees
        // we always get Some device back.
        let d = CandleBackend::pick_device();
        // Device doesn't impl PartialEq; `Debug` is enough to confirm.
        let printed = format!("{d:?}");
        assert!(!printed.is_empty(), "device debug should render");
    }

    #[test]
    fn ctx_window_reads_max_position_embeddings() {
        let json = br#"{"max_position_embeddings": 8192, "hidden_size": 2048}"#;
        assert_eq!(CandleBackend::read_ctx_window(json), 8192);
    }

    #[test]
    fn ctx_window_falls_back_when_field_missing() {
        let json = br#"{"hidden_size": 2048}"#;
        assert_eq!(CandleBackend::read_ctx_window(json), 2048);
    }

    #[test]
    fn ctx_window_falls_back_on_malformed_json() {
        assert_eq!(CandleBackend::read_ctx_window(b"not json"), 2048);
    }

    #[test]
    fn backend_starts_unloaded() {
        let b = CandleBackend::new();
        assert!(!b.is_model_loaded());
        assert_eq!(b.n_ctx(), 0);
    }

    #[test]
    fn unload_is_idempotent_when_empty() {
        let b = CandleBackend::new();
        b.unload_model();
        b.unload_model();
        assert!(!b.is_model_loaded());
    }

    #[test]
    fn settings_version_increments_on_upload() {
        let b = CandleBackend::new();
        assert_eq!(b.settings_version(), 0);
        b.upload_settings(Settings::default()).unwrap();
        assert_eq!(b.settings_version(), 1);
        b.upload_settings(Settings::default()).unwrap();
        assert_eq!(b.settings_version(), 2);
    }

    #[test]
    fn start_session_without_model_fails_cleanly() {
        let b = CandleBackend::new();
        let err = b.start_session(SessionSpec::default()).unwrap_err();
        assert!(matches!(err, ExecError::Unimplemented));
    }

    #[test]
    fn backend_name_is_stable() {
        assert_eq!(CandleBackend::new().backend_name(), "candle");
    }

    #[test]
    fn first_token_tier_is_slow() {
        assert_eq!(CandleBackend::new().first_token_tier(), LatencyTier::Slow);
    }
}
