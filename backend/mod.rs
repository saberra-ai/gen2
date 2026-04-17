//! Backend-specific inference implementations.
//!
//! Multiple backends can be compiled simultaneously — the runtime picks
//! the right one based on model format (GGUF → llamacpp, safetensors dir → MLX,
//! ONNX file → onnx).  MLX is only available on Apple platforms.

pub mod common;

#[cfg(feature = "backend-llamacpp")]
pub mod llama;

#[cfg(feature = "backend-mlx")]
pub(crate) mod mlx;

#[cfg(feature = "backend-onnx")]
pub(crate) mod onnx;

#[cfg(feature = "backend-external-api")]
pub(crate) mod external_api;

// Compile-time guard: at least one backend must be selected.
#[cfg(not(any(
    feature = "backend-llamacpp",
    feature = "backend-mlx",
    feature = "backend-onnx",
    feature = "backend-external-api"
)))]
compile_error!(
    "No inference backend selected. Enable at least one of: backend-llamacpp, backend-mlx, backend-onnx, backend-external-api"
);

pub mod caps;
mod dispatch;
pub mod health;
pub mod traits;
pub use caps::{BackendCaps, LatencyTier};
pub use dispatch::{Engine, ModelBundle, Session, SessionId, TokenPuller};
pub use health::SessionHealth;
pub use traits::KvSnapshot as KvSnapshotTrait;
pub use traits::{
    Backend, BackendSession, Embeddings, LocalBackend, Multimodal, RemoteBackend, SessionTokenizer,
    TokenPullerDyn,
};

// Llama-specific internal types (needed by session_rt only when llamacpp is active)
#[cfg(feature = "backend-llamacpp")]
pub(crate) use llama::{DecodeState, SessionCtxCell};
