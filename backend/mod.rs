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

// mlxcel — fast MLX inference embedded via mlxcel-core (docs/plans/mlxcel-embedding-roadmap.md).
// backend-mlx (mlx-rs) and backend-mlxcel (mlxcel-core's cxx MLX C++) BOTH link MLX C++, so
// enabling both duplicates the MLX symbol surface → mutually exclusive (guard below).
#[cfg(feature = "backend-mlxcel")]
pub(crate) mod mlxcel;

#[cfg(all(feature = "backend-mlx", feature = "backend-mlxcel"))]
compile_error!(
    "backend-mlx and backend-mlxcel both link MLX C++ and cannot be enabled together; \
     enable exactly one (backend-mlxcel replaces backend-mlx on the macOS/daemon path)."
);

#[cfg(feature = "backend-onnx")]
pub(crate) mod onnx;

#[cfg(feature = "backend-external-api")]
pub(crate) mod external_api;

// Phase D week 14 — mobile + Rust-native fallback. Both are scaffolds
// (trait impls returning Unimplemented) until the real integration ships;
// module layout reserved so the rest of gen2 can reference them.
#[cfg(feature = "backend-executorch")]
pub(crate) mod executorch;

#[cfg(feature = "backend-candle")]
pub(crate) mod candle;

// Compile-time guard: at least one backend must be selected.
#[cfg(not(any(
    feature = "backend-llamacpp",
    feature = "backend-mlx",
    feature = "backend-mlxcel",
    feature = "backend-onnx",
    feature = "backend-external-api",
    feature = "backend-executorch",
    feature = "backend-candle"
)))]
compile_error!(
    "No inference backend selected. Enable at least one of: backend-llamacpp, backend-mlx, backend-mlxcel, backend-onnx, backend-external-api, backend-executorch, backend-candle"
);

pub mod caps;
mod facade;
pub mod health;
pub mod traits;
pub use caps::{BackendCaps, LatencyTier};
pub use facade::{Engine, ModelBundle, Session, SessionId, TokenPuller};
pub use health::SessionHealth;
pub use traits::KvSnapshot as KvSnapshotTrait;
pub use traits::{
    Backend, BackendSession, Embeddings, LocalBackend, Multimodal, RemoteBackend, SessionTokenizer,
    TokenPullerDyn,
};

// Llama-specific internal types (needed by session_rt only when llamacpp is active)
#[cfg(feature = "backend-llamacpp")]
pub(crate) use llama::{DecodeState, SessionCtxCell};
