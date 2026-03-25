//! Backend-specific inference implementations.
//!
//! Multiple backends can be compiled simultaneously — the runtime picks
//! the right one based on model format (GGUF → llamacpp, safetensors dir → MLX,
//! ONNX file → onnx).  MLX is only available on Apple platforms.

pub(crate) mod common;

#[cfg(feature = "backend-llamacpp")]
pub(crate) mod llama;

#[cfg(feature = "backend-mlx")]
pub(crate) mod mlx;

#[cfg(feature = "backend-onnx")]
pub(crate) mod onnx;

// Compile-time guard: at least one backend must be selected.
#[cfg(not(any(feature = "backend-llamacpp", feature = "backend-mlx", feature = "backend-onnx")))]
compile_error!("No inference backend selected. Enable at least one of: backend-llamacpp, backend-mlx, backend-onnx");

mod dispatch;
pub use dispatch::{Engine, ModelBundle, Session, SessionId, TokenPuller};

// Llama-specific internal types (needed by session_rt only when llamacpp is active)
#[cfg(feature = "backend-llamacpp")]
pub(crate) use llama::{DecodeState, SessionCtxCell};
