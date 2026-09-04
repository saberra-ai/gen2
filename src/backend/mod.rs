//! Backend-specific inference implementations.
//!
//! Multiple backends can be compiled simultaneously — the runtime picks
//! the right one based on model format (GGUF → llamacpp, safetensors dir → MLX,
//! `.litertlm` → LiteRT-LM).  MLX is only available on Apple platforms.
//! A backend built outside the crate joins the same routing through
//! [`crate::advanced::BackendPlugin`], and is asked before any of these.

pub mod common;

#[cfg(feature = "backend-llamacpp")]
pub mod llama;

#[cfg(feature = "backend-mlx")]
pub(crate) mod mlx;

#[cfg(feature = "backend-external-api")]
pub(crate) mod external_api;

// There is deliberately no compile-time "at least one backend" guard. A build
// with no backend feature is a legitimate one: a consumer that brings its own
// backend through `crate::advanced::BackendPlugin` (the way the companion crate
// under `crates/` does) has nothing to enable here. With no feature and no
// plugin, `Engine::new` starts `Uninit` and the first load fails naming both
// ways out — see `facade::no_backend_error`.

/// One contract every compiled backend must satisfy. See the module docs for
/// which half needs a real model and which does not.
#[cfg(test)]
mod conformance;

#[cfg(feature = "backend-mistralrs")]
pub(crate) mod mistralrs;

/// Google's on-device runtime, loaded from its C ABI at run time. Nothing is
/// vendored or linked, so this compiles on a machine that has never had
/// LiteRT-LM installed.
#[cfg(feature = "backend-litertlm")]
pub(crate) mod litertlm;

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
