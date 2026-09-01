//! llama-cpp-2 inference backend.

mod bundle;
pub mod embedder;
pub(crate) mod engine;
/// iOS runtime memory-budgeting (jetsam-aware load preflight).
///
/// The module compiles on every platform so its PURE helpers (headroom math,
/// quant/param ceiling) can be unit-tested off-iOS, but the iOS FFI
/// (`os_proc_available_memory`) and the `preflight_ios` entry point are
/// individually `#[cfg(target_os = "ios")]`-gated, and the loader only *calls*
/// them under `#[cfg(target_os = "ios")]`. Off-iOS nothing here runs, so
/// desktop/flagship behavior is byte-identical; the helpers are dead code off
/// iOS (silenced below).
#[cfg_attr(not(target_os = "ios"), allow(dead_code))]
mod ios_memory;
pub mod llama_config;
mod loader;
mod puller;
mod session;
mod tokenizer_adapter;

pub use bundle::ModelBundle;
pub use engine::Engine;
pub use puller::TokenPuller;
pub use session::Session;
pub(crate) use session::{DecodeState, SessionCtxCell};
