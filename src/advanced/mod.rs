//! Below the happy path.
//!
//! Everything a normal consumer needs is at the crate root: [`Engine`],
//! [`Session`], the three ways to call. This module is for the consumer who
//! needs to reach past that — today, to bring a backend of their own.
//!
//! - [`plugin`] — implement [`LocalBackend`](crate::advanced::plugin::LocalBackend)
//!   outside the crate, wrap it in a [`BackendPlugin`], and register it with
//!   [`EngineBuilder::backend`](crate::EngineBuilder::backend).
//! - [`runtime`] — the residency controls of api_spec.md §4.5: what a
//!   [`Runtime`](crate::Runtime) has resident, and the types behind
//!   `hardware()`, `residency()`, `preload`, `evict`, and `stats()`.
//!
//! [`Engine`]: crate::Engine
//! [`Session`]: crate::Session

pub mod plugin;

/// Runtime-level residency and hardware, below the happy path (api_spec.md
/// §4.5). The methods are on [`Runtime`](crate::Runtime); the types they
/// return live here so the root stays boring.
pub mod runtime {
    pub use crate::api::{ModelResidency, ResidencySnapshot, RuntimeStats};
    pub use crate::hardware::{GpuBackend, HardwareProfile};
}

pub use plugin::BackendPlugin;
