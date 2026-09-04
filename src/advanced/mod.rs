//! Below the happy path.
//!
//! Everything a normal consumer needs is at the crate root: [`Engine`],
//! [`Session`], the three ways to call. This module is for the consumer who
//! needs to reach past that — today, to bring a backend of their own.
//!
//! - [`plugin`] — implement [`LocalBackend`](crate::advanced::plugin::LocalBackend)
//!   outside the crate, wrap it in a [`BackendPlugin`], and register it with
//!   [`EngineBuilder::backend`](crate::EngineBuilder::backend).
//!
//! [`Engine`]: crate::Engine
//! [`Session`]: crate::Session

pub mod plugin;

pub use plugin::BackendPlugin;
