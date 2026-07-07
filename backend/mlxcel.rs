//! `backend-mlxcel` — fast MLX inference embedded via [`mlxcel_core`] (Rust + MLX C++
//! bindings, decode ≈ mlx-lm). Replaces `backend-mlx` (mlx-rs) on the macOS/daemon
//! path, which was measured ~8–11× slower than mlx-lm. See the roadmap at
//! `docs/plans/mlxcel-embedding-roadmap.md`.
//!
//! **Slice 1 (this module):** pin the `mlxcel-core` dependency and prove pio-core
//! links its MLX-C++ symbol surface cleanly. Because `backend-mlx` and
//! `backend-mlxcel` are mutually exclusive (guard in `super`), an mlxcel build never
//! links mlx-rs too, so there is no duplicate-MLX-symbol clash. The real gen2
//! [`Backend`](super::traits::Backend) impl (load → stream → grammar-masked sample)
//! lands in Slice 2, mirroring mlxcel's `src/commands/generate.rs`
//! (`load_generation_model → build_sampling_config → sample_token_optimized`).

/// Link probe (Slice 1): referencing a public `mlxcel-core` item forces the crate
/// (and its `cmake`-built MLX C++ + Metal kernels) to compile and link into
/// pio-core. If this builds, embedding is viable on this target. Scaffold —
/// replaced by the real `Backend` impl in Slice 2.
#[allow(dead_code)]
pub fn linked() -> bool {
    // `TokenBiasMap` is re-exported at the crate root (mlxcel-core lib.rs); naming
    // it pulls the crate in as a real, non-pruned dependency.
    let _bias: Option<mlxcel_core::TokenBiasMap> = None;
    true
}

#[cfg(test)]
mod tests {
    /// Slice 1 gate: pio-core compiles + links against mlxcel-core (MLX C++ via cxx)
    /// with no symbol clash (mlx-rs is absent in a backend-mlxcel build).
    #[test]
    fn mlxcel_core_links() {
        assert!(super::linked());
    }
}
