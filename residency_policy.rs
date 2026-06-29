use crate::diagnostics::{MachineMemoryTier, MemoryPressureLevel};

/// Residency sizing helpers and context-budget policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub max_context_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyPolicy {
    pub helper_idle_timeout_secs: u64,
    pub llm_swap_requires_unload: bool,
}

impl Default for ResidencyPolicy {
    fn default() -> Self {
        Self {
            helper_idle_timeout_secs: 300,
            llm_swap_requires_unload: true,
        }
    }
}

pub fn default_context_budget_for_tier(tier: MachineMemoryTier) -> ContextBudget {
    let max_context_tokens = match tier {
        MachineMemoryTier::MobileConstrained => 4_096,
        MachineMemoryTier::DesktopConstrained => 8_192,
        MachineMemoryTier::DesktopMainstream => 16_384,
        MachineMemoryTier::DesktopPower => 24_576,
        MachineMemoryTier::Workstation => 32_768,
    };
    ContextBudget { max_context_tokens }
}

pub fn effective_context_budget(
    tier: MachineMemoryTier,
    pressure: MemoryPressureLevel,
    multimodal_active: bool,
) -> ContextBudget {
    let mut max_context_tokens = default_context_budget_for_tier(tier).max_context_tokens;

    max_context_tokens = match pressure {
        MemoryPressureLevel::Normal => max_context_tokens,
        MemoryPressureLevel::Constrained => max_context_tokens.saturating_mul(3) / 4,
        MemoryPressureLevel::Severe | MemoryPressureLevel::Emergency => {
            max_context_tokens.saturating_mul(1) / 2
        }
    };

    if multimodal_active {
        max_context_tokens = max_context_tokens.saturating_mul(3) / 4;
    }

    ContextBudget {
        max_context_tokens: max_context_tokens.max(2048),
    }
}

use crate::hardware::GpuBackend;

fn file_mb_of(path: &std::path::Path) -> u64 {
    std::fs::metadata(path)
        .ok()
        .map(|md| (md.len().saturating_add(1024 * 1024 - 1)) / (1024 * 1024))
        .unwrap_or(0)
}

/// Estimate resident memory for a local runtime path.
///
/// Conservative, GPU-agnostic: assumes **no** offload (all weights in host RAM).
/// Used for runtimes we don't model GPU offload for (e.g. embedders). For LLM
/// loads that may offload to VRAM, use [`estimate_resident_mb_for_path_offloaded`].
pub fn estimate_resident_mb_for_path(path: &std::path::Path) -> u64 {
    resident_mb_from_file_mb(file_mb_of(path), GpuBackend::Cpu, Some(0))
}

/// Host (system-RAM) resident MB for a model file, **accounting for GPU offload**.
///
/// Weights for layers offloaded to the GPU live in **VRAM, not host RAM**. Counting
/// a model's whole on-disk size against the system-RAM budget wrongly denies
/// admission on RAM-tight hosts — e.g. a ~2 GB model bound for a 10 GB GPU on a box
/// with 2.5 GB free RAM gets `can_admit == false` and the load 500s before llama
/// even runs. When `gpu_layers > 0` we estimate only the **non-offloaded** layers'
/// weights (which stay in host RAM) plus a fixed host overhead (KV/compute scratch +
/// the CPU-mapped metadata buffer — empirically ~300 MB for a fully-offloaded 2 GB
/// model). `None`/`Some(0)` keeps the conservative full-file estimate (pure CPU).
///
/// The layer-count proxy (`file_mb / 100`, clamped 12..80) mirrors the auto-tune
/// heuristic in `crate::app::models::service_impl` (service_impl.rs:457).
///
/// **Backend matters, not just `gpu_layers`.** On a GPU host the offload default is
/// `gpu_layers == None` (= "auto / all layers" — `hardware.rs:208`, `config.rs:994`),
/// *not* `Some(n)`. A CPU-only desktop *also* defaults to `None`. So `None` alone is
/// ambiguous; we resolve it against the detected [`GpuBackend`]: only a real GPU
/// backend (Metal/Cuda/Vulkan) puts weights in VRAM.
pub fn estimate_resident_mb_for_path_offloaded(
    path: &std::path::Path,
    gpu_layers: Option<u32>,
) -> u64 {
    let backend = crate::hardware::HardwareProfile::detect().gpu_backend;
    resident_mb_from_file_mb(file_mb_of(path), backend, gpu_layers)
}

/// Pure host-resident estimate (testable without a real file or hardware). See
/// [`estimate_resident_mb_for_path_offloaded`].
///
/// `gpu_layers`: `None` = auto (all layers offloaded **iff** a GPU backend is
/// present); `Some(0)` = CPU-only; `Some(n)` = offload `n` layers.
pub(crate) fn resident_mb_from_file_mb(
    file_mb: u64,
    gpu_backend: GpuBackend,
    gpu_layers: Option<u32>,
) -> u64 {
    const MIN_ESTIMATE_MB: u64 = 256;
    // Host RAM that stays resident even under full GPU offload: KV-cache + compute
    // scratch + the CPU-mapped metadata buffer.
    const GPU_HOST_OVERHEAD_MB: u64 = 384;

    // No GPU backend → every weight is resident in host RAM, whatever gpu_layers says.
    if gpu_backend == GpuBackend::Cpu {
        return file_mb.max(MIN_ESTIMATE_MB);
    }

    // Same layer-count proxy as the auto-tune path (service_impl.rs:457).
    let est_layers = ((file_mb / 100) as u32).clamp(12, 80) as u64;
    let offloaded = match gpu_layers {
        None => est_layers,                  // auto = offload all layers to VRAM
        Some(n) => (n as u64).min(est_layers),
    };
    if offloaded == 0 {
        // Explicit CPU-only even though a GPU is present — respect it.
        return file_mb.max(MIN_ESTIMATE_MB);
    }
    let resident_layers = est_layers.saturating_sub(offloaded);
    let resident_weights_mb = file_mb.saturating_mul(resident_layers) / est_layers.max(1);
    resident_weights_mb
        .saturating_add(GPU_HOST_OVERHEAD_MB)
        .max(MIN_ESTIMATE_MB)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_budget_grows_with_tier() {
        let mobile = default_context_budget_for_tier(MachineMemoryTier::MobileConstrained);
        let desk = default_context_budget_for_tier(MachineMemoryTier::DesktopMainstream);
        let work = default_context_budget_for_tier(MachineMemoryTier::Workstation);
        assert!(mobile.max_context_tokens < desk.max_context_tokens);
        assert!(desk.max_context_tokens < work.max_context_tokens);
    }

    #[test]
    fn residency_policy_defaults_to_helper_eviction_and_llm_swap_unload() {
        let policy = ResidencyPolicy::default();
        assert!(policy.helper_idle_timeout_secs > 0);
        assert!(policy.llm_swap_requires_unload);
    }

    // ── GPU-offload aware host-RAM estimate (the RTX-GPU-brain fix) ──────────
    // Repro: a ~2 GB model bound for a 10 GB GPU on a 15 GB host that's down to
    // ~2.5 GB free. The CPU estimate (full file size) trips `can_admit` →
    // "admission denied" → 500 before llama runs. The offload-aware estimate
    // counts only the non-offloaded weights + host overhead.
    const TIGHT_BUDGET_MB: u64 = 1536; // a real desktop-tier inference_resident_mb

    #[test]
    fn cpu_backend_is_full_file_size_regardless_of_layers() {
        // No GPU → every weight resident, even with auto/explicit "offload" asked.
        assert_eq!(resident_mb_from_file_mb(2000, GpuBackend::Cpu, None), 2000);
        assert_eq!(resident_mb_from_file_mb(2000, GpuBackend::Cpu, Some(999)), 2000);
        assert!(resident_mb_from_file_mb(2000, GpuBackend::Cpu, None) > TIGHT_BUDGET_MB);
    }

    #[test]
    fn gpu_auto_offload_drops_host_estimate_below_tight_budget() {
        // THE RTX REPRO: 2 GB model, CUDA host, gpu_layers == None (the real default
        // on a GPU desktop — config.rs:994). Weights go to VRAM, so only host
        // overhead stays resident → it ADMITS where the CPU full-file estimate denies.
        let cuda_auto = resident_mb_from_file_mb(2000, GpuBackend::Cuda, None);
        assert!(
            cuda_auto <= TIGHT_BUDGET_MB,
            "auto offload on a GPU host should fit a tight budget, got {cuda_auto}MB"
        );
        assert!(cuda_auto < resident_mb_from_file_mb(2000, GpuBackend::Cpu, None));
        assert!(cuda_auto >= 256, "must stay above MIN floor, got {cuda_auto}");
        // Explicit "all layers" lands at the same place.
        assert_eq!(
            cuda_auto,
            resident_mb_from_file_mb(2000, GpuBackend::Cuda, Some(999))
        );
    }

    #[test]
    fn gpu_explicit_zero_layers_is_full_file() {
        // A GPU is present but the user forced CPU-only (gpu_layers = 0): respect it.
        assert_eq!(resident_mb_from_file_mb(2000, GpuBackend::Cuda, Some(0)), 2000);
    }

    #[test]
    fn partial_offload_scales_between_cpu_and_full() {
        // est_layers(2000MB) = clamp(20,12,80) = 20; offloading 10 leaves ~half
        // the weights resident.
        let half = resident_mb_from_file_mb(2000, GpuBackend::Cuda, Some(10));
        let cpu = resident_mb_from_file_mb(2000, GpuBackend::Cpu, None);
        let full = resident_mb_from_file_mb(2000, GpuBackend::Cuda, None);
        assert!(half > full && half < cpu, "partial={half} full={full} cpu={cpu}");
    }

    #[test]
    fn tiny_fixtures_floor_to_minimum() {
        // CPU path floors at MIN_ESTIMATE_MB (256).
        assert_eq!(resident_mb_from_file_mb(10, GpuBackend::Cpu, None), 256);
        // Offloaded path always carries host overhead, so it floors at
        // GPU_HOST_OVERHEAD_MB (384) even for a 0-byte fixture.
        assert_eq!(resident_mb_from_file_mb(0, GpuBackend::Cuda, None), 384);
    }

    #[test]
    fn metal_auto_offload_matches_cuda() {
        // Mac (Metal) auto-offload behaves like CUDA — the flock's other GPU host.
        assert_eq!(
            resident_mb_from_file_mb(2000, GpuBackend::Metal, None),
            resident_mb_from_file_mb(2000, GpuBackend::Cuda, None),
        );
    }

    #[test]
    fn effective_context_budget_shrinks_under_pressure() {
        let normal = effective_context_budget(
            MachineMemoryTier::DesktopMainstream,
            MemoryPressureLevel::Normal,
            false,
        );
        let severe = effective_context_budget(
            MachineMemoryTier::DesktopMainstream,
            MemoryPressureLevel::Severe,
            false,
        );
        assert!(severe.max_context_tokens < normal.max_context_tokens);
    }

    #[test]
    fn multimodal_context_budget_is_lower_than_text_only() {
        let text_only = effective_context_budget(
            MachineMemoryTier::DesktopPower,
            MemoryPressureLevel::Normal,
            false,
        );
        let multimodal = effective_context_budget(
            MachineMemoryTier::DesktopPower,
            MemoryPressureLevel::Normal,
            true,
        );
        assert!(multimodal.max_context_tokens < text_only.max_context_tokens);
    }
}
