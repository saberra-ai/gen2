//! Runtime hardware detection and adaptive model/context recommendation.
//!
//! Detects system RAM, CPU cores, and GPU backend at runtime.
//! Recommends optimal model size + context window, prioritizing context length.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Hardware profile
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct HardwareProfile {
    pub total_ram_bytes: u64,
    pub cpu_cores: usize,
    pub gpu_backend: GpuBackend,
    /// Total dedicated GPU VRAM in bytes, summed across all detected GPUs.
    ///
    /// Mirrors Jan's per-GPU VRAM probe + sum
    /// (`jan/src-tauri/plugins/tauri-plugin-hardware/src/vendor/nvidia.rs:174`
    /// reports each GPU's `memory_info.total` in MiB; Jan sums them in
    /// `jan/src-tauri/plugins/tauri-plugin-llamacpp/src/gguf/commands.rs:133-137`).
    /// `0` means "no measured VRAM" — the conventional value for Apple-Silicon
    /// unified memory (no separate pool, budget rides on `total_ram_bytes`),
    /// CPU-only machines, and backends we can't probe (Vulkan; see
    /// `detect_vram_bytes`). Consumers must treat `0` as "unknown / use RAM",
    /// never as "0 bytes of GPU".
    #[serde(default)]
    pub vram_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    Metal,
    Cuda,
    Vulkan,
    Cpu,
}

impl HardwareProfile {
    /// Detect hardware at runtime. Safe to call multiple times (cheap).
    pub fn detect() -> Self {
        let gpu_backend = detect_gpu_backend();
        let vram_bytes = detect_vram_bytes(&gpu_backend);
        Self {
            total_ram_bytes: detect_total_ram(),
            cpu_cores: detect_cpu_cores(),
            gpu_backend,
            vram_bytes,
        }
    }

    /// Detect once and cache for the process lifetime. Use on hot paths
    /// (e.g. per-session context fitting) — `detect()` can shell out to
    /// `nvidia-smi` on CUDA hosts, which is too slow to repeat.
    pub fn cached() -> &'static Self {
        static CACHE: std::sync::OnceLock<HardwareProfile> = std::sync::OnceLock::new();
        CACHE.get_or_init(Self::detect)
    }

    /// Context-token cap for this machine's memory tier — the same
    /// ladder as `residency_policy::default_context_budget_for_tier`,
    /// derivable from RAM alone so load paths don't need a governor.
    pub fn tier_context_cap(&self) -> u32 {
        if cfg!(any(target_os = "android", target_os = "ios")) {
            return 4_096;
        }
        match self.total_ram_gb() {
            0..=7 => 8_192,
            8..=15 => 16_384,
            16..=31 => 24_576,
            _ => 32_768,
        }
    }

    /// Total RAM in gigabytes (rounded down).
    pub fn total_ram_gb(&self) -> u64 {
        self.total_ram_bytes / (1024 * 1024 * 1024)
    }

    /// Usable RAM budget for inference (total minus OS/app reserve).
    pub fn inference_budget_bytes(&self) -> u64 {
        let reserve = if cfg!(any(target_os = "android", target_os = "ios")) {
            2 * 1024 * 1024 * 1024_u64 // 2 GB reserve on mobile
        } else {
            4 * 1024 * 1024 * 1024_u64 // 4 GB reserve on desktop
        };
        self.total_ram_bytes.saturating_sub(reserve)
    }
}

// ---------------------------------------------------------------------------
// Model recommendation
// ---------------------------------------------------------------------------

/// A recommended model + settings for the detected hardware.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct ModelRecommendation {
    /// HuggingFace repo ID
    pub model_id: String,
    /// GGUF filename within the repo (e.g. "qwen2.5-3b-instruct-q4_k_m.gguf")
    pub filename: String,
    /// Human-readable label (e.g. "Qwen 3B")
    pub model_label: String,
    /// Quantization level
    pub quant: String,
    /// Recommended context window
    pub ctx_size: u32,
    /// GPU layers (None = offload all)
    pub gpu_layers: Option<u32>,
    /// Batch size for prefill
    pub batch_size: u32,
    /// Thread count
    pub threads: u32,
    /// Estimated total RAM usage in MB (model + KV cache)
    pub estimated_ram_mb: u32,
}

/// Static model catalog entry.
struct ModelSpec {
    repo_id: &'static str,
    filename: &'static str,
    label: &'static str,
    quant: &'static str,
    size_mb: u32,      // approximate file size in MB
    kv_per_token: u32, // KV cache bytes per token (all layers)
    ctx_train: u32,    // max context from training
}

/// Qwen 3 GGUF catalog via unsloth (context-first priority).
/// Used when the llama.cpp backend is active (default).
const MODELS: &[ModelSpec] = &[
    ModelSpec {
        repo_id: "unsloth/Qwen3-0.6B-GGUF",
        filename: "Qwen3-0.6B-Q4_K_M.gguf",
        label: "Qwen3 0.6B",
        quant: "Q4_K_M",
        size_mb: 378,
        kv_per_token: 57_344, // 28 layers, 8 KV heads, 128 dim
        ctx_train: 40_960,
    },
    ModelSpec {
        repo_id: "unsloth/Qwen3-1.7B-GGUF",
        filename: "Qwen3-1.7B-Q4_K_M.gguf",
        label: "Qwen3 1.7B",
        quant: "Q4_K_M",
        size_mb: 1_056,
        kv_per_token: 57_344, // 28 layers, 8 KV heads, 128 dim
        ctx_train: 40_960,
    },
    ModelSpec {
        // The quality "knee" (early 2026): the 2507 refresh posts 30B-class
        // scores at ~2.5 GB Q4 and extends context to 262K. Supersedes the
        // April Qwen3-4B (hybrid-thinking, dropped). Reasoning is a *separate*
        // checkpoint now (Qwen3-4B-Thinking-2507) — routed via `model_tiers`,
        // not this general-default ladder. See docs/plans/capability-tiered-model-set.md.
        repo_id: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        label: "Qwen3 4B Instruct 2507",
        quant: "Q4_K_M",
        size_mb: 2_500,
        kv_per_token: 73_728, // 36 layers, 8 KV heads, 128 dim
        ctx_train: 262_144,
    },
    ModelSpec {
        repo_id: "unsloth/Qwen3-8B-GGUF",
        filename: "Qwen3-8B-Q4_K_M.gguf",
        label: "Qwen3 8B",
        quant: "Q4_K_M",
        size_mb: 4_795,
        kv_per_token: 73_728, // 36 layers, 8 KV heads, 128 dim
        ctx_train: 40_960,
    },
];

/// Qwen 3 MLX (safetensors) catalog — preferred on Apple Silicon for faster generation.
/// Used when the MLX backend is active (feature `backend-mlx`).
/// NOTE (Maya): On iOS, reduce inference budget by ~2 GB vs desktop — leave room for OS.
#[cfg(feature = "backend-mlx")]
#[allow(dead_code)] // model catalog retained for the MLX device-profile picker
const MLX_MODELS: &[ModelSpec] = &[
    ModelSpec {
        repo_id: "mlx-community/Qwen3-0.6B-4bit",
        filename: "", // MLX loads from directory, not single file
        label: "Qwen3 0.6B (MLX)",
        quant: "4bit",
        size_mb: 400,
        kv_per_token: 57_344,
        ctx_train: 40_960,
    },
    ModelSpec {
        repo_id: "lmstudio-community/Qwen3-1.7B-MLX-4bit",
        filename: "",
        label: "Qwen3 1.7B (MLX)",
        quant: "4bit",
        size_mb: 1_100,
        kv_per_token: 57_344,
        ctx_train: 40_960,
    },
    ModelSpec {
        // 2507 knee, MLX build. Exact mlx-community repo id confirmed at fetch
        // time — see docs/plans/capability-tiered-model-set.md §7.
        repo_id: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        filename: "",
        label: "Qwen3 4B Instruct 2507 (MLX)",
        quant: "4bit",
        size_mb: 2_400,
        kv_per_token: 73_728,
        ctx_train: 262_144,
    },
    ModelSpec {
        repo_id: "Qwen/Qwen3-8B-MLX-4bit",
        filename: "",
        label: "Qwen3 8B (MLX)",
        quant: "4bit",
        size_mb: 4_800,
        kv_per_token: 73_728,
        ctx_train: 40_960,
    },
];

/// Context tiers to try, largest first (context-priority algorithm).
const CTX_TIERS: &[u32] = &[40_960, 32_768, 16_384, 8_192, 4_096, 2_048];

/// Pick the optimal model + context for the detected hardware.
/// Algorithm: for each context tier (largest first), find the largest model
/// whose total memory (weights + KV cache) fits in the inference budget.
/// This prioritizes context length over model size.
pub fn recommend_model(hw: &HardwareProfile) -> ModelRecommendation {
    let budget_bytes = hw.inference_budget_bytes();
    let budget_mb = (budget_bytes / (1024 * 1024)) as u32;

    // Try each context tier, largest first
    for &ctx in CTX_TIERS {
        // Try each model, largest first (reverse order)
        for spec in MODELS.iter().rev() {
            let cap = ctx.min(spec.ctx_train);
            let kv_mb = (spec.kv_per_token as u64 * cap as u64 / (1024 * 1024)) as u32;
            let total_mb = spec.size_mb + kv_mb + 256; // +256 MB overhead
            if total_mb <= budget_mb {
                let threads = ((hw.cpu_cores / 2) as u32).max(2);
                let batch_size = if budget_mb >= 12_000 {
                    512
                } else if budget_mb >= 4_000 {
                    256
                } else {
                    128
                };
                let gpu_layers = match hw.gpu_backend {
                    GpuBackend::Metal | GpuBackend::Cuda | GpuBackend::Vulkan => None, // all layers
                    GpuBackend::Cpu => Some(0),
                };

                return ModelRecommendation {
                    model_id: spec.repo_id.to_string(),
                    filename: spec.filename.to_string(),
                    model_label: spec.label.to_string(),
                    quant: spec.quant.to_string(),
                    ctx_size: cap,
                    gpu_layers,
                    batch_size,
                    threads,
                    estimated_ram_mb: total_mb,
                };
            }
        }
    }

    // Absolute fallback: smallest model, smallest context
    let fallback = &MODELS[0];
    ModelRecommendation {
        model_id: fallback.repo_id.to_string(),
        filename: fallback.filename.to_string(),
        model_label: fallback.label.to_string(),
        quant: fallback.quant.to_string(),
        ctx_size: 2_048,
        gpu_layers: Some(0),
        batch_size: 128,
        threads: 2,
        estimated_ram_mb: fallback.size_mb + 64,
    }
}

/// Given a loaded model's file size and layer count, compute the optimal
/// context window for the current hardware. Used for user-imported models
/// where we don't have a catalog entry.
pub fn auto_tune_ctx(hw: &HardwareProfile, model_file_size: u64, n_layer: u32) -> u32 {
    // Conservative dims when architecture metadata is unavailable: 2 KV
    // heads and 128 head dim (common for GQA models).
    auto_tune_ctx_with_dims(hw, model_file_size, n_layer, 2, 128)
}

/// Like [`auto_tune_ctx`] but with real architecture dims (from parsed
/// GGUF metadata or a live model), sharing the fit formula with the
/// load-time context clamp.
pub fn auto_tune_ctx_with_dims(
    hw: &HardwareProfile,
    model_file_size: u64,
    n_layer: u32,
    n_head_kv: u32,
    head_dim: u32,
) -> u32 {
    use crate::bundle::gguf::{fit_context, kv_bytes_per_token};
    let model_mem = (model_file_size as f64 * 1.2) as u64;
    let kv = kv_bytes_per_token(
        n_layer as u64,
        n_head_kv.max(1) as u64,
        head_dim.max(1) as u64,
    );
    let fitted = fit_context(hw.inference_budget_bytes(), model_mem, kv, u32::MAX, None);
    // Snap down to the recommendation tiers (largest first), floor 2048.
    for &tier in CTX_TIERS {
        if tier <= fitted {
            return tier;
        }
    }
    2_048 // absolute minimum
}

// ---------------------------------------------------------------------------
// Platform-aware defaults
// ---------------------------------------------------------------------------

/// Returns platform-aware default inference settings based on detected hardware.
/// Used by the frontend "Reset to defaults" action when config parameters are invalid.
pub fn platform_defaults(hw: &HardwareProfile) -> crate::engine::Settings {
    let mut s = crate::engine::Settings::default();
    let ram_gb = hw.total_ram_gb();

    s.system.ctx_size = Some(match ram_gb {
        0..=3 => 2048,
        4..=7 => 4096,
        8..=15 => 8192,
        _ => 16384,
    });

    let cores = (hw.cpu_cores as u32).clamp(1, 8);
    s.system.threads = Some(cores);
    s.system.threads_batch = Some(cores);

    s.system.batch_size = Some(match ram_gb {
        0..=3 => 128,
        4..=7 => 512,
        8..=15 => 1024,
        _ => 2048,
    });

    if hw.gpu_backend != GpuBackend::Cpu {
        s.system.gpu_layers = Some(match ram_gb {
            0..=7 => 20,
            8..=15 => 32,
            _ => 99,
        });
        s.system.flash_attn = Some(true);
    }

    s.sampling.temperature = Some(0.7);
    s.sampling.top_p = Some(0.9);
    s.sampling.top_k = Some(40);
    s.sampling.min_p = Some(0.05);

    s
}

// ---------------------------------------------------------------------------
// Platform-specific detection
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn detect_total_ram() -> u64 {
    use std::mem;
    let mut size: u64 = 0;
    let mut len = mem::size_of::<u64>();
    let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
    unsafe {
        libc::sysctl(
            mib.as_ptr() as *mut _,
            2,
            &mut size as *mut u64 as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        );
    }
    size
}

#[cfg(target_os = "linux")]
fn detect_total_ram() -> u64 {
    let mut info: libc::sysinfo = unsafe { std::mem::zeroed() };
    unsafe { libc::sysinfo(&mut info) };
    info.totalram as u64 * info.mem_unit as u64
}

#[cfg(target_os = "windows")]
fn detect_total_ram() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    unsafe { GlobalMemoryStatusEx(&mut status) };
    status.ullTotalPhys
}

#[cfg(target_os = "ios")]
fn detect_total_ram() -> u64 {
    // iOS uses the same sysctl as macOS
    use std::mem;
    let mut size: u64 = 0;
    let mut len = mem::size_of::<u64>();
    let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
    unsafe {
        libc::sysctl(
            mib.as_ptr() as *mut _,
            2,
            &mut size as *mut u64 as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        );
    }
    size
}

#[cfg(target_os = "android")]
fn detect_total_ram() -> u64 {
    use std::mem;
    let mut info: libc::sysinfo = unsafe { mem::zeroed() };
    unsafe { libc::sysinfo(&mut info) };
    info.totalram as u64 * info.mem_unit as u64
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows",
    target_os = "ios",
    target_os = "android"
)))]
fn detect_total_ram() -> u64 {
    8 * 1024 * 1024 * 1024 // 8 GB fallback
}

fn detect_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Free space (bytes) available to an unprivileged user on the volume holding
/// `path`. Returns `None` if the volume can't be queried — callers (the
/// "will it fit?" verdict, which the host derives from this profile) then simply skip the
/// free-disk gate rather than guess. Local-only, no network.
///
/// This is the input the spec's §4 verdict folds in on top of Jan's RAM-only
/// `estimateModelFit` (Jan gates disk separately, in its download flow).
pub fn free_disk_bytes(path: &std::path::Path) -> Option<u64> {
    free_disk_bytes_impl(path)
}

#[cfg(unix)]
fn free_disk_bytes_impl(path: &std::path::Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: c_path is a valid NUL-terminated C string; stat is zero-init.
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return None;
    }
    // f_bavail = blocks free to an unprivileged process; f_frsize = block size.
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(windows)]
fn free_disk_bytes_impl(path: &std::path::Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut free_to_caller: u64 = 0;
    // SAFETY: wide is NUL-terminated; out-params are valid local u64s.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut free_to_caller,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 { None } else { Some(free_to_caller) }
}

#[cfg(not(any(unix, windows)))]
fn free_disk_bytes_impl(_path: &std::path::Path) -> Option<u64> {
    None
}

/// True when running inside the iOS Simulator. The sim's Metal driver
/// (`MTLSimDriver`) crashes GGML's Metal backend (whisper.cpp / llama.cpp), so
/// we fall back to CPU there — the sim runs on the host Mac's CPU, which handles
/// the small bundled models fine. Real iOS devices and macOS/Linux desktop are
/// unaffected. Detected via the `SIMULATOR_*` env vars the simulator injects
/// into every app process (absent on a physical device).
pub fn is_ios_simulator() -> bool {
    #[cfg(target_os = "ios")]
    {
        std::env::var_os("SIMULATOR_UDID").is_some()
            || std::env::var_os("SIMULATOR_DEVICE_NAME").is_some()
    }
    #[cfg(not(target_os = "ios"))]
    {
        false
    }
}

fn detect_gpu_backend() -> GpuBackend {
    // The iOS simulator's Metal driver crashes GGML — force CPU there.
    if is_ios_simulator() {
        return GpuBackend::Cpu;
    }

    #[cfg(feature = "backend-mlx")]
    return GpuBackend::Metal;

    #[cfg(all(not(feature = "backend-mlx"), feature = "metal"))]
    return GpuBackend::Metal;

    #[cfg(feature = "cuda")]
    return GpuBackend::Cuda;

    #[cfg(feature = "vulkan")]
    return GpuBackend::Vulkan;

    #[allow(unreachable_code)]
    GpuBackend::Cpu
}

// ---------------------------------------------------------------------------
// Runtime VRAM detection
// ---------------------------------------------------------------------------
//
// ## Mirrored reference
// Jan probes per-GPU VRAM and sums it. We mirror Jan's *values and decision
// logic* (sum-across-GPUs, MiB→bytes, the discrete vs. unified-memory split)
// faithfully, but with one deliberate transport divergence:
//
// - **Jan uses NVML** (the `nvml_wrapper` crate, an in-process binding to
//   `libnvidia-ml`) in
//   `jan/src-tauri/plugins/tauri-plugin-hardware/src/vendor/nvidia.rs:160-184`
//   — `create_gpu_info` reads `device.memory_info().total` (bytes) and converts
//   to MiB (`/ (1024 * 1024)`, line 174). Jan then **sums** every GPU's MiB and
//   converts back to bytes in
//   `jan/src-tauri/plugins/tauri-plugin-llamacpp/src/gguf/commands.rs:133-137`:
//   `gpus.iter().map(|g| g.total_memory * 1024 * 1024).sum::<u64>()`.
// - **Pio shells out to `nvidia-smi`** instead, per this phase's spec. NVML would
//   pull in the `nvml_wrapper` dep + a runtime `libnvidia-ml` load; `nvidia-smi`
//   ships with every NVIDIA driver and is what Jan's own users have installed
//   anyway. The *number* is identical (both read the driver's reported total
//   VRAM); only the access path differs. The parsing is factored into the pure,
//   unit-testable [`parse_nvidia_smi_total_mib`] so the contract is verified
//   without the binary. We keep Jan's sum-across-GPUs choice (matches its router
//   semantics: total fleet VRAM, not a single card).

/// Detect total dedicated VRAM (bytes), summed across all GPUs. Mirrors Jan's
/// sum-across-GPUs choice. Returns `0` for "no measured VRAM" (unified memory,
/// CPU, or an unprobeable backend) — see [`HardwareProfile::vram_bytes`].
fn detect_vram_bytes(backend: &GpuBackend) -> u64 {
    match backend {
        // Discrete NVIDIA: shell out to nvidia-smi (mirrors Jan's NVML read).
        GpuBackend::Cuda => detect_cuda_vram_bytes(),

        // Apple-Silicon unified memory: there is NO separate VRAM pool — CPU and
        // GPU share one physical bank. Jan models this by taking the
        // `gpus.is_empty()` branch (no discrete GPU) and reasoning over
        // `total_memory` directly in `check_apple_silicon_compatibility`
        // (`gguf/commands.rs:112-117, 195-235`). Pio's `fit.rs` does the same:
        // its Apple-Silicon branch keys off `GpuBackend::Metal` and budgets
        // against `total_ram_bytes`, never `vram_bytes`. So we report `0` by
        // convention here — the unified pool is already carried by
        // `total_ram_bytes`, and double-counting it as VRAM would mislead the
        // discrete-GPU fit branch and the flock router.
        GpuBackend::Metal => 0,

        // Vulkan: Jan reads Vulkan device memory via its hardware plugin, but the
        // heap-vs-VRAM mapping (DEVICE_LOCAL heaps, shared-vs-dedicated) is
        // genuinely ambiguous and not worth a fragile vulkaninfo scrape here.
        // TODO(vram): add a `vulkaninfo --json` heap-sum probe when a Vulkan
        // target needs precise routing. Honest `0` (= "unknown, use RAM") until
        // then — a documented gap, not a guess.
        GpuBackend::Vulkan => 0,

        // CPU-only: no GPU, no VRAM.
        GpuBackend::Cpu => 0,
    }
}

/// Query `nvidia-smi` for total VRAM and sum across GPUs. Returns `0` if the
/// binary is missing or the call fails (graceful: a CUDA build on a box without
/// the CLI degrades to "unknown", not a crash). The pure parse step is
/// [`parse_nvidia_smi_total_mib`].
///
/// Mirrors Jan reading every GPU's total and summing
/// (`gguf/commands.rs:133-137`); we issue the equivalent query the driver CLI
/// exposes: `--query-gpu=memory.total --format=csv,noheader,nounits` prints one
/// integer MiB per GPU, one per line.
fn detect_cuda_vram_bytes() -> u64 {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"]);

    // Don't pop a console window on Windows when launched from a GUI process.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    // Hard timeout: a wedged GPU driver — or a PATH-hijacked `nvidia-smi`
    // impostor that sleeps — must not stall `HardwareProfile::detect()`, which
    // runs on the boot / flock-gossip path. Degrade to "unknown" (0) on expiry.
    let stdout = match run_capturing_stdout_with_timeout(cmd, std::time::Duration::from_secs(2)) {
        Some(bytes) => bytes,
        None => {
            log::debug!(
                "nvidia-smi unavailable / failed / timed out; reporting VRAM as unknown (0)"
            );
            return 0;
        }
    };

    let stdout = String::from_utf8_lossy(&stdout);
    parse_nvidia_smi_total_mib(&stdout)
        .and_then(mib_total_to_bytes) // MiB → bytes (Jan: `* 1024 * 1024`, gguf/commands.rs:136)
        .unwrap_or(0)
}

/// Run `cmd` with a hard wall-clock timeout, returning its stdout bytes only on
/// a successful (zero-exit) completion within `timeout`; `None` on spawn
/// failure, non-zero exit, I/O error, or timeout (on which the child is
/// **killed and reaped** so no zombie / runaway probe survives).
///
/// std-only (no extra deps): spawn detached from our stdio, then poll
/// [`Child::try_wait`] to a deadline, sleeping briefly between polls. The poll
/// granularity adds negligible latency for a fast probe like `nvidia-smi`
/// (sub-100ms in practice) while bounding the worst case. Output is read only
/// after the child has exited — safe for the small single-integer-per-GPU
/// `nvidia-smi` output (a process that floods >64 KiB without exiting would
/// block on the pipe and be caught by the timeout instead).
fn run_capturing_stdout_with_timeout(
    mut cmd: std::process::Command,
    timeout: std::time::Duration,
) -> Option<Vec<u8>> {
    use std::process::Stdio;
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let mut buf = Vec::new();
                if let Some(mut out) = child.stdout.take() {
                    use std::io::Read;
                    out.read_to_end(&mut buf).ok()?;
                }
                return Some(buf);
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait(); // reap
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(15));
            }
            Err(_) => return None,
        }
    }
}

/// Convert a MiB total to a byte count, failing **closed** on overflow.
///
/// Jan does a naked `total_memory * 1024 * 1024` (`gguf/commands.rs:136`). That
/// is safe for *real* driver values but `nvidia-smi` stdout is, in principle,
/// driver/attacker-controlled: a single absurd line like `17592187092992`
/// parses as a valid `u64` MiB, and `* 1024 * 1024` then **wraps** a `u64` — e.g.
/// `17592187092992 MiB` wraps to exactly `1_099_511_627_776` bytes (a believable
/// *false* 1 TiB), and `17592186044416 MiB` wraps to `0`. A wrapped nonzero VRAM
/// is the dangerous case: it inflates [`HardwareProfile::vram_bytes`], makes a
/// Red model look Green in `fit.rs` branch 3 (over-commit on load), and is
/// gossiped to flock peers as `StaticFacts.vram_bytes` (mis-routes work).
///
/// So we use `checked_mul` and return `None` on overflow. The caller maps `None`
/// to `0` = "unknown / use RAM" — the same honest, conservative sentinel used
/// for missing/erroring `nvidia-smi`. Invariant: the VRAM byte count is either a
/// faithful conversion or `0`, **never** a wrapped/garbage nonzero.
fn mib_total_to_bytes(mib: u64) -> Option<u64> {
    mib.checked_mul(1024 * 1024)
}

/// Parse `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
/// output into total VRAM in **MiB**, summed across all GPUs. Pure — no I/O — so
/// it's unit-tested against captured driver output without an NVIDIA card.
///
/// Each non-empty line is one GPU's total VRAM as an integer count of MiB
/// (the `nounits` format strips the " MiB" suffix the driver would otherwise
/// print). Lines that don't parse as a `u64` are skipped (defensive against
/// driver-error rows like "[Insufficient Permissions]"). Returns `None` when no
/// line yields a number (empty / all-garbage) so the caller reports `0` =
/// "unknown" rather than a fake `0 MiB` total.
///
/// Sum-across-GPUs mirrors Jan (`gguf/commands.rs:133-137`).
pub fn parse_nvidia_smi_total_mib(out: &str) -> Option<u64> {
    let mut any = false;
    let mut sum: u64 = 0;
    for line in out.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(mib) = t.parse::<u64>() {
            sum = sum.saturating_add(mib);
            any = true;
        }
    }
    if any { Some(sum) } else { None }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_nonzero_ram() {
        let hw = HardwareProfile::detect();
        assert!(hw.total_ram_bytes > 0, "RAM detection failed");
        assert!(hw.cpu_cores > 0, "CPU core detection failed");
    }

    // ── VRAM detection: nvidia-smi output parsing ────────────────
    // Fixtures are real `nvidia-smi --query-gpu=memory.total
    // --format=csv,noheader,nounits` outputs (one integer MiB per GPU, one per
    // line). Mirrors Jan's sum-across-GPUs (gguf/commands.rs:133-137).

    #[test]
    fn nvidia_smi_single_gpu() {
        // RTX 3080 (10 GiB card reports 10240 MiB).
        assert_eq!(parse_nvidia_smi_total_mib("10240\n"), Some(10240));
    }

    #[test]
    fn nvidia_smi_single_gpu_no_trailing_newline() {
        assert_eq!(parse_nvidia_smi_total_mib("24576"), Some(24576));
    }

    #[test]
    fn nvidia_smi_multi_gpu_sums() {
        // Two A100 80GB (81920 MiB each) -> Jan sums them.
        assert_eq!(parse_nvidia_smi_total_mib("81920\n81920\n"), Some(163840));
        // Mixed cards: 24564 (4090) + 8192 (3070).
        assert_eq!(parse_nvidia_smi_total_mib("24564\n8192\n"), Some(32756));
    }

    #[test]
    fn nvidia_smi_skips_blank_lines() {
        assert_eq!(
            parse_nvidia_smi_total_mib("\n10240\n\n6144\n\n"),
            Some(16384)
        );
    }

    // ── MiB→bytes overflow: fail closed to "unknown", never wrap ─────────
    // `nvidia-smi` stdout is driver/attacker-controlled in principle. A single
    // absurd-but-valid-u64 MiB line must NOT wrap the `* 1024 * 1024` into a
    // garbage nonzero byte count that inflates HardwareProfile.vram_bytes (Red
    // model looks Green in fit branch 3 → over-commit; gossiped to flock peers).
    // Invariant: byte count is a faithful conversion or 0/None, never wrapped.

    #[test]
    fn mib_to_bytes_overflow_wraps_to_one_tib_must_be_none() {
        // 17_592_187_092_992 MiB * 1024 * 1024 wraps (mod 2^64) to exactly
        // 1_099_511_627_776 bytes (a believable FALSE 1 TiB). The naked multiply
        // would advertise that phantom VRAM. checked_mul → None (caller → 0).
        let evil_mib: u64 = 17_592_187_092_992;
        // Prove the wrapping multiply this replaces would have produced the lie:
        assert_eq!(
            evil_mib.wrapping_mul(1024 * 1024),
            1_099_511_627_776,
            "fixture must reproduce the wrap the fix prevents"
        );
        assert_eq!(
            mib_total_to_bytes(evil_mib),
            None,
            "overflowing MiB must fail closed, not wrap to a phantom 1 TiB"
        );
    }

    #[test]
    fn mib_to_bytes_overflow_wraps_to_zero_must_be_none() {
        // 17_592_186_044_416 MiB wraps to exactly 0 — also rejected (None),
        // even though 0 is the safe sentinel: the conversion is not faithful.
        let evil_mib: u64 = 17_592_186_044_416;
        assert_eq!(evil_mib.wrapping_mul(1024 * 1024), 0);
        assert_eq!(mib_total_to_bytes(evil_mib), None);
    }

    #[test]
    fn mib_to_bytes_largest_faithful_value_is_kept() {
        // The largest MiB that fits a u64 in bytes converts exactly (no false
        // None) — proves the guard is at the boundary, not over-eager.
        let max_ok: u64 = u64::MAX / (1024 * 1024); // 17_592_186_044_415
        assert_eq!(mib_total_to_bytes(max_ok), Some(max_ok * 1024 * 1024));
        assert_eq!(mib_total_to_bytes(max_ok + 1), None); // first overflow
    }

    #[test]
    fn mib_to_bytes_realistic_values_convert_exactly() {
        assert_eq!(mib_total_to_bytes(10240), Some(10_737_418_240)); // 10 GiB
        assert_eq!(mib_total_to_bytes(0), Some(0));
    }

    // A wedged `nvidia-smi` must not stall detect(): the timeout kills it and
    // returns None (→ VRAM unknown / 0) at ~the deadline, not the run length.
    #[cfg(unix)]
    #[test]
    fn run_with_timeout_kills_a_hung_command() {
        use std::time::{Duration, Instant};
        let mut cmd = std::process::Command::new("sleep");
        cmd.arg("10");
        let start = Instant::now();
        let out = run_capturing_stdout_with_timeout(cmd, Duration::from_millis(200));
        let elapsed = start.elapsed();
        assert!(
            out.is_none(),
            "a command exceeding the timeout must yield None"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "must return ~at the timeout, not wait for the command (took {elapsed:?})"
        );
    }

    #[cfg(unix)]
    #[test]
    fn run_with_timeout_captures_fast_command_stdout() {
        use std::time::Duration;
        let mut cmd = std::process::Command::new("printf");
        cmd.arg("10240");
        let out = run_capturing_stdout_with_timeout(cmd, Duration::from_secs(5));
        assert_eq!(out.as_deref(), Some(&b"10240"[..]));
    }

    #[cfg(unix)]
    #[test]
    fn run_with_timeout_rejects_nonzero_exit() {
        use std::time::Duration;
        let cmd = std::process::Command::new("false");
        let out = run_capturing_stdout_with_timeout(cmd, Duration::from_secs(5));
        assert!(
            out.is_none(),
            "a non-zero exit must yield None, not parsed garbage"
        );
    }

    #[test]
    fn nvidia_smi_empty_is_none() {
        // No GPUs / empty output -> None (caller reports 0 = "unknown").
        assert_eq!(parse_nvidia_smi_total_mib(""), None);
        assert_eq!(parse_nvidia_smi_total_mib("\n\n  \n"), None);
    }

    #[test]
    fn nvidia_smi_garbage_is_none() {
        // Driver-error rows / non-numeric junk yield no number -> None, never a
        // fake 0 MiB total.
        assert_eq!(parse_nvidia_smi_total_mib("No devices were found"), None);
        assert_eq!(
            parse_nvidia_smi_total_mib("[Insufficient Permissions]"),
            None
        );
        assert_eq!(parse_nvidia_smi_total_mib("N/A\nN/A"), None);
    }

    #[test]
    fn nvidia_smi_garbage_rows_skipped_valid_kept() {
        // One unreadable GPU + one readable: keep the number we can trust.
        assert_eq!(
            parse_nvidia_smi_total_mib("[Insufficient Permissions]\n10240\n"),
            Some(10240)
        );
    }

    // The ADR-0036 `vram-detect` captest that lived here stayed with the host:
    // it asserts against `fit::model_fit`, whose three-tier verdict is the
    // host's placement policy, not the engine's hardware read. The detector it
    // exercises (`parse_nvidia_smi_total_mib`) is covered by the parse tests above.

    #[test]
    fn detect_vram_cpu_and_metal_are_zero() {
        // CPU: no GPU. Metal: unified memory, by convention 0 (budget on RAM).
        assert_eq!(detect_vram_bytes(&GpuBackend::Cpu), 0);
        assert_eq!(detect_vram_bytes(&GpuBackend::Metal), 0);
        // Vulkan: documented honest 0 (probe deferred).
        assert_eq!(detect_vram_bytes(&GpuBackend::Vulkan), 0);
    }

    #[test]
    fn detect_returns_vram_field() {
        // On a Mac (Metal/CPU), vram_bytes is 0 by convention; the field exists.
        let hw = HardwareProfile::detect();
        // Never panics; on this host (no CUDA feature) VRAM is 0.
        let _ = hw.vram_bytes;
    }

    #[test]
    fn recommend_8gb_machine() {
        let hw = HardwareProfile {
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
            cpu_cores: 8,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        // 8GB - 4GB reserve = 4GB budget → should get 3B + 8K ctx
        assert!(rec.ctx_size >= 4_096);
        assert!(rec.estimated_ram_mb <= 4_096);
    }

    #[test]
    fn recommend_16gb_machine() {
        let hw = HardwareProfile {
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            cpu_cores: 10,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        // 16GB - 4GB = 12GB budget → should get 3B or 7B + 32K ctx
        assert!(
            rec.ctx_size >= 16_384,
            "expected ≥16K ctx, got {}",
            rec.ctx_size
        );
    }

    #[test]
    fn recommend_4gb_phone() {
        let hw = HardwareProfile {
            total_ram_bytes: 4 * 1024 * 1024 * 1024,
            cpu_cores: 4,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        // 4GB - 4GB reserve = 0 budget → fallback
        assert!(rec.ctx_size <= 4_096);
    }

    #[test]
    fn auto_tune_respects_model_size() {
        let hw = HardwareProfile {
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            cpu_cores: 10,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        // 2GB model, 36 layers → should get large context
        let ctx = auto_tune_ctx(&hw, 2_000_000_000, 36);
        assert!(ctx >= 8_192, "expected ≥8K ctx, got {}", ctx);
    }

    #[test]
    fn context_first_over_model_size() {
        // 6 GB machine: should prefer smaller model + larger context
        // over larger model + smaller context
        let hw = HardwareProfile {
            total_ram_bytes: 6 * 1024 * 1024 * 1024,
            cpu_cores: 4,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        // 6GB - 4GB = 2GB budget
        assert!(
            rec.ctx_size >= 4_096,
            "context-first should get ≥4K, got {}",
            rec.ctx_size
        );
    }

    // ── [Suki] Edge hardware profiles ────────────────────────────

    #[test]
    fn recommend_64gb_workstation() {
        let hw = HardwareProfile {
            total_ram_bytes: 64 * 1024 * 1024 * 1024,
            cpu_cores: 24,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        // 60GB budget → should get largest model + max context
        assert_eq!(rec.ctx_size, 40_960, "64GB should get max 40K context");
        assert!(rec.estimated_ram_mb > 4_000, "should pick a large model");
        assert!(rec.gpu_layers.is_none(), "Metal should auto-offload all");
    }

    #[test]
    fn recommend_2gb_phone() {
        let hw = HardwareProfile {
            total_ram_bytes: 2 * 1024 * 1024 * 1024,
            cpu_cores: 2,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        // 2GB - 4GB = saturating_sub → 0 budget → fallback
        assert_eq!(rec.ctx_size, 2_048, "zero budget should fallback to 2K");
        assert_eq!(rec.gpu_layers, Some(0), "fallback should be CPU-only");
        assert_eq!(rec.threads, 2, "fallback threads should be 2");
    }

    #[test]
    fn recommend_3gb_phone() {
        let hw = HardwareProfile {
            total_ram_bytes: 3 * 1024 * 1024 * 1024,
            cpu_cores: 4,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        // 3GB - 4GB reserve = 0 → fallback
        assert!(rec.ctx_size <= 2_048);
    }

    #[test]
    fn recommend_always_returns_valid() {
        // Even with impossible hardware, we get a valid recommendation
        let hw = HardwareProfile {
            total_ram_bytes: 0,
            cpu_cores: 1,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        assert!(!rec.model_id.is_empty());
        assert!(!rec.filename.is_empty());
        assert!(rec.ctx_size >= 2_048);
        assert!(rec.threads >= 2);
        assert!(rec.batch_size >= 128);
    }

    #[test]
    fn recommend_vulkan_offloads_all() {
        let hw = HardwareProfile {
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            cpu_cores: 8,
            gpu_backend: GpuBackend::Vulkan,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        assert!(rec.gpu_layers.is_none(), "Vulkan should auto-offload");
    }

    #[test]
    fn recommend_cuda_offloads_all() {
        let hw = HardwareProfile {
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            cpu_cores: 8,
            gpu_backend: GpuBackend::Cuda,
            vram_bytes: 0,
        };
        let rec = recommend_model(&hw);
        assert!(rec.gpu_layers.is_none(), "CUDA should auto-offload");
    }

    // ── [Suki] auto_tune_ctx edge cases ──────────────────────────

    #[test]
    fn auto_tune_huge_model_on_small_machine() {
        let hw = HardwareProfile {
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
            cpu_cores: 4,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        // 7GB model on 8GB machine → almost no room for KV
        let ctx = auto_tune_ctx(&hw, 7_000_000_000, 32);
        assert_eq!(ctx, 2_048, "should fallback to minimum");
    }

    #[test]
    fn auto_tune_tiny_model_on_big_machine() {
        let hw = HardwareProfile {
            total_ram_bytes: 64 * 1024 * 1024 * 1024,
            cpu_cores: 24,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        // 500MB model on 64GB → tons of room
        let ctx = auto_tune_ctx(&hw, 500_000_000, 28);
        assert!(ctx >= 32_768, "should get large context, got {}", ctx);
    }

    #[test]
    fn auto_tune_zero_layers_safe() {
        let hw = HardwareProfile {
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            cpu_cores: 8,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        // 0 layers → kv_per_token would be 0, capped to 1
        let ctx = auto_tune_ctx(&hw, 1_000_000_000, 0);
        assert!(ctx >= 2_048);
    }

    // ── [Maya] Phone inference budget math ───────────────────────

    #[test]
    fn inference_budget_desktop_reserve() {
        let hw = HardwareProfile {
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            cpu_cores: 8,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        // Desktop reserve = 4GB
        let budget = hw.inference_budget_bytes();
        assert_eq!(budget, 12 * 1024 * 1024 * 1024);
    }

    #[test]
    fn inference_budget_saturating_sub() {
        let hw = HardwareProfile {
            total_ram_bytes: 2 * 1024 * 1024 * 1024,
            cpu_cores: 2,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        // 2GB - 4GB reserve = saturating_sub → 0
        assert_eq!(hw.inference_budget_bytes(), 0);
    }

    #[test]
    fn total_ram_gb_rounds_down() {
        let hw = HardwareProfile {
            total_ram_bytes: 15_999_999_999,
            cpu_cores: 8,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        assert_eq!(hw.total_ram_gb(), 14); // rounds down from 14.9
    }

    // ── [Maya] GpuBackend serde ──────────────────────────────────

    #[test]
    fn gpu_backend_serde_roundtrip() {
        for backend in [
            GpuBackend::Metal,
            GpuBackend::Cuda,
            GpuBackend::Vulkan,
            GpuBackend::Cpu,
        ] {
            let json = serde_json::to_string(&backend).unwrap();
            let back: GpuBackend = serde_json::from_str(&json).unwrap();
            assert_eq!(backend, back);
        }
    }

    #[test]
    fn gpu_backend_snake_case() {
        let json = serde_json::to_string(&GpuBackend::Metal).unwrap();
        assert_eq!(json, "\"metal\"");
    }

    // ── Platform defaults ──────────────────────────────────────

    #[test]
    fn platform_defaults_low_ram() {
        let hw = HardwareProfile {
            total_ram_bytes: 4 * 1024 * 1024 * 1024,
            cpu_cores: 4,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        let s = super::platform_defaults(&hw);
        assert_eq!(s.system.ctx_size, Some(4096));
        assert_eq!(s.system.batch_size, Some(512));
        assert_eq!(s.system.gpu_layers, None);
    }

    #[test]
    fn platform_defaults_high_ram_gpu() {
        let hw = HardwareProfile {
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            cpu_cores: 10,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        let s = super::platform_defaults(&hw);
        assert_eq!(s.system.ctx_size, Some(16384));
        assert_eq!(s.system.batch_size, Some(2048));
        assert_eq!(s.system.gpu_layers, Some(99));
        assert_eq!(s.system.flash_attn, Some(true));
    }

    #[test]
    fn platform_defaults_threads_capped_at_8() {
        let hw = HardwareProfile {
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            cpu_cores: 24,
            gpu_backend: GpuBackend::Cpu,
            vram_bytes: 0,
        };
        let s = super::platform_defaults(&hw);
        assert_eq!(s.system.threads, Some(8));
        assert_eq!(s.system.threads_batch, Some(8));
    }

    #[test]
    fn platform_defaults_sampling_values() {
        let hw = HardwareProfile {
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
            cpu_cores: 4,
            gpu_backend: GpuBackend::Cuda,
            vram_bytes: 0,
        };
        let s = super::platform_defaults(&hw);
        assert_eq!(s.sampling.temperature, Some(0.7));
        assert_eq!(s.sampling.top_p, Some(0.9));
        assert_eq!(s.sampling.top_k, Some(40));
    }
}
