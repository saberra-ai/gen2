//! Model zoo — Phase D week 14.
//!
//! Maps a canonical model id (`gemma-4`, `llama-3.1-8b`, …) to the
//! per-platform bundle that should actually load: backend + source +
//! file + minimum RAM. Consumers ask the zoo for a model by id and get
//! back the right artifact for *this* device, with auto-quant selection
//! when the user doesn't specify.
//!
//! No Supabase. No cloud registry. The zoo is a JSON resource bundled
//! with the app and editable at `resources/models/zoo.json`. Future
//! enhancement: a signed override fetched on-demand from a public URL so
//! the model zoo can expand without shipping a new binary — but that's
//! opt-in and always falls back to the bundled manifest.
//!
//! # Platform detection
//!
//! We return the platform key the running device should use. Mapping:
//! - `macos` + `aarch64` → `macos_arm64`
//! - `macos` + `x86_64` → `macos_x86`
//! - `ios` (any arch) → `ios`
//! - `android` (any arch) → `android`
//! - `linux` + CUDA available → `linux_cuda`
//! - `linux` + no CUDA → `linux_cpu`
//! - `windows` (any) → `windows`
//!
//! CUDA detection is a best-effort check of env vars / common paths
//! because full runtime detection pulls in a heavy dep chain.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Embedded at compile time so the app always has a working zoo even
/// before the user downloads any overrides.
const BUNDLED_ZOO_JSON: &str = include_str!("../../../resources/models/zoo.json");

/// One per-platform entry for a model. Tells the gen2 loader exactly
/// which backend to instantiate and which file/repo to pull from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct PlatformBundle {
    /// Which gen2 backend to load: `"mlx"`, `"llamacpp"`, `"candle"`,
    /// `"executorch"`, `"onnx"`. Backend id strings match
    /// `gen2::backend::Backend::name()`.
    pub backend: String,
    /// HuggingFace repo id, URL, or local path. Examples:
    /// `"unsloth/gemma-4-E4B-it-UD-MLX-4bit"`,
    /// `"unsloth/gemma-4-E2B-it-GGUF"`.
    pub source: String,
    /// Specific file within `source`. `None` means "entire repo snapshot"
    /// (typical for MLX bundles shipped as a dir). For GGUF, this is the
    /// `.gguf` filename we want from the repo.
    #[serde(default)]
    pub file: Option<String>,
    /// Minimum physical RAM in MB the device must have to load this
    /// bundle. Selection logic skips bundles the device can't host.
    #[serde(default = "default_min_ram_mb")]
    pub min_ram_mb: u32,
}

fn default_min_ram_mb() -> u32 {
    4096
}

/// Full entry for one canonical model in the zoo.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelZooEntry {
    /// User-facing model name. Shown in the model picker UI.
    pub display_name: String,
    /// Family (`google/gemma`, `meta/llama`, etc.) — used for model-
    /// family-wide policy (e.g. a chat template per family).
    pub family: String,
    /// Default quantization tier. See [`QuantTier`] for the strings we
    /// accept. Used for model-level sanity checking; per-platform bundle
    /// already pins the exact quant.
    #[serde(default = "default_quant")]
    pub default_quant: String,
    /// Per-platform bundles, keyed by platform id (see module docs).
    pub platforms: HashMap<String, PlatformBundle>,
}

fn default_quant() -> String {
    "q4_k_m".to_string()
}

/// The manifest as loaded from disk or the bundled default.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelZoo {
    pub schema_version: u32,
    pub models: HashMap<String, ModelZooEntry>,
}

impl ModelZoo {
    /// Load from a user-supplied path. Used when the user points at an
    /// alternate zoo file (power-user override); falls back to [`Self::bundled`]
    /// on any error.
    pub fn load(path: &Path) -> Result<Self, ZooError> {
        let bytes = std::fs::read(path).map_err(ZooError::Io)?;
        serde_json::from_slice(&bytes).map_err(ZooError::Parse)
    }

    /// Load the compiled-in default manifest. Always succeeds unless the
    /// JSON is malformed at build time (CI catches this via the test
    /// below).
    pub fn bundled() -> Self {
        serde_json::from_str(BUNDLED_ZOO_JSON).expect("bundled zoo json is malformed — CI should catch this")
    }

    /// Convenience: look up a canonical model id.
    pub fn get(&self, model_id: &str) -> Option<&ModelZooEntry> {
        self.models.get(model_id)
    }
}

#[derive(Debug)]
pub enum ZooError {
    Io(std::io::Error),
    Parse(serde_json::Error),
}

impl std::fmt::Display for ZooError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "zoo load io: {e}"),
            Self::Parse(e) => write!(f, "zoo parse: {e}"),
        }
    }
}

impl std::error::Error for ZooError {}

// ── Platform detection ────────────────────────────────────────────────

/// Logical platform identifier matching the keys in `zoo.json`. Built
/// from compile-time `cfg` so there's no runtime surprise.
pub fn current_platform_id() -> &'static str {
    if cfg!(target_os = "ios") {
        return "ios";
    }
    if cfg!(target_os = "android") {
        return "android";
    }
    if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            return "macos_arm64";
        }
        return "macos_x86";
    }
    if cfg!(target_os = "windows") {
        return "windows";
    }
    if cfg!(target_os = "linux") {
        if linux_has_cuda() {
            return "linux_cuda";
        }
        return "linux_cpu";
    }
    // Unknown target — fall back to linux_cpu as the safest broadly-
    // compatible option (it expects Candle Rust-native fallback).
    "linux_cpu"
}

/// Heuristic CUDA presence check on Linux. Looks for common indicators
/// without pulling in `cuda-sys`. A false positive is safe (we'll fail
/// on load and downshift), false negative just means we don't use GPU
/// (also safe, just slower).
fn linux_has_cuda() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    if std::env::var_os("CUDA_PATH").is_some() || std::env::var_os("CUDA_HOME").is_some() {
        return true;
    }
    // Standard install path.
    Path::new("/usr/local/cuda/bin/nvcc").exists() || Path::new("/opt/cuda/bin/nvcc").exists()
}

// ── Selection ─────────────────────────────────────────────────────────

/// Pick the platform bundle for `model_id` on the current device, with
/// RAM budget awareness. If the primary platform bundle's `min_ram_mb`
/// exceeds the given RAM, we try other bundles in the same entry as
/// fallbacks (e.g. iOS might accept the Android GGUF if it has enough
/// RAM and a GGUF runtime — handled upstream by the backend loader).
///
/// Returns `None` when:
/// - the model id isn't in the zoo
/// - there's no bundle for this platform AND no viable fallback
pub fn select_for_device<'a>(
    zoo: &'a ModelZoo,
    model_id: &str,
    available_ram_mb: u32,
) -> Option<&'a PlatformBundle> {
    let entry = zoo.get(model_id)?;
    let platform = current_platform_id();

    // Primary: the bundle declared for this exact platform id.
    if let Some(primary) = entry.platforms.get(platform)
        && available_ram_mb >= primary.min_ram_mb
    {
        return Some(primary);
    }

    // Fallback: any bundle in this entry whose RAM budget fits. Prefer
    // backends likely to run on the current OS.
    let os_compatible = |b: &PlatformBundle| -> bool {
        match b.backend.as_str() {
            "mlx" => cfg!(target_os = "macos") || cfg!(target_os = "ios"),
            "llamacpp" | "candle" | "onnx" => true,
            "executorch" => cfg!(target_os = "ios") || cfg!(target_os = "android"),
            _ => true,
        }
    };

    entry
        .platforms
        .values()
        .filter(|b| os_compatible(b) && available_ram_mb >= b.min_ram_mb)
        // Prefer the bundle with the highest min_ram_mb that still fits —
        // heuristic for "largest quant the device can handle."
        .max_by_key(|b| b.min_ram_mb)
}

// ── Auto-quant helper ────────────────────────────────────────────────

/// Quantization tiers we understand. Ordered by memory footprint.
/// Used to pick a default when the user says "smartest this device can
/// run" or when a model has multiple quantization options in its
/// platform bundle list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum QuantTier {
    /// 2-bit (Q2_K, IQ2_*) — emergency only, noticeable quality loss
    Q2,
    /// 3-bit (Q3_K_M)
    Q3,
    /// 4-bit — Q4_K_M or equivalent MLX 4-bit. Default sweet spot.
    Q4KM,
    /// 5-bit (Q5_K_M)
    Q5,
    /// 6-bit (Q6_K)
    Q6,
    /// 8-bit (Q8_0)
    Q8,
    /// Full precision
    F16,
}

impl QuantTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q2 => "q2",
            Self::Q3 => "q3",
            Self::Q4KM => "q4_k_m",
            Self::Q5 => "q5",
            Self::Q6 => "q6",
            Self::Q8 => "q8",
            Self::F16 => "f16",
        }
    }
}

/// Pick the default quant tier for a device with `ram_mb` of RAM. Rules
/// are conservative — we'd rather underclaim and let the model run
/// comfortably than pick Q8 and thrash.
///
/// Bands (for a 7-8B parameter model):
/// - ≤  6 GB → Q4_K_M (Gemma-E2B / Phi-mini territory)
/// - ≤ 12 GB → Q4_K_M (typical 7B default)
/// - ≤ 20 GB → Q6_K (8B comfortable, 13B tight)
/// - ≤ 40 GB → Q8_0 (32B practical)
/// - >  40 GB → F16 (70B+ territory)
pub fn auto_quant_for_ram(ram_mb: u32) -> QuantTier {
    match ram_mb {
        r if r <= 6 * 1024 => QuantTier::Q4KM,
        r if r <= 12 * 1024 => QuantTier::Q4KM,
        r if r <= 20 * 1024 => QuantTier::Q6,
        r if r <= 40 * 1024 => QuantTier::Q8,
        _ => QuantTier::F16,
    }
}

/// Best-effort RAM detection. Reads from `sysinfo` when available in the
/// process; otherwise falls back to a safe default. Callers that need
/// authoritative values should query their platform's system APIs.
pub fn detect_ram_mb() -> u32 {
    // `sysinfo` crate is on the `backend-*` feature paths but not core;
    // avoid a heavy dep just for this. Fall through to a platform sniff.
    #[cfg(target_os = "macos")]
    {
        // sysctl hw.memsize. Avoid libc dep — shell out, but wrapped in
        // a best-effort. This is cheap and runs at most once per session.
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            && out.status.success()
            && let Ok(s) = std::str::from_utf8(&out.stdout)
            && let Ok(bytes) = s.trim().parse::<u64>()
        {
            return (bytes / (1024 * 1024)) as u32;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb: u64 = rest
                        .split_whitespace()
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    if kb > 0 {
                        return (kb / 1024) as u32;
                    }
                }
            }
        }
    }
    // Safe default when detection fails.
    8 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_zoo_parses() {
        let zoo = ModelZoo::bundled();
        assert!(zoo.schema_version >= 1);
        assert!(!zoo.models.is_empty(), "bundled zoo must contain at least one model");
    }

    #[test]
    fn gemma_4_has_all_platforms() {
        let zoo = ModelZoo::bundled();
        let gemma = zoo.get("gemma-4").expect("gemma-4 bundled");
        for plat in [
            "macos_arm64",
            "macos_x86",
            "ios",
            "android",
            "linux_cuda",
            "linux_cpu",
            "windows",
        ] {
            assert!(
                gemma.platforms.contains_key(plat),
                "gemma-4 must define platform `{plat}`"
            );
        }
    }

    #[test]
    fn gemma_4_macos_arm64_routes_to_mlx_e4b() {
        let zoo = ModelZoo::bundled();
        let gemma = zoo.get("gemma-4").expect("gemma-4 bundled");
        let bundle = gemma.platforms.get("macos_arm64").unwrap();
        assert_eq!(bundle.backend, "mlx");
        assert!(bundle.source.contains("E4B"));
        assert!(bundle.source.contains("MLX"));
    }

    #[test]
    fn gemma_4_ios_routes_to_mlx_e2b() {
        let zoo = ModelZoo::bundled();
        let gemma = zoo.get("gemma-4").expect("gemma-4 bundled");
        let bundle = gemma.platforms.get("ios").unwrap();
        assert_eq!(bundle.backend, "mlx");
        assert!(bundle.source.contains("E2B"), "mobile = smaller model");
    }

    #[test]
    fn gemma_4_android_routes_to_gguf() {
        let zoo = ModelZoo::bundled();
        let gemma = zoo.get("gemma-4").expect("gemma-4 bundled");
        let bundle = gemma.platforms.get("android").unwrap();
        assert_eq!(bundle.backend, "llamacpp");
        assert!(bundle.source.contains("E2B"));
        assert!(bundle.source.contains("GGUF"));
    }

    #[test]
    fn gemma_4_linux_cpu_routes_to_candle_safetensors() {
        let zoo = ModelZoo::bundled();
        let gemma = zoo.get("gemma-4").expect("gemma-4 bundled");
        let bundle = gemma.platforms.get("linux_cpu").unwrap();
        assert_eq!(bundle.backend, "candle");
        assert!(bundle.source.ends_with("safetensors"));
    }

    #[test]
    fn current_platform_id_is_real() {
        let p = current_platform_id();
        assert!(
            matches!(
                p,
                "macos_arm64"
                    | "macos_x86"
                    | "ios"
                    | "android"
                    | "linux_cuda"
                    | "linux_cpu"
                    | "windows"
            ),
            "unknown platform id: {p}"
        );
    }

    #[test]
    fn select_for_device_respects_ram_budget() {
        let zoo = ModelZoo::bundled();
        // Fake a very low-RAM device: nothing should match.
        let bundle = select_for_device(&zoo, "gemma-4", 512);
        assert!(bundle.is_none(), "512 MB device can't run any gemma-4 bundle");
    }

    #[test]
    fn select_for_device_returns_bundle_with_enough_ram() {
        let zoo = ModelZoo::bundled();
        let bundle = select_for_device(&zoo, "gemma-4", 32 * 1024)
            .expect("32 GB device has more than enough for gemma-4");
        assert!(bundle.min_ram_mb <= 32 * 1024);
    }

    #[test]
    fn select_unknown_model_returns_none() {
        let zoo = ModelZoo::bundled();
        assert!(select_for_device(&zoo, "nonexistent-model-xyz", 32 * 1024).is_none());
    }

    #[test]
    fn auto_quant_bands_monotonic() {
        // Higher RAM → higher-or-equal quant tier.
        assert!(auto_quant_for_ram(4096) <= auto_quant_for_ram(12 * 1024));
        assert!(auto_quant_for_ram(12 * 1024) <= auto_quant_for_ram(24 * 1024));
        assert!(auto_quant_for_ram(24 * 1024) <= auto_quant_for_ram(64 * 1024));
    }

    #[test]
    fn auto_quant_low_ram_picks_4bit() {
        assert_eq!(auto_quant_for_ram(4 * 1024), QuantTier::Q4KM);
        assert_eq!(auto_quant_for_ram(8 * 1024), QuantTier::Q4KM);
    }

    #[test]
    fn auto_quant_high_ram_picks_f16() {
        assert_eq!(auto_quant_for_ram(64 * 1024), QuantTier::F16);
        assert_eq!(auto_quant_for_ram(128 * 1024), QuantTier::F16);
    }

    #[test]
    fn detect_ram_mb_returns_nonzero() {
        let ram = detect_ram_mb();
        assert!(ram >= 1024, "any real device should report ≥1 GB RAM, got {ram} MB");
    }
}
