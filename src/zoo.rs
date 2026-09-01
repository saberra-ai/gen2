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
const BUNDLED_ZOO_JSON: &str = include_str!("../resources/models/zoo.json");

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
        serde_json::from_str(BUNDLED_ZOO_JSON)
            .expect("bundled zoo json is malformed — CI should catch this")
    }

    /// Convenience: look up a canonical model id.
    pub fn get(&self, model_id: &str) -> Option<&ModelZooEntry> {
        self.models.get(model_id)
    }

    /// Recommended thinking-mode default for the toggle UI.
    ///
    /// - `None` → model has no thinking mode; frontend should hide the toggle.
    /// - `Some(ThinkingMode::Auto)` → toggleable, default position is Auto
    ///   (Qwen3.5, Gemma4, GLM4).
    /// - `Some(ThinkingMode::On)` → always reasoning (DeepSeek-R1, gpt-oss);
    ///   toggle exists but reads as "always on" — UI may render it disabled
    ///   or label it accordingly.
    ///
    /// Returns `None` when the model id isn't in the zoo.
    pub fn recommended_thinking(&self, model_id: &str) -> Option<crate::generation::ThinkingMode> {
        use crate::generation::ThinkingMode;
        let entry = self.get(model_id)?;
        match entry.defaults().thinking {
            ThinkingDefault::Unsupported => None,
            ThinkingDefault::Off => Some(ThinkingMode::Off),
            ThinkingDefault::On => Some(ThinkingMode::On),
            ThinkingDefault::Auto => Some(ThinkingMode::Auto),
        }
    }

    /// Recommended sampling settings for a model — the family-derived
    /// "Load recommended" payload. Used as the runtime default when a
    /// model loads with no user overrides, and as the value the
    /// Settings UI's "Load recommended" button writes back.
    ///
    /// Returns `None` when the model id isn't in the zoo.
    pub fn recommended_sampling(&self, model_id: &str) -> Option<crate::engine::SamplingSettings> {
        let entry = self.get(model_id)?;
        let d = entry.defaults();
        Some(crate::engine::SamplingSettings {
            temperature: Some(d.temperature),
            top_p: d.top_p,
            top_k: d.top_k,
            min_p: d.min_p,
            penalty_repeat: d.repetition_penalty,
            penalty_present: d.presence_penalty,
            ..Default::default()
        })
    }
}

impl ModelZooEntry {
    /// Parse the `family` string into a [`ModelFamily`] enum.
    ///
    /// **Strict** match against [`ModelFamily::as_str`] values: anything
    /// that doesn't match returns [`ModelFamily::Unknown`]. We deliberately
    /// do NOT fall back to [`ModelFamily::detect`] from the display_name
    /// here, because that silently rescued typos (e.g. `family: "qwen35"`
    /// on a "Qwen 3.5 …" entry would resolve correctly via the display
    /// name and the typo never surfaced). The
    /// `every_zoo_entry_resolves_to_known_family` test makes the typo
    /// loud at build time.
    pub fn family_kind(&self) -> ModelFamily {
        match self.family.as_str() {
            "llama3" => ModelFamily::Llama3,
            "qwen2.5" => ModelFamily::Qwen25,
            "qwen3.5" => ModelFamily::Qwen35,
            "gemma2" => ModelFamily::Gemma2,
            "gemma3" => ModelFamily::Gemma3,
            "gemma4" => ModelFamily::Gemma4,
            "mistral-small-3" => ModelFamily::MistralSmall3,
            "glm4" => ModelFamily::Glm4,
            "minimax-m2" => ModelFamily::MiniMaxM2,
            "deepseek-coder-v2" => ModelFamily::DeepSeekCoderV2,
            "deepseek-r1" => ModelFamily::DeepSeekR1,
            "smollm2" => ModelFamily::SmolLm2,
            "gpt-oss" => ModelFamily::GptOss,
            _ => ModelFamily::Unknown,
        }
    }

    /// Sampling/template/thinking defaults for this entry, sourced from
    /// upstream best-practices (see [`ModelFamily::defaults`]).
    pub fn defaults(&self) -> FamilyDefaults {
        self.family_kind().defaults()
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

// ── Family taxonomy + sampling defaults ──────────────────────────────
//
// The platform/bundle layer above answers "which file do I load on this
// device". This layer answers "what sampling/template config does this
// model want when I run it". They're independent — a Pio user with their
// own GGUF that isn't in `zoo.json` still benefits from family-derived
// defaults via runtime detection from GGUF arch + repo hints.
//
// Source for every default below: upstream README "Best Practices" /
// "Recommended Settings" / "Inference Parameters" section, falling back
// to upstream `generation_config.json` defaults, falling back to
// family convention (e.g. Meta's published Llama-3 inference recipe).
// See `pio-core/bench-models/inference-test-matrix.toml` for per-model
// citations and any deltas from family defaults.

/// Coarse model family. One variant per "sampling-distinct" group —
/// models that share a common Best Practices recipe collapse together.
/// Architectures with the same defaults but different sub-versions
/// (e.g. Qwen3.5 dense + Qwen3.5-A3B MoE + Qwen3.6) are unified;
/// architectures with the same arch tag but different recipes (e.g.
/// Qwen2.5 vs Qwen3.5) are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    /// Meta Llama 3.x family (3.1, 3.2 instruct).
    Llama3,
    /// Qwen 2.5 family (incl Coder variants — distinct recipe from 3.x).
    Qwen25,
    /// Qwen 3.5 / 3.6 family (incl A3B MoE). Has /think /no_think toggle.
    Qwen35,
    /// google/gemma-2 (older). No thinking mode.
    Gemma2,
    /// google/gemma-3 (incl QAT 4-bit variants). No thinking mode.
    Gemma3,
    /// google/gemma-4 (incl 26B-A4B MoE). Has `<|think|>` control tokens.
    Gemma4,
    /// mistralai/Mistral-Small-3.x — distinct low-temperature recipe.
    MistralSmall3,
    /// zai-org/GLM-4.x family. Has Preserved Thinking mode.
    Glm4,
    /// MiniMaxAI/MiniMax-M2.x.
    MiniMaxM2,
    /// deepseek-ai/DeepSeek-Coder-V2-* (Lite + full). Not a reasoning model.
    DeepSeekCoderV2,
    /// deepseek-ai/DeepSeek-R1 + R1-distill-* (any base). Reasoning model.
    DeepSeekR1,
    /// HuggingFaceTB/SmolLM2-*.
    SmolLm2,
    /// openai/gpt-oss-* (uses harmony chat format).
    GptOss,
    /// Couldn't identify — fall back to conservative cross-family defaults.
    Unknown,
}

/// Which chat-template parser/family to use. Matches what
/// `gen2/backend/common/chat_template.rs` expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateKind {
    /// Llama 3 format (`<|begin_of_text|><|start_header_id|>…<|eot_id|>`).
    Llama3,
    /// ChatML (`<|im_start|>role…<|im_end|>`). Used by Qwen 2.x, SmolLM2.
    ChatMl,
    /// Qwen 3 with /think /no_think toggles.
    Qwen3,
    /// Gemma 2/3 (`<start_of_turn>role\n…<end_of_turn>`).
    Gemma,
    /// Gemma 4 (adds `<|think|>` control tokens for thinking mode).
    Gemma4,
    /// Mistral [INST]…[/INST].
    Mistral,
    /// DeepSeek (varies by sub-family — auto-detect from tokenizer).
    DeepSeek,
    /// GLM 4 chat format.
    Glm4,
    /// MiniMax chat format.
    MiniMax,
    /// OpenAI harmony format (gpt-oss).
    Harmony,
    /// Don't override — let `chat_template.rs` parse from the model's
    /// own `tokenizer_config.json` chat_template field.
    Auto,
}

/// Sampling + template + thinking defaults sourced from upstream best
/// practices. Optional fields (`top_k`, `presence_penalty`, etc.) are
/// `None` when the family's recipe doesn't specify a value — callers
/// should leave the corresponding `GenSpec` field unset rather than
/// defaulting to 0.
#[derive(Debug, Clone, Copy)]
pub struct FamilyDefaults {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    /// Repetition penalty (`1.0` = no penalty).
    pub repetition_penalty: Option<f32>,
    /// Presence penalty — Qwen3.5 specifically calls for `2.0` in its
    /// non-thinking text recipe; most families leave this unset.
    pub presence_penalty: Option<f32>,
    pub template_kind: TemplateKind,
    /// `On` for always-reasoning models (DeepSeek-R1, gpt-oss),
    /// `Auto` for toggleable families (Qwen3.5, Gemma4, GLM4),
    /// `Off` for the rest.
    pub thinking: ThinkingDefault,
}

/// Family-level thinking-mode classification. Distinct from
/// [`gen2::generation::thinking::ThinkingMode`] in one critical way:
/// adds `Unsupported` to express "this model has no thinking mode at
/// all — UI should hide the toggle". The runtime `ThinkingMode` enum
/// (Off/On/Auto) doesn't need that variant because by the time you're
/// at runtime you're already known to support thinking.
///
/// Cycle-5 originally collapsed Unsupported and Off into one variant;
/// the model-card check during cycle 6 caught the bug — Llama3
/// (no thinking) and Qwen3.5-9B (supports it, defaults off) were
/// indistinguishable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingDefault {
    /// Model has no thinking mode in its chat template; the UI toggle
    /// must be hidden. Llama3, Qwen2.5, Gemma2/3, Mistral, MiniMax,
    /// DeepSeek-Coder-V2, SmolLM2 — none of these emit reasoning.
    Unsupported,
    /// Toggleable; default position is OFF. Qwen3.5/3.6 (template
    /// hard-codes `enable_thinking: false` for the 9B+ sizes), Gemma4
    /// (`<|think|>` is a *trigger* token — opt-in), GLM-4.7-Flash
    /// (README: "please turn ON Preserved Thinking mode for ...").
    Off,
    /// Toggleable; default position is ON. DeepSeek-R1 (always
    /// reasons; the model's whole identity), gpt-oss (harmony format
    /// always emits the reasoning channel).
    On,
    /// Toggleable; let the runtime/template decide. No current family
    /// uses Auto post-correction, but kept as a valid state for
    /// future entries that genuinely don't pin a default.
    Auto,
}

impl ModelFamily {
    /// Best-effort detection from GGUF `general.architecture` (lowercase)
    /// plus an optional repo/path hint (the user's HF repo id, model
    /// directory name, or display id). Both signals together resolve
    /// ambiguities the arch tag alone can't:
    ///
    /// - `arch="qwen2"` could be Qwen2.5 OR Qwen2 base — repo hint
    ///   distinguishes.
    /// - `arch="llama"` covers Llama-3.x AND DeepSeek-R1-distill-Llama.
    /// - `arch="qwen3"` covers Qwen3.5 AND DeepSeek-R1-distill-Qwen3.
    /// - Some quantizers strip the version suffix, leaving `arch="gemma"`
    ///   for both Gemma-2 and Gemma-3.
    pub fn detect(arch: Option<&str>, repo_hint: Option<&str>) -> Self {
        let arch = arch.unwrap_or("").to_ascii_lowercase();
        let repo = repo_hint.unwrap_or("").to_ascii_lowercase();

        // Stage 1: repo-specific overrides. These run BEFORE arch-based
        // matching because repo names are precise where arch tags are
        // generic — e.g. SmolLM2 ships as `arch=llama`, R1 distills as
        // whatever their base is. Repo wins.

        // R1 distills (regardless of base — Llama or Qwen3).
        if repo.contains("deepseek-r1") || repo.contains("r1-distill") || repo.contains("r1-0528") {
            return Self::DeepSeekR1;
        }

        // Brand-specific overrides where arch alone misroutes:
        if repo.contains("smollm2") {
            return Self::SmolLm2;
        }
        if repo.contains("gemma-4") {
            return Self::Gemma4;
        }
        if repo.contains("gemma-3") {
            return Self::Gemma3;
        }
        if repo.contains("gemma-2") {
            return Self::Gemma2;
        }
        if repo.contains("qwen3.5") || repo.contains("qwen3.6") {
            return Self::Qwen35;
        }
        if repo.contains("qwen2.5") {
            return Self::Qwen25;
        }
        if repo.contains("mistral-small-3") {
            return Self::MistralSmall3;
        }
        if repo.contains("glm-4") {
            return Self::Glm4;
        }
        if repo.contains("minimax-m2") {
            return Self::MiniMaxM2;
        }
        if repo.contains("deepseek-coder-v2") {
            return Self::DeepSeekCoderV2;
        }
        if repo.contains("gpt-oss") {
            return Self::GptOss;
        }

        // Stage 2: arch-based fallback (when no repo hint distinguishes).
        if arch.starts_with("gemma4") {
            return Self::Gemma4;
        }
        if arch.starts_with("gemma3") {
            return Self::Gemma3;
        }
        if arch.starts_with("gemma2") {
            return Self::Gemma2;
        }
        if arch.starts_with("gemma") {
            // Unversioned. Pick Gemma-2 (oldest still shipped) since
            // Gemma-3/4 quantizers usually keep the suffix.
            return Self::Gemma2;
        }
        if arch.starts_with("qwen3") {
            return Self::Qwen35;
        }
        if arch.starts_with("qwen2") {
            return Self::Qwen25;
        }
        if arch == "llama" || arch.starts_with("llama") {
            return Self::Llama3;
        }
        if arch.starts_with("mistral") {
            // Older Mistral (7B-Instruct etc.) — falls back to Llama3
            // defaults rather than the Small-3 low-temp recipe.
            return Self::Llama3;
        }
        if arch.starts_with("glm") {
            return Self::Glm4;
        }
        if arch.starts_with("minimax") {
            return Self::MiniMaxM2;
        }
        if arch.starts_with("deepseek2") {
            return Self::DeepSeekCoderV2;
        }
        if arch.starts_with("gpt_oss") || arch.starts_with("gpt-oss") {
            return Self::GptOss;
        }

        Self::Unknown
    }

    /// Sampling + template defaults for this family. Values traceable to
    /// upstream README / generation_config.json — see the matrix at
    /// `pio-core/bench-models/inference-test-matrix.toml` for citations.
    pub fn defaults(self) -> FamilyDefaults {
        match self {
            Self::Llama3 => FamilyDefaults {
                temperature: 0.6,
                top_p: Some(0.9),
                top_k: Some(50),
                min_p: None,
                repetition_penalty: Some(1.0),
                presence_penalty: None,
                template_kind: TemplateKind::Llama3,
                thinking: ThinkingDefault::Unsupported,
            },
            Self::Qwen25 => FamilyDefaults {
                // Qwen2.5 generation_config.json defaults.
                temperature: 0.7,
                top_p: Some(0.8),
                top_k: Some(20),
                min_p: None,
                repetition_penalty: Some(1.05),
                presence_penalty: None,
                template_kind: TemplateKind::ChatMl,
                thinking: ThinkingDefault::Unsupported,
            },
            Self::Qwen35 => FamilyDefaults {
                // Qwen3.5/3.6 upstream README "Best Practices" —
                // "Instruct (or non-thinking) mode for general tasks".
                // Source verified live 2026-05-02 against
                // `Qwen/Qwen3.5-9B/raw/main/README.md`. ADR-0015 cites
                // the same recipe via the Qwen3-0.6B model card.
                // Earlier cycle baked thinking-mode-ish values
                // (temp=1.0, top_p=1.0, presence=2.0) — corrected.
                temperature: 0.7,
                top_p: Some(0.8),
                top_k: Some(20),
                min_p: Some(0.0),
                repetition_penalty: Some(1.0),
                presence_penalty: Some(1.5),
                template_kind: TemplateKind::Qwen3,
                // Qwen3.5/3.6 9B+ chat templates hard-code
                // `enable_thinking: false`; smaller sizes (0.8B, 2B)
                // default true but using family-level Off matches the
                // most common case and what upstream's Best Practices
                // explicitly recommends ("non-thinking mode for text").
                thinking: ThinkingDefault::Off,
            },
            Self::Gemma2 => FamilyDefaults {
                temperature: 1.0,
                top_p: Some(0.95),
                top_k: Some(64),
                min_p: None,
                repetition_penalty: Some(1.0),
                presence_penalty: None,
                template_kind: TemplateKind::Gemma,
                thinking: ThinkingDefault::Unsupported,
            },
            Self::Gemma3 => FamilyDefaults {
                temperature: 1.0,
                top_p: Some(0.95),
                top_k: Some(64),
                min_p: None,
                repetition_penalty: Some(1.0),
                presence_penalty: None,
                template_kind: TemplateKind::Gemma,
                thinking: ThinkingDefault::Unsupported,
            },
            Self::Gemma4 => FamilyDefaults {
                // Gemma 4 README: "standardized sampling configuration".
                temperature: 1.0,
                top_p: Some(0.95),
                top_k: Some(64),
                min_p: None,
                repetition_penalty: Some(1.0),
                presence_penalty: None,
                template_kind: TemplateKind::Gemma4,
                // Gemma 4 thinking is opt-in via `<|think|>` *trigger*
                // tokens — the model card frames it as a feature you
                // enable, so default Off matches the documented UX.
                thinking: ThinkingDefault::Off,
            },
            Self::MistralSmall3 => FamilyDefaults {
                // Mistral upstream Note 1: "We recommend a relatively low
                // temperature, such as temperature=0.15."
                temperature: 0.15,
                top_p: Some(1.0),
                top_k: None,
                min_p: None,
                repetition_penalty: None,
                presence_penalty: None,
                template_kind: TemplateKind::Mistral,
                thinking: ThinkingDefault::Unsupported,
            },
            Self::Glm4 => FamilyDefaults {
                // GLM-4.7-Flash README Terminal/SWE benchmark recipe.
                temperature: 0.7,
                top_p: Some(1.0),
                top_k: None,
                min_p: None,
                repetition_penalty: None,
                presence_penalty: None,
                template_kind: TemplateKind::Glm4,
                // GLM-4.7-Flash README: "for multi-turn agentic tasks
                // please TURN ON Preserved Thinking mode" — explicit
                // opt-in language, so Off is the documented default.
                thinking: ThinkingDefault::Off,
            },
            Self::MiniMaxM2 => FamilyDefaults {
                // MiniMax-M2.7 README "Inference Parameters".
                temperature: 1.0,
                top_p: Some(0.95),
                top_k: Some(40),
                min_p: None,
                repetition_penalty: None,
                presence_penalty: None,
                template_kind: TemplateKind::MiniMax,
                thinking: ThinkingDefault::Unsupported,
            },
            Self::DeepSeekCoderV2 => FamilyDefaults {
                temperature: 0.6,
                top_p: Some(0.95),
                top_k: Some(50),
                min_p: None,
                repetition_penalty: Some(1.0),
                presence_penalty: None,
                template_kind: TemplateKind::DeepSeek,
                thinking: ThinkingDefault::Unsupported,
            },
            Self::DeepSeekR1 => FamilyDefaults {
                // R1 family — always reasoning. Template comes from the
                // distill base (Qwen3 or Llama3); use Auto so the model's
                // own tokenizer_config drives template selection.
                temperature: 0.6,
                top_p: Some(0.95),
                top_k: Some(50),
                min_p: None,
                repetition_penalty: Some(1.0),
                presence_penalty: None,
                template_kind: TemplateKind::Auto,
                thinking: ThinkingDefault::On,
            },
            Self::SmolLm2 => FamilyDefaults {
                // gen_config.json sets temp=2 which is wrong for chat —
                // use the HF blog's recommended 0.6.
                temperature: 0.6,
                top_p: Some(0.95),
                top_k: Some(50),
                min_p: None,
                repetition_penalty: Some(1.2),
                presence_penalty: None,
                template_kind: TemplateKind::ChatMl,
                thinking: ThinkingDefault::Unsupported,
            },
            Self::GptOss => FamilyDefaults {
                temperature: 1.0,
                top_p: Some(1.0),
                top_k: None,
                min_p: None,
                repetition_penalty: None,
                presence_penalty: None,
                template_kind: TemplateKind::Harmony,
                thinking: ThinkingDefault::On,
            },
            Self::Unknown => FamilyDefaults {
                // Conservative cross-family fallback. Slightly cooler
                // than the Qwen3.5 recipe (most popular family), no
                // unusual penalties.
                temperature: 0.7,
                top_p: Some(0.95),
                top_k: Some(40),
                min_p: None,
                repetition_penalty: None,
                presence_penalty: None,
                template_kind: TemplateKind::Auto,
                // Don't pretend to know what unknown families need;
                // hide the toggle until someone classifies the family.
                thinking: ThinkingDefault::Unsupported,
            },
        }
    }

    /// Whether a backend that renders this family's chat template should
    /// pass `enable_thinking=true` when it has *no* runtime thinking
    /// toggle to consult (the llama-cpp prompt path). This is a
    /// chat-template rendering concern, distinct from the UI-facing
    /// [`FamilyDefaults::thinking`] default:
    ///
    /// Gemma 4's IT template gates its `<|think|>` trigger tokens on
    /// `enable_thinking`; without it the rendered prompt omits the
    /// `<|think|>\n` marker and the think-trained model emits `<turn|>`
    /// (EOS, token 106) inside markdown instead of answering. We mirror
    /// llama-cli's `--jinja` behavior of defaulting `enable_thinking=true`
    /// for the Gemma family. Every other family renders correctly without
    /// the flag, so they default `false`.
    pub fn default_enable_thinking(self) -> bool {
        matches!(self, Self::Gemma4)
    }

    /// Streaming reasoning-channel markers for this family — the
    /// open/close string pairs [`crate::generation::ReplyStateMachine`]
    /// scans for to split visible content from the reasoning channel.
    ///
    /// This is the single owner of "which channel markers does family X
    /// use". [`crate::generation::ChannelMarkers::gemma4`] /
    /// [`qwen3_deepseek`](crate::generation::ChannelMarkers::qwen3_deepseek)
    /// and [`ChannelMarkers::from_architecture`](crate::generation::ChannelMarkers::from_architecture)
    /// read from here so the marker set can't drift from the family's
    /// `template_kind` / `thinking` defaults above.
    ///
    /// Only the reasoning-channel families return non-empty markers:
    /// - Gemma 4 emits `<|channel>thought…<channel|>` when its `<|think|>`
    ///   control tokens fire.
    /// - Qwen3.5/3.6 and DeepSeek-R1 both use the `<think>…</think>` text
    ///   form (R1 distills inherit their base's `<think>` tags).
    ///
    /// Every other family (including DeepSeek-Coder-V2, which is *not* a
    /// reasoning model) has no channel and returns
    /// [`ChannelMarkers::none`](crate::generation::ChannelMarkers::none).
    pub fn channel_markers(self) -> crate::generation::ChannelMarkers {
        use crate::generation::ChannelMarkers;
        match self {
            Self::Gemma4 => ChannelMarkers::gemma4(),
            Self::Qwen35 | Self::DeepSeekR1 => ChannelMarkers::qwen3_deepseek(),
            _ => ChannelMarkers::none(),
        }
    }

    /// Stable string id, suitable for logs / metrics / `zoo.json`'s
    /// `family` field.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Llama3 => "llama3",
            Self::Qwen25 => "qwen2.5",
            Self::Qwen35 => "qwen3.5",
            Self::Gemma2 => "gemma2",
            Self::Gemma3 => "gemma3",
            Self::Gemma4 => "gemma4",
            Self::MistralSmall3 => "mistral-small-3",
            Self::Glm4 => "glm4",
            Self::MiniMaxM2 => "minimax-m2",
            Self::DeepSeekCoderV2 => "deepseek-coder-v2",
            Self::DeepSeekR1 => "deepseek-r1",
            Self::SmolLm2 => "smollm2",
            Self::GptOss => "gpt-oss",
            Self::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_zoo_parses() {
        let zoo = ModelZoo::bundled();
        assert!(zoo.schema_version >= 1);
        assert!(
            !zoo.models.is_empty(),
            "bundled zoo must contain at least one model"
        );
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
    fn gemma_4_linux_cpu_routes_to_a_backend_that_can_generate() {
        // It used to route to Candle and safetensors. Candle's `start_session`
        // returns `Unimplemented` and no feature bundle compiles it in, so a
        // Linux CPU box selecting gemma-4 downloaded weights and then could not
        // generate from them. It now takes the same GGUF as every other
        // non-Apple platform.
        let zoo = ModelZoo::bundled();
        let gemma = zoo.get("gemma-4").expect("gemma-4 bundled");
        let bundle = gemma.platforms.get("linux_cpu").unwrap();
        assert_eq!(bundle.backend, "llamacpp");
        assert!(bundle.source.contains("GGUF"));
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
        assert!(
            bundle.is_none(),
            "512 MB device can't run any gemma-4 bundle"
        );
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
        assert!(
            ram >= 1024,
            "any real device should report ≥1 GB RAM, got {ram} MB"
        );
    }

    // ── ModelFamily detection ────────────────────────────────────────

    #[test]
    fn detect_llama3_from_arch() {
        assert_eq!(
            ModelFamily::detect(Some("llama"), Some("meta-llama/Llama-3.1-8B-Instruct")),
            ModelFamily::Llama3,
        );
    }

    #[test]
    fn detect_qwen25_distinct_from_qwen35() {
        // Same arch tag, different recipe — repo hint disambiguates.
        assert_eq!(
            ModelFamily::detect(Some("qwen2"), Some("Qwen/Qwen2.5-7B-Instruct")),
            ModelFamily::Qwen25,
        );
        assert_eq!(
            ModelFamily::detect(Some("qwen3"), Some("Qwen/Qwen3.5-9B")),
            ModelFamily::Qwen35,
        );
        // Qwen3.5 and Qwen3.6 collapse — same recipe.
        assert_eq!(
            ModelFamily::detect(Some("qwen3_5_moe"), Some("Qwen/Qwen3.6-35B-A3B")),
            ModelFamily::Qwen35,
        );
    }

    #[test]
    fn detect_gemma_versions_from_arch_tag() {
        assert_eq!(
            ModelFamily::detect(Some("gemma2"), Some("google/gemma-2-2b-it")),
            ModelFamily::Gemma2,
        );
        assert_eq!(
            ModelFamily::detect(Some("gemma3"), Some("google/gemma-3-4b-it")),
            ModelFamily::Gemma3,
        );
        assert_eq!(
            ModelFamily::detect(Some("gemma4"), Some("google/gemma-4-31B-it")),
            ModelFamily::Gemma4,
        );
    }

    #[test]
    fn detect_gemma_falls_back_when_arch_loses_version() {
        // Some quantizers strip the version. Repo hint rescues it.
        assert_eq!(
            ModelFamily::detect(Some("gemma"), Some("unsloth/gemma-4-31B-it-GGUF")),
            ModelFamily::Gemma4,
        );
        assert_eq!(
            ModelFamily::detect(Some("gemma"), Some("google/gemma-3-12b-it")),
            ModelFamily::Gemma3,
        );
    }

    #[test]
    fn detect_r1_distill_overrides_base_arch() {
        // DeepSeek-R1-Distill-Llama: arch is llama, but recipe is R1.
        assert_eq!(
            ModelFamily::detect(
                Some("llama"),
                Some("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
            ),
            ModelFamily::DeepSeekR1,
        );
        // DeepSeek-R1-0528-Qwen3-8B: arch is qwen3.
        assert_eq!(
            ModelFamily::detect(Some("qwen3"), Some("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")),
            ModelFamily::DeepSeekR1,
        );
    }

    #[test]
    fn detect_mistral_small_3_distinct_recipe() {
        // Mistral-Small-3.x: low-temp recipe.
        assert_eq!(
            ModelFamily::detect(
                Some("mistral3"),
                Some("mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
            ),
            ModelFamily::MistralSmall3,
        );
        // Older Mistral-7B: doesn't get the low-temp treatment.
        assert_eq!(
            ModelFamily::detect(Some("mistral"), Some("mistralai/Mistral-7B-Instruct-v0.3")),
            ModelFamily::Llama3,
        );
    }

    #[test]
    fn detect_misc_families() {
        assert_eq!(
            ModelFamily::detect(Some("glm4_moe_lite"), Some("zai-org/GLM-4.7-Flash")),
            ModelFamily::Glm4,
        );
        assert_eq!(
            ModelFamily::detect(Some("minimax_m2"), Some("MiniMaxAI/MiniMax-M2.7")),
            ModelFamily::MiniMaxM2,
        );
        assert_eq!(
            ModelFamily::detect(
                Some("deepseek2"),
                Some("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"),
            ),
            ModelFamily::DeepSeekCoderV2,
        );
        assert_eq!(
            ModelFamily::detect(Some("llama"), Some("HuggingFaceTB/SmolLM2-1.7B-Instruct")),
            ModelFamily::SmolLm2,
        );
        assert_eq!(
            ModelFamily::detect(Some("gpt_oss"), Some("openai/gpt-oss-20b")),
            ModelFamily::GptOss,
        );
    }

    #[test]
    fn detect_unknown_falls_through() {
        assert_eq!(
            ModelFamily::detect(Some("phi4"), Some("microsoft/phi-4-not-in-zoo")),
            ModelFamily::Unknown,
        );
        assert_eq!(ModelFamily::detect(None, None), ModelFamily::Unknown);
    }

    // ── FamilyDefaults sanity ────────────────────────────────────────

    #[test]
    fn mistral_small_3_keeps_distinctive_low_temp() {
        // Guard against a future refactor accidentally normalising this
        // to 0.7 — silently degrades Mistral output.
        let d = ModelFamily::MistralSmall3.defaults();
        assert!(
            d.temperature < 0.5,
            "Mistral-Small-3 must keep low temp; got {}",
            d.temperature
        );
    }

    #[test]
    fn qwen35_family_matches_upstream_non_thinking_recipe() {
        // Source: live Qwen/Qwen3.5-9B README "Best Practices",
        // "Instruct (or non-thinking) mode for general tasks":
        //   temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
        //   presence_penalty=1.5, repetition_penalty=1.0
        // ADR-0015's table cites the same values via the Qwen3-0.6B
        // model card — both upstream sources agree. An earlier cycle
        // baked temp=1.0/top_p=1.0/presence=2.0 (stale README extract);
        // this guard prevents regression.
        let d = ModelFamily::Qwen35.defaults();
        assert_eq!(d.temperature, 0.7, "temperature");
        assert_eq!(d.top_p, Some(0.8), "top_p");
        assert_eq!(d.top_k, Some(20), "top_k");
        assert_eq!(d.min_p, Some(0.0), "min_p");
        assert_eq!(d.presence_penalty, Some(1.5), "presence_penalty");
        assert_eq!(d.repetition_penalty, Some(1.0), "repetition_penalty");
    }

    #[test]
    fn r1_family_always_reasons() {
        assert_eq!(
            ModelFamily::DeepSeekR1.defaults().thinking,
            ThinkingDefault::On,
        );
        assert_eq!(ModelFamily::GptOss.defaults().thinking, ThinkingDefault::On);
    }

    #[test]
    fn non_reasoning_families_have_no_thinking_mode() {
        // These families' chat templates have no `enable_thinking` flag
        // and no `<think>` machinery. UI must hide the toggle entirely.
        for f in [
            ModelFamily::Llama3,
            ModelFamily::Qwen25,
            ModelFamily::Gemma2,
            ModelFamily::Gemma3,
            ModelFamily::MistralSmall3,
            ModelFamily::MiniMaxM2,
            ModelFamily::DeepSeekCoderV2,
            ModelFamily::SmolLm2,
        ] {
            assert_eq!(
                f.defaults().thinking,
                ThinkingDefault::Unsupported,
                "{:?} must default to thinking=Unsupported (no chat-template support)",
                f,
            );
        }
    }

    #[test]
    fn toggleable_families_default_to_off_per_model_card() {
        // Per chat-template inspection (Qwen3.5/3.6 9B+: enable_thinking
        // hard-coded false) and prose ("Gemma 4 `<|think|>` is a trigger
        // token"; "GLM-4.7-Flash: please TURN ON Preserved Thinking
        // mode"), the documented default for all three is Off — not
        // Auto. The toggle exists but starts in OFF position.
        for f in [ModelFamily::Qwen35, ModelFamily::Gemma4, ModelFamily::Glm4] {
            assert_eq!(
                f.defaults().thinking,
                ThinkingDefault::Off,
                "{:?} must default to thinking=Off per model card",
                f,
            );
        }
    }

    #[test]
    fn smollm2_overrides_misconfigured_temperature() {
        // SmolLM2's own gen_config.json sets temp=2 (almost certainly a
        // bug upstream). Pio should NOT respect that.
        let d = ModelFamily::SmolLm2.defaults();
        assert!(
            d.temperature <= 1.0,
            "SmolLm2 must override upstream gen_config; got {}",
            d.temperature
        );
    }

    // ── Public interface: recommended_sampling ───────────────────────

    /// Tracer bullet for the "Load recommended" feature: looking up a
    /// model id should yield SamplingSettings populated from its family
    /// defaults. Mistral-Small's 0.15 temp is the most distinctive
    /// signal — won't pass if `recommended_sampling` returns
    /// `SamplingSettings::default()`.
    #[test]
    fn recommended_sampling_for_mistral_small_uses_low_temp() {
        let zoo = ModelZoo::bundled();
        let s = zoo
            .recommended_sampling("mistral-small-3.2-24b")
            .expect("mistral-small-3.2-24b must be in zoo");
        assert_eq!(s.temperature, Some(0.15));
    }

    /// Qwen3.5 needs `presence_penalty=1.5` (upstream non-thinking
    /// recipe) to mitigate its "repeats forever" failure mode. The
    /// "Load recommended" path must preserve it — temperature alone
    /// isn't enough.
    #[test]
    fn recommended_sampling_for_qwen35_preserves_presence_penalty() {
        let zoo = ModelZoo::bundled();
        let s = zoo
            .recommended_sampling("qwen3.5-9b")
            .expect("qwen3.5-9b must be in zoo");
        assert_eq!(s.penalty_present, Some(1.5));
    }

    /// Tracer bullet for the thinking-mode toggle UI: Qwen3.5-9B
    /// supports toggleable thinking (chat template has `enable_thinking`
    /// alongside the `/think` and `/no_think` slash-commands) — so
    /// `recommended_thinking` returns `Some`, telling the UI to show
    /// the toggle. The default position is OFF because Qwen3.5-9B's
    /// chat template hard-codes `enable_thinking: false` (verified in
    /// `tokenizer_config.json` chat_template) and upstream's "Best
    /// Practices" recipe says "non-thinking mode for text tasks".
    /// Cycle-5's original assertion of `Auto` was based on a wrong
    /// reading of the README.
    #[test]
    fn recommended_thinking_for_qwen35_9b_returns_off_per_template() {
        let zoo = ModelZoo::bundled();
        let t = zoo
            .recommended_thinking("qwen3.5-9b")
            .expect("qwen3.5-9b supports thinking");
        assert_eq!(t, crate::generation::ThinkingMode::Off);
    }

    /// Llama 3 has no thinking mode in its chat template — UI must hide
    /// the toggle. Tests the Unsupported branch of the recommendation
    /// match, which the `qwen35` test alone can't exercise.
    #[test]
    fn recommended_thinking_for_llama3_returns_none() {
        let zoo = ModelZoo::bundled();
        assert_eq!(zoo.recommended_thinking("llama-3.1-8b"), None);
    }

    /// DeepSeek R1 always reasons (R1's whole identity is reasoning;
    /// chat template emits `<think>` unconditionally). UI should show
    /// the toggle but seeded ON.
    #[test]
    fn recommended_thinking_for_r1_returns_on() {
        let zoo = ModelZoo::bundled();
        let t = zoo
            .recommended_thinking("deepseek-r1-0528-qwen3-8b")
            .expect("R1 supports thinking");
        assert_eq!(t, crate::generation::ThinkingMode::On);
    }

    /// Conversion fidelity: every FamilyDefaults sampling field that
    /// has a SamplingSettings counterpart must round-trip through
    /// `recommended_sampling`. Llama3 is the third family we test, so
    /// any "hardcoded for Mistral / Qwen" bug surfaces here.
    #[test]
    fn recommended_sampling_for_llama3_preserves_full_recipe() {
        let zoo = ModelZoo::bundled();
        let s = zoo
            .recommended_sampling("llama-3.1-8b")
            .expect("llama-3.1-8b must be in zoo");
        let f = ModelFamily::Llama3.defaults();
        assert_eq!(s.temperature, Some(f.temperature), "temperature");
        assert_eq!(s.top_p, f.top_p, "top_p");
        assert_eq!(s.top_k, f.top_k, "top_k");
        assert_eq!(s.min_p, f.min_p, "min_p");
        assert_eq!(s.penalty_repeat, f.repetition_penalty, "penalty_repeat");
        assert_eq!(s.penalty_present, f.presence_penalty, "penalty_present");
    }

    // ── ModelZooEntry::family_kind / defaults ────────────────────────

    #[test]
    fn bundled_zoo_has_30_plus_entries() {
        let zoo = ModelZoo::bundled();
        assert!(
            zoo.models.len() >= 30,
            "expected ≥30 entries (the inference test matrix), got {}",
            zoo.models.len(),
        );
    }

    #[test]
    fn every_zoo_entry_resolves_to_known_family() {
        // Guard: any new entry whose `family` field doesn't match
        // ModelFamily::as_str (or get rescued by detect()) silently
        // collapses to Unknown defaults, which is almost never what
        // the user intended. Catch at parse time.
        let zoo = ModelZoo::bundled();
        let unresolved: Vec<_> = zoo
            .models
            .iter()
            .filter(|(_, e)| e.family_kind() == ModelFamily::Unknown)
            .map(|(id, e)| format!("{} (family={:?})", id, e.family))
            .collect();
        assert!(
            unresolved.is_empty(),
            "{} entries fell through to Unknown family: {:?}",
            unresolved.len(),
            unresolved,
        );
    }

    #[test]
    fn zoo_entries_cover_every_family_we_implement() {
        // Every ModelFamily variant (except Unknown) should have at
        // least one zoo entry. If we add a new family but never wire a
        // canonical model, the family code is dead.
        let zoo = ModelZoo::bundled();
        let mut families_seen: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        for e in zoo.models.values() {
            families_seen.insert(e.family_kind().as_str());
        }
        for f in [
            ModelFamily::Llama3,
            ModelFamily::Qwen25,
            ModelFamily::Qwen35,
            ModelFamily::Gemma2,
            ModelFamily::Gemma3,
            ModelFamily::Gemma4,
            ModelFamily::MistralSmall3,
            ModelFamily::Glm4,
            ModelFamily::MiniMaxM2,
            ModelFamily::DeepSeekCoderV2,
            ModelFamily::DeepSeekR1,
            ModelFamily::SmolLm2,
            ModelFamily::GptOss,
        ] {
            assert!(
                families_seen.contains(f.as_str()),
                "no zoo entry for family `{}` — either add a canonical \
                 model or remove the family",
                f.as_str(),
            );
        }
    }

    #[test]
    fn mistral_zoo_entry_carries_low_temp_defaults() {
        // End-to-end: zoo lookup → defaults → low temp survives.
        // Catches a misclassification of mistral-small-3.2-24b → some
        // other family that would silently lose the 0.15 recipe.
        let zoo = ModelZoo::bundled();
        let entry = zoo
            .get("mistral-small-3.2-24b")
            .expect("mistral-small-3.2-24b must be in zoo");
        let d = entry.defaults();
        assert!(
            d.temperature < 0.5,
            "mistral entry must keep low temp via family lookup; got {}",
            d.temperature,
        );
    }

    #[test]
    fn qwen35_zoo_entries_carry_presence_penalty() {
        // Qwen3.5/3.6 entries should all hit the Qwen35 family and
        // pick up the presence_penalty=1.5 recipe (upstream
        // non-thinking general-tasks default).
        let zoo = ModelZoo::bundled();
        for id in ["qwen3.5-0.8b", "qwen3.5-9b", "qwen3.6-35b-a3b"] {
            let entry = zoo.get(id).unwrap_or_else(|| panic!("missing {id}"));
            assert_eq!(
                entry.defaults().presence_penalty,
                Some(1.5),
                "{id} must inherit Qwen3.5 presence_penalty=1.5",
            );
        }
    }

    #[test]
    fn r1_probe_entry_defaults_to_thinking_on() {
        let zoo = ModelZoo::bundled();
        let entry = zoo.get("deepseek-r1-0528-qwen3-8b").expect("R1 probe");
        assert_eq!(entry.defaults().thinking, ThinkingDefault::On);
    }

    #[test]
    fn gguf_bundles_have_well_formed_filenames() {
        // Catches: wrong-filename typos in zoo.json's `file` field.
        //
        // Rules per platform-bundle:
        //   - backend == "llamacpp" → file must be Some, end in .gguf,
        //     AND contain the entry's `default_quant` tier (e.g. Q4_K_M)
        //     so a paste from the wrong repo's filename gets caught.
        //   - backend == "mlx" → file must be None (whole-dir snapshot).
        //   - other backends (candle, onnx, executorch) — out of scope
        //     for now; their conventions are less uniform.
        let zoo = ModelZoo::bundled();
        let mut errs: Vec<String> = Vec::new();
        for (id, entry) in &zoo.models {
            let quant_upper = entry.default_quant.to_uppercase();
            for (plat, bundle) in &entry.platforms {
                match bundle.backend.as_str() {
                    "llamacpp" => {
                        let Some(file) = bundle.file.as_deref() else {
                            errs.push(format!("{id}/{plat}: llamacpp bundle missing `file`"));
                            continue;
                        };
                        if !file.ends_with(".gguf") {
                            errs.push(format!("{id}/{plat}: file `{file}` not .gguf"));
                        }
                        if !file.to_uppercase().contains(&quant_upper) {
                            errs.push(format!(
                                "{id}/{plat}: file `{file}` doesn't contain quant `{quant_upper}` \
                                 — likely a wrong-repo paste",
                            ));
                        }
                    }
                    "mlx" if bundle.file.is_some() => {
                        errs.push(format!(
                            "{id}/{plat}: mlx bundle should have file=null \
                             (whole-dir snapshot), got {:?}",
                            bundle.file,
                        ));
                    }
                    _ => {} // candle/onnx/executorch out of scope
                }
            }
        }
        assert!(
            errs.is_empty(),
            "zoo.json filename validation failed:\n  - {}",
            errs.join("\n  - "),
        );
    }

    #[test]
    fn ram_budgets_scale_with_size() {
        // Per-tier sanity: bigger models should NOT have lower RAM
        // budgets than smaller siblings on the same platform. Catches
        // typos in `min_ram_mb`.
        let zoo = ModelZoo::bundled();
        let plat = "macos_arm64";
        let pairs: &[(&str, &str)] = &[
            ("qwen3.5-0.8b", "qwen3.5-2b"),
            ("qwen3.5-2b", "qwen3.5-4b"),
            ("qwen3.5-4b", "qwen3.5-9b"),
            ("qwen3.5-9b", "qwen3.5-27b"),
            ("gemma-3-1b", "gemma-3-4b"),
            ("gemma-3-4b", "gemma-3-12b"),
            ("gemma-4-e2b", "gemma-4-e4b"),
            ("llama-3.2-1b", "llama-3.2-3b"),
            ("llama-3.2-3b", "llama-3.1-8b"),
        ];
        for (small, big) in pairs {
            let s = zoo.get(small).unwrap().platforms.get(plat).unwrap();
            let b = zoo.get(big).unwrap().platforms.get(plat).unwrap();
            assert!(
                b.min_ram_mb >= s.min_ram_mb,
                "{} ({}MB) must require ≥ RAM than {} ({}MB)",
                big,
                b.min_ram_mb,
                small,
                s.min_ram_mb,
            );
        }
    }

    // ── zoo.json as executable configuration ─────────────────────────
    //
    // The manifest is compiled in and drives which artifact every device
    // downloads and which backend loads it, so a typo is a shipped bug,
    // not a bad config file the user can edit. These checks are the
    // compiler zoo.json doesn't get: they run over the raw text as well
    // as the parsed struct, because serde's `HashMap` silently swallows
    // the two failure modes that survive parsing — a duplicate key and
    // a misspelled field.

    /// Backend ids this module knows how to route — the same set
    /// `select_for_device`'s `os_compatible` switch matches on. A bundle
    /// naming anything else falls into that switch's `_ => true` arm and
    /// is offered to a device that can never load it.
    const KNOWN_BACKENDS: &[&str] = &["mlx", "llamacpp", "candle", "onnx", "executorch"];

    /// Backends whose `start_session` still returns `Unimplemented`
    /// (`src/backend/candle/mod.rs`, `src/backend/executorch/mod.rs`), and
    /// which no platform feature bundle in Cargo.toml enables. Routing a
    /// device here gets it a download and then a failed generate.
    const BACKENDS_THAT_CANNOT_GENERATE_YET: &[&str] = &["candle", "executorch"];

    /// Platform ids `current_platform_id` can return.
    const KNOWN_PLATFORMS: &[&str] = &[
        "macos_arm64",
        "macos_x86",
        "ios",
        "android",
        "linux_cuda",
        "linux_cpu",
        "windows",
    ];

    /// Which backends each platform can actually load, from the crate's own
    /// feature layout. MLX links Apple's Metal framework and is built only
    /// for `apple`/`ios` targets — and only on Apple Silicon, so an Intel
    /// Mac entry is a mis-route even though the OS matches. ExecuTorch is
    /// the mobile scaffold. Everything else is portable C/Rust.
    fn backends_supported_on(platform: &str) -> &'static [&'static str] {
        match platform {
            "macos_arm64" => &["mlx", "llamacpp", "candle", "onnx"],
            "macos_x86" => &["llamacpp", "candle", "onnx"],
            "ios" => &["mlx", "llamacpp", "executorch"],
            "android" => &["llamacpp", "onnx", "executorch"],
            "linux_cuda" | "linux_cpu" | "windows" => &["llamacpp", "candle", "onnx"],
            _ => &[],
        }
    }

    /// Every key the loader reads out of a bundle. Anything else is a typo
    /// serde would drop on the floor — including a misspelled `min_ram_mb`,
    /// which would silently fall back to the 4096 default.
    const BUNDLE_KEYS: &[&str] = &["backend", "source", "file", "min_ram_mb", "sha256"];
    const ENTRY_KEYS: &[&str] = &["display_name", "family", "default_quant", "platforms"];
    const ROOT_KEYS: &[&str] = &["schema_version", "models"];

    /// Duplicate object keys in `value`, at `path`, reported depth-first.
    ///
    /// `serde_json::Value` is a map, so a duplicate is already gone by the
    /// time it parses — the second wins and the first vanishes without a
    /// diagnostic. Re-scanning the raw text is the only way to see it.
    fn duplicate_keys(raw: &str) -> Vec<String> {
        // Re-parse preserving order, then compare each object's key count
        // against the de-duplicated map serde produced.
        fn walk(
            de: &mut serde_json::Deserializer<serde_json::de::StrRead<'_>>,
        ) -> Result<Vec<String>, serde_json::Error> {
            use serde::de::Deserialize;
            let ordered = OrderedJson::deserialize(de)?;
            Ok(ordered.duplicates("$"))
        }
        let mut de = serde_json::Deserializer::from_str(raw);
        walk(&mut de).expect("bundled zoo json must parse")
    }

    /// A JSON tree that keeps every key an object declared, duplicates and
    /// all, so they can be counted.
    enum OrderedJson {
        Object(Vec<(String, OrderedJson)>),
        Other,
    }

    impl<'de> serde::Deserialize<'de> for OrderedJson {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> serde::de::Visitor<'de> for V {
                type Value = OrderedJson;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("any JSON value")
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> Result<OrderedJson, A::Error> {
                    let mut entries = Vec::new();
                    while let Some((k, v)) = map.next_entry::<String, OrderedJson>()? {
                        entries.push((k, v));
                    }
                    Ok(OrderedJson::Object(entries))
                }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> Result<OrderedJson, A::Error> {
                    while seq.next_element::<OrderedJson>()?.is_some() {}
                    Ok(OrderedJson::Other)
                }
                fn visit_unit<E>(self) -> Result<OrderedJson, E> {
                    Ok(OrderedJson::Other)
                }
                fn visit_none<E>(self) -> Result<OrderedJson, E> {
                    Ok(OrderedJson::Other)
                }
                fn visit_bool<E>(self, _: bool) -> Result<OrderedJson, E> {
                    Ok(OrderedJson::Other)
                }
                fn visit_i64<E>(self, _: i64) -> Result<OrderedJson, E> {
                    Ok(OrderedJson::Other)
                }
                fn visit_u64<E>(self, _: u64) -> Result<OrderedJson, E> {
                    Ok(OrderedJson::Other)
                }
                fn visit_f64<E>(self, _: f64) -> Result<OrderedJson, E> {
                    Ok(OrderedJson::Other)
                }
                fn visit_str<E>(self, _: &str) -> Result<OrderedJson, E> {
                    Ok(OrderedJson::Other)
                }
            }
            d.deserialize_any(V)
        }
    }

    impl OrderedJson {
        fn duplicates(&self, path: &str) -> Vec<String> {
            let OrderedJson::Object(entries) = self else {
                return Vec::new();
            };
            let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
            let mut out = Vec::new();
            for (key, value) in entries {
                if !seen.insert(key.as_str()) {
                    out.push(format!("{path}.{key}"));
                }
                out.extend(value.duplicates(&format!("{path}.{key}")));
            }
            out
        }
    }

    #[test]
    fn no_key_in_the_zoo_manifest_is_declared_twice() {
        // A repeated model id or platform key parses cleanly and the last
        // one silently wins, so the entry someone thought they added is
        // simply absent at runtime.
        let dupes = duplicate_keys(BUNDLED_ZOO_JSON);
        assert!(
            dupes.is_empty(),
            "zoo.json declares these keys more than once (the later one silently wins): {dupes:?}",
        );
    }

    #[test]
    fn the_duplicate_key_scan_catches_a_repeat_serde_would_swallow() {
        // Teeth for the test above: serde keeps the *second* `min_ram_mb`
        // and reports nothing, so without this scan the manifest could ship
        // a value nobody reviewed.
        let doubled = r#"{
            "schema_version": 1,
            "models": {
                "a": {"display_name": "A", "family": "llama3", "platforms": {
                    "ios": {"backend": "llamacpp", "source": "o/r", "min_ram_mb": 2048,
                            "min_ram_mb": 99999}
                }},
                "a": {"display_name": "A again", "family": "llama3", "platforms": {}}
            }
        }"#;
        let dupes = duplicate_keys(doubled);
        assert!(
            dupes.iter().any(|d| d.ends_with(".min_ram_mb")),
            "a repeated bundle field must be reported, got {dupes:?}",
        );
        assert!(
            dupes.iter().any(|d| d.ends_with(".a")),
            "a repeated model id must be reported, got {dupes:?}",
        );
    }

    #[test]
    fn every_zoo_field_is_one_the_loader_reads() {
        // serde drops unknown fields without complaint, so `min_ram` or
        // `filename` would parse and then quietly take the default —
        // shipping a bundle sized for the wrong device.
        let errs = field_problems(BUNDLED_ZOO_JSON);
        assert!(
            errs.is_empty(),
            "zoo.json field validation failed:\n  - {}",
            errs.join("\n  - ")
        );
    }

    #[test]
    fn the_field_scan_catches_a_misspelling_serde_would_default_away() {
        let typo = r#"{
            "schema_version": 1,
            "models": {
                "a": {"display_name": "A", "family": "llama3", "platforms": {
                    "ios": {"backend": "llamacpp", "source": "o/r", "min_ram": 2048,
                            "sha256": "not-a-digest"}
                }}
            }
        }"#;
        let errs = field_problems(typo);
        assert!(
            errs.iter().any(|e| e.contains("min_ram")),
            "a misspelled field must be reported, got {errs:?}",
        );
        assert!(
            errs.iter().any(|e| e.contains("sha256")),
            "a malformed checksum must be reported, got {errs:?}",
        );
    }

    fn field_problems(raw: &str) -> Vec<String> {
        let root: serde_json::Value = serde_json::from_str(raw).expect("zoo json parses");
        let mut errs = Vec::new();

        for key in root.as_object().expect("zoo root is an object").keys() {
            if !ROOT_KEYS.contains(&key.as_str()) {
                errs.push(format!("$.{key} is not a field the loader reads"));
            }
        }
        let models = root["models"].as_object().expect("models is an object");
        for (id, entry) in models {
            for key in entry.as_object().expect("entry is an object").keys() {
                if !ENTRY_KEYS.contains(&key.as_str()) {
                    errs.push(format!("{id}.{key} is not a field the loader reads"));
                }
            }
            let platforms = entry["platforms"]
                .as_object()
                .unwrap_or_else(|| panic!("{id}.platforms is an object"));
            for (plat, bundle) in platforms {
                for key in bundle.as_object().expect("bundle is an object").keys() {
                    if !BUNDLE_KEYS.contains(&key.as_str()) {
                        errs.push(format!("{id}/{plat}.{key} is not a field the loader reads"));
                    }
                }
                // Checksums are optional, but a malformed one is worse than
                // none: it fails verification after a multi-GB download.
                if let Some(sum) = bundle.get("sha256").and_then(|v| v.as_str())
                    && (sum.len() != 64 || !sum.bytes().all(|b| b.is_ascii_hexdigit()))
                {
                    errs.push(format!(
                        "{id}/{plat}.sha256 `{sum}` is not a 64-char hex digest"
                    ));
                }
            }
        }
        errs
    }

    #[test]
    fn every_zoo_bundle_names_a_backend_platform_and_source_the_loader_can_act_on() {
        let errs = bundle_problems(&ModelZoo::bundled());
        assert!(
            errs.is_empty(),
            "zoo.json validation failed:\n  - {}",
            errs.join("\n  - ")
        );
    }

    #[test]
    fn the_bundle_scan_catches_the_routing_mistakes_that_only_fail_on_the_device() {
        // Each of these parses fine and only fails once a real device tries
        // to load it — an MLX bundle on Intel Mac has no Metal build, a
        // platform key nothing returns is unreachable, and min_ram_mb: 0
        // admits a 27B model onto a phone.
        let broken = r#"{
            "schema_version": 1,
            "models": {
                "bad": {"display_name": "", "family": "llama3", "platforms": {
                    "macos_x86": {"backend": "mlx", "source": "o/r", "min_ram_mb": 8192},
                    "linux_arm64": {"backend": "llamacpp", "source": "o/r", "min_ram_mb": 8192},
                    "windows": {"backend": "tensorrt", "source": "o/r", "min_ram_mb": 8192},
                    "android": {"backend": "llamacpp", "source": "o/r", "min_ram_mb": 0},
                    "ios": {"backend": "llamacpp", "source": "not a repo id", "min_ram_mb": 4096}
                }},
                "empty": {"display_name": "E", "family": "llama3", "platforms": {}}
            }
        }"#;
        let zoo: ModelZoo = serde_json::from_str(broken).expect("fixture parses");
        let errs = bundle_problems(&zoo);
        for expected in [
            "cannot be built for this platform",
            "unreachable",
            "not one the loader can instantiate",
            "min_ram_mb of 0",
            "neither a URL nor a HuggingFace",
            "empty display_name",
            "no platform bundles",
        ] {
            assert!(
                errs.iter().any(|e| e.contains(expected)),
                "expected a `{expected}` complaint, got {errs:?}",
            );
        }
    }

    fn bundle_problems(zoo: &ModelZoo) -> Vec<String> {
        let mut errs: Vec<String> = Vec::new();

        for (id, entry) in &zoo.models {
            if id.trim().is_empty() || id.contains(char::is_whitespace) {
                errs.push(format!("`{id}` is not usable as a canonical model id"));
            }
            if entry.display_name.trim().is_empty() {
                errs.push(format!(
                    "{id}: empty display_name — the picker would show a blank row"
                ));
            }
            if entry.platforms.is_empty() {
                errs.push(format!(
                    "{id}: no platform bundles, so it can never be selected"
                ));
            }

            for (plat, bundle) in &entry.platforms {
                if !KNOWN_PLATFORMS.contains(&plat.as_str()) {
                    errs.push(format!(
                        "{id}/{plat}: no `current_platform_id()` ever returns this key, so the \
                         bundle is unreachable",
                    ));
                }
                if !KNOWN_BACKENDS.contains(&bundle.backend.as_str()) {
                    errs.push(format!(
                        "{id}/{plat}: backend `{}` is not one the loader can instantiate",
                        bundle.backend,
                    ));
                } else if !backends_supported_on(plat).contains(&bundle.backend.as_str()) {
                    errs.push(format!(
                        "{id}/{plat}: backend `{}` cannot be built for this platform — the \
                         download would succeed and the load would fail",
                        bundle.backend,
                    ));
                }
                if bundle.min_ram_mb == 0 {
                    errs.push(format!(
                        "{id}/{plat}: min_ram_mb of 0 admits the bundle on any device",
                    ));
                }
                if bundle.file.as_deref().is_some_and(|f| f.trim().is_empty()) {
                    errs.push(format!(
                        "{id}/{plat}: empty `file` — use null for a whole-repo snapshot",
                    ));
                }
                errs.extend(
                    source_problem(&bundle.source).map(|why| format!("{id}/{plat}: {why}")),
                );
            }
        }

        errs
    }

    /// A `source` is either an absolute URL or a HuggingFace `owner/repo`
    /// id; the downloader resolves it one of those two ways and has no
    /// third branch to fall into.
    fn source_problem(source: &str) -> Option<String> {
        if source.trim() != source || source.is_empty() {
            return Some(format!(
                "source `{source}` is empty or padded with whitespace"
            ));
        }
        if source.contains("://") {
            return match url::Url::parse(source) {
                Ok(url) if url.has_host() => None,
                Ok(_) => Some(format!("source `{source}` parses but names no host")),
                Err(e) => Some(format!("source `{source}` is not a URL: {e}")),
            };
        }
        let segments: Vec<&str> = source.split('/').collect();
        if segments.len() != 2 || segments.iter().any(|s| s.is_empty()) {
            return Some(format!(
                "source `{source}` is neither a URL nor a HuggingFace `owner/repo` id",
            ));
        }
        if source.contains(char::is_whitespace) {
            return Some(format!("source `{source}` contains whitespace"));
        }
        None
    }

    #[test]
    fn no_bundle_is_routed_to_a_backend_that_cannot_generate() {
        // `gemma-4/linux_cpu` used to point at Candle, whose `start_session`
        // returns `Unimplemented`. That is now fixed, and this holds the line:
        // routing a download at a backend that cannot consume it wastes the
        // user's bandwidth and fails at the last possible moment.
        let zoo = ModelZoo::bundled();
        let mut routes: Vec<String> = zoo
            .models
            .iter()
            .flat_map(|(id, entry)| {
                entry
                    .platforms
                    .iter()
                    .filter(|(_, bundle)| {
                        BACKENDS_THAT_CANNOT_GENERATE_YET.contains(&bundle.backend.as_str())
                    })
                    .map(move |(plat, bundle)| format!("{id}/{plat} -> {}", bundle.backend))
            })
            .collect();
        routes.sort();
        assert_eq!(
            routes,
            Vec::<String>::new(),
            "a bundle was routed to a backend that cannot generate yet; either finish that \
             backend or point the platform at llamacpp",
        );
    }

    #[test]
    fn every_zoo_entry_is_selectable_by_some_device() {
        // An entry whose cheapest bundle asks for more RAM than any device
        // in the fleet has is dead weight in the picker.
        let zoo = ModelZoo::bundled();
        for (id, entry) in &zoo.models {
            let cheapest = entry
                .platforms
                .values()
                .map(|b| b.min_ram_mb)
                .min()
                .unwrap_or(u32::MAX);
            assert!(
                cheapest <= 512 * 1024,
                "{id} needs at least {cheapest} MB on every platform — no device can select it",
            );
        }
    }

    #[test]
    fn family_str_is_round_trippable_label() {
        // Every variant produces a non-empty stable id usable in logs.
        for f in [
            ModelFamily::Llama3,
            ModelFamily::Qwen25,
            ModelFamily::Qwen35,
            ModelFamily::Gemma2,
            ModelFamily::Gemma3,
            ModelFamily::Gemma4,
            ModelFamily::MistralSmall3,
            ModelFamily::Glm4,
            ModelFamily::MiniMaxM2,
            ModelFamily::DeepSeekCoderV2,
            ModelFamily::DeepSeekR1,
            ModelFamily::SmolLm2,
            ModelFamily::GptOss,
            ModelFamily::Unknown,
        ] {
            let s = f.as_str();
            assert!(!s.is_empty());
            assert!(!s.contains(' '));
        }
    }

    // ── Single-owner: all three family readers agree ─────────────────

    /// The drift this consolidation prevents: `ModelFamily` is now the
    /// single owner of `{ template_kind, thinking_default, channel_markers,
    /// default_enable_thinking }`, so the streaming marker reader
    /// (`ChannelMarkers::from_architecture`), the chat-template reader
    /// (`ModelFamily::default_enable_thinking`, consumed by the llama-cpp
    /// prompt path), and the zoo `defaults()` reader can't disagree about
    /// what a reasoning family does.
    ///
    /// Gemma 4 (control-token channel) and Qwen3.5 (`<think>` tag channel)
    /// are the two reasoning families keyed off the arch string alone;
    /// asserting both catches a per-family marker/template divergence.
    #[test]
    fn all_family_readers_agree_for_gemma4_and_qwen35() {
        use crate::generation::ChannelMarkers;

        // Gemma 4: reasoning family with control-token markers, its own
        // Gemma4 template, and the enable_thinking template flag set.
        let g = ModelFamily::Gemma4;
        assert_eq!(g.channel_markers(), ChannelMarkers::gemma4());
        assert_eq!(g.defaults().template_kind, TemplateKind::Gemma4);
        assert!(g.default_enable_thinking());
        // The streaming reader keyed off the raw arch tag must resolve to
        // the SAME markers the family owns — no independent re-derivation.
        assert_eq!(
            ChannelMarkers::from_architecture(Some("gemma4")),
            g.channel_markers(),
        );

        // Qwen3.5: reasoning family with `<think>` markers, the Qwen3
        // template, and no Gemma-style enable_thinking flag.
        let q = ModelFamily::Qwen35;
        assert_eq!(q.channel_markers(), ChannelMarkers::qwen3_deepseek());
        assert_eq!(q.defaults().template_kind, TemplateKind::Qwen3);
        assert!(!q.default_enable_thinking());
        for arch in ["qwen3", "qwen3moe"] {
            assert_eq!(
                ChannelMarkers::from_architecture(Some(arch)),
                q.channel_markers(),
                "arch `{arch}` must resolve to the Qwen3.5-owned markers",
            );
        }

        // Non-reasoning families own empty markers and don't set the
        // Gemma template flag — the reader must stay silent for them.
        for f in [
            ModelFamily::Llama3,
            ModelFamily::Gemma2,
            ModelFamily::MistralSmall3,
        ] {
            assert_eq!(f.channel_markers(), ChannelMarkers::none());
            assert!(!f.default_enable_thinking());
        }
    }
}
