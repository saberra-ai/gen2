//! Will this model run on this machine, and at what context size?
//!
//! The question every local-inference app has to answer before it loads
//! anything. Answering it needs the model's own header (layer count, KV heads,
//! embedding width, training context) and the machine's memory, so it lives
//! here rather than in the caller.

use std::path::Path;

use crate::bundle::gguf::{
    build_model_metadata, estimate_ram_bytes, fit_context, kv_bytes_per_token, parse_gguf_metadata,
};
use crate::hardware::HardwareProfile;
use crate::types::ModelMetadata;

use super::error::{Error, Result};

/// What a model file says about itself.
///
/// Read from the file's header without loading any weights, so it's cheap
/// enough to run over a directory of candidates.
///
/// GGUF only today — MLX and ONNX carry their metadata differently, and
/// [`ModelInfo::read`] says so rather than guessing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModelInfo {
    /// Size of the model file on disk.
    pub file_bytes: u64,
    /// Architecture family: `"llama"`, `"qwen2"`, `"gemma3"`, …
    pub architecture: Option<String>,
    /// Quantization label, e.g. `"Q4_K_M"`.
    pub quantization: Option<String>,
    /// Estimated parameter count.
    pub parameters: Option<u64>,
    /// Context length the model was trained for.
    pub train_context: Option<u32>,
    /// Whether the chat template references tool-use variables.
    pub supports_tools: bool,
    /// The full header metadata, for anything the fields above don't cover.
    pub metadata: ModelMetadata,
}

impl ModelInfo {
    /// Read a model's header. Does not load weights.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file_bytes = std::fs::metadata(path)
            .map(|m| m.len())
            .map_err(|e| Error::Load(format!("cannot read {}: {e}", path.display())))?;

        let gguf = parse_gguf_metadata(path)?;
        let metadata = build_model_metadata(&gguf, Some(file_bytes)).ok_or_else(|| {
            Error::Load(format!(
                "{} has no usable architecture metadata",
                path.display()
            ))
        })?;

        Ok(Self {
            file_bytes,
            architecture: metadata.architecture.clone(),
            quantization: metadata.quantization.clone(),
            parameters: metadata.parameter_count,
            train_context: metadata.context_length.map(|c| c as u32),
            supports_tools: metadata.supports_tools.unwrap_or(false),
            metadata,
        })
    }

    /// Bytes of memory needed to run at `context`.
    pub fn memory_needed(&self, context: u32) -> u64 {
        estimate_ram_bytes(&self.metadata, self.file_bytes, context)
    }

    /// Memory cost of one token of context.
    fn kv_per_token(&self) -> u64 {
        match (
            self.metadata.block_count,
            self.metadata.head_count_kv,
            self.metadata.embedding_length,
            self.metadata.head_count,
        ) {
            (Some(layers), Some(kv_heads), Some(width), Some(heads)) if heads > 0 => {
                kv_bytes_per_token(layers, kv_heads, width / heads)
            }
            // No architecture detail: assume a token is free, so context sizing
            // falls back to the training context rather than pretending to a
            // precision the header doesn't support.
            _ => 1,
        }
    }

    /// The largest context this machine can give the model.
    pub fn max_context(&self, hw: &HardwareProfile) -> u32 {
        fit_context(
            budget_bytes(hw),
            self.file_bytes,
            self.kv_per_token(),
            self.train_context.unwrap_or(4096),
            None,
        )
    }

    /// Whether this machine can run the model, and how comfortably.
    ///
    /// `context` is what you intend to use; `None` asks for the largest that
    /// fits.
    pub fn fits(&self, hw: &HardwareProfile, context: Option<u32>) -> Fit {
        let budget = budget_bytes(hw);
        let max_context = self.max_context(hw);
        let context = context.unwrap_or(max_context);
        let needed = self.memory_needed(context);

        let verdict = if self.file_bytes >= budget {
            // The weights alone don't fit; no context size rescues that.
            FitVerdict::TooLarge
        } else if needed <= budget {
            FitVerdict::Fits
        } else {
            FitVerdict::ContextTooLarge
        };

        Fit {
            verdict,
            context,
            max_context,
            needed_bytes: needed,
            available_bytes: budget,
            model_bytes: self.file_bytes,
        }
    }
}

/// How much memory the engine may use.
///
/// Discrete VRAM when there is some, otherwise system RAM — which is also the
/// right answer for Apple Silicon's unified memory, where `vram_bytes` is 0 by
/// convention. Leaves a quarter of the machine to everything else.
fn budget_bytes(hw: &HardwareProfile) -> u64 {
    let total = if hw.vram_bytes > 0 {
        hw.vram_bytes
    } else {
        hw.total_ram_bytes
    };
    total / 4 * 3
}

/// Whether a model fits, and by how much.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Fit {
    /// The verdict.
    pub verdict: FitVerdict,
    /// The context this verdict is about.
    pub context: u32,
    /// The largest context that would fit.
    pub max_context: u32,
    /// Memory needed at `context`.
    pub needed_bytes: u64,
    /// Memory the engine may use.
    pub available_bytes: u64,
    /// Size of the weights alone.
    pub model_bytes: u64,
}

impl Fit {
    /// Whether the model can run at this context.
    pub fn ok(&self) -> bool {
        self.verdict == FitVerdict::Fits
    }
}

impl std::fmt::Display for Fit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
        match self.verdict {
            FitVerdict::Fits => write!(
                f,
                "fits at {} context ({:.1} GB of {:.1} GB)",
                self.context,
                gb(self.needed_bytes),
                gb(self.available_bytes)
            ),
            FitVerdict::ContextTooLarge => write!(
                f,
                "{} context needs {:.1} GB but only {:.1} GB is available; \
                 {} context would fit",
                self.context,
                gb(self.needed_bytes),
                gb(self.available_bytes),
                self.max_context
            ),
            FitVerdict::TooLarge => write!(
                f,
                "the weights alone are {:.1} GB, more than the {:.1} GB available",
                gb(self.model_bytes),
                gb(self.available_bytes)
            ),
        }
    }
}

/// The three answers to "will it fit".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FitVerdict {
    /// Runs at the requested context.
    Fits,
    /// The weights fit but the requested context doesn't. A smaller context
    /// works — see [`Fit::max_context`].
    ContextTooLarge,
    /// The weights alone exceed the budget. No context size helps; this
    /// machine needs a smaller model or a heavier quantization.
    TooLarge,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::GpuBackend;

    fn machine(ram_gb: u64) -> HardwareProfile {
        HardwareProfile {
            total_ram_bytes: ram_gb * 1024 * 1024 * 1024,
            cpu_cores: 8,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        }
    }

    fn model(file_gb: u64) -> ModelInfo {
        let metadata = ModelMetadata {
            block_count: Some(32),
            head_count_kv: Some(8),
            head_count: Some(32),
            embedding_length: Some(4096),
            context_length: Some(32768),
            ..Default::default()
        };
        ModelInfo {
            file_bytes: file_gb * 1024 * 1024 * 1024,
            architecture: Some("llama".into()),
            quantization: Some("Q4_K_M".into()),
            parameters: None,
            train_context: Some(32768),
            supports_tools: false,
            metadata,
        }
    }

    #[test]
    fn a_small_model_on_a_big_machine_fits() {
        let fit = model(4).fits(&machine(64), Some(8192));
        assert_eq!(fit.verdict, FitVerdict::Fits);
        assert!(fit.ok());
        assert!(fit.to_string().contains("fits at 8192"));
    }

    #[test]
    fn weights_larger_than_the_budget_are_too_large_at_any_context() {
        let fit = model(64).fits(&machine(16), Some(2048));
        assert_eq!(fit.verdict, FitVerdict::TooLarge);
        assert!(!fit.ok());
        // No context size rescues it, so don't suggest one.
        assert!(fit.to_string().contains("weights alone"));
    }

    #[test]
    fn an_oversized_context_is_reported_as_such_with_one_that_would_work() {
        // Weights fit; the requested context does not.
        let info = model(8);
        let hw = machine(16);
        let fit = info.fits(&hw, Some(1_000_000));
        assert_eq!(fit.verdict, FitVerdict::ContextTooLarge);
        assert!(
            fit.max_context < 1_000_000,
            "should report a smaller workable context"
        );
        assert!(fit.to_string().contains("would fit"));
    }

    #[test]
    fn max_context_never_exceeds_what_the_model_was_trained_for() {
        // A huge machine must not offer more context than the model supports.
        let fit = model(1).fits(&machine(512), None);
        assert!(fit.max_context <= 32768, "capped by train_context");
        assert_eq!(fit.verdict, FitVerdict::Fits);
    }

    #[test]
    fn asking_for_no_context_uses_the_largest_that_fits() {
        let info = model(4);
        let hw = machine(64);
        let fit = info.fits(&hw, None);
        assert_eq!(fit.context, info.max_context(&hw));
    }

    #[test]
    fn vram_is_the_budget_when_a_discrete_gpu_reports_some() {
        let mut hw = machine(128);
        hw.vram_bytes = 8 * 1024 * 1024 * 1024;
        hw.gpu_backend = GpuBackend::Cuda;
        // 128 GB of RAM must not excuse a model that won't fit in 8 GB of VRAM.
        let fit = model(16).fits(&hw, Some(4096));
        assert_eq!(fit.verdict, FitVerdict::TooLarge);
    }
}
