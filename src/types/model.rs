//! Model descriptors — the catalog record a backend is asked to load.
//!
//! Extracted from `pio-core`'s `types` module during the gen2 crate split.
//! These are plain serde records with no host-app coupling: a `Model` names a
//! file on disk plus the knobs a session starts with, and `ModelMetadata`
//! carries what the loader read out of the model's own header.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelConfig {
    pub context_size: u32,
    pub seed: Option<u32>,
    pub batch_size: usize,
    pub gpu_layers: Option<i32>,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_p_keep: usize,
    pub top_k: i32,
    pub repeat_penalty: i32,
}

/// Rich metadata extracted from model files at import time (GGUF headers, HF config.json).
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelMetadata {
    /// Model architecture family (e.g. "llama", "qwen2", "phi3").
    pub architecture: Option<String>,
    /// Human-readable quantization label (e.g. "Q4_K_M", "F16").
    pub quantization: Option<String>,
    /// Raw GGUF `general.file_type` enum value.
    pub file_type: Option<u32>,
    /// Estimated total parameter count.
    pub parameter_count: Option<u64>,
    /// Training context length from GGUF header.
    pub context_length: Option<u64>,
    /// Hidden dimension (`{arch}.embedding_length`).
    pub embedding_length: Option<u64>,
    /// Number of transformer layers (`{arch}.block_count`).
    pub block_count: Option<u64>,
    /// Number of attention heads (`{arch}.attention.head_count`).
    pub head_count: Option<u64>,
    /// Number of KV heads for GQA (`{arch}.attention.head_count_kv`).
    pub head_count_kv: Option<u64>,
    /// Vocabulary size.
    pub vocab_size: Option<u64>,
    /// FFN intermediate dimension (`{arch}.feed_forward_length`).
    pub feed_forward_length: Option<u64>,
    /// Whether the chat template references tool-use variables.
    pub supports_tools: Option<bool>,
    /// Total number of experts (MoE models only).
    pub expert_count: Option<u64>,
    /// Number of experts used per forward pass (MoE models only).
    pub expert_used_count: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Model {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub provider: Option<String>,
    pub model_path: Option<String>,
    /// Model format: "gguf", "mlx", "onnx", or None (legacy/unknown).
    pub format: Option<String>,
    /// SHA-256 hex digest of the model file, computed after download.
    #[serde(default)]
    pub sha256: Option<String>,
    pub is_selected: bool,
    pub created_at: i64,
    pub updated_at: i64,
    pub config: ModelConfig,
    /// Rich metadata extracted from GGUF headers (None for non-GGUF or legacy models).
    #[serde(default)]
    pub metadata: Option<ModelMetadata>,
}
