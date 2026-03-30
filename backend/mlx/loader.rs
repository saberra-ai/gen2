//! Safetensors model loader for MLX backend.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use super::model::{LlamaModel, ModelConfig};
use crate::gen2::engine::ExecError;

/// Load model config from `config.json` in the model directory.
pub fn load_config(model_dir: &Path) -> Result<ModelConfig, ExecError> {
    let config_path = model_dir.join("config.json");
    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| ExecError::Io(format!("failed to read config.json: {}", e)))?;
    let config: ModelConfig = serde_json::from_str(&config_str)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("failed to parse config.json: {}", e)))?;
    Ok(config)
}

/// Discover all safetensors files in a model directory and load all tensors.
fn load_all_tensors(model_dir: &Path) -> Result<HashMap<String, Array>, ExecError> {
    let mut tensors = HashMap::new();

    // Find all .safetensors files
    let mut safetensor_files: Vec<_> = fs::read_dir(model_dir)
        .map_err(|e| ExecError::Io(format!("failed to read model dir: {}", e)))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    safetensor_files.sort(); // deterministic order

    for path in &safetensor_files {
        // Use mlx_rs built-in safetensors loading which handles dtype correctly
        let file_tensors = Array::load_safetensors(path).map_err(|e| {
            ExecError::Other(anyhow::anyhow!("failed to load {}: {}", path.display(), e))
        })?;

        tensors.extend(file_tensors);
    }

    Ok(tensors)
}

/// Build a LlamaModel from a model directory containing config.json and *.safetensors.
pub fn build_model(model_dir: &Path) -> Result<(LlamaModel, ModelConfig), ExecError> {
    let config = load_config(model_dir)?;
    let tensors = load_all_tensors(model_dir)?;

    let mut model = LlamaModel::new(&config);

    // Map safetensors tensor names to model weights.
    // HuggingFace Llama naming convention:
    //   model.embed_tokens.weight
    //   model.layers.{i}.self_attn.q_proj.weight
    //   model.layers.{i}.self_attn.k_proj.weight
    //   model.layers.{i}.self_attn.v_proj.weight
    //   model.layers.{i}.self_attn.o_proj.weight
    //   model.layers.{i}.mlp.gate_proj.weight
    //   model.layers.{i}.mlp.up_proj.weight
    //   model.layers.{i}.mlp.down_proj.weight
    //   model.layers.{i}.input_layernorm.weight
    //   model.layers.{i}.post_attention_layernorm.weight
    //   model.norm.weight
    //   lm_head.weight

    if let Some(w) = tensors.get("model.embed_tokens.weight") {
        model.embed_tokens = w.clone();
    }
    if let Some(w) = tensors.get("lm_head.weight") {
        model.lm_head = w.clone();
    } else if config.tie_word_embeddings {
        model.lm_head = model.embed_tokens.clone();
    } else {
        // Fallback: tie anyway (common default)
        model.lm_head = model.embed_tokens.clone();
    }
    if let Some(w) = tensors.get("model.norm.weight") {
        model.norm.weight = w.clone();
    }

    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}", i);
        if let Some(layer) = model.layers.get_mut(i) {
            if let Some(w) = tensors.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.attention.q_proj = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                layer.attention.k_proj = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                layer.attention.v_proj = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                layer.attention.o_proj = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                layer.ffn.gate_proj = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                layer.ffn.up_proj = w.clone();
            }
            // Fused gate_up_proj fallback (some models fuse gate+up into one tensor)
            if tensors
                .get(&format!("{}.mlp.gate_proj.weight", prefix))
                .is_none()
            {
                if let Some(w) = tensors.get(&format!("{}.mlp.gate_up_proj.weight", prefix)) {
                    // Shape: (2 * intermediate_size, hidden_size) — split into gate and up
                    let shape = w.shape();
                    if shape.len() == 2 {
                        let half = shape[0] / 2;
                        layer.ffn.gate_proj = w.index((0..half as i32, ..));
                        layer.ffn.up_proj = w.index((half as i32..shape[0] as i32, ..));
                    }
                }
            }
            if let Some(w) = tensors.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                layer.ffn.down_proj = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_norm.weight = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.post_attn_norm.weight = w.clone();
            }
        }
    }

    tracing::info!(
        "loaded {} tensors for {}-layer model",
        tensors.len(),
        config.num_hidden_layers
    );

    Ok((model, config))
}
