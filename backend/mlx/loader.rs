//! Safetensors model loader for MLX backend.
//!
//! Supports both plain float and quantized (1-bit/2-bit/4-bit/8-bit) models.
//! Quantized models store (weight, scales, biases) triples per tensor;
//! the loader detects these and wraps them in `Weight::quantized`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use super::model::{LlamaModel, ModelConfig, Weight};
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

/// Detect the quantization bit-width from a weight tensor.
///
/// Quantized weights are packed uint32: each uint32 holds `32/bits` values.
/// Given a weight shape of `(out_features, packed_cols)` and the expected
/// full dimension `full_dim`, the bits = `32 * packed_cols / full_dim`.
fn detect_bits(weight_shape: &[i32], full_dim: usize) -> Option<i32> {
    if weight_shape.len() != 2 {
        return None;
    }
    let packed_cols = weight_shape[1] as usize;
    if packed_cols == 0 || full_dim == 0 {
        return None;
    }
    // packed_cols = full_dim * bits / 32, so bits = 32 * packed_cols / full_dim
    let bits = (32 * packed_cols) / full_dim;
    if bits == 0 || bits > 8 || (32 * packed_cols) % full_dim != 0 {
        return None;
    }
    Some(bits as i32)
}

/// Detect group_size from scales shape.
///
/// scales shape: `(out_features, num_groups)` where `group_size = full_dim / num_groups`.
fn detect_group_size(scales_shape: &[i32], full_dim: usize) -> i32 {
    if scales_shape.len() != 2 || scales_shape[1] == 0 {
        return 128; // default fallback
    }
    let num_groups = scales_shape[1] as usize;
    (full_dim / num_groups) as i32
}

/// Try to build a quantized `Weight` from the tensor map.
/// Looks for `{name}.weight`, `{name}.scales`, `{name}.biases`.
/// Returns `None` if the weight isn't quantized.
fn try_quantized_weight(
    tensors: &HashMap<String, Array>,
    name: &str,
    full_dim: usize,
) -> Option<Weight> {
    let weight = tensors.get(&format!("{}.weight", name))?;
    let scales = tensors.get(&format!("{}.scales", name))?;
    let biases = tensors.get(&format!("{}.biases", name))?;

    let bits = detect_bits(weight.shape(), full_dim)?;
    let group_size = detect_group_size(scales.shape(), full_dim);

    Some(Weight::quantized(
        weight.clone(),
        scales.clone(),
        biases.clone(),
        group_size,
        bits,
    ))
}

/// Load a weight: try quantized first, fall back to plain float.
fn load_weight(tensors: &HashMap<String, Array>, name: &str, full_dim: usize) -> Weight {
    // Check for quantized triple: name.weight + name.scales + name.biases
    if let Some(qw) = try_quantized_weight(tensors, name, full_dim) {
        return qw;
    }
    // Plain float: just name.weight
    if let Some(w) = tensors.get(&format!("{}.weight", name)) {
        return Weight::plain(w.clone());
    }
    // Nothing found — return the default zero placeholder
    Weight::default()
}

/// Build a LlamaModel from a model directory containing config.json and *.safetensors.
pub fn build_model(model_dir: &Path) -> Result<(LlamaModel, ModelConfig), ExecError> {
    let config = load_config(model_dir)?;
    let tensors = load_all_tensors(model_dir)?;

    let mut model = LlamaModel::new(&config);
    let hidden = config.hidden_size;

    // ── Embedding + LM head ─────────────────────────────────────
    model.embed_tokens = load_weight(&tensors, "model.embed_tokens", hidden);

    // lm_head: check quantized, then plain, then tie to embeddings
    if let Some(qw) = try_quantized_weight(&tensors, "lm_head", hidden) {
        model.lm_head = qw;
    } else if let Some(w) = tensors.get("lm_head.weight") {
        model.lm_head = Weight::plain(w.clone());
    } else {
        // Tie to embeddings (dequantize if needed)
        model.lm_head = Weight::plain(model.embed_tokens.to_full());
    }

    if let Some(w) = tensors.get("model.norm.weight") {
        model.norm.weight = w.clone();
    }

    // ── Transformer layers ──────────────────────────────────────
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}", i);
        if let Some(layer) = model.layers.get_mut(i) {
            let head_dim = config.head_dim();

            // Attention projections
            layer.attention.q_proj =
                load_weight(&tensors, &format!("{}.self_attn.q_proj", prefix), hidden);
            layer.attention.k_proj =
                load_weight(&tensors, &format!("{}.self_attn.k_proj", prefix), hidden);
            layer.attention.v_proj =
                load_weight(&tensors, &format!("{}.self_attn.v_proj", prefix), hidden);
            layer.attention.o_proj = load_weight(
                &tensors,
                &format!("{}.self_attn.o_proj", prefix),
                config.num_attention_heads * head_dim,
            );

            // FFN projections
            let gate_key = format!("{}.mlp.gate_proj", prefix);
            let up_key = format!("{}.mlp.up_proj", prefix);
            let down_key = format!("{}.mlp.down_proj", prefix);

            layer.ffn.gate_proj = load_weight(&tensors, &gate_key, hidden);
            layer.ffn.up_proj = load_weight(&tensors, &up_key, hidden);
            layer.ffn.down_proj = load_weight(&tensors, &down_key, config.intermediate_size);

            // Fused gate_up_proj fallback (some models fuse gate+up into one tensor)
            if tensors.get(&format!("{}.weight", gate_key)).is_none()
                && try_quantized_weight(&tensors, &gate_key, hidden).is_none()
            {
                let fused_key = format!("{}.mlp.gate_up_proj", prefix);
                if let Some(w) = tensors.get(&format!("{}.weight", fused_key)) {
                    // Shape: (2 * intermediate_size, hidden_size) — split into gate and up
                    let shape = w.shape();
                    if shape.len() == 2 {
                        let half = shape[0] / 2;
                        layer.ffn.gate_proj = Weight::plain(w.index((0..half, ..)));
                        layer.ffn.up_proj = Weight::plain(w.index((half..shape[0], ..)));
                    }
                }
            }

            // Layer norms (always plain float)
            if let Some(w) = tensors.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_norm.weight = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.post_attn_norm.weight = w.clone();
            }
        }
    }

    let quantized_count = tensors.keys().filter(|k| k.ends_with(".scales")).count();
    tracing::info!(
        "loaded {} tensors ({} quantized groups) for {}-layer model",
        tensors.len(),
        quantized_count,
        config.num_hidden_layers
    );

    Ok((model, config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_bits_1bit() {
        // embed_tokens: (151669, 128) uint32, hidden_size=4096
        // 128 packed cols * 32 bits = 4096 values → 1 bit per value
        assert_eq!(detect_bits(&[151669, 128], 4096), Some(1));
    }

    #[test]
    fn detect_bits_4bit() {
        // q_proj: (4096, 512) uint32, hidden_size=4096
        // 512 packed cols * 32 bits = 16384, 16384/4096 = 4 bits
        assert_eq!(detect_bits(&[4096, 512], 4096), Some(4));
    }

    #[test]
    fn detect_bits_8bit() {
        // 1024 packed cols * 32 bits = 32768, 32768/4096 = 8 bits
        assert_eq!(detect_bits(&[4096, 1024], 4096), Some(8));
    }

    #[test]
    fn detect_bits_plain_float_returns_none() {
        // Plain f16: (4096, 4096) — packed_cols == full_dim → 32 bits, which is > 8
        assert_eq!(detect_bits(&[4096, 4096], 4096), None);
    }

    #[test]
    fn detect_bits_wrong_rank_returns_none() {
        assert_eq!(detect_bits(&[4096], 4096), None);
        assert_eq!(detect_bits(&[1, 2, 3], 4096), None);
    }

    #[test]
    fn detect_bits_zero_dims_returns_none() {
        assert_eq!(detect_bits(&[0, 128], 4096), None);
        assert_eq!(detect_bits(&[4096, 0], 4096), None);
        assert_eq!(detect_bits(&[4096, 128], 0), None);
    }

    #[test]
    fn detect_bits_non_divisible_returns_none() {
        // 100 packed cols * 32 = 3200, 3200/4096 is not an integer
        assert_eq!(detect_bits(&[4096, 100], 4096), None);
    }

    #[test]
    fn detect_group_size_128() {
        // scales: (4096, 32), full_dim=4096 → 4096/32 = 128
        assert_eq!(detect_group_size(&[4096, 32], 4096), 128);
    }

    #[test]
    fn detect_group_size_64() {
        // scales: (4096, 64), full_dim=4096 → 4096/64 = 64
        assert_eq!(detect_group_size(&[4096, 64], 4096), 64);
    }

    #[test]
    fn detect_group_size_fallback() {
        // Invalid scales shape → fallback to 128
        assert_eq!(detect_group_size(&[4096], 4096), 128);
        assert_eq!(detect_group_size(&[4096, 0], 4096), 128);
    }
}
