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

use super::model::{Gemma4Model, LlamaModel, Model, ModelConfig, Weight};
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
/// Quantized weights are packed uint32 along the *last* axis: each uint32 holds
/// `32/bits` values. For standard 2D weights `(out_features, packed_cols)` and
/// for 3D expert weights `(n_experts, out_features, packed_cols)` the formula
/// is the same: `bits = 32 * packed_cols / full_dim`. Accepts any N ≥ 2.
fn detect_bits(weight_shape: &[i32], full_dim: usize) -> Option<i32> {
    if weight_shape.len() < 2 || full_dim == 0 {
        return None;
    }
    let packed_cols = *weight_shape.last()? as usize;
    if packed_cols == 0 {
        return None;
    }
    let bits = (32 * packed_cols) / full_dim;
    if bits == 0 || bits > 8 || (32 * packed_cols) % full_dim != 0 {
        return None;
    }
    Some(bits as i32)
}

/// Detect group_size from scales shape.
///
/// scales shape ends with `num_groups` along the last axis:
/// 2D: `(out_features, num_groups)`, 3D: `(n_experts, out_features, num_groups)`.
/// `group_size = full_dim / num_groups` in either layout.
fn detect_group_size(scales_shape: &[i32], full_dim: usize) -> i32 {
    let Some(&last) = scales_shape.last() else {
        return 128;
    };
    if last == 0 {
        return 128;
    }
    let num_groups = last as usize;
    (full_dim / num_groups) as i32
}

/// Try to build a quantized `Weight` from the tensor map.
/// Looks for `{name}.weight`, `{name}.scales`, `{name}.biases`.
/// Returns `None` if the weight isn't quantized.
///
/// All Apple-Silicon-supported bit widths (2/3/4/5/6/8) stay quantized — the
/// forward-time `quantized_matmul` + `embedding_lookup` paths handle every
/// width we've seen in practice.
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
        // DIAGNOSTIC: if PIO_FORCE_DEQUANT=1, dequantize immediately and store
        // as plain float. Isolates bugs in quantized_matmul (e.g. 8-bit path)
        // vs correctness bugs elsewhere. Revert this once diagnosis is done.
        if std::env::var("PIO_FORCE_DEQUANT").is_ok() {
            return Weight::plain(qw.to_full());
        }
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

/// Load raw config.json as an untyped JSON value (used for nested Gemma 4 configs).
fn load_raw_config(model_dir: &Path) -> Result<serde_json::Value, ExecError> {
    let path = model_dir.join("config.json");
    let s = fs::read_to_string(&path)
        .map_err(|e| ExecError::Io(format!("failed to read config.json: {}", e)))?;
    serde_json::from_str(&s)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("failed to parse config.json: {}", e)))
}

/// Build a Gemma 4 model from a model directory.
///
/// Config is read from the top-level `text_config` sub-object when present
/// (Gemma 4 is a multimodal model that nests the LM config there).
/// All weights use the prefix `language_model.model.`.
pub fn build_gemma4_model(model_dir: &Path) -> Result<(Gemma4Model, ModelConfig), ExecError> {
    // Parse config: prefer text_config sub-object
    let raw = load_raw_config(model_dir)?;
    let text_cfg_value = raw
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| raw.clone());
    let mut config: ModelConfig = serde_json::from_value(text_cfg_value).map_err(|e| {
        ExecError::Other(anyhow::anyhow!(
            "failed to parse Gemma 4 text_config: {}",
            e
        ))
    })?;

    // Gemma 4 ships rope config nested under `rope_parameters` instead of the
    // flat `rope_theta` / `rope_local_base_freq` / `global_partial_rotary_factor`
    // fields. If the flat fields are absent and the nested dict is present,
    // pull per-layer-type values out and seed the flat fields so downstream
    // construction (Gemma4Model::new, Gemma4TransformerBlock::new) sees them.
    if let Some(rp) = config.rope_parameters.clone() {
        let full = rp.get("full_attention");
        let sliding = rp.get("sliding_attention");
        let as_f32 = |v: Option<&serde_json::Value>| v.and_then(|x| x.as_f64()).map(|x| x as f32);
        if let Some(t) = as_f32(full.and_then(|v| v.get("rope_theta"))) {
            config.rope_theta = t;
        }
        if config.rope_local_base_freq.is_none() {
            if let Some(t) = as_f32(sliding.and_then(|v| v.get("rope_theta"))) {
                config.rope_local_base_freq = Some(t);
            }
        }
        if config.global_partial_rotary_factor.is_none() {
            if let Some(f) = as_f32(full.and_then(|v| v.get("partial_rotary_factor"))) {
                config.global_partial_rotary_factor = Some(f);
            }
        }
    }

    let tensors = load_all_tensors(model_dir)?;
    let mut model = Gemma4Model::new(&config);

    let hidden = config.hidden_size;
    let n = config.num_hidden_layers;
    // PLE is Gemma 3n / Gemma 4 E-series only. 12B / 26B / 31B ship `0` here
    // and have no `embed_tokens_per_layer` / `per_layer_model_projection`
    // tensors in the checkpoint — skip those loads on the truthy gate.
    let hpl_raw = config.hidden_size_per_layer_input.unwrap_or(0);
    let has_ple = hpl_raw > 0;
    let hpl = hpl_raw;
    let hpl_all = hpl * n;

    // ── Global embeddings ────────────────────────────────────────────────────
    let pfx = "language_model.model";
    model.embed_tokens = load_weight(&tensors, &format!("{pfx}.embed_tokens"), hidden);
    if has_ple {
        model.embed_tokens_per_layer =
            load_weight(&tensors, &format!("{pfx}.embed_tokens_per_layer"), hpl_all);
        // Linear(hidden → num_layers × hpl) — feeds the residual stream into per-layer inputs.
        model.per_layer_model_projection = load_weight(
            &tensors,
            &format!("{pfx}.per_layer_model_projection"),
            hidden,
        );
        if let Some(w) = tensors.get(&format!("{pfx}.per_layer_projection_norm.weight")) {
            model.per_layer_projection_norm.weight = w.clone();
        }
    }

    if let Some(w) = tensors.get(&format!("{pfx}.norm.weight")) {
        model.norm.weight = w.clone();
    }

    // ── Transformer layers ───────────────────────────────────────────────────
    let n_non_shared = n.saturating_sub(config.num_kv_shared_layers.unwrap_or(0));
    let double_wide = config.use_double_wide_mlp.unwrap_or(false);
    for i in 0..n {
        let lp = format!("{pfx}.layers.{i}");
        let layer = &mut model.layers[i];
        let is_sliding = layer.attention.is_sliding;
        let head_dim = layer.attention.head_dim;

        // Shared-KV layers may have a 2× wider FFN intermediate size.
        let is_shared_kv = i >= n_non_shared;
        let layer_intermediate = if is_shared_kv && double_wide {
            config.intermediate_size * 2
        } else {
            config.intermediate_size
        };

        // Attention projections
        layer.attention.q_proj = load_weight(&tensors, &format!("{lp}.self_attn.q_proj"), hidden);
        layer.attention.k_proj = load_weight(&tensors, &format!("{lp}.self_attn.k_proj"), hidden);
        // 31B `attention_k_eq_v`: full-attention layers ship without v_proj.
        if layer.attention.use_k_eq_v {
            layer.attention.v_proj = None;
        } else {
            layer.attention.v_proj = Some(load_weight(
                &tensors,
                &format!("{lp}.self_attn.v_proj"),
                hidden,
            ));
        }
        let o_in = config.num_attention_heads * head_dim;
        layer.attention.o_proj = load_weight(&tensors, &format!("{lp}.self_attn.o_proj"), o_in);

        // Per-head norms (plain float, loaded directly)
        if let Some(w) = tensors.get(&format!("{lp}.self_attn.q_norm.weight")) {
            layer.attention.q_norm.weight = w.clone();
        }
        if let Some(w) = tensors.get(&format!("{lp}.self_attn.k_norm.weight")) {
            layer.attention.k_norm.weight = w.clone();
        }

        // FFN — gate/up use hidden as full_dim; down uses layer_intermediate
        layer.ffn.gate_proj = load_weight(&tensors, &format!("{lp}.mlp.gate_proj"), hidden);
        layer.ffn.up_proj = load_weight(&tensors, &format!("{lp}.mlp.up_proj"), hidden);
        layer.ffn.down_proj =
            load_weight(&tensors, &format!("{lp}.mlp.down_proj"), layer_intermediate);

        // Per-layer input system (PLE — E-series only).
        if has_ple {
            layer.per_layer_input.per_layer_input_gate =
                load_weight(&tensors, &format!("{lp}.per_layer_input_gate"), hidden);
            layer.per_layer_input.per_layer_projection =
                load_weight(&tensors, &format!("{lp}.per_layer_projection"), hpl);
            if let Some(w) = tensors.get(&format!("{lp}.post_per_layer_input_norm.weight")) {
                layer.per_layer_input.post_per_layer_input_norm.weight = w.clone();
            }
        }

        // Layer norms (plain float)
        if let Some(w) = tensors.get(&format!("{lp}.input_layernorm.weight")) {
            layer.input_layernorm.weight = w.clone();
        }
        if let Some(w) = tensors.get(&format!("{lp}.post_attention_layernorm.weight")) {
            layer.post_attention_layernorm.weight = w.clone();
        }
        if let Some(w) = tensors.get(&format!("{lp}.pre_feedforward_layernorm.weight")) {
            layer.pre_feedforward_layernorm.weight = w.clone();
        }
        if let Some(w) = tensors.get(&format!("{lp}.post_feedforward_layernorm.weight")) {
            layer.post_feedforward_layernorm.weight = w.clone();
        }

        // Layer scalar
        if let Some(w) = tensors.get(&format!("{lp}.layer_scalar")) {
            layer.layer_scalar = w.clone();
        }

        // ── MoE weights (Gemma 4 26B) ─────────────────────────────────────
        // Key naming follows mlx-lm `Model.sanitize()` in gemma4.py:
        // raw checkpoint keys are unsuffixed (`.experts.gate_up_proj`,
        // `.experts.down_proj`) — NOT `.weight`. Quantized variants would
        // use `.weight/.scales/.biases` triples on the same base path.
        if let Some(moe) = layer.moe.as_mut() {
            let moe_inter = config
                .moe_intermediate_size
                .unwrap_or(config.intermediate_size);

            // Router: `router.scale` (norm), `router.proj.weight`, `router.per_expert_scale`.
            if let Some(w) = tensors.get(&format!("{lp}.router.scale")) {
                moe.router.norm.weight = w.clone();
            }
            moe.router.proj = load_weight(&tensors, &format!("{lp}.router.proj"), hidden);
            if let Some(w) = tensors.get(&format!("{lp}.router.per_expert_scale")) {
                moe.router.per_expert_scale = w.clone();
            }

            // Experts — three observed layouts in the wild:
            //
            //   (a) `experts.switch_glu.{gate,up,down}_proj` (+ `.scales` / `.biases`
            //       when quantized). This is the actual layout in Gemma 4 26B
            //       (`gemma-4-26b-a4b-4bit` and related MLX packs).
            //   (b) `experts.{gate,up,down}_proj` (pre-split, unquantized).
            //   (c) `experts.gate_up_proj` fused (unquantized).
            //
            // Use `load_weight` so quantized triples decode correctly for (a).
            // Quantized experts are 3D `[n_experts, out, in_packed]` — the
            // generalized `detect_bits` / `detect_group_size` handle that.
            let gate_switch = format!("{lp}.experts.switch_glu.gate_proj");
            let up_switch = format!("{lp}.experts.switch_glu.up_proj");
            let down_switch = format!("{lp}.experts.switch_glu.down_proj");
            if tensors.contains_key(&format!("{gate_switch}.weight"))
                || tensors.contains_key(&gate_switch)
            {
                moe.experts.gate_proj = load_weight(&tensors, &gate_switch, hidden);
                moe.experts.up_proj = load_weight(&tensors, &up_switch, hidden);
                moe.experts.down_proj = load_weight(&tensors, &down_switch, moe_inter);
            } else {
                let bare_gate_up = format!("{lp}.experts.gate_up_proj");
                let weight_gate_up = format!("{lp}.experts.gate_up_proj.weight");
                let fused = tensors
                    .get(&bare_gate_up)
                    .or_else(|| tensors.get(&weight_gate_up));
                if let Some(gate_up) = fused {
                    let shape = gate_up.shape();
                    if shape.len() == 3 && shape[1] as usize == moe_inter * 2 {
                        let half = moe_inter as i32;
                        moe.experts.gate_proj = Weight::plain(gate_up.index((.., 0..half, ..)));
                        moe.experts.up_proj =
                            Weight::plain(gate_up.index((.., half..(2 * half), ..)));
                    }
                } else {
                    if let Some(w) = tensors.get(&format!("{lp}.experts.gate_proj")) {
                        moe.experts.gate_proj = Weight::plain(w.clone());
                    }
                    if let Some(w) = tensors.get(&format!("{lp}.experts.up_proj")) {
                        moe.experts.up_proj = Weight::plain(w.clone());
                    }
                }
                if let Some(w) = tensors
                    .get(&format!("{lp}.experts.down_proj"))
                    .or_else(|| tensors.get(&format!("{lp}.experts.down_proj.weight")))
                {
                    moe.experts.down_proj = Weight::plain(w.clone());
                }
            }

            // Extra MoE-only norms.
            if let Some(w) = tensors.get(&format!("{lp}.pre_feedforward_layernorm_2.weight")) {
                moe.pre_feedforward_layernorm_2.weight = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{lp}.post_feedforward_layernorm_1.weight")) {
                moe.post_feedforward_layernorm_1.weight = w.clone();
            }
            if let Some(w) = tensors.get(&format!("{lp}.post_feedforward_layernorm_2.weight")) {
                moe.post_feedforward_layernorm_2.weight = w.clone();
            }
        }

        let _ = is_sliding; // layer type already encoded in the struct at construction
    }

    let quantized_count = tensors.keys().filter(|k| k.ends_with(".scales")).count();
    tracing::info!(
        "loaded {} tensors ({} quantized groups) for Gemma 4 {}-layer model",
        tensors.len(),
        quantized_count,
        n
    );

    Ok((model, config))
}

/// Dispatch: detect model type from config.json and call the right builder.
pub fn build_any_model(model_dir: &Path) -> Result<(Model, ModelConfig), ExecError> {
    let raw = load_raw_config(model_dir)?;
    // Gemma 4 nests model_type in text_config; fall back to root model_type
    let model_type = raw
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .or_else(|| raw.get("model_type"))
        .and_then(|v| v.as_str());

    let is_gemma = model_type.map_or(false, |t| t.contains("gemma"));

    if is_gemma {
        let (m, c) = build_gemma4_model(model_dir)?;
        return Ok((Model::Gemma4(m), c));
    }

    if let Some(t) = model_type {
        tracing::warn!(
            "unknown model_type {:?}, defaulting to Llama — add to build_any_model() if wrong",
            t
        );
    }
    let (m, c) = build_model(model_dir)?;
    Ok((Model::Llama(m), c))
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
