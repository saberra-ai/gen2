//! Import-time metadata extraction for HuggingFace-convention model directories.

use crate::types::ModelMetadata;
use std::path::Path;

/// Build a [`ModelMetadata`] from a HuggingFace-convention model directory
/// (MLX safetensors or ONNX). Reads `config.json` for architecture dims,
/// `tokenizer_config.json` for chat template / tool support, and
/// `quantize_config.json` for quantization info (MLX).
pub fn parse_hf_model_metadata(model_dir: &Path) -> Option<ModelMetadata> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&config_str).ok()?;

    let architecture = cfg
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            cfg.get("architectures")
                .and_then(|a| a.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });

    let embedding_length = cfg.get("hidden_size").and_then(|v| v.as_u64());
    let block_count = cfg.get("num_hidden_layers").and_then(|v| v.as_u64());
    let head_count = cfg.get("num_attention_heads").and_then(|v| v.as_u64());
    let head_count_kv = cfg.get("num_key_value_heads").and_then(|v| v.as_u64());
    let vocab_size = cfg.get("vocab_size").and_then(|v| v.as_u64());
    let feed_forward_length = cfg.get("intermediate_size").and_then(|v| v.as_u64());
    let context_length = cfg.get("max_position_embeddings").and_then(|v| v.as_u64());

    // MoE fields (HF convention: num_local_experts / num_experts_per_tok)
    let expert_count = cfg.get("num_local_experts").and_then(|v| v.as_u64());
    let expert_used_count = cfg.get("num_experts_per_tok").and_then(|v| v.as_u64());

    // Need at least some architecture info to be useful
    if architecture.is_none() && block_count.is_none() {
        return None;
    }

    // Parameter count: standard transformer formula (attention + FFN + embeddings)
    // For MoE: FFN is multiplied by expert count (each expert has own FFN)
    let parameter_count = match (embedding_length, block_count) {
        (Some(d), Some(n_layer)) => {
            let d_ff = feed_forward_length.unwrap_or((d as f64 * 2.67) as u64);
            let vocab = vocab_size.unwrap_or(32000);
            let attention_params = n_layer * 4 * d * d;
            let n_experts = expert_count.unwrap_or(1);
            let ffn_params = n_layer * 3 * d * d_ff * n_experts;
            let embed_params = vocab * d;
            Some(attention_params + ffn_params + embed_params)
        }
        _ => None,
    };

    // Quantization from quantize_config.json (MLX convention)
    let quantization = detect_quantization(model_dir);

    // Chat template → tool support
    let supports_tools = super::load_chat_template(model_dir).map(|tpl| tpl.contains("tools"));

    Some(ModelMetadata {
        architecture,
        quantization,
        file_type: None,
        parameter_count,
        context_length,
        embedding_length,
        block_count,
        head_count,
        head_count_kv,
        vocab_size,
        feed_forward_length,
        supports_tools,
        expert_count,
        expert_used_count,
    })
}

/// Detect quantization from `quantize_config.json` (MLX convention).
///
/// Returns a label like "4-bit (group 64)" or "8-bit".
fn detect_quantization(model_dir: &Path) -> Option<String> {
    let qc_path = model_dir.join("quantize_config.json");
    let content = std::fs::read_to_string(&qc_path).ok()?;
    let qc: serde_json::Value = serde_json::from_str(&content).ok()?;

    let bits = qc.get("bits").and_then(|v| v.as_u64())?;
    let group_size = qc.get("group_size").and_then(|v| v.as_u64());

    Some(match group_size {
        Some(gs) => format!("{bits}-bit (group {gs})"),
        None => format!("{bits}-bit"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn write_config(dir: &Path, json: &str) {
        fs::write(dir.join("config.json"), json).unwrap();
    }

    fn write_tokenizer_config(dir: &Path, json: &str) {
        fs::write(dir.join("tokenizer_config.json"), json).unwrap();
    }

    fn write_quantize_config(dir: &Path, json: &str) {
        fs::write(dir.join("quantize_config.json"), json).unwrap();
    }

    #[test]
    fn llama_config() {
        let dir = TempDir::new().unwrap();
        write_config(
            dir.path(),
            r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "intermediate_size": 11008,
                "max_position_embeddings": 4096
            }"#,
        );

        let meta = parse_hf_model_metadata(dir.path()).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.embedding_length, Some(4096));
        assert_eq!(meta.block_count, Some(32));
        assert_eq!(meta.head_count, Some(32));
        assert_eq!(meta.head_count_kv, Some(8));
        assert_eq!(meta.vocab_size, Some(32000));
        assert_eq!(meta.feed_forward_length, Some(11008));
        assert_eq!(meta.context_length, Some(4096));
        assert!(meta.parameter_count.unwrap() > 0);
        assert!(meta.quantization.is_none());
        assert!(meta.supports_tools.is_none());
    }

    #[test]
    fn architectures_array_fallback() {
        let dir = TempDir::new().unwrap();
        write_config(
            dir.path(),
            r#"{
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 24,
                "hidden_size": 2048
            }"#,
        );

        let meta = parse_hf_model_metadata(dir.path()).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("Qwen2ForCausalLM"));
    }

    #[test]
    fn moe_fields_and_param_count() {
        let dir = TempDir::new().unwrap();
        write_config(
            dir.path(),
            r#"{
                "model_type": "mixtral",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "intermediate_size": 14336,
                "num_local_experts": 8,
                "num_experts_per_tok": 2
            }"#,
        );

        let meta = parse_hf_model_metadata(dir.path()).unwrap();
        assert_eq!(meta.expert_count, Some(8));
        assert_eq!(meta.expert_used_count, Some(2));
        let moe_ffn = 32 * 3 * 4096 * 14336 * 8;
        assert!(meta.parameter_count.unwrap() > moe_ffn);
    }

    #[test]
    fn returns_none_for_empty_config() {
        let dir = TempDir::new().unwrap();
        write_config(dir.path(), r#"{"torch_dtype": "float16"}"#);
        assert!(parse_hf_model_metadata(dir.path()).is_none());
    }

    #[test]
    fn returns_none_for_missing_config() {
        let dir = TempDir::new().unwrap();
        assert!(parse_hf_model_metadata(dir.path()).is_none());
    }

    #[test]
    fn quantization_with_group_size() {
        let dir = TempDir::new().unwrap();
        write_quantize_config(dir.path(), r#"{"bits": 4, "group_size": 64}"#);
        assert_eq!(
            detect_quantization(dir.path()).as_deref(),
            Some("4-bit (group 64)")
        );
    }

    #[test]
    fn quantization_without_group_size() {
        let dir = TempDir::new().unwrap();
        write_quantize_config(dir.path(), r#"{"bits": 8}"#);
        assert_eq!(detect_quantization(dir.path()).as_deref(), Some("8-bit"));
    }

    #[test]
    fn quantization_returns_none_when_missing() {
        let dir = TempDir::new().unwrap();
        assert!(detect_quantization(dir.path()).is_none());
    }

    #[test]
    fn tool_support_detected() {
        let dir = TempDir::new().unwrap();
        write_config(
            dir.path(),
            r#"{"model_type": "llama", "num_hidden_layers": 32}"#,
        );
        write_tokenizer_config(
            dir.path(),
            r#"{"chat_template": "{% if tools %}use tools{% endif %}"}"#,
        );
        let meta = parse_hf_model_metadata(dir.path()).unwrap();
        assert_eq!(meta.supports_tools, Some(true));
    }

    #[test]
    fn no_tool_support_when_template_lacks_tools() {
        let dir = TempDir::new().unwrap();
        write_config(
            dir.path(),
            r#"{"model_type": "llama", "num_hidden_layers": 32}"#,
        );
        write_tokenizer_config(
            dir.path(),
            r#"{"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}"#,
        );
        let meta = parse_hf_model_metadata(dir.path()).unwrap();
        assert_eq!(meta.supports_tools, Some(false));
    }
}
