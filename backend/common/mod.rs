//! Shared utilities across inference backends.

pub(crate) mod chat_template;
mod fingerprint;
mod hf_meta;
pub mod sampler;
pub mod tokenizer;

use std::path::Path;

// Re-export the public API so call sites don't change.
pub use fingerprint::compute_hf_model_meta;
pub use hf_meta::parse_hf_model_metadata;

/// Load the Jinja2 chat template from `tokenizer_config.json` in a model directory.
///
/// Used by import-time metadata extraction, runtime fingerprinting, and
/// engine bundle construction.
pub(crate) fn load_chat_template(model_dir: &Path) -> Option<String> {
    let config_path = model_dir.join("tokenizer_config.json");
    let content = std::fs::read_to_string(&config_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
    parsed
        .get("chat_template")?
        .as_str()
        .map(|s| s.to_string())
}

/// Fallback Llama 3 Instruct chat template for models without `tokenizer_config.json`.
pub(crate) fn default_llama3_template() -> String {
    r#"{% for message in messages %}{% if message.role == 'system' %}<|start_header_id|>system<|end_header_id|>

{{ message.content }}<|eot_id|>{% elif message.role == 'user' %}<|start_header_id|>user<|end_header_id|>

{{ message.content }}<|eot_id|>{% elif message.role == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

{{ message.content }}<|eot_id|>{% endif %}{% endfor %}<|start_header_id|>assistant<|end_header_id|>

"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn load_chat_template_extracts_string() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}"#,
        )
        .unwrap();

        let tpl = load_chat_template(dir.path()).unwrap();
        assert!(tpl.contains("messages"));
    }

    #[test]
    fn load_chat_template_returns_none_when_missing() {
        let dir = TempDir::new().unwrap();
        assert!(load_chat_template(dir.path()).is_none());
    }
}
