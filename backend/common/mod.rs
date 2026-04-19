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

/// Conservative fallback template for Hugging Face-style chat models when
/// `tokenizer_config.json` does not declare one.
#[allow(dead_code)] // Wired when tokenizer metadata omits template; keep for upcoming HF path.
pub(crate) fn default_llama3_template() -> String {
    r#"
{%- for message in messages -%}
<|start_header_id|>{{ message.role }}<|end_header_id|>

{{ message.content }}<|eot_id|>
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>

{%- endif -%}
"#
    .trim()
    .to_string()
}

/// Load the Jinja2 chat template from `tokenizer_config.json` in a model directory.
///
/// Used by import-time metadata extraction, runtime fingerprinting, and
/// engine bundle construction.
pub(crate) fn load_chat_template(model_dir: &Path) -> Option<String> {
    // Preferred: embedded in tokenizer_config.json.
    if let Ok(content) = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content)
        && let Some(s) = parsed.get("chat_template").and_then(|v| v.as_str())
    {
        return Some(s.to_string());
    }
    // Fallback: sidecar `chat_template.jinja` (Gemma 4+ stores it this way).
    std::fs::read_to_string(model_dir.join("chat_template.jinja")).ok()
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
