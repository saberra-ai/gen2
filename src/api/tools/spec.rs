//! What the model sees: a tool's name, purpose, and argument schema.

use serde::{Deserialize, Serialize};

/// The model-visible description of an executable tool.
///
/// A [`Tool`](super::Tool) is the thing that runs; a `ToolSpec` is the context
/// that tells the model it exists and how to call it. Backends render this into
/// whatever shape their chat template expects, so the same spec works across
/// llama.cpp, MLX, and an OpenAI-compatible endpoint.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolSpec {
    /// Identifier the model calls. Unique within a registry.
    pub name: String,
    /// What the tool does, in the model's terms. This is the primary signal
    /// tool search matches against, so write it for retrieval, not just docs.
    pub description: String,
    /// JSON Schema for the arguments object.
    pub input_schema: serde_json::Value,
}

impl ToolSpec {
    /// A spec with a hand-written schema.
    ///
    /// Prefer [`FunctionTool`](super::FunctionTool), which derives the schema
    /// from the argument type. Use this for dynamic sources — MCP servers,
    /// plugins — where the schema arrives as data.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }

    /// The text tool search indexes for this spec.
    ///
    /// Deliberately more than the description: argument names and their doc
    /// comments carry the exact terminology a query is likely to use
    /// (`repository`, `namespace`, `invoice_id`), which is precisely what
    /// lexical search matches and a prose description often omits.
    pub fn searchable_text(&self) -> String {
        let mut out = format!("{} {}", self.name, self.description);
        if let Some(props) = self
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
        {
            for (arg, schema) in props {
                out.push(' ');
                out.push_str(arg);
                if let Some(d) = schema.get("description").and_then(|d| d.as_str()) {
                    out.push(' ');
                    out.push_str(d);
                }
            }
        }
        out
    }
}

/// Whether a tool's spec sits in the prompt or is discovered on demand.
///
/// A property of the *registry*, not the tool: the same `github_search` may be
/// resident in a coding agent and deferred in a general assistant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ToolLoading {
    /// In the prompt from the first turn. Keep this set small — Anthropic's
    /// guidance is roughly 3–5 common tools.
    #[default]
    Resident,
    /// Absent from the prompt until tool search finds it, at which point its
    /// spec is appended to the conversation rather than the prefix.
    Deferred,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ToolSpec {
        ToolSpec::new(
            "github_create_pull_request",
            "Open a pull request",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "repository": { "type": "string", "description": "owner/name" },
                    "draft": { "type": "boolean" }
                }
            }),
        )
    }

    #[test]
    fn searchable_text_includes_argument_names_and_descriptions() {
        // The exact terms a query uses often live in the arguments, not the
        // prose — indexing the description alone loses them.
        let text = spec().searchable_text();
        assert!(text.contains("github_create_pull_request"));
        assert!(text.contains("Open a pull request"));
        assert!(text.contains("repository"), "argument names are indexed");
        assert!(text.contains("owner/name"), "argument docs are indexed");
        assert!(text.contains("draft"), "arguments without docs still index");
    }

    #[test]
    fn a_schema_without_properties_still_indexes() {
        let s = ToolSpec::new("now", "Current time", serde_json::json!({"type":"object"}));
        assert_eq!(s.searchable_text(), "now Current time");
    }
}
