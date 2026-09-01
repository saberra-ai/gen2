//! [`Skill`] — instructions loaded when they become relevant.
//!
//! The same trade as deferred tools, applied to prose. A skill's description
//! sits in the prompt; its body — the actual procedure, conventions, examples —
//! arrives only when the model asks for it. Twenty skills cost twenty lines of
//! context instead of twenty documents.
//!
//! Implemented as a tool because that is all it needs to be: the model calls
//! `load_skill`, and the body comes back as a tool result — which lands in the
//! conversation, not the prompt prefix, leaving the warm cache intact.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use super::{Tool, ToolContext, ToolError, ToolOutput, ToolSpec};

/// A named body of instructions the model can pull in on demand.
#[derive(Debug, Clone)]
pub struct Skill {
    /// How the model refers to it.
    pub name: String,
    /// One line, always in context. This is what the model reads when deciding
    /// whether the skill is worth loading, so write it as a trigger condition
    /// ("when writing a migration") rather than a summary.
    pub description: String,
    /// The full instructions, loaded only when asked for.
    pub body: String,
}

impl Skill {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            body: body.into(),
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct LoadArgs {
    /// Name of the skill to load.
    skill: String,
}

/// The tool that loads skills, plus the catalogue it loads from.
///
/// Register one of these and every skill becomes reachable through it.
pub struct SkillLibrary {
    spec: ToolSpec,
    skills: HashMap<String, Skill>,
    loaded: Arc<Mutex<Vec<String>>>,
}

impl SkillLibrary {
    /// Build a library from a set of skills.
    ///
    /// Each skill's name and description go into the tool's own description, so
    /// the model can see what exists without any body being in context.
    pub fn new(skills: impl IntoIterator<Item = Skill>) -> Self {
        let skills: HashMap<String, Skill> =
            skills.into_iter().map(|s| (s.name.clone(), s)).collect();

        let mut names: Vec<&Skill> = skills.values().collect();
        // Stable order so the prompt prefix doesn't churn between runs — a
        // reordered tool description invalidates the cached prefill.
        names.sort_by(|a, b| a.name.cmp(&b.name));

        let catalogue = names
            .iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");

        Self {
            spec: ToolSpec::new(
                "load_skill",
                format!("Load detailed instructions for a task. Available skills:\n{catalogue}"),
                serde_json::to_value(schemars::schema_for!(LoadArgs))
                    .unwrap_or_else(|_| serde_json::json!({"type": "object"})),
            ),
            skills,
            loaded: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Skills pulled in so far.
    pub fn loaded(&self) -> Vec<String> {
        self.loaded.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// How many skills the library holds.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

#[async_trait]
impl Tool for SkillLibrary {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(
        &self,
        _ctx: &ToolContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let args: LoadArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let Some(skill) = self.skills.get(&args.skill) else {
            // Naming a skill that doesn't exist is the model's to fix, and the
            // list of real ones is already in the tool description.
            return Err(ToolError::InvalidArguments(format!(
                "no skill named '{}'",
                args.skill
            )));
        };

        if let Ok(mut l) = self.loaded.lock()
            && !l.contains(&skill.name)
        {
            l.push(skill.name.clone());
        }

        Ok(ToolOutput::Text(skill.body.clone()))
    }
}

impl std::fmt::Debug for SkillLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillLibrary")
            .field("skills", &self.skills.len())
            .field("loaded", &self.loaded())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn library() -> SkillLibrary {
        SkillLibrary::new([
            Skill::new(
                "migrations",
                "when writing a database migration",
                "Always add a down migration. Never drop a column in the same release.",
            ),
            Skill::new(
                "commits",
                "when writing a commit message",
                "Imperative mood.",
            ),
        ])
    }

    #[test]
    fn the_catalogue_is_in_context_but_the_bodies_are_not() {
        // This is the whole trade: descriptions cost two lines, bodies cost
        // nothing until asked for.
        let lib = library();
        let d = &lib.spec().description;
        assert!(d.contains("migrations"), "{d}");
        assert!(d.contains("when writing a database migration"));
        assert!(
            !d.contains("Always add a down migration"),
            "a body leaked into the prompt: {d}"
        );
    }

    #[test]
    fn the_catalogue_order_is_stable() {
        // A reordered description changes the prompt prefix, which throws away
        // the warm KV cache built over it.
        let a = library().spec().description.clone();
        let b = library().spec().description.clone();
        assert_eq!(a, b);
        assert!(a.find("commits") < a.find("migrations"), "sorted by name");
    }

    #[tokio::test]
    async fn loading_returns_the_body_and_records_it() {
        let lib = library();
        let ctx = ToolContext::new("s1");
        let out = lib
            .call(&ctx, serde_json::json!({"skill": "migrations"}))
            .await
            .unwrap();
        assert!(out.to_model_text().contains("down migration"));
        assert_eq!(lib.loaded(), ["migrations"]);
    }

    #[tokio::test]
    async fn loading_the_same_skill_twice_records_it_once() {
        let lib = library();
        let ctx = ToolContext::new("s1");
        for _ in 0..2 {
            lib.call(&ctx, serde_json::json!({"skill": "commits"}))
                .await
                .unwrap();
        }
        assert_eq!(lib.loaded(), ["commits"]);
    }

    #[tokio::test]
    async fn an_unknown_skill_is_correctable_not_fatal() {
        let lib = library();
        let ctx = ToolContext::new("s1");
        let err = lib
            .call(&ctx, serde_json::json!({"skill": "nope"}))
            .await
            .unwrap_err();
        assert!(
            err.is_model_actionable(),
            "the real names are in the description"
        );
    }
}
