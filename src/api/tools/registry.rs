//! Where tools live, and how deferred ones are found.

use std::collections::HashMap;
use std::sync::Arc;

use super::search::{Bm25Index, rank_by_similarity, reciprocal_rank_fusion};
use super::{Tool, ToolLoading, ToolSearch, ToolSpec};

/// Why an agent's tool configuration is unusable.
#[derive(Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum ToolConfigError {
    /// Two tools share a name; the model could not address either
    /// unambiguously.
    #[error("duplicate tool name: {0}")]
    DuplicateName(String),
    /// Tools were deferred with no way to find them, so they can never be
    /// called. Register a search strategy or make them resident.
    #[error("{0} tools are deferred but no tool search is configured — they could never be found")]
    DeferredToolsWithoutSearch(usize),
    /// A tool's description is what search matches on and what tells the model
    /// when to reach for it.
    #[error("tool '{0}' has no description")]
    MissingDescription(String),
}

/// The tools an agent can reach, and their loading state.
///
/// Deferred specs are kept out of the prompt entirely. When search finds one,
/// the agent records a hydration event and appends the spec to the
/// *conversation*, leaving the prompt prefix — and the warm KV cache built over
/// it — untouched.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    resident: Vec<String>,
    deferred: Vec<String>,
    /// Deferred tools already hydrated into this conversation.
    hydrated: Vec<String>,
    search: Option<ToolSearch>,
    lexical: Bm25Index,
    /// Deferred specs' embeddings, when a semantic strategy is in use.
    embeddings: Vec<(String, Vec<f32>)>,
}

// The constructor and hydration hooks are driven by the agent builder, which
// lands next. Kept together here so the registry is complete and testable on
// its own rather than half-written across two commits.
#[allow(dead_code)]
impl ToolRegistry {
    pub(crate) fn build(
        entries: Vec<(Arc<dyn Tool>, ToolLoading)>,
        search: Option<ToolSearch>,
    ) -> Result<Self, ToolConfigError> {
        // An empty registry is legal: an agent with no tools is a plain
        // reasoning loop, and rejecting it would make the tool-less case an
        // error rather than the simplest one.
        let mut tools = HashMap::new();
        let (mut resident, mut deferred) = (Vec::new(), Vec::new());

        for (tool, loading) in entries {
            let spec = tool.spec();
            if spec.description.trim().is_empty() {
                return Err(ToolConfigError::MissingDescription(spec.name.clone()));
            }
            let name = spec.name.clone();
            if tools.contains_key(&name) {
                return Err(ToolConfigError::DuplicateName(name));
            }
            match loading {
                ToolLoading::Resident => resident.push(name.clone()),
                ToolLoading::Deferred => deferred.push(name.clone()),
            }
            tools.insert(name, tool);
        }

        // Deferring everything without a way to search is the one
        // configuration that silently does nothing: the model is never told
        // those tools exist and has no way to ask.
        if !deferred.is_empty() && search.is_none() {
            return Err(ToolConfigError::DeferredToolsWithoutSearch(deferred.len()));
        }

        let deferred_specs: Vec<ToolSpec> = deferred
            .iter()
            .filter_map(|n| tools.get(n).map(|t| t.spec().clone()))
            .collect();

        Ok(Self {
            lexical: Bm25Index::build(&deferred_specs),
            tools,
            resident,
            deferred,
            hydrated: Vec::new(),
            search,
            embeddings: Vec::new(),
        })
    }

    /// Attach embeddings for the deferred specs, enabling semantic search.
    pub(crate) fn set_embeddings(&mut self, embeddings: Vec<(String, Vec<f32>)>) {
        self.embeddings = embeddings;
    }

    /// Text to embed for each deferred tool, in registry order.
    pub(crate) fn deferred_search_text(&self) -> Vec<(String, String)> {
        self.deferred
            .iter()
            .filter_map(|n| {
                self.tools
                    .get(n)
                    .map(|t| (n.clone(), t.spec().searchable_text()))
            })
            .collect()
    }

    /// Look up a tool by the name the model used.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Specs the model can see right now: the resident set plus anything
    /// hydrated so far this conversation.
    pub fn visible_specs(&self) -> Vec<ToolSpec> {
        self.resident
            .iter()
            .chain(&self.hydrated)
            .filter_map(|n| self.tools.get(n).map(|t| t.spec().clone()))
            .collect()
    }

    /// Whether a name is registered but not yet visible to the model.
    pub fn is_deferred(&self, name: &str) -> bool {
        self.deferred.contains(&name.to_string()) && !self.hydrated.contains(&name.to_string())
    }

    /// Configured search strategy, if any.
    pub fn search_strategy(&self) -> Option<ToolSearch> {
        self.search
    }

    pub fn resident_names(&self) -> &[String] {
        &self.resident
    }

    pub fn deferred_names(&self) -> &[String] {
        &self.deferred
    }

    /// Rank deferred, not-yet-hydrated tools against a query.
    ///
    /// `query_embedding` is required for the semantic half; without it Hybrid
    /// degrades to lexical rather than failing, so a missing embedder costs
    /// recall instead of the whole search.
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        query_embedding: Option<&[f32]>,
    ) -> Vec<ToolSpec> {
        let Some(strategy) = self.search else {
            return Vec::new();
        };

        let lexical = || self.lexical.search(query, limit * 2);
        let semantic = || match query_embedding {
            Some(q) if !self.embeddings.is_empty() => {
                rank_by_similarity(q, &self.embeddings, limit * 2)
            }
            _ => Vec::new(),
        };

        let ranked = match strategy {
            ToolSearch::Bm25 => lexical(),
            ToolSearch::Semantic => semantic(),
            ToolSearch::Hybrid => {
                let (l, s) = (lexical(), semantic());
                if s.is_empty() {
                    l
                } else if l.is_empty() {
                    s
                } else {
                    reciprocal_rank_fusion(&[l, s], limit * 2)
                }
            }
        };

        ranked
            .into_iter()
            .filter(|n| self.is_deferred(n))
            .take(limit)
            .filter_map(|n| self.tools.get(&n).map(|t| t.spec().clone()))
            .collect()
    }

    /// Record that a deferred tool's spec has entered the conversation.
    ///
    /// Idempotent: hydrating twice must not append the spec twice.
    pub(crate) fn mark_hydrated(&mut self, name: &str) -> bool {
        if self.is_deferred(name) {
            self.hydrated.push(name.to_string());
            true
        } else {
            false
        }
    }

    /// A stable fingerprint of the registered tool set.
    ///
    /// Over what is *registered*, not what is currently visible: hydration is
    /// designed to leave the prompt prefix alone, so a tool becoming visible
    /// mid-run must not read as a changed set and trigger a re-prefill.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut names: Vec<&String> = self.tools.keys().collect();
        names.sort();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for name in names {
            name.hash(&mut h);
            if let Some(t) = self.tools.get(name) {
                t.spec().input_schema.to_string().hash(&mut h);
            }
        }
        h.finish()
    }

    /// Deferred tools hydrated so far, in the order they were found.
    pub fn hydrated_names(&self) -> &[String] {
        &self.hydrated
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("resident", &self.resident)
            .field("deferred", &self.deferred.len())
            .field("hydrated", &self.hydrated)
            .field("search", &self.search)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tools::{FunctionTool, ToolOutput};
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}

    fn tool(name: &str, desc: &str) -> Arc<dyn Tool> {
        Arc::new(FunctionTool::new(name, desc, |_c, _a: NoArgs| async move {
            Ok(ToolOutput::from("ok"))
        }))
    }

    fn registry() -> ToolRegistry {
        ToolRegistry::build(
            vec![
                (
                    tool("read_file", "Read a file from disk"),
                    ToolLoading::Resident,
                ),
                (
                    tool("github_create_pull_request", "Open a pull request"),
                    ToolLoading::Deferred,
                ),
                (
                    tool("kubectl_apply", "Apply a manifest to a cluster"),
                    ToolLoading::Deferred,
                ),
            ],
            Some(ToolSearch::Bm25),
        )
        .unwrap()
    }

    #[test]
    fn only_resident_specs_are_visible_before_any_search() {
        let r = registry();
        let visible: Vec<String> = r.visible_specs().into_iter().map(|s| s.name).collect();
        assert_eq!(
            visible,
            ["read_file"],
            "deferred specs stay out of the prompt"
        );
    }

    #[test]
    fn deferring_everything_without_search_is_rejected_at_build() {
        // Otherwise the tools exist but nothing can ever reach them.
        let err = ToolRegistry::build(vec![(tool("a", "does a"), ToolLoading::Deferred)], None)
            .unwrap_err();
        assert_eq!(err, ToolConfigError::DeferredToolsWithoutSearch(1));
    }

    #[test]
    fn an_agent_with_no_tools_is_legal() {
        // The tool-less case is a plain reasoning loop, not a misconfiguration.
        let r = ToolRegistry::build(Vec::new(), None).expect("no tools is valid");
        assert!(r.visible_specs().is_empty());
        assert!(r.deferred_names().is_empty());
    }

    #[test]
    fn resident_tools_need_no_search() {
        assert!(
            ToolRegistry::build(vec![(tool("a", "does a"), ToolLoading::Resident)], None).is_ok()
        );
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let err = ToolRegistry::build(
            vec![
                (tool("a", "first"), ToolLoading::Resident),
                (tool("a", "second"), ToolLoading::Resident),
            ],
            None,
        )
        .unwrap_err();
        assert_eq!(err, ToolConfigError::DuplicateName("a".into()));
    }

    #[test]
    fn a_tool_without_a_description_is_rejected() {
        // Search matches on the description; an empty one is unfindable.
        let err =
            ToolRegistry::build(vec![(tool("a", "  "), ToolLoading::Resident)], None).unwrap_err();
        assert_eq!(err, ToolConfigError::MissingDescription("a".into()));
    }

    #[test]
    fn search_finds_a_deferred_tool_and_hydration_makes_it_visible() {
        let mut r = registry();
        let hits = r.search("pull request", 2, None);
        assert_eq!(hits[0].name, "github_create_pull_request");

        assert!(r.mark_hydrated("github_create_pull_request"));
        let visible: Vec<String> = r.visible_specs().into_iter().map(|s| s.name).collect();
        assert_eq!(visible, ["read_file", "github_create_pull_request"]);
    }

    #[test]
    fn an_already_hydrated_tool_is_not_returned_again() {
        // Re-appending a spec already in the conversation wastes context and
        // would show the model the same tool twice.
        let mut r = registry();
        r.mark_hydrated("github_create_pull_request");
        assert!(r.search("pull request", 2, None).is_empty());
        assert!(!r.mark_hydrated("github_create_pull_request"));
    }

    #[test]
    fn hybrid_degrades_to_lexical_when_no_embeddings_are_available() {
        let r = ToolRegistry::build(
            vec![
                (tool("read_file", "Read a file"), ToolLoading::Resident),
                (
                    tool("kubectl_apply", "Apply a manifest"),
                    ToolLoading::Deferred,
                ),
            ],
            Some(ToolSearch::Hybrid),
        )
        .unwrap();
        // Losing recall is acceptable; losing search entirely is not.
        assert_eq!(r.search("kubectl", 2, None)[0].name, "kubectl_apply");
    }

    #[test]
    fn the_fingerprint_tracks_registration_not_visibility() {
        // Hydration deliberately avoids touching the prompt prefix, so a tool
        // becoming visible must not look like a changed tool set.
        let mut r = registry();
        let before = r.fingerprint();
        r.mark_hydrated("github_create_pull_request");
        assert_eq!(r.fingerprint(), before, "hydration is not a set change");
    }

    #[test]
    fn different_tool_sets_fingerprint_differently() {
        let a = registry().fingerprint();
        let b = ToolRegistry::build(
            vec![(
                tool("read_file", "Read a file from disk"),
                ToolLoading::Resident,
            )],
            None,
        )
        .unwrap()
        .fingerprint();
        assert_ne!(a, b);
    }

    #[test]
    fn the_fingerprint_ignores_registration_order() {
        let one = ToolRegistry::build(
            vec![
                (tool("a", "does a"), ToolLoading::Resident),
                (tool("b", "does b"), ToolLoading::Resident),
            ],
            None,
        )
        .unwrap()
        .fingerprint();
        let two = ToolRegistry::build(
            vec![
                (tool("b", "does b"), ToolLoading::Resident),
                (tool("a", "does a"), ToolLoading::Resident),
            ],
            None,
        )
        .unwrap()
        .fingerprint();
        assert_eq!(one, two, "the same tools registered in either order match");
    }

    #[test]
    fn resident_tools_never_surface_as_search_results() {
        // Only deferred tools are indexed — a resident tool is already visible,
        // so returning it would hydrate something that needs no hydration.
        let r = registry();
        let hits: Vec<String> = r
            .search("read a file from disk", 3, None)
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert!(!hits.contains(&"read_file".to_string()), "got {hits:?}");
    }
}
