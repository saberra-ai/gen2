//! Tool definitions as data: what the model is told, with nothing to run.
//!
//! A [`ToolDefinition`] is a name, a description, and an argument schema. It
//! carries no closure, because the inference core never executes a tool
//! (api_spec.md §10, invariant 4): the model declares a call, the caller runs
//! it, and the result comes back through
//! [`Session::push_tool_result`](crate::Session::push_tool_result).
//!
//! A [`ToolSet`] is an ordered collection of them, and it is first-order
//! session state — [`Session::set_tools`](crate::Session::set_tools) — rather
//! than something an agent layer owns. Ordering is deterministic so the
//! rendered tool prefix, and therefore a cached prefill, is stable across
//! turns.
//!
//! The executable [`Tool`](crate::Tool) / [`ToolSpec`](crate::api::tools::ToolSpec)
//! pair in [`api::tools`](crate::api::tools) is the agent layer's; a spec
//! converts into a definition with [`From`], so the two do not drift.

use serde::{Deserialize, Serialize};

use crate::types::message::{FunctionDefinition, ToolSpec as WireToolSpec};

/// What the model is told about one tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The identifier the model calls it by. Unique within a [`ToolSet`].
    pub name: String,
    /// What the tool does, in the model's terms.
    pub description: String,
    /// JSON Schema for the arguments object.
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// A definition with an empty description and an unconstrained argument
    /// object. Fill it in with [`ToolDefinition::description`] and
    /// [`ToolDefinition::input_schema`] or [`ToolDefinition::schema`].
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            input_schema: serde_json::json!({ "type": "object" }),
        }
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Derive the argument schema from a Rust type.
    ///
    /// Uses the [`schemars`] this crate compiled against — reach it as
    /// `gen2::schemars` rather than adding your own dependency.
    #[must_use]
    pub fn input_schema<T: schemars::JsonSchema>(mut self) -> Self {
        self.input_schema = serde_json::to_value(schemars::schema_for!(T))
            .unwrap_or_else(|_| serde_json::json!({ "type": "object" }));
        self
    }

    /// Set the argument schema from JSON — for schemas that arrive as data.
    #[must_use]
    pub fn schema(mut self, input_schema: serde_json::Value) -> Self {
        self.input_schema = input_schema;
        self
    }
}

impl From<crate::api::tools::ToolSpec> for ToolDefinition {
    fn from(spec: crate::api::tools::ToolSpec) -> Self {
        Self {
            name: spec.name,
            description: spec.description,
            input_schema: spec.input_schema,
        }
    }
}

impl From<&crate::api::tools::ToolSpec> for ToolDefinition {
    fn from(spec: &crate::api::tools::ToolSpec) -> Self {
        Self::from(spec.clone())
    }
}

impl From<WireToolSpec> for ToolDefinition {
    fn from(spec: WireToolSpec) -> Self {
        Self {
            name: spec.function.name,
            description: spec.function.description.unwrap_or_default(),
            input_schema: spec.function.arguments,
        }
    }
}

impl From<ToolDefinition> for WireToolSpec {
    fn from(def: ToolDefinition) -> Self {
        WireToolSpec {
            r#type: "function".into(),
            function: FunctionDefinition {
                description: Some(def.description),
                name: def.name,
                arguments: def.input_schema,
            },
        }
    }
}

impl From<&ToolDefinition> for WireToolSpec {
    fn from(def: &ToolDefinition) -> Self {
        Self::from(def.clone())
    }
}

/// An ordered set of tool definitions.
///
/// Insertion order is the order the model sees, and adding a name that is
/// already present replaces that entry *in place* — so the same set built the
/// same way always renders the same prefix, which is what lets a cached
/// prefill survive a turn.
///
/// ```
/// use gen2::tool_defs::{ToolDefinition, ToolSet};
///
/// let tools = ToolSet::new()
///     .with(ToolDefinition::new("read").description("Read a file"))
///     .with(ToolDefinition::new("write").description("Write a file"));
/// assert_eq!(tools.names(), ["read", "write"]);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolSet {
    tools: Vec<ToolDefinition>,
}

impl ToolSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a definition, builder-style.
    #[must_use]
    pub fn with(mut self, tool: impl Into<ToolDefinition>) -> Self {
        self.add(tool);
        self
    }

    /// Add a definition. An existing tool of the same name is replaced where
    /// it stands rather than moved to the end.
    pub fn add(&mut self, tool: impl Into<ToolDefinition>) {
        let tool = tool.into();
        match self.tools.iter_mut().find(|t| t.name == tool.name) {
            Some(slot) => *slot = tool,
            None => self.tools.push(tool),
        }
    }

    /// Remove a tool by name. Returns what was removed, if anything.
    pub fn remove(&mut self, name: &str) -> Option<ToolDefinition> {
        let at = self.tools.iter().position(|t| t.name == name)?;
        Some(self.tools.remove(at))
    }

    /// Look a tool up by name.
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Whether a tool of this name is in the set.
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// The definitions, in the order the model sees them.
    pub fn iter(&self) -> std::slice::Iter<'_, ToolDefinition> {
        self.tools.iter()
    }

    /// The tool names, in set order.
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_str()).collect()
    }

    /// How many tools the set holds.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// A stable fingerprint of the set's contents.
    ///
    /// Two sets with the same definitions in the same order fingerprint the
    /// same, across processes and Rust versions — it hashes the canonical
    /// text with FNV-1a, not the standard library's randomised or
    /// version-dependent hasher — so it is fit to key a prompt-prefix cache.
    pub fn fingerprint(&self) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let mut feed = |bytes: &[u8]| {
            for b in bytes {
                h ^= u64::from(*b);
                h = h.wrapping_mul(PRIME);
            }
            // A separator no field can contain, so `ab`+`c` ≠ `a`+`bc`.
            h ^= 0xff;
            h = h.wrapping_mul(PRIME);
        };
        for t in &self.tools {
            feed(t.name.as_bytes());
            feed(t.description.as_bytes());
            feed(t.input_schema.to_string().as_bytes());
        }
        h
    }

    /// The set as the wire specs a backend renders.
    pub fn to_wire(&self) -> Vec<WireToolSpec> {
        self.tools.iter().map(WireToolSpec::from).collect()
    }
}

impl<'a> IntoIterator for &'a ToolSet {
    type Item = &'a ToolDefinition;
    type IntoIter = std::slice::Iter<'a, ToolDefinition>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.iter()
    }
}

impl IntoIterator for ToolSet {
    type Item = ToolDefinition;
    type IntoIter = std::vec::IntoIter<ToolDefinition>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.into_iter()
    }
}

impl<T: Into<ToolDefinition>> FromIterator<T> for ToolSet {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new();
        for t in iter {
            set.add(t);
        }
        set
    }
}

impl From<Vec<WireToolSpec>> for ToolSet {
    fn from(specs: Vec<WireToolSpec>) -> Self {
        specs.into_iter().collect()
    }
}

impl From<ToolSet> for Vec<WireToolSpec> {
    fn from(set: ToolSet) -> Self {
        set.to_wire()
    }
}

impl From<Vec<ToolDefinition>> for ToolSet {
    fn from(tools: Vec<ToolDefinition>) -> Self {
        tools.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(name: &str) -> ToolDefinition {
        ToolDefinition::new(name).description(format!("{name} tool"))
    }

    #[test]
    fn order_is_insertion_order_and_a_repeat_replaces_in_place() {
        let mut set = ToolSet::new().with(def("a")).with(def("b")).with(def("c"));
        set.add(ToolDefinition::new("b").description("replaced"));
        assert_eq!(set.names(), ["a", "b", "c"], "a replacement must not move");
        assert_eq!(set.get("b").unwrap().description, "replaced");
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn remove_returns_the_definition_and_keeps_the_rest_in_order() {
        let mut set = ToolSet::new().with(def("a")).with(def("b")).with(def("c"));
        assert_eq!(set.remove("b").map(|t| t.name), Some("b".into()));
        assert_eq!(set.remove("zzz"), None);
        assert_eq!(set.names(), ["a", "c"]);
    }

    #[test]
    fn the_fingerprint_depends_on_content_and_order_only() {
        let one = ToolSet::new().with(def("a")).with(def("b"));
        let same = ToolSet::new().with(def("a")).with(def("b"));
        let reordered = ToolSet::new().with(def("b")).with(def("a"));
        let changed = ToolSet::new()
            .with(def("a"))
            .with(def("b").schema(serde_json::json!({"type":"object","properties":{}})));
        assert_eq!(one.fingerprint(), same.fingerprint());
        assert_ne!(
            one.fingerprint(),
            reordered.fingerprint(),
            "order is prefix-visible"
        );
        assert_ne!(one.fingerprint(), changed.fingerprint());
        assert_ne!(ToolSet::new().fingerprint(), one.fingerprint());
        // Pinned: a fingerprint that drifted between builds would invalidate
        // every persisted cache key for no reason.
        assert_eq!(ToolSet::new().fingerprint(), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn a_wire_spec_round_trips_through_a_definition() {
        let wire = WireToolSpec {
            r#type: "function".into(),
            function: FunctionDefinition {
                description: Some("Current weather".into()),
                name: "get_weather".into(),
                arguments: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
            },
        };
        let set = ToolSet::from(vec![wire.clone()]);
        assert_eq!(set.names(), ["get_weather"]);
        let back: Vec<WireToolSpec> = set.into();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].function.name, wire.function.name);
        assert_eq!(back[0].function.description, wire.function.description);
        assert_eq!(back[0].function.arguments, wire.function.arguments);
        assert_eq!(back[0].r#type, "function");
    }

    #[test]
    fn an_executable_spec_converts_to_a_definition() {
        let spec = crate::api::tools::ToolSpec::new("read", "Read a file", serde_json::json!({}));
        let def: ToolDefinition = spec.into();
        assert_eq!(def.name, "read");
        assert_eq!(def.description, "Read a file");
    }

    #[test]
    fn a_schema_derives_from_a_rust_type() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Args {
            city: String,
        }
        let def = ToolDefinition::new("weather").input_schema::<Args>();
        assert!(
            def.input_schema["properties"]["city"].is_object(),
            "got {}",
            def.input_schema
        );
    }

    #[test]
    fn serde_is_a_plain_list() {
        let set = ToolSet::new().with(def("a"));
        let json = serde_json::to_string(&set).unwrap();
        assert!(json.starts_with('['), "transparent over the list: {json}");
        let back: ToolSet = serde_json::from_str(&json).unwrap();
        assert_eq!(back, set);
    }
}
