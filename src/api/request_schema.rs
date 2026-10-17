//! Immutable, typed request schemas. Refinements only narrow string domains.
use super::{Error, Result};
use serde_json::Value;
use std::{collections::BTreeSet, marker::PhantomData};

const MAX_SCHEMA_BYTES: usize = 2 * 1024 * 1024;
const MAX_ENUM_VALUES: usize = 4096;
const MAX_ENUM_BYTES: usize = 64 * 1024;

/// The schema of `T` with request-owned allowed string values.
///
/// JSON pointers address the schema document, including explicit `$defs`, not
/// instance data. Shared definitions are narrowed at every reference site.
/// Empty domains become `false` schemas; existing null/union alternatives stay
/// intact. No request state is global or stored in the model or session.
pub struct RequestSchema<T> {
    schema: Value,
    paths: BTreeSet<String>,
    enum_bytes: usize,
    marker: PhantomData<fn() -> T>,
}
impl<T> Clone for RequestSchema<T> {
    fn clone(&self) -> Self {
        Self {
            schema: self.schema.clone(),
            paths: self.paths.clone(),
            enum_bytes: self.enum_bytes,
            marker: PhantomData,
        }
    }
}
impl<T> std::fmt::Debug for RequestSchema<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestSchema")
            .field("type", &std::any::type_name::<T>())
            .field("bindings", &self.paths.len())
            .finish_non_exhaustive()
    }
}
impl<T: schemars::JsonSchema> RequestSchema<T> {
    /// Generate the schema once. Malformed/remote references and oversized
    /// schema documents are refused without loading or invoking a model.
    pub fn new() -> Result<Self> {
        let schema = super::turn::schema_of::<T>()?;
        validate_size(&schema)?;
        schema_locations(&schema)?;
        Ok(Self {
            schema,
            paths: BTreeSet::new(),
            enum_bytes: 0,
            marker: PhantomData,
        })
    }
    /// Intersect an existing scalar string schema with these permitted values.
    ///
    /// This never follows `$ref` implicitly, widens existing enums/constants,
    /// removes sibling constraints or replaces a previous binding. The target
    /// must explicitly declare `"type":"string"`; null alternatives belong in
    /// its existing enclosing schema. Empty intersections match no strings.
    pub fn string_enum_at<I, S>(mut self, pointer: &str, values: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        if self.paths.contains(pointer) || self.paths.len() >= 128 {
            return Err(invalid("duplicate or excessive request enum bindings"));
        }
        let mut permitted = BTreeSet::new();
        for (index, value) in values.into_iter().enumerate() {
            let value = value.into();
            self.enum_bytes = self
                .enum_bytes
                .checked_add(value.len())
                .ok_or_else(|| invalid("request enum size"))?;
            if index >= MAX_ENUM_VALUES || value.len() > 1024 || self.enum_bytes > MAX_ENUM_BYTES {
                return Err(invalid("request enum size limit exceeded"));
            }
            permitted.insert(value);
        }
        if !schema_locations(&self.schema)?.contains(pointer) {
            return Err(invalid("request enum target must be a schema location"));
        }
        let target = self
            .schema
            .pointer_mut(pointer)
            .and_then(Value::as_object_mut)
            .ok_or_else(|| invalid("request enum target must be an existing string schema"))?;
        if target.get("type").and_then(Value::as_str) != Some("string")
            || target.contains_key("$ref")
            || target.contains_key("$dynamicRef")
        {
            return Err(invalid(
                "request enum target must declare a scalar string type",
            ));
        }
        if let Some(original) = target.get("enum") {
            let original = original
                .as_array()
                .ok_or_else(|| invalid("invalid existing string enum"))?;
            permitted.retain(|value| {
                original
                    .iter()
                    .any(|allowed| allowed.as_str() == Some(value))
            });
        }
        if let Some(constant) = target.get("const") {
            permitted.retain(|value| constant.as_str() == Some(value));
        }
        let empty_domain = permitted.is_empty();
        if empty_domain {
            *self
                .schema
                .pointer_mut(pointer)
                .ok_or_else(|| invalid("request enum target disappeared"))? = Value::Bool(false);
        } else {
            target.insert(
                "enum".into(),
                Value::Array(permitted.into_iter().map(Value::String).collect()),
            );
        }
        self.paths.insert(pointer.into());
        if empty_domain {
            prune_empty_references(&mut self.schema)?;
        }
        validate_size(&self.schema)?;
        Ok(self)
    }
    /// The exact read-only schema used by `Turn::request_schema`. Embed this
    /// value in instructions when the model should see its output contract.
    pub fn as_json(&self) -> &Value {
        &self.schema
    }
}
fn invalid(message: &str) -> Error {
    Error::InvalidRequest(message.into())
}
fn validate_size(schema: &Value) -> Result<()> {
    if serde_json::to_vec(schema)
        .map_err(|_| invalid("request schema encoding"))?
        .len()
        > MAX_SCHEMA_BYTES
    {
        return Err(invalid("request schema size limit exceeded"));
    }
    Ok(())
}
fn schema_locations(root: &Value) -> Result<BTreeSet<String>> {
    let mut pending = vec![(String::new(), root, 0)];
    let mut locations = BTreeSet::new();
    while let Some((path, schema, depth)) = pending.pop() {
        if depth > 128 || locations.len() >= 16_384 {
            return Err(invalid("request schema traversal limit exceeded"));
        }
        if schema.is_boolean() {
            locations.insert(path);
            continue;
        }
        let Some(object) = schema.as_object() else {
            continue;
        };
        locations.insert(path.clone());
        // The first API supports positive schema refinements only. Narrowing
        // beneath these keywords (including via shared definitions) can widen
        // the root language or affect only annotations. Refuse the whole schema
        // rather than attempting a use-site polarity specializer.
        if [
            "not",
            "if",
            "then",
            "else",
            "oneOf",
            "contains",
            "maxContains",
            "dependencies",
            "dependentSchemas",
            "unevaluatedProperties",
            "unevaluatedItems",
            "contentSchema",
            "$id",
            "$anchor",
            "$dynamicAnchor",
            "$recursiveAnchor",
        ]
        .iter()
        .any(|key| object.contains_key(*key))
        {
            return Err(invalid(
                "request schemas require supported positive applicators",
            ));
        }
        if object.contains_key("$dynamicRef") || object.contains_key("$recursiveRef") {
            return Err(invalid("dynamic schema references are unsupported"));
        }
        if let Some(reference) = object.get("$ref") {
            let reference = reference
                .as_str()
                .ok_or_else(|| invalid("invalid schema reference"))?;
            if reference != "#"
                && (!reference.starts_with("#/") || root.pointer(&reference[1..]).is_none())
            {
                return Err(invalid(
                    "request schemas require resolvable local references",
                ));
            }
        }
        for key in [
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
            "dependencies",
        ] {
            if let Some(values) = object.get(key).and_then(Value::as_object) {
                for (name, value) in values {
                    let name = name.replace('~', "~0").replace('/', "~1");
                    pending.push((format!("{path}/{key}/{name}"), value, depth + 1));
                }
            }
        }
        for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
            if let Some(values) = object.get(key).and_then(Value::as_array) {
                for (index, value) in values.iter().enumerate() {
                    pending.push((format!("{path}/{key}/{index}"), value, depth + 1));
                }
            }
        }
        for key in [
            "items",
            "contains",
            "not",
            "if",
            "then",
            "else",
            "additionalProperties",
            "unevaluatedProperties",
            "propertyNames",
            "additionalItems",
            "unevaluatedItems",
            "contentSchema",
        ] {
            if let Some(value) = object.get(key) {
                if let Some(values) = value.as_array() {
                    for (index, value) in values.iter().enumerate() {
                        pending.push((format!("{path}/{key}/{index}"), value, depth + 1));
                    }
                } else {
                    pending.push((format!("{path}/{key}"), value, depth + 1));
                }
            }
        }
    }
    // A resolvable JSON pointer can still address literal annotation/const data.
    for path in &locations {
        if let Some(reference) = root
            .pointer(path)
            .and_then(|s| s.get("$ref"))
            .and_then(Value::as_str)
            && !locations.contains(&reference[1..])
        {
            return Err(invalid("request schema references must target schemas"));
        }
    }
    Ok(locations)
}

// llguidance resolves definitions after compiling their callers. An impossible
// referenced object would therefore abort compilation even when its caller has
// a valid null alternative. Replace only references proven to match nothing;
// retain definitions/constraints and never infer that an unknown branch is empty.
fn prune_empty_references(root: &mut Value) -> Result<()> {
    fn empty(
        schema: &Value,
        root: &Value,
        visiting: &mut BTreeSet<String>,
        budget: &mut usize,
    ) -> bool {
        if visiting.len() >= 128 {
            *budget = 0;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        if schema == &Value::Bool(false) {
            return true;
        }
        let Some(object) = schema.as_object() else {
            return false;
        };
        if let Some(reference) = object.get("$ref").and_then(Value::as_str)
            && visiting.insert(reference.into())
        {
            let result = root
                .pointer(reference.strip_prefix('#').unwrap_or(reference))
                .is_some_and(|target| empty(target, root, visiting, budget));
            visiting.remove(reference);
            if result {
                return true;
            }
        }
        if object
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|branches| {
                branches
                    .iter()
                    .all(|branch| empty(branch, root, visiting, budget))
            })
            || object
                .get("allOf")
                .and_then(Value::as_array)
                .is_some_and(|branches| {
                    branches
                        .iter()
                        .any(|branch| empty(branch, root, visiting, budget))
                })
        {
            return true;
        }
        if object.get("type").and_then(Value::as_str) == Some("object")
            && let (Some(required), Some(properties)) = (
                object.get("required").and_then(Value::as_array),
                object.get("properties").and_then(Value::as_object),
            )
        {
            return required.iter().filter_map(Value::as_str).any(|key| {
                properties
                    .get(key)
                    .is_some_and(|value| empty(value, root, visiting, budget))
            });
        }
        object.get("type").and_then(Value::as_str) == Some("array")
            && object
                .get("minItems")
                .and_then(Value::as_u64)
                .is_some_and(|n| n > 0)
            && !object.contains_key("prefixItems")
            && object
                .get("items")
                .is_some_and(|item| empty(item, root, visiting, budget))
    }
    let locations = schema_locations(root)?;
    let snapshot = root.clone();
    let mut budget = 100_000;
    for path in locations {
        let Some(schema) = snapshot.pointer(&path) else {
            continue;
        };
        if schema.get("$ref").is_some()
            && empty(schema, &snapshot, &mut BTreeSet::new(), &mut budget)
            && let Some(target) = root.pointer_mut(&path)
        {
            *target = Value::Bool(false);
        }
        if budget == 0 {
            return Err(invalid("request schema reference analysis limit exceeded"));
        }
    }
    Ok(())
}
