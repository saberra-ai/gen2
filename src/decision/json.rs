use std::fmt;

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{DecisionError, Result};

/// JSON whose object insertion order survives parsing and serialization.
///
/// Laya tokenizes the serialized state, so sorting an object's keys can change
/// its prediction. Unlike an arbitrary string, this type also distinguishes a
/// conversation list (which keeps its newest entries when truncated).
#[derive(Debug, Clone, PartialEq)]
pub enum OrderedJson {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedJson>),
    Object(Vec<(String, OrderedJson)>),
}

impl OrderedJson {
    /// Parse JSON without sorting keys. Duplicate object keys are rejected.
    pub fn parse(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| DecisionError::InvalidRequest(e.to_string()))
    }

    /// Serialize as Python `json.dumps(..., ensure_ascii=False)` does for the
    /// supported JSON number range, including spaces after separators.
    pub fn to_state_text(&self) -> Result<String> {
        let mut output = String::new();
        self.write_python(&mut output)?;
        Ok(output)
    }

    pub(crate) fn write_python(&self, output: &mut String) -> Result<()> {
        match self {
            Self::Null => output.push_str("null"),
            Self::Bool(b) => output.push_str(if *b { "true" } else { "false" }),
            Self::Number(n) => {
                if n.is_f64() {
                    // Rust and Python use shortest-round-trip formatting. Python
                    // additionally spells the exponent with a sign and two digits.
                    let value = n.as_f64().ok_or_else(|| {
                        DecisionError::InvalidRequest("invalid JSON number".into())
                    })?;
                    if !value.is_finite() {
                        return Err(DecisionError::InvalidRequest(
                            "non-finite JSON number".into(),
                        ));
                    }
                    let raw = format!("{value:?}");
                    if let Some((base, exponent)) = raw.split_once('e') {
                        let e: i32 = exponent.parse().map_err(|_| {
                            DecisionError::InvalidRequest("invalid exponent".into())
                        })?;
                        output.push_str(&format!("{base}e{e:+03}"));
                    } else {
                        output.push_str(&raw);
                    }
                } else {
                    output.push_str(&n.to_string());
                }
            }
            Self::String(s) => {
                output.push_str(&serde_json::to_string(s).expect("strings serialize"))
            }
            Self::Array(items) => {
                output.push('[');
                for (i, value) in items.iter().enumerate() {
                    if i > 0 {
                        output.push_str(", ");
                    }
                    value.write_python(output)?;
                }
                output.push(']');
            }
            Self::Object(entries) => {
                let mut keys = std::collections::HashSet::new();
                output.push('{');
                for (i, (key, value)) in entries.iter().enumerate() {
                    if !keys.insert(key) {
                        return Err(DecisionError::InvalidRequest(format!(
                            "duplicate JSON key {key:?}"
                        )));
                    }
                    if i > 0 {
                        output.push_str(", ");
                    }
                    output.push_str(&serde_json::to_string(key).expect("strings serialize"));
                    output.push_str(": ");
                    value.write_python(output)?;
                }
                output.push('}');
            }
        }
        Ok(())
    }

    pub(crate) fn criterion(&self) -> Result<String> {
        match self {
            Self::String(s) => Ok(s.clone()),
            _ => self.to_state_text(),
        }
    }

    pub(crate) fn label_text(&self) -> Result<String> {
        match self {
            Self::String(s) => Ok(s.clone()),
            Self::Number(_) => self.to_state_text(),
            Self::Bool(b) => Ok(if *b { "True" } else { "False" }.into()),
            Self::Null => Ok("None".into()),
            _ => Err(DecisionError::InvalidRequest(
                "choice labels must be JSON scalars".into(),
            )),
        }
    }
}

impl From<&str> for OrderedJson {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}
impl From<String> for OrderedJson {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}
impl From<bool> for OrderedJson {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
impl From<i64> for OrderedJson {
    fn from(value: i64) -> Self {
        Self::Number(value.into())
    }
}

impl Serialize for OrderedJson {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Null => s.serialize_unit(),
            Self::Bool(v) => s.serialize_bool(*v),
            Self::Number(v) => v.serialize(s),
            Self::String(v) => s.serialize_str(v),
            Self::Array(v) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for x in v {
                    seq.serialize_element(x)?;
                }
                seq.end()
            }
            Self::Object(v) => {
                let mut map = s.serialize_map(Some(v.len()))?;
                for (k, x) in v {
                    map.serialize_entry(k, x)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = OrderedJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("ordered JSON")
            }
            fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_bool<E: serde::de::Error>(
                self,
                v: bool,
            ) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Bool(v))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Number(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Number(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(OrderedJson::Number)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_str<E: serde::de::Error>(
                self,
                v: &str,
            ) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::String(v.into()))
            }
            fn visit_string<E: serde::de::Error>(
                self,
                v: String,
            ) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::String(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(OrderedJson::Array(v))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut v = Vec::new();
                let mut keys = std::collections::HashSet::new();
                while let Some((k, x)) = a.next_entry::<String, OrderedJson>()? {
                    if !keys.insert(k.clone()) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON key {k:?}"
                        )));
                    }
                    v.push((k, x));
                }
                Ok(OrderedJson::Object(v))
            }
        }
        d.deserialize_any(JsonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_serialization_preserves_order_spaces_and_unicode() {
        let v = OrderedJson::parse(r#"{"z":["é",true,null],"a":{"y":1,"x":2}}"#).unwrap();
        assert_eq!(
            v.to_state_text().unwrap(),
            r#"{"z": ["é", true, null], "a": {"y": 1, "x": 2}}"#
        );
        let roundtrip: OrderedJson =
            serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
        assert_eq!(roundtrip, v);
    }
    #[test]
    fn duplicate_keys_are_not_silently_changed() {
        assert!(OrderedJson::parse(r#"{"x":1,"x":2}"#).is_err());
        let v = OrderedJson::Object(vec![("x".into(), 1i64.into()), ("x".into(), 2i64.into())]);
        assert!(v.to_state_text().is_err());
    }
    #[test]
    fn float_spelling_matches_python_examples() {
        for (json, expected) in [
            ("1e-7", "1e-07"),
            ("1e20", "1e+20"),
            ("-0.0", "-0.0"),
            ("1.0", "1.0"),
            ("0.0001", "0.0001"),
        ] {
            assert_eq!(
                OrderedJson::parse(json).unwrap().to_state_text().unwrap(),
                expected
            );
        }
    }
}
