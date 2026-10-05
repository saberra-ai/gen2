use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::{DecisionError, OrderedJson, Result};

/// A choice label and its optional description. Their order is model input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    pub label: OrderedJson,
    pub description: Option<OrderedJson>,
}

/// A typed question. The answer type follows this variant, never generated JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Question {
    Choice {
        #[serde(deserialize_with = "deserialize_instructions")]
        instructions: String,
        options: Vec<ChoiceOption>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        option_order: Option<Vec<usize>>,
    },
    Score {
        #[serde(deserialize_with = "deserialize_instructions")]
        instructions: String,
        levels: Vec<OrderedJson>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        option_order: Option<Vec<usize>>,
    },
    YesNo {
        #[serde(deserialize_with = "deserialize_instructions")]
        instructions: String,
        false_label: String,
        true_label: String,
        false_description: Option<OrderedJson>,
        true_description: Option<OrderedJson>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        option_order: Option<Vec<usize>>,
    },
}

impl Question {
    /// Select presentation slots without changing canonical labels or scores.
    /// `order[slot]` is the canonical option index presented in that slot.
    pub fn with_option_order(mut self, order: Vec<usize>) -> Result<Self> {
        match &mut self {
            Self::Choice { option_order, .. }
            | Self::Score { option_order, .. }
            | Self::YesNo { option_order, .. } => *option_order = Some(order),
        }
        self.validate()?;
        Ok(self)
    }
    pub fn option_order(&self) -> Option<&[usize]> {
        match self {
            Self::Choice { option_order, .. }
            | Self::Score { option_order, .. }
            | Self::YesNo { option_order, .. } => option_order.as_deref(),
        }
    }
    pub(crate) fn render_model_options(&self) -> Result<Vec<String>> {
        let canonical = self.render_options()?;
        Ok(match self.option_order() {
            Some(order) => order.iter().map(|&i| canonical[i].clone()).collect(),
            None => canonical,
        })
    }
    /// Render structured instructions with the reference JSON conventions.
    pub fn with_instructions_json(mut self, instructions: OrderedJson) -> Result<Self> {
        let text = instructions.criterion()?;
        match &mut self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::YesNo { instructions, .. } => *instructions = text,
        };
        Ok(self)
    }
    pub fn choice<I, L, D>(instructions: impl Into<String>, options: I) -> Self
    where
        I: IntoIterator<Item = (L, D)>,
        L: Into<OrderedJson>,
        D: Into<OrderedJson>,
    {
        Self::Choice {
            instructions: instructions.into(),
            option_order: None,
            options: options
                .into_iter()
                .map(|(label, description)| ChoiceOption {
                    label: label.into(),
                    description: Some(description.into()),
                })
                .collect(),
        }
    }
    pub fn score<I, T>(instructions: impl Into<String>, levels: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<OrderedJson>,
    {
        Self::Score {
            instructions: instructions.into(),
            option_order: None,
            levels: levels.into_iter().map(Into::into).collect(),
        }
    }
    pub fn yes_no(instructions: impl Into<String>) -> Self {
        Self::YesNo {
            instructions: instructions.into(),
            option_order: None,
            false_label: "false".into(),
            true_label: "true".into(),
            false_description: None,
            true_description: None,
        }
    }
    pub fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::YesNo { instructions, .. } => instructions,
        }
    }
    pub(crate) fn qtype(&self) -> usize {
        match self {
            Self::Choice { .. } => 0,
            Self::Score { .. } => 1,
            Self::YesNo { .. } => 2,
        }
    }
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
            Self::YesNo { .. } => "noul",
        }
    }
    /// Render the exact option text before tokenization, useful for inspection.
    pub fn render_options(&self) -> Result<Vec<String>> {
        self.validate()?;
        let described = |label: String, description: &Option<OrderedJson>| -> Result<String> {
            match description {
                None | Some(OrderedJson::Null) => Ok(label),
                Some(OrderedJson::String(s)) if s.is_empty() => Ok(label),
                Some(d) => Ok(format!("{label}: {}", d.criterion()?)),
            }
        };
        match self {
            Self::Choice { options, .. } => options
                .iter()
                .map(|o| described(o.label.label_text()?, &o.description))
                .collect(),
            Self::Score { levels, .. } => levels
                .iter()
                .enumerate()
                .map(|(i, v)| Ok(format!("level {i}: {}", v.criterion()?)))
                .collect(),
            Self::YesNo {
                false_label,
                true_label,
                false_description,
                true_description,
                ..
            } => {
                let desc = |d: &Option<OrderedJson>, default: &str| -> Result<String> {
                    match d {
                        None | Some(OrderedJson::Null) => Ok(default.into()),
                        Some(OrderedJson::String(s)) if s.is_empty() => Ok(default.into()),
                        Some(v) => v.criterion(),
                    }
                };
                Ok(vec![
                    format!(
                        "{}: {}",
                        false_label.trim(),
                        desc(false_description, "no, the statement does not hold")?
                    ),
                    format!(
                        "{}: {}",
                        true_label.trim(),
                        desc(true_description, "yes, the statement holds")?
                    ),
                ])
            }
        }
    }
    pub fn validate(&self) -> Result<()> {
        let count = match self {
            Self::Choice { options, .. } => options.len(),
            Self::Score { levels, .. } => levels.len(),
            Self::YesNo { .. } => 2,
        };
        if let Some(order) = self.option_order() {
            let distinct: HashSet<_> = order.iter().copied().collect();
            if order.len() != count || distinct.len() != count || order.iter().any(|&i| i >= count)
            {
                return Err(DecisionError::InvalidRequest(
                    "option_order must be a permutation of canonical option indices".into(),
                ));
            }
        }
        match self {
            Self::Choice { options, .. } => {
                if options.is_empty() {
                    return Err(DecisionError::InvalidRequest(
                        "choice needs at least one option".into(),
                    ));
                }
                let mut labels = HashSet::new();
                for option in options {
                    let label = option.label.label_text()?;
                    if label.is_empty() || !labels.insert(label) {
                        return Err(DecisionError::InvalidRequest(
                            "choice labels must be non-empty and distinct".into(),
                        ));
                    }
                }
            }
            Self::Score { levels, .. } if levels.is_empty() => {
                return Err(DecisionError::InvalidRequest(
                    "score needs at least one level".into(),
                ));
            }
            Self::YesNo {
                false_label,
                true_label,
                ..
            } if false_label.trim().is_empty()
                || true_label.trim().is_empty()
                || false_label.trim() == true_label.trim() =>
            {
                return Err(DecisionError::InvalidRequest(
                    "yes/no labels must be distinct non-empty strings".into(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

fn deserialize_instructions<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<String, D::Error> {
    OrderedJson::deserialize(deserializer)?
        .criterion()
        .map_err(serde::de::Error::custom)
}

/// Input state; conversations keep their newest tokens under explicit truncation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum DecisionState {
    Text(String),
    Json(OrderedJson),
    Conversation(Vec<OrderedJson>),
}
impl DecisionState {
    pub fn serialize(&self) -> Result<String> {
        match self {
            Self::Text(s) => Ok(s.clone()),
            Self::Json(v) => v.to_state_text(),
            Self::Conversation(v) => OrderedJson::Array(v.clone()).to_state_text(),
        }
    }
    pub(crate) fn truncate_left(&self) -> bool {
        matches!(
            self,
            Self::Conversation(_) | Self::Json(OrderedJson::Array(_))
        )
    }
}

/// Ordered named questions over one state. An empty question set is a no-op.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub state: DecisionState,
    pub questions: Vec<(String, Question)>,
}
impl DecisionRequest {
    pub fn text(text: impl Into<String>) -> Self {
        Self::new(DecisionState::Text(text.into()))
    }
    pub fn new(state: DecisionState) -> Self {
        Self {
            state,
            questions: Vec::new(),
        }
    }
    pub fn question(mut self, id: impl Into<String>, question: Question) -> Self {
        self.questions.push((id.into(), question));
        self
    }
    pub fn validate(&self) -> Result<()> {
        let mut ids = HashSet::new();
        for (id, q) in &self.questions {
            if id.is_empty() || !ids.insert(id) {
                return Err(DecisionError::InvalidRequest(format!(
                    "empty or duplicate question ID {id:?}"
                )));
            }
            q.validate()
                .map_err(|e| DecisionError::InvalidRequest(format!("question {id:?}: {e}")))?;
        }
        self.state.serialize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn option_order_and_yes_polarity_are_preserved() {
        assert_eq!(
            Question::choice("route", [("z", "Last"), ("a", "First")])
                .render_options()
                .unwrap(),
            ["z: Last", "a: First"]
        );
        assert_eq!(
            Question::yes_no("urgent?").render_options().unwrap(),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
    }
    #[test]
    fn questions_do_not_overwrite_duplicate_ids() {
        assert!(DecisionRequest::text("").validate().is_ok());
        assert!(
            DecisionRequest::text("x")
                .question("a", Question::yes_no("x"))
                .question("a", Question::yes_no("y"))
                .validate()
                .is_err()
        );
    }
    #[test]
    fn structured_criteria_and_scalar_labels_are_rendered() {
        let q = Question::choice(
            "route",
            [(
                OrderedJson::from(3i64),
                OrderedJson::parse(r#"{"z":true,"a":0}"#).unwrap(),
            )],
        );
        assert_eq!(q.render_options().unwrap(), [r#"3: {"z": true, "a": 0}"#]);
        assert!(DecisionState::Json(OrderedJson::Array(vec![])).truncate_left());
    }
    #[test]
    fn structured_instructions_use_ordered_json_not_debug_text() {
        let q: Question = serde_json::from_str(
            r#"{"type":"score","instructions":{"z":true,"a":"é"},"levels":["low"]}"#,
        )
        .unwrap();
        assert_eq!(q.instructions(), r#"{"z": true, "a": "é"}"#);
    }
}
