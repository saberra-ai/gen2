//! Probability and calibration semantics adapted from the pinned Laya reference.
use super::{DecisionError, OrderedJson, Question, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Temperatures after upstream's [0.5, 5] clamp has been applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureMap {
    pub temperature: [f64; 3],
    pub temperature_by_options: BTreeMap<String, f64>,
}
impl Default for TemperatureMap {
    fn default() -> Self {
        Self {
            temperature: [1.0; 3],
            temperature_by_options: BTreeMap::new(),
        }
    }
}
/// One histogram calibration table. Values represent empirical correctness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub bins: usize,
    pub values: Vec<f64>,
}
/// A clamped or replaced temperature; confidence from this entry is unqualified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDiagnostic {
    pub field: String,
    pub raw: serde_json::Value,
    pub applied: f64,
}
/// Immutable calibration selected when a model is loaded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Calibration {
    pub base: TemperatureMap,
    pub languages: BTreeMap<String, TemperatureMap>,
    pub binning_map: BTreeMap<String, Histogram>,
    pub diagnostics: Vec<CalibrationDiagnostic>,
}
impl Calibration {
    /// Parse checkpoint or calibration JSON. Bundle loading verifies checkpoint
    /// identity separately; this function only interprets probability parameters.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        if !value.is_object() {
            return Err(invalid("calibration must be an object"));
        }
        if let Some(v) = value.get("version")
            && !matches!(v.as_u64(), Some(1 | 2))
        {
            return Err(invalid("unsupported calibration version"));
        }
        let mut result = Self::default();
        result.base = parse_map(value, [1.0; 3], "base", &mut result.diagnostics)?;
        if let Some(langs) = value.get("lang_temperatures").filter(|v| !v.is_null()) {
            for (lang, cfg) in langs
                .as_object()
                .ok_or_else(|| invalid("lang_temperatures must be an object"))?
            {
                let norm = lang.split('-').next().unwrap_or("").to_lowercase();
                if norm.is_empty() || result.languages.contains_key(&norm) {
                    return Err(invalid("empty or duplicate normalized language"));
                }
                let map = parse_map(cfg, result.base.temperature, lang, &mut result.diagnostics)?;
                result.languages.insert(norm, map);
            }
        }
        if let Some(bins) = value.get("binning_map").filter(|v| !v.is_null()) {
            result.binning_map = serde_json::from_value(bins.clone())
                .map_err(|e| invalid(&format!("binning_map: {e}")))?;
        }
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<()> {
        for map in std::iter::once(&self.base).chain(self.languages.values()) {
            if map
                .temperature
                .iter()
                .chain(map.temperature_by_options.values())
                .any(|v| !v.is_finite() || !(0.5..=5.0).contains(v))
            {
                return Err(invalid("temperatures must be finite and in [0.5, 5]"));
            }
        }
        for h in self.binning_map.values() {
            if h.bins == 0
                || h.values.len() != h.bins
                || h.values
                    .iter()
                    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            {
                return Err(invalid("invalid histogram bins or values"));
            }
        }
        Ok(())
    }
    /// Decode one row, excluding padding slots. Logits are in presentation
    /// order; an explicit question permutation is restored before interpretation.
    pub fn decode(
        &self,
        question: &Question,
        logits: &[f32],
        action_logits: [f32; 2],
        language: Option<&str>,
    ) -> Result<Answer> {
        self.validate()?;
        question.validate()?;
        let k = question.render_options()?.len();
        if logits.len() < k {
            return Err(DecisionError::InvalidOutput("too few option logits".into()));
        }
        let bucket = bucket(question, k);
        let override_map = language.and_then(|s| {
            self.languages
                .get(&s.split('-').next().unwrap_or("").to_lowercase())
        });
        let map = override_map.unwrap_or(&self.base);
        let temperature = *map
            .temperature_by_options
            .get(&bucket)
            .unwrap_or(&map.temperature[question.qtype()]);
        let mut probabilities = softmax(&logits[..k], temperature)?;
        if let Some(order) = question.option_order() {
            let slots = probabilities.clone();
            for (slot, &canonical) in order.iter().enumerate() {
                probabilities[canonical] = slots[slot];
            }
        }
        let winner = probabilities.iter().enumerate().fold(0, |best, (i, p)| {
            if *p > probabilities[best] { i } else { best }
        });
        let raw = probabilities[winner];
        let answer_confidence = if override_map.is_none() {
            self.binning_map.get(&bucket).map_or(raw, |h| {
                h.values[((raw * h.bins as f64) as usize).min(h.bins - 1)]
            })
        } else {
            raw
        };
        let entropy = -probabilities
            .iter()
            .map(|p| p * p.clamp(1e-12, 1.0).ln())
            .sum::<f64>();
        let confidence = if question.qtype() == 2 {
            raw
        } else if k < 2 {
            1.0
        } else {
            (1.0 - entropy / (k as f64).ln()).clamp(0.0, 1.0)
        };
        let value = match question {
            Question::Choice { options, .. } => AnswerValue::Choice {
                index: winner,
                label: options[winner].label.clone(),
            },
            Question::Score { levels, .. } => AnswerValue::Score {
                expectation: probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum(),
                legend: levels
                    .iter()
                    .map(OrderedJson::criterion)
                    .collect::<Result<_>>()?,
            },
            Question::YesNo { .. } => AnswerValue::YesNo {
                probability_true: probabilities[1],
            },
        };
        Ok(Answer {
            value,
            probabilities,
            confidence,
            answer_confidence,
            raw_answer_confidence: raw,
            act_probability: softmax(&action_logits, 1.0)?[0],
            temperature,
            temperature_bucket: bucket,
            language_override: override_map.is_some(),
        })
    }
}
fn invalid(message: &str) -> DecisionError {
    DecisionError::InvalidBundle(message.into())
}
fn parse_map(
    v: &serde_json::Value,
    fallback: [f64; 3],
    prefix: &str,
    diagnostics: &mut Vec<CalibrationDiagnostic>,
) -> Result<TemperatureMap> {
    if !v.is_null() && !v.is_object() {
        return Err(invalid("temperature configuration must be an object"));
    }
    let mut map = TemperatureMap {
        temperature: fallback,
        ..Default::default()
    };
    if let Some(t) = v.get("temperature").filter(|v| !v.is_null()) {
        let t = t
            .as_array()
            .filter(|a| a.len() == 3)
            .ok_or_else(|| invalid("temperature must contain three entries"))?;
        for (i, raw) in t.iter().enumerate() {
            map.temperature[i] = clamp(raw, format!("{prefix}.temperature[{i}]"), diagnostics);
        }
    }
    if let Some(t) = v.get("temperature_by_options").filter(|v| !v.is_null()) {
        for (key, raw) in t
            .as_object()
            .ok_or_else(|| invalid("temperature_by_options must be an object"))?
        {
            map.temperature_by_options.insert(
                key.clone(),
                clamp(raw, format!("{prefix}.{key}"), diagnostics),
            );
        }
    }
    Ok(map)
}
fn clamp(
    v: &serde_json::Value,
    field: String,
    diagnostics: &mut Vec<CalibrationDiagnostic>,
) -> f64 {
    let number = v
        .as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()));
    let applied = number
        .filter(|v| v.is_finite())
        .map_or(1.0, |v| v.clamp(0.5, 5.0));
    if number != Some(applied) {
        diagnostics.push(CalibrationDiagnostic {
            field,
            raw: v.clone(),
            applied,
        });
    }
    applied
}
fn bucket(q: &Question, k: usize) -> String {
    format!(
        "{}:{}",
        q.type_name(),
        match k {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        }
    )
}
fn softmax(logits: &[f32], temperature: f64) -> Result<Vec<f64>> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err(DecisionError::InvalidOutput(
            "empty or non-finite logits".into(),
        ));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<_> = logits
        .iter()
        .map(|v| ((*v as f64 - max) / temperature).exp())
        .collect();
    let sum: f64 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    Ok(p)
}
/// Distribution in the original option order. No display rounding is applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub value: AnswerValue,
    pub probabilities: Vec<f64>,
    pub confidence: f64,
    pub answer_confidence: f64,
    pub raw_answer_confidence: f64,
    /// Diagnostic only; this never authorizes or executes an action.
    pub act_probability: f64,
    pub temperature: f64,
    pub temperature_bucket: String,
    pub language_override: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnswerValue {
    Choice {
        index: usize,
        label: OrderedJson,
    },
    Score {
        expectation: f64,
        legend: Vec<String>,
    },
    YesNo {
        probability_true: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn permutation_restores_choice_score_and_yes_polarity() {
        let calibration = Calibration::default();
        let choice = Question::choice("", [("a", ""), ("b", ""), ("c", "")])
            .with_option_order(vec![2, 0, 1])
            .unwrap();
        let result = calibration
            .decode(&choice, &[8., 0., 1.], [0., 0.], None)
            .unwrap();
        assert!(matches!(result.value, AnswerValue::Choice { index: 2, .. }));
        let score = Question::score("", ["low", "mid", "high"])
            .with_option_order(vec![2, 0, 1])
            .unwrap();
        let result = calibration
            .decode(&score, &[8., 0., 1.], [0., 0.], None)
            .unwrap();
        assert!(
            matches!(result.value, AnswerValue::Score { expectation, .. } if expectation > 1.99)
        );
        let yes = Question::yes_no("").with_option_order(vec![1, 0]).unwrap();
        let result = calibration.decode(&yes, &[8., 0.], [0., 0.], None).unwrap();
        assert!(
            matches!(result.value, AnswerValue::YesNo { probability_true } if probability_true > 0.999)
        );
        for invalid in [vec![], vec![0, 0], vec![1, 2]] {
            assert!(Question::yes_no("").with_option_order(invalid).is_err());
        }
    }
    #[test]
    fn padding_ties_single_option_and_score_expectation() {
        let c = Calibration::default();
        let a = c
            .decode(
                &Question::choice("", [("b", ""), ("a", "")]),
                &[0., 0., f32::NAN],
                [0., 0.],
                None,
            )
            .unwrap();
        assert!(matches!(a.value, AnswerValue::Choice { index: 0, .. }));
        assert_eq!(a.probabilities, vec![0.5, 0.5]);
        assert_eq!(a.confidence, 0.);
        let a = c
            .decode(
                &Question::score("", ["low", "mid", "high"]),
                &[0., 0., 0.],
                [0., 0.],
                None,
            )
            .unwrap();
        assert!(matches!(
            a.value,
            AnswerValue::Score {
                expectation: 1.,
                ..
            }
        ));
        let a = c
            .decode(
                &Question::choice("", [("only", "")]),
                &[100.],
                [0., 0.],
                None,
            )
            .unwrap();
        assert_eq!(a.confidence, 1.);
        assert_eq!(a.answer_confidence, 1.);
    }
    #[test]
    fn language_override_bypasses_base_buckets_and_binning() {
        let c=Calibration::from_json(&serde_json::json!({"temperature":[1,1,1],"temperature_by_options":{"noul:2":5},"lang_temperatures":{"DE-de":{"temperature":[2,2,2]}},"binning_map":{"noul:2":{"bins":1,"values":[0.3]}}})).unwrap();
        let q = Question::yes_no("");
        let a = c.decode(&q, &[0., 2.], [0., 0.], None).unwrap();
        assert_eq!(a.answer_confidence, 0.3);
        assert_eq!(a.temperature, 5.);
        let a = c.decode(&q, &[0., 2.], [0., 0.], Some("de-AT")).unwrap();
        assert_eq!(a.temperature, 2.);
        assert_eq!(a.answer_confidence, a.raw_answer_confidence);
        assert!(
            matches!(a.value,AnswerValue::YesNo {probability_true:p} if (p-0.7310585786).abs()<1e-9)
        );
    }
    #[test]
    fn bad_calibration_and_logits_fail_clearly() {
        let c = Calibration::from_json(&serde_json::json!({"temperature":[true,0.1,"2"]})).unwrap();
        assert_eq!(c.base.temperature, [1., 0.5, 2.]);
        assert_eq!(c.diagnostics.len(), 2);
        assert!(Calibration::from_json(&serde_json::json!({"version":3})).is_err());
        assert!(
            Calibration::from_json(
                &serde_json::json!({"binning_map":{"noul:2":{"bins":2,"values":[0.2]}}})
            )
            .is_err()
        );
        assert!(
            c.decode(&Question::yes_no(""), &[0., f32::INFINITY], [0., 0.], None)
                .is_err()
        );
    }
}
