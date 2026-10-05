//! Sequence construction adapted from Laya common.py, revision
//! 8a6e1328cce2460a0e5aa348ad465bb1b5821cd2 (Apache-2.0).
//! See THIRD_PARTY_NOTICES for attribution and license.

use std::collections::HashSet;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::{DecisionError, DecisionRequest, Result};

/// Special-token IDs must come from the selected checkpoint's tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
    pub mask: u32,
    pub mask_text: String,
}

/// Whether input evidence may be dropped to fit the selected token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverflowPolicy {
    #[default]
    Reject,
    AllowWithDiagnostics,
}

/// Per-call sequence budgets, independent of device and language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeOptions {
    pub max_len: usize,
    pub head_max_len: usize,
    pub overflow: OverflowPolicy,
}

/// One question row before padding. The diagnostics describe actual token loss.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodedQuestion {
    pub id: String,
    pub input_ids: Vec<i64>,
    pub marker_pos: Vec<i64>,
    pub qtype: i64,
    pub state_tokens: usize,
    pub state_tokens_dropped: usize,
    pub options_total: usize,
    pub options_distinct: usize,
    pub tokens_per_option: Option<usize>,
    pub instruction_tokens_dropped: usize,
    pub option_tokens_dropped: Vec<usize>,
}

/// Tokenizer and sequence formatter for an inspected Laya checkpoint.
///
/// This can be used without ONNX to diagnose budgets and verify preprocessing.
pub struct LayaTokenizer {
    tokenizer: Tokenizer,
    special: SpecialTokens,
    #[cfg_attr(not(feature = "backend-laya-onnx"), allow(dead_code))]
    cleanup_spaces: bool,
}

impl std::fmt::Debug for LayaTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayaTokenizer")
            .field("special", &self.special)
            .finish_non_exhaustive()
    }
}

impl LayaTokenizer {
    #[cfg(feature = "backend-laya-onnx")]
    pub(crate) fn state_ids(&self, state: &super::DecisionState) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(
                state.serialize()?.replace(&self.special.mask_text, " "),
                false,
            )
            .map(|v| v.get_ids().to_vec())
            .map_err(|e| DecisionError::InvalidRequest(e.to_string()))
    }
    #[cfg(feature = "backend-laya-onnx")]
    pub(crate) fn decode_state(&self, ids: &[u32]) -> Result<String> {
        let mut text = self
            .tokenizer
            .decode(ids, false)
            .map_err(|e| DecisionError::InvalidRequest(e.to_string()))?;
        if self.cleanup_spaces {
            // Transformers 4.57.6's fast-tokenizer cleanup, used when the
            // reference decodes long-state windows back into text.
            for (from, to) in [
                (" .", "."),
                (" ?", "?"),
                (" !", "!"),
                (" ,", ","),
                (" ' ", "'"),
                (" n't", "n't"),
                (" 'm", "'m"),
                (" 's", "'s"),
                (" 've", "'ve"),
                (" 're", "'re"),
            ] {
                text = text.replace(from, to);
            }
        }
        Ok(text)
    }
    pub fn from_file(path: impl AsRef<Path>, special: SpecialTokens) -> Result<Self> {
        let path = path.as_ref();
        let mut tokenizer =
            Tokenizer::from_file(path).map_err(|e| DecisionError::InvalidBundle(e.to_string()))?;
        let config_path = path.with_file_name("tokenizer_config.json");
        let config: serde_json::Value = match std::fs::read(&config_path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| DecisionError::InvalidBundle(format!("tokenizer config: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Null,
            Err(e) => return Err(DecisionError::InvalidBundle(e.to_string())),
        };
        for (key, id) in [
            ("cls_token", special.cls),
            ("sep_token", special.sep),
            ("pad_token", special.pad),
            ("mask_token", special.mask),
        ] {
            if let Some(value) = config.get(key) {
                let text = value
                    .as_str()
                    .or_else(|| value.get("content").and_then(|v| v.as_str()))
                    .ok_or_else(|| DecisionError::InvalidBundle(format!("invalid {key}")))?;
                if tokenizer.token_to_id(text) != Some(id) {
                    return Err(DecisionError::InvalidBundle(format!(
                        "{key} ID differs from tokenizer config"
                    )));
                }
            }
        }
        let cleanup_spaces = config
            .get("clean_up_tokenization_spaces")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if special.mask_text.is_empty()
            || tokenizer.token_to_id(&special.mask_text) != Some(special.mask)
            || [special.cls, special.sep, special.pad, special.mask]
                .into_iter()
                .collect::<HashSet<_>>()
                .len()
                != 4
        {
            return Err(DecisionError::InvalidBundle(
                "special token IDs must be distinct and mask must match tokenizer".into(),
            ));
        }
        for id in [special.cls, special.sep, special.pad] {
            if tokenizer.id_to_token(id).is_none() {
                return Err(DecisionError::InvalidBundle(format!(
                    "unknown special token ID {id}"
                )));
            }
        }
        tokenizer
            .with_truncation(None)
            .map_err(|e| DecisionError::InvalidBundle(e.to_string()))?;
        tokenizer.with_padding(None);
        Ok(Self {
            tokenizer,
            special,
            cleanup_spaces,
        })
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    /// Build one row per question, preserving input order and reporting cuts.
    pub fn encode(
        &self,
        request: &DecisionRequest,
        options: &EncodeOptions,
    ) -> Result<Vec<EncodedQuestion>> {
        encode(request, options, &self.special, &|text| {
            self.tokenizer
                .encode(text, false)
                .map(|v| v.get_ids().iter().map(|&id| i64::from(id)).collect())
                .map_err(|e| DecisionError::InvalidRequest(e.to_string()))
        })
    }
}

fn encode(
    request: &DecisionRequest,
    options: &EncodeOptions,
    special: &SpecialTokens,
    tokenize: &impl Fn(&str) -> Result<Vec<i64>>,
) -> Result<Vec<EncodedQuestion>> {
    request.validate()?;
    if options.max_len == 0 || options.head_max_len == 0 {
        return Err(DecisionError::InvalidRequest(
            "token budgets must be positive".into(),
        ));
    }
    if request.questions.is_empty() {
        return Ok(Vec::new());
    }
    let state_text = request.state.serialize()?.replace(&special.mask_text, " ");
    let state_ids = tokenize(&state_text)?;
    request
        .questions
        .iter()
        .map(|(id, q)| {
            let instructions = q.instructions().replace(&special.mask_text, " ");
            let mut head = tokenize(&format!("{} question: {instructions}", q.type_name()))?;
            let original_head = head.len();
            let mut original_options = Vec::new();
            let mut options_ids = q
                .render_model_options()?
                .into_iter()
                .map(|text| {
                    let mut v = tokenize(&format!(" {}", text.replace(&special.mask_text, " ")))?;
                    original_options.push(v.len() + 1);
                    v.truncate(48);
                    v.insert(0, i64::from(special.mask));
                    Ok(v)
                })
                .collect::<Result<Vec<_>>>()?;
            let total = options_ids.len();
            let sum = options_ids
                .iter()
                .try_fold(0usize, |a, v| a.checked_add(v.len()))
                .ok_or_else(|| DecisionError::InvalidRequest("option size overflow".into()))?;
            let mut per_option = None;
            if options.head_max_len.saturating_sub(sum) < 16 {
                let per = (options.head_max_len.saturating_sub(16) / total).max(4);
                for v in &mut options_ids {
                    v.truncate(per);
                }
                per_option = Some(per);
            }
            let distinct = options_ids.iter().collect::<HashSet<_>>().len();
            // Identical descriptions cannot denote different choices after token clipping.
            if distinct < total {
                return Err(DecisionError::InputOverflow(format!(
                    "question {id:?}: only {distinct} of {total} option spans remain distinct"
                )));
            }
            let sum: usize = options_ids.iter().map(Vec::len).sum();
            head.truncate(options.head_max_len.saturating_sub(sum).max(8));
            let instruction_tokens_dropped = original_head - head.len();
            let option_tokens_dropped: Vec<_> = original_options
                .iter()
                .zip(&options_ids)
                .map(|(original, actual)| original - actual.len())
                .collect();
            if options.overflow == OverflowPolicy::Reject
                && (instruction_tokens_dropped > 0 || option_tokens_dropped.iter().any(|n| *n > 0))
            {
                return Err(DecisionError::InputOverflow(format!(
                    "question {id:?} would truncate instructions or options"
                )));
            }
            let mut ids = vec![i64::from(special.cls)];
            ids.extend(head);
            ids.push(i64::from(special.sep));
            let mut markers = Vec::with_capacity(total);
            for v in options_ids {
                markers.push(ids.len() as i64);
                ids.extend(v);
            }
            ids.push(i64::from(special.sep));
            if ids.len() >= options.max_len {
                return Err(DecisionError::InputOverflow(format!(
                    "question {id:?} header does not fit"
                )));
            }
            let room = options.max_len.saturating_sub(ids.len().saturating_add(1));
            let used = room.min(state_ids.len());
            let dropped = state_ids.len() - used;
            if dropped > 0 && options.overflow == OverflowPolicy::Reject {
                return Err(DecisionError::InputOverflow(format!(
                    "question {id:?} would drop {dropped} state tokens"
                )));
            }
            if request.state.truncate_left() {
                ids.extend_from_slice(&state_ids[state_ids.len() - used..]);
            } else {
                ids.extend_from_slice(&state_ids[..used]);
            }
            ids.push(i64::from(special.sep));
            ids.truncate(options.max_len);
            markers.retain(|&p| p >= 0 && (p as usize) < ids.len());
            if markers.len() != total {
                return Err(DecisionError::InputOverflow(format!(
                    "question {id:?}: only {} of {total} option markers fit",
                    markers.len()
                )));
            }
            Ok(EncodedQuestion {
                id: id.clone(),
                input_ids: ids,
                marker_pos: markers,
                qtype: q.qtype() as i64,
                state_tokens: state_ids.len(),
                state_tokens_dropped: dropped,
                options_total: total,
                options_distinct: distinct,
                tokens_per_option: per_option,
                instruction_tokens_dropped,
                option_tokens_dropped,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Question;
    use crate::decision::{DecisionState, OrderedJson};
    fn special() -> SpecialTokens {
        SpecialTokens {
            cls: 1,
            sep: 2,
            pad: 0,
            mask: 3,
            mask_text: "[MASK]".into(),
        }
    }
    fn tokens(text: &str) -> Result<Vec<i64>> {
        Ok(text.bytes().map(|b| i64::from(b) + 10).collect())
    }
    #[test]
    fn zero_room_keeps_no_conversation_tokens() {
        let r = DecisionRequest::new(DecisionState::Conversation(vec![OrderedJson::from(
            "newest",
        )]))
        .question("q", Question::yes_no("x"));
        let options = EncodeOptions {
            max_len: 100,
            head_max_len: 60,
            overflow: OverflowPolicy::AllowWithDiagnostics,
        };
        let full = encode(&r, &options, &special(), &tokens).unwrap().remove(0);
        let header = full.input_ids.len() - (full.state_tokens - full.state_tokens_dropped) - 1;
        let tiny = encode(
            &r,
            &EncodeOptions {
                max_len: header + 1,
                ..options
            },
            &special(),
            &tokens,
        )
        .unwrap()
        .remove(0);
        assert_eq!(tiny.state_tokens_dropped, tiny.state_tokens);
        assert_eq!(tiny.input_ids.last(), Some(&2));
    }
    #[test]
    fn default_rejects_lost_evidence_but_explicit_policy_reports_it() {
        let r = DecisionRequest::text("state ".repeat(100)).question("q", Question::yes_no("x"));
        let mut options = EncodeOptions {
            max_len: 100,
            head_max_len: 60,
            overflow: OverflowPolicy::Reject,
        };
        assert!(matches!(
            encode(&r, &options, &special(), &tokens),
            Err(DecisionError::InputOverflow(_))
        ));
        options.overflow = OverflowPolicy::AllowWithDiagnostics;
        assert!(encode(&r, &options, &special(), &tokens).unwrap()[0].state_tokens_dropped > 0);
    }
    #[test]
    fn injected_mask_text_cannot_add_option_markers() {
        let r = DecisionRequest::text("[MASK] state").question(
            "q",
            Question::choice("[MASK]", [("a", "[MASK]"), ("b", "other")]),
        );
        let rows = encode(
            &r,
            &EncodeOptions {
                max_len: 200,
                head_max_len: 100,
                overflow: OverflowPolicy::Reject,
            },
            &special(),
            &tokens,
        )
        .unwrap();
        assert_eq!(rows[0].input_ids.iter().filter(|&&id| id == 3).count(), 2);
        assert_eq!(rows[0].marker_pos.len(), 2);
    }
    #[test]
    fn upstream_recorded_sequences_match_without_model_weights() {
        for name in [
            "preprocessing.json",
            "preprocessing-multilingual.json",
            "preprocessing-typed-decisions.json",
            "permutations-english.json",
            "permutations-multilingual.json",
            "permutations-typed-decisions.json",
        ] {
            let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/laya")
                .join(name);
            let fixture: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(file).expect("record upstream fixtures"),
            )
            .unwrap();
            let special: SpecialTokens =
                serde_json::from_value(fixture["special_tokens"].clone()).unwrap();
            for case in fixture["cases"].as_array().unwrap() {
                let request: DecisionRequest =
                    serde_json::from_value(case["request"].clone()).unwrap();
                let options: EncodeOptions =
                    serde_json::from_value(case["options"].clone()).unwrap();
                let expected: Vec<EncodedQuestion> =
                    serde_json::from_value(case["expected"].clone()).unwrap();
                let token_map = &case["tokenizations"];
                let result = encode(&request, &options, &special, &|s| {
                    serde_json::from_value(
                        token_map
                            .get(s)
                            .unwrap_or_else(|| panic!("unexpected tokenizer input {s:?}"))
                            .clone(),
                    )
                    .map_err(|e| DecisionError::InvalidRequest(e.to_string()))
                })
                .unwrap();
                assert_eq!(result, expected, "case {}", case["name"]);
            }
        }
    }
    #[test]
    #[ignore = "set GEN2_LAYA_TOKENIZER to the pinned real tokenizer.json; no model weights needed"]
    fn real_tokenizer_matches_upstream_recorded_sequences() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/laya/preprocessing.json"))
                .unwrap();
        let special = serde_json::from_value(fixture["special_tokens"].clone()).unwrap();
        let tokenizer = LayaTokenizer::from_file(
            std::env::var("GEN2_LAYA_TOKENIZER").expect("tokenizer path"),
            special,
        )
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let request = serde_json::from_value(case["request"].clone()).unwrap();
            let options = serde_json::from_value(case["options"].clone()).unwrap();
            let expected: Vec<EncodedQuestion> =
                serde_json::from_value(case["expected"].clone()).unwrap();
            assert_eq!(
                tokenizer.encode(&request, &options).unwrap(),
                expected,
                "case {}",
                case["name"]
            );
        }
    }
}
