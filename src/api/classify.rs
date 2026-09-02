//! [`Classify`] — pick one of a fixed set of labels.
//!
//! The smallest useful thing a language model does, and the one most often
//! reimplemented badly: prompt for a category, get back "Positive." or
//! "positive (the customer sounds happy)" or a paragraph, then write a parser
//! for it. This does it once, properly.
//!
//! Two guarantees make it worth having. The label set is enforced by
//! grammar-constrained decoding, so the model cannot produce prose that has to
//! be salvaged afterwards. And the returned string is checked against the
//! caller's own list before it is handed back — a backend with no grammar
//! support would otherwise let arbitrary text through a function whose type
//! says it returns one of *these* labels.
//!
//! Deliberately no confidence score. Nothing here measures one, and a number
//! invented to look like a probability is worse than no number at all.

use std::collections::BTreeSet;

use crate::backend::common::grammar::GrammarSpec;

use super::engine::Engine;
use super::error::{Error, Result};

/// Room for the longest label plus JSON quoting, with slack for a tokenizer
/// that splits it more finely than expected. Small on purpose: the grammar
/// already bounds the output, and a large budget only delays a model that has
/// gone wrong.
const TOKENS_PER_LABEL_CHAR: usize = 2;
const MIN_TOKEN_BUDGET: usize = 16;

/// One prompt, one label out of a set the caller fixes.
///
/// Built by [`Engine::classify`].
///
/// ```no_run
/// # fn main() -> Result<(), gen2::Error> {
/// # let engine = gen2::Engine::load("model.gguf")?;
/// let label = engine
///     .classify("The service was fantastic")
///     .labels(["positive", "negative", "neutral"])
///     .label()?;
/// # Ok(())
/// # }
/// ```
#[must_use = "a Classify does nothing until .label() is called"]
pub struct Classify<'e> {
    engine: &'e Engine,
    text: String,
    labels: Vec<String>,
    instructions: Option<String>,
    temperature: Option<f32>,
}

impl<'e> Classify<'e> {
    pub(crate) fn new(engine: &'e Engine, text: String) -> Self {
        Self {
            engine,
            text,
            labels: Vec::new(),
            instructions: None,
            temperature: None,
        }
    }

    /// The labels to choose between. Required.
    ///
    /// Calling this twice replaces the set rather than adding to it, because
    /// a classifier's label set is one decision, not an accumulation.
    pub fn labels<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.labels = labels.into_iter().map(Into::into).collect();
        self
    }

    /// Replace the instruction the model is given.
    ///
    /// The default says what the task is and nothing else. Override it when
    /// the labels need explaining — "urgent" means something different in a
    /// support queue than in a newsroom.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = Some(text.into());
        self
    }

    /// Sample instead of deciding.
    ///
    /// Classification is deterministic by default. This exists because a
    /// caller who wants to sample several labels to see where a model is
    /// uncertain has a real use for it — but that is an unusual thing to want,
    /// which is why it is not the default.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Run it and return the chosen label.
    ///
    /// The returned string is always one of the labels passed to
    /// [`labels`](Self::labels), spelled exactly as the caller spelled it.
    pub fn label(self) -> Result<String> {
        let labels = validate(&self.labels)?;

        let budget = self
            .labels
            .iter()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0)
            * TOKENS_PER_LABEL_CHAR;

        let prompt = self.instructions.unwrap_or_else(|| {
            format!(
                "Classify the text into exactly one of these categories: {}.\n\
                 Reply with the category and nothing else.",
                labels.join(", ")
            )
        });

        let reply = self
            .engine
            .infer(&self.text)
            .system(prompt)
            // A root string enum: the decoder cannot produce anything that is
            // not one of these, so there is no prose to salvage afterwards.
            .grammar(GrammarSpec::JsonSchema(serde_json::json!({
                "type": "string",
                "enum": labels,
            })))
            .temperature(self.temperature.unwrap_or(0.0))
            .max_tokens(budget.max(MIN_TOKEN_BUDGET))
            .text()?;

        // Checked rather than trusted. A backend that does not enforce
        // grammars would otherwise return arbitrary model prose from a
        // function whose contract is "one of these labels".
        pick(&reply, &self.labels).ok_or_else(|| Error::Extraction {
            type_name: "a label",
            message: format!(
                "the model answered with something outside the label set {:?}",
                self.labels
            ),
            raw: reply,
        })
    }
}

/// Check the label set before anything is generated.
fn validate(labels: &[String]) -> Result<Vec<String>> {
    let cleaned: Vec<String> = labels.iter().map(|l| l.trim().to_string()).collect();

    if cleaned.iter().any(|l| l.is_empty()) {
        return Err(Error::InvalidRequest(
            "a classification label is empty; every label has to name something".into(),
        ));
    }
    if cleaned.len() < 2 {
        return Err(Error::InvalidRequest(format!(
            "classification needs at least two labels to choose between, got {}",
            cleaned.len()
        )));
    }
    // Case-insensitively, because two labels a model cannot tell apart are not
    // two labels — and the answer would be unattributable to either.
    let distinct: BTreeSet<String> = cleaned.iter().map(|l| l.to_lowercase()).collect();
    if distinct.len() != cleaned.len() {
        return Err(Error::InvalidRequest(format!(
            "classification labels must be distinct, got {cleaned:?}"
        )));
    }
    Ok(cleaned)
}

/// Which label the model chose, if any.
///
/// The grammar makes the reply a JSON string, so that is tried first. The
/// looser matches after it are for backends that do not enforce grammars:
/// they still only ever return a label the caller supplied, so the contract
/// holds either way.
fn pick(reply: &str, labels: &[String]) -> Option<String> {
    let decoded = serde_json::from_str::<String>(reply.trim()).ok();
    let candidate = decoded.as_deref().unwrap_or(reply).trim();

    labels
        .iter()
        .find(|l| l.trim() == candidate)
        .or_else(|| {
            labels
                .iter()
                .find(|l| l.trim().eq_ignore_ascii_case(candidate))
        })
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::session::Session;
    use crate::test_support::Script;

    fn engine_saying(reply: &str) -> Engine {
        Engine::scripted(Script::new().say([reply]))
    }

    /// The guarantee the whole type exists for.
    #[test]
    fn a_label_outside_the_caller_s_set_is_never_returned() {
        // A backend that ignores grammars, which is exactly the case the
        // post-check is for. Returning "maybe" from a function documented to
        // return one of three labels would put a value into the caller's
        // program that their own `match` cannot handle.
        let engine = engine_saying("maybe");
        let outcome = engine
            .classify("hello")
            .labels(["positive", "negative", "neutral"])
            .label();

        match outcome {
            Err(Error::Extraction { raw, .. }) => assert_eq!(raw, "maybe"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_label_comes_back_spelled_the_way_the_caller_spelled_it() {
        // The grammar makes the model answer with a JSON string; the caller
        // gets their own plain label back, not a quoted one.
        let engine = engine_saying("\"positive\"");
        let label = engine
            .classify("great")
            .labels(["positive", "negative"])
            .label()
            .expect("a grammar-shaped reply should classify");
        assert_eq!(label, "positive");
    }

    /// Backends differ in whether they enforce a grammar, so a bare reply has
    /// to work too — and still map onto the caller's spelling.
    #[test]
    fn an_unquoted_reply_still_maps_onto_the_caller_s_label() {
        let engine = engine_saying("Positive");
        let label = engine
            .classify("great")
            .labels(["positive", "negative"])
            .label()
            .expect("case should not decide whether classification works");
        assert_eq!(
            label, "positive",
            "the caller's spelling is what their code compares against"
        );
    }

    #[test]
    fn fewer_than_two_labels_fails_before_anything_is_generated() {
        let engine = engine_saying("positive");
        let script = engine.script().clone();

        let outcome = engine.classify("hello").labels(["positive"]).label();
        assert!(matches!(outcome, Err(Error::InvalidRequest(_))));
        assert!(
            script.seen().is_empty(),
            "a request that cannot succeed must not cost a generation"
        );
    }

    #[test]
    fn duplicate_labels_are_refused() {
        let engine = engine_saying("positive");
        // Differing only by case: the model cannot distinguish them, so an
        // answer could not be attributed to either.
        let outcome = engine
            .classify("hello")
            .labels(["positive", "Positive"])
            .label();
        assert!(
            matches!(outcome, Err(Error::InvalidRequest(m)) if m.contains("distinct")),
            "duplicate labels should be caught before inference"
        );
    }

    #[test]
    fn an_empty_label_is_refused() {
        let engine = engine_saying("positive");
        let outcome = engine.classify("hello").labels(["positive", "  "]).label();
        assert!(matches!(outcome, Err(Error::InvalidRequest(_))));
    }

    /// The label set has to reach the model as a constraint, not a suggestion.
    #[test]
    fn the_labels_are_sent_as_a_grammar_the_decoder_must_obey() {
        let engine = engine_saying("\"positive\"");
        let script = engine.script().clone();
        let _ = engine
            .classify("great")
            .labels(["positive", "negative"])
            .label();

        let grammar = script
            .grammars_seen()
            .into_iter()
            .next()
            .expect("classification must constrain decoding, not just ask nicely");
        let GrammarSpec::JsonSchema(schema) = grammar else {
            panic!("expected a JSON-schema constraint, got {grammar:?}");
        };
        assert_eq!(schema["type"], "string");
        assert_eq!(
            schema["enum"],
            serde_json::json!(["positive", "negative"]),
            "the grammar must carry exactly the caller's labels"
        );
    }

    /// Classification decides; it does not sample.
    #[test]
    fn classification_is_deterministic_unless_the_caller_says_otherwise() {
        let engine = engine_saying("\"positive\"");
        let script = engine.script().clone();
        let _ = engine
            .classify("great")
            .labels(["positive", "negative"])
            .label();
        assert_eq!(
            script.temperatures_seen().first().copied(),
            Some(0.0),
            "a classifier that samples gives different answers to the same input"
        );

        let engine = engine_saying("\"positive\"");
        let script = engine.script().clone();
        let _ = engine
            .classify("great")
            .labels(["positive", "negative"])
            .temperature(0.7)
            .label();
        assert_eq!(script.temperatures_seen().first().copied(), Some(0.7));
    }

    /// What a classification costs the engine, recorded rather than assumed.
    ///
    /// It runs through `Engine::infer`, which opens a throwaway conversation
    /// and never asks the controller to close it — there is no command for
    /// that. So each call holds a residency slot until the controller evicts
    /// it, and a caller classifying in a loop will push a real conversation
    /// out and make it re-prefill on its next turn.
    ///
    /// Transparent rather than broken: an evicted session reopens (see
    /// `a_conversation_evicted_for_capacity_still_works_when_you_come_back`).
    /// This test exists so the cost is written down and a future change that
    /// removes it is noticed.
    #[test]
    fn a_classification_holds_a_residency_slot_until_it_is_evicted() {
        let engine = engine_saying("\"positive\"");
        let script = engine.script().clone();
        let _ = engine
            .classify("great")
            .labels(["positive", "negative"])
            .label();
        assert_eq!(
            script.live_sessions(),
            1,
            "one throwaway conversation per classification is the current cost"
        );
    }

    /// A classification still does not disturb a conversation the caller owns.
    #[test]
    fn classifying_does_not_touch_a_session_the_caller_holds() {
        let engine = engine_saying("\"positive\"");
        let mut session = Session::new();
        engine.chat(&mut session).user("hello").send().unwrap();
        let before = session.messages().len();

        let _ = engine
            .classify("great")
            .labels(["positive", "negative"])
            .label();

        assert_eq!(
            session.messages().len(),
            before,
            "a throwaway classification must not append to the caller's transcript"
        );
    }
}
