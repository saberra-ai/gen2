//! [`Extract`] — pull a typed value out of unstructured text.
//!
//! The schema is generated from the exact type the caller will deserialize
//! into, handed to the decoder as a grammar, and the reply is parsed back into
//! that same type. One type declaration drives all three, so they cannot drift
//! apart the way a hand-written prompt and a hand-written parser do.
//!
//! ```no_run
//! # fn main() -> Result<(), gen2::Error> {
//! # let engine = gen2::Engine::load("model.gguf")?;
//! #[derive(serde::Deserialize, schemars::JsonSchema)]
//! struct Invoice {
//!     vendor: String,
//!     total: f64,
//! }
//!
//! let invoice: Invoice = engine.extract("Acme Ltd — total due $1,240.00").value()?;
//! # Ok(())
//! # }
//! ```
//!
//! Generation failing and decoding failing are different errors on purpose.
//! "The model is not loaded" and "the model answered, but `total` was a
//! sentence" need different fixes, and collapsing them into one message sends
//! the reader to the wrong place.

use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::backend::common::grammar::GrammarSpec;

use super::engine::Engine;
use super::error::{Error, Result};

/// Enough for a substantial object without letting a model that has started
/// repeating itself run indefinitely. Overridable, because a document-sized
/// struct is a legitimate thing to ask for.
const DEFAULT_MAX_TOKENS: usize = 1024;

/// One prompt, one typed value out.
///
/// Built by [`Engine::extract`]. `T` is inferred from whatever the result is
/// assigned to, so it usually needs no turbofish.
#[must_use = "an Extract does nothing until .value() is called"]
pub struct Extract<'e, T> {
    engine: &'e Engine,
    text: String,
    instructions: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    _target: PhantomData<fn() -> T>,
}

impl<'e, T> Extract<'e, T>
where
    T: DeserializeOwned + JsonSchema,
{
    pub(crate) fn new(engine: &'e Engine, text: String) -> Self {
        Self {
            engine,
            text,
            instructions: None,
            max_tokens: None,
            temperature: None,
            _target: PhantomData,
        }
    }

    /// Replace the instruction the model is given.
    ///
    /// The schema already says what the shape is, so the default only says
    /// what the job is. Override it to say what the fields *mean* when the
    /// names alone are not enough.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = Some(text.into());
        self
    }

    /// Alias for [`instructions`](Self::instructions), for callers who think
    /// of it as the prompt.
    pub fn prompt(self, text: impl Into<String>) -> Self {
        self.instructions(text)
    }

    /// Cap how many tokens the value may take.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Sample instead of extracting deterministically.
    ///
    /// Extraction is deterministic by default: the same document should give
    /// the same record twice.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Run it and return the value.
    pub fn value(self) -> Result<T> {
        let schema = schema_of::<T>()?;

        let prompt = self.instructions.unwrap_or_else(|| {
            "Read the text and return the requested fields as JSON. \
             Use only what the text actually says."
                .to_string()
        });

        let reply = self
            .engine
            .infer(&self.text)
            .system(prompt)
            // The same schema the reply is decoded with, so the decoder cannot
            // produce a shape `T` will then reject.
            .grammar(GrammarSpec::JsonSchema(schema))
            .temperature(self.temperature.unwrap_or(0.0))
            .max_tokens(self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS))
            .text()?;

        // Generation succeeded. Anything wrong from here is a decode problem,
        // and saying so — with what the model actually wrote — is the
        // difference between a fixable report and a shrug.
        serde_json::from_str::<T>(reply.trim()).map_err(|e| Error::Extraction {
            type_name: std::any::type_name::<T>(),
            message: e.to_string(),
            raw: reply,
        })
    }
}

/// The JSON schema for `T`.
///
/// Generated from the type itself rather than written by hand, which is the
/// point: the constraint the model decodes under and the type the reply is
/// parsed into are the same declaration.
fn schema_of<T: JsonSchema>() -> Result<serde_json::Value> {
    serde_json::to_value(schemars::schema_for!(T)).map_err(|e| {
        Error::InvalidRequest(format!(
            "could not build a JSON schema for {}: {e}",
            std::any::type_name::<T>()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::Script;

    #[derive(Debug, serde::Deserialize, JsonSchema, PartialEq)]
    struct Invoice {
        vendor: String,
        total: f64,
    }

    fn engine_saying(reply: &str) -> Engine {
        Engine::scripted(Script::new().say([reply]))
    }

    #[test]
    fn a_valid_reply_decodes_into_the_requested_type() {
        let engine = engine_saying(r#"{"vendor":"Acme Ltd","total":1240.0}"#);
        let invoice: Invoice = engine
            .extract("Acme Ltd, $1,240.00")
            .value()
            .expect("decode");
        assert_eq!(
            invoice,
            Invoice {
                vendor: "Acme Ltd".into(),
                total: 1240.0
            }
        );
    }

    /// The grammar has to come from `T`, not from a prompt describing `T`.
    ///
    /// This is what stops the schema and the parser drifting apart: one type
    /// declaration produces both, so a field renamed in Rust is renamed in the
    /// constraint the model decodes under.
    #[test]
    fn the_grammar_sent_is_the_schema_of_the_type_being_decoded() {
        let engine = engine_saying(r#"{"vendor":"Acme","total":1.0}"#);
        let script = engine.script().clone();
        let _: Result<Invoice> = engine.extract("Acme").value();

        let grammar = script
            .grammars_seen()
            .into_iter()
            .next()
            .expect("extraction must constrain decoding");
        let GrammarSpec::JsonSchema(schema) = grammar else {
            panic!("expected a JSON-schema constraint, got {grammar:?}");
        };
        let properties = &schema["properties"];
        assert!(
            properties.get("vendor").is_some() && properties.get("total").is_some(),
            "the schema must describe the fields of T, got {schema}"
        );
    }

    /// Valid JSON that is the wrong shape is a decode failure, not a
    /// generation failure.
    ///
    /// The two need different fixes — one is the caller's schema or prompt,
    /// the other is the engine — and a caller routing on the error has to be
    /// able to tell them apart.
    #[test]
    fn json_that_does_not_fit_the_type_is_reported_as_an_extraction_failure() {
        let engine = engine_saying(r#"{"vendor":"Acme","total":"a lot"}"#);
        let outcome: Result<Invoice> = engine.extract("Acme").value();

        match outcome {
            Err(Error::Extraction {
                type_name,
                raw,
                message,
            }) => {
                assert!(type_name.contains("Invoice"));
                assert!(
                    raw.contains("a lot"),
                    "the raw reply is what tells the caller what went wrong"
                );
                assert!(!message.is_empty());
            }
            other => panic!("expected an extraction failure, got {other:?}"),
        }
    }

    #[test]
    fn text_that_is_not_json_at_all_is_also_an_extraction_failure() {
        let engine = engine_saying("I could not find an invoice in that.");
        let outcome: Result<Invoice> = engine.extract("nothing here").value();
        assert!(matches!(outcome, Err(Error::Extraction { .. })));
    }

    /// An extraction failure is not a generation failure, and its code says so.
    #[test]
    fn the_two_failure_kinds_carry_different_codes() {
        let engine = engine_saying("not json");
        let err: Error = engine
            .extract::<Invoice>("x")
            .value()
            .expect_err("this should not decode");
        assert_eq!(err.code(), Some("extraction_failed"));
        assert!(
            !err.is_retryable(),
            "retrying the same prompt against the same model will fail the same way"
        );
    }

    #[test]
    fn extraction_is_deterministic_unless_the_caller_says_otherwise() {
        let engine = engine_saying(r#"{"vendor":"Acme","total":1.0}"#);
        let script = engine.script().clone();
        let _: Result<Invoice> = engine.extract("Acme").value();
        assert_eq!(script.temperatures_seen().first().copied(), Some(0.0));
    }

    /// The same residency cost `Engine::infer` has, recorded here too.
    ///
    /// See `classify::tests::a_classification_holds_a_residency_slot_until_it_is_evicted`
    /// — extraction opens a throwaway conversation the controller is never
    /// asked to close, because no command exists to close one.
    #[test]
    fn an_extraction_holds_a_residency_slot_until_it_is_evicted() {
        let engine = engine_saying(r#"{"vendor":"Acme","total":1.0}"#);
        let script = engine.script().clone();
        let _: Result<Invoice> = engine.extract("Acme").value();
        assert_eq!(script.live_sessions(), 1);
    }
}
