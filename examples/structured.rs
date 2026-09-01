//! Getting output you can parse: grammar-constrained decoding.
//!
//! The grammar is enforced *during* decoding, so the model cannot emit anything
//! that violates it. No "please reply with JSON" in the prompt, no retry loop,
//! no salvaging a half-valid object — and it behaves identically on every
//! backend.
//!
//! ```sh
//! cargo run --example structured --no-default-features --features metal -- /path/model.gguf
//! ```

use gen2::{Engine, GrammarSpec};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Sentiment {
    label: String,
    confidence: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .ok_or("usage: structured <model.gguf>")?;
    let engine = Engine::load(&model)?;

    // ── A schema the reply must satisfy ─────────────────────────────────────
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "label": { "type": "string", "enum": ["positive", "negative", "neutral"] },
            "confidence": { "type": "number" }
        },
        "required": ["label", "confidence"]
    });

    let raw = engine
        .infer("Classify the sentiment of: 'this crate finally has a decent API'")
        .grammar(GrammarSpec::JsonSchema(schema.clone()))
        .max_tokens(128)
        .greedy()
        .text()?;

    // Sound because the grammar made the alternative unreachable, not because
    // we're hoping.
    let parsed: Sentiment = serde_json::from_str(&raw)?;
    println!("label={} confidence={}", parsed.label, parsed.confidence);

    // ── Other shapes ────────────────────────────────────────────────────────
    // Any JSON object, when you don't want to write a schema:
    let json = engine
        .infer("Describe Rust in JSON with keys 'name' and 'year'.")
        .grammar(GrammarSpec::JsonObject)
        .max_tokens(64)
        .text()?;
    println!("json: {json}");

    // A regex, when you want one token-shaped thing:
    let year = engine
        .infer("In what year was Rust 1.0 released?")
        .grammar(GrammarSpec::Regex(r"\d{4}".into()))
        .max_tokens(8)
        .text()?;
    println!("year: {year}");

    // `GrammarSpec::Lark(..)` takes a full grammar when the shape is more than
    // a schema or a pattern can say.

    // ── Or fix the shape at build time ──────────────────────────────────────
    // When an engine exists to produce one shape, set it once. A turn can
    // still override it, or drop it with `.unconstrained()`.
    let classifier = Engine::builder()
        .model(&model)
        .grammar(GrammarSpec::JsonSchema(schema))
        .greedy()
        .max_tokens(128)
        .build()?;

    let raw = classifier.infer("Classify: 'this is terrible'").text()?;
    println!("classifier: {raw}");

    Ok(())
}
