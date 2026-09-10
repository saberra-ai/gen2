//! Output you can parse: a type, enforced during decoding.
//!
//! Where the backend can constrain decoding (llama.cpp), the schema derived
//! from your type is enforced token by token, so the model cannot emit
//! anything that fails to deserialize. No "please reply with JSON", no retry
//! loop, no salvaging a half-valid object. A backend that cannot (a remote
//! endpoint) is asked for the schema in the prompt and the reply is parsed.
//!
//! ```sh
//! cargo run --example structured -- /path/model.gguf
//! ```

use serde::Deserialize;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct Sentiment {
    label: Label,
    confidence: f32,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
enum Label {
    Positive,
    Negative,
    Neutral,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct Invoice {
    vendor: String,
    total: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: structured <model.gguf>")?;
    let model = gen2::load(&path)?;
    println!(
        "grammar-enforced: {}",
        model.capabilities().structured_output
    );

    // ── A type the reply must satisfy ───────────────────────────────────────
    let s: Sentiment = model
        .generate("Classify the sentiment of: 'this crate finally has a decent API'")
        .greedy()
        .structured()?;
    println!("label={:?} confidence={}", s.label, s.confidence);

    // ── With instructions ───────────────────────────────────────────────────
    let invoice: Invoice = model
        .generate("Acme Ltd — total $1,240.00")
        .system("Extract the invoice fields")
        .greedy()
        .structured()?;
    println!("vendor={} total={}", invoice.vendor, invoice.total);

    // ── On a turn, inside a conversation ────────────────────────────────────
    let mut session = gen2::Session::new().with_system("You classify customer messages.");
    let s: Sentiment = model
        .turn(&mut session)
        .user("Classify: 'this is terrible'")
        .greedy()
        .structured()?;
    println!("in conversation: {:?}", s.label);

    // A reply that does not decode is `Error::Extraction`, carrying the raw
    // text — generation succeeded; reading it as your type did not.
    Ok(())
}
