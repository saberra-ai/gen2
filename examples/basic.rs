//! The three things you'll do most: ask, stream, converse.
//!
//! ```sh
//! cargo run --example basic --features metal -- /path/model.gguf
//! ```

use std::io::Write;

use gen2::{Event, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: basic <model.gguf>")?;

    // Loading blocks until the weights are resident. The backend is picked from
    // the file — GGUF here, but an MLX directory would work the same.
    let model = gen2::load(&path)?;

    // ── 1. Ask a question, get a string ─────────────────────────────────────
    let answer = model
        .generate("Explain entropy in one sentence.")
        .max_tokens(128)
        .text()?;
    println!("answer: {answer}\n");

    // ── 2. Stream it instead, so the user sees tokens as they land ──────────
    print!("streaming: ");
    let mut stream = model
        .generate("Write a haiku about the borrow checker.")
        .max_tokens(64)
        .stream()?;
    for event in &mut stream {
        if let Event::TextDelta(text) = event? {
            print!("{text}");
            let _ = std::io::stdout().flush();
        }
    }
    stream.finish()?;
    println!("\n");

    // ── 3. Hold a conversation ──────────────────────────────────────────────
    // The session carries the history, so the second turn can rely on the
    // first without you resending anything.
    let mut session = Session::new();
    model
        .turn(&mut session)
        .user("My favourite colour is blue. Reply with just: ok")
        .max_tokens(16)
        .run()?;
    let recalled = model
        .turn(&mut session)
        .user("What is my favourite colour? Answer in one word.")
        .max_tokens(16)
        .run()?;
    println!("recalled: {}\n", recalled.text());
    println!("transcript: {} messages\n", session.len());

    // ── 4. The text plus what happened, in one value ────────────────────────
    let response = model.generate("Name three colours.").max_tokens(64).run()?;
    println!("text: {}", response.text());
    println!("finished: {}", response.finish_reason());
    let stats = response.stats();
    if stats.reported() {
        println!(
            "{} tokens at {:.1} tok/s",
            response.usage().completion_tokens,
            stats.tokens_per_second()
        );
    }

    // Dropping `model` drops its runtime, which stops the controller and waits
    // for the backend to be released. Nothing to remember.
    Ok(())
}
