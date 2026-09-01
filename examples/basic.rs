//! The three things you'll do most: ask, stream, converse.
//!
//! ```sh
//! cargo run --example basic --no-default-features --features metal -- /path/model.gguf
//! ```

use std::io::Write;

use pio_gen2::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).ok_or("usage: basic <model.gguf>")?;

    // Loading blocks until the weights are resident. The backend is picked from
    // the file — GGUF here, but an MLX or ONNX directory would work the same.
    let engine = Engine::load(&model)?;

    // ── 1. Ask a question, get a string ─────────────────────────────────────
    let answer = engine
        .prompt("Explain entropy in one sentence.")
        .max_tokens(128)
        .text()?;
    println!("answer: {answer}\n");

    // ── 2. Stream it instead, so the user sees tokens as they land ──────────
    print!("streaming: ");
    engine
        .prompt("Write a haiku about the borrow checker.")
        .max_tokens(64)
        .text_streaming(|token| {
            print!("{token}");
            let _ = std::io::stdout().flush();
        })?;
    println!("\n");

    // ── 3. Hold a conversation ──────────────────────────────────────────────
    // Reusing a chat id continues that conversation, so the second turn can
    // rely on the first without you resending the history.
    engine
        .chat("demo")
        .user("My favourite colour is blue. Reply with just: ok")
        .max_tokens(16)
        .text()?;

    let recalled = engine
        .chat("demo")
        .user("What is my favourite colour? Answer in one word.")
        .max_tokens(16)
        .text()?;
    println!("recalled: {recalled}\n");

    // ── 4. The text plus what happened, in one value ────────────────────────
    let done = engine
        .prompt("Name three colours.")
        .max_tokens(64)
        .complete()?;

    println!("text: {}", done.text);
    println!("finished: {:?}", done.finish);
    if let Some(s) = &done.stats {
        println!("{} tokens at {:.1} tok/s", s.decode_tokens, s.avg_tps);
    }
    if done.context_was_shed() {
        eprintln!(
            "shed context: {} dropped, {} compacted",
            done.dropped, done.compacted
        );
    }

    // ── 5. Or just the tokens, as an iterator ───────────────────────────────
    print!("tokens: ");
    for token in engine.prompt("Count to three.").max_tokens(32).tokens()? {
        print!("{}", token?);
    }
    println!();

    // The full `Event` stream is still there via `.stream()` when you need
    // media boundaries or tool calls as they arrive.

    // Dropping `engine` here stops the controller and waits for the backend to
    // be released. Nothing to remember.
    Ok(())
}
