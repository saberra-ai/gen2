//! The smallest useful program: load a model, hold a conversation, stream.
//!
//! ```sh
//! cargo run --example minimal -- /path/model.gguf
//! ```
//!
//! (`hello` needs no path: it fetches a small model from Hugging Face.)

use gen2::{Event, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: minimal <model.gguf | hf:owner/repo>")?;

    // 1. A model. Its runtime is private and lives as long as the handle.
    let model = gen2::load(&path)?;

    // 2. A conversation you own.
    let mut session = Session::new();

    // 3. A turn, streamed. The reply is appended to the session as it lands.
    let mut stream = model
        .turn(&mut session)
        .user("Name two colours.")
        .stream()?;
    for event in &mut stream {
        if let Event::TextDelta(text) = event? {
            print!("{text}");
        }
    }
    stream.finish()?;
    println!();

    // 4. A follow-up. The history is already in the session; nothing is resent.
    let response = model.turn(&mut session).user("Now one more.").run()?;
    println!("{}", response.text());

    // The transcript is yours to render, persist, or edit.
    println!("\n{} messages:", session.len());
    for m in session.messages() {
        println!("  {}", m.role);
    }
    Ok(())
}
