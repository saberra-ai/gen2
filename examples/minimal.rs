//! The smallest useful program: make a chat, stream the reply.

use pio_gen2::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).unwrap();

    // 1. One engine for the whole app.
    let engine = Engine::load(&model)?;

    // 2. A chat is just an id. Streaming it is a for-loop over text.
    for token in engine.chat("chat-1").user("Name two colours.").tokens()? {
        print!("{}", token?);
    }
    println!();

    // 3. Same id = same conversation. It remembers turn 1.
    for token in engine.chat("chat-1").user("Now one more.").tokens()? {
        print!("{}", token?);
    }
    println!();

    Ok(())
}
