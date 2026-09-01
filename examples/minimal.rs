//! The smallest useful program: make a session, stream the reply.

use gen2::{Engine, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).unwrap();

    // 1. One engine for the whole app.
    let engine = Engine::load(&model)?;

    // 2. A conversation you own.
    let mut session = Session::new();

    // 3. A turn. The reply is appended to the session as it streams.
    engine
        .chat(&mut session)
        .user("Name two colours.")
        .send_streaming(|t| print!("{t}"))?;
    println!();

    // 4. A follow-up. The history is already in the session; you don't resend it.
    engine
        .chat(&mut session)
        .user("Now one more.")
        .send_streaming(|t| print!("{t}"))?;
    println!();

    // The transcript is yours to render, persist, or edit.
    println!("\n{} messages:", session.len());
    for m in session.messages() {
        println!("  {}", m.role);
    }

    Ok(())
}
