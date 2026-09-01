//! The async API, behind the `tokio` feature.
//!
//! ```sh
//! cargo run --example async_chat --no-default-features --features metal,tokio -- /path/model.gguf
//! ```

use std::sync::Arc;

use futures::StreamExt;
use pio_gen2::{Engine, Session, Update};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .ok_or("usage: async_chat <model.gguf>")?;

    let engine = Arc::new(Engine::load(&model)?);

    // ── Await a whole turn ──────────────────────────────────────────────────
    let (completion, session) = engine
        .chat_owned(Session::new())
        .user("Name two colours.")
        .max_tokens(64)
        .send_async()
        .await?;
    println!("awaited: {}", completion.text.trim());

    // ── Stream one ──────────────────────────────────────────────────────────
    // The session carries the history, so this continues the conversation.
    print!("streamed: ");
    let mut turn = engine
        .chat_owned(session)
        .user("Now name one more.")
        .max_tokens(64)
        .spawn_async();

    let mut session = None;
    while let Some(update) = turn.next().await {
        match update {
            Update::Delta(t) => print!("{t}"),
            Update::Done { session: s, .. } => session = Some(s),
            Update::Failed { error, .. } => eprintln!("failed: {error}"),
            _ => {}
        }
    }
    println!();

    if let Some(s) = session {
        println!("transcript: {} messages", s.len());
    }

    // ── Cancel from a task ──────────────────────────────────────────────────
    let mut turn = engine
        .chat_owned(Session::new())
        .user("Write a very long essay about rust.")
        .max_tokens(512)
        .spawn_async();

    let canceller = turn.canceller();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let _ = canceller.cancel();
    });

    let mut fragments = 0;
    while let Some(update) = turn.next().await {
        match update {
            Update::Delta(_) => fragments += 1,
            Update::Done { completion, .. } => println!(
                "cancelled after {fragments} fragments, kept {} chars ({:?})",
                completion.text.len(),
                completion.finish
            ),
            _ => {}
        }
    }

    Ok(())
}
