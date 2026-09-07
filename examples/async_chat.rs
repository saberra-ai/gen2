//! The async API, behind the `tokio` feature.
//!
//! The same turns and the same types; decoding runs on a blocking task and
//! events are bridged through a bounded channel.
//!
//! ```sh
//! cargo run --example async_chat --features metal,tokio -- /path/model.gguf
//! ```

use futures::StreamExt;
use gen2::{Event, Session};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: async_chat <model.gguf>")?;
    let model = gen2::load(&path)?;
    let mut session = Session::new();

    // ── Await a whole turn ──────────────────────────────────────────────────
    let response = model
        .turn(&mut session)
        .user("Name two colours.")
        .max_tokens(64)
        .run_async()
        .await?;
    println!("awaited: {}", response.text().trim());

    // ── Stream one ──────────────────────────────────────────────────────────
    // The session carries the history, so this continues the conversation.
    print!("streamed: ");
    let mut stream = model
        .turn(&mut session)
        .user("Now name one more.")
        .max_tokens(64)
        .stream_async()
        .await?;
    while let Some(event) = stream.next().await {
        if let Event::TextDelta(text) = event? {
            print!("{text}");
        }
    }
    let response = stream.finish().await?;
    println!(
        "\n[{}] transcript: {} messages",
        response.finish_reason(),
        session.len()
    );

    // ── Cancel from a task ──────────────────────────────────────────────────
    let mut stream = model
        .turn(&mut session)
        .user("Write a very long essay about rust.")
        .max_tokens(512)
        .stream_async()
        .await?;
    let canceller = stream.canceller();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        canceller.cancel();
    });
    let mut fragments = 0;
    while let Some(event) = stream.next().await {
        if let Event::TextDelta(_) = event? {
            fragments += 1;
        }
    }
    // A cancelled turn is a finish, not an error; the partial reply is in the
    // session under an id you can `remove_message`.
    let response = stream.finish().await?;
    println!(
        "cancelled after {fragments} fragments, kept {} chars ({})",
        response.text().len(),
        response.finish_reason()
    );
    Ok(())
}
