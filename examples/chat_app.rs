//! The shape of a chat application.
//!
//! A UI can't block on a generation, so inference runs on a worker and the UI
//! reads a channel. That's the whole architecture; everything below is detail.
//!
//! Covers the four things a real chat app needs beyond "call the model":
//! background generation, cancellation, concurrent conversations, and owning
//! the transcript.
//!
//! ```sh
//! cargo run --example chat_app --no-default-features --features metal -- /path/model.gguf
//! ```

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Duration;

use pio_gen2::{Engine, Error};

/// What the worker sends the UI. Yours would carry whatever your renderer
/// needs — this is deliberately small.
enum UiMsg {
    Delta(String),
    Done { text: String, tokens: u32 },
    Failed(String),
}

/// Kick off a turn on a worker thread and hand back a channel of updates.
///
/// The `Arc<Engine>` clone is why this works: `Engine` is `Send + Sync`, so one
/// engine serves every conversation in the app.
fn send_turn(engine: Arc<Engine>, chat_id: &str, text: &str) -> Receiver<UiMsg> {
    let (tx, rx) = channel();
    let chat_id = chat_id.to_string();
    let text = text.to_string();

    thread::spawn(move || {
        let ui = tx.clone();
        let result = engine
            .chat(&chat_id)
            .user(text)
            .max_tokens(256)
            // Fragments reach the UI as they're decoded, so the user sees the
            // reply forming instead of waiting for the whole thing.
            .complete_streaming(|fragment| {
                let _ = ui.send(UiMsg::Delta(fragment.to_string()));
            });

        let _ = match result {
            Ok(done) => tx.send(UiMsg::Done {
                text: done.text,
                tokens: done.stats.map(|s| s.decode_tokens).unwrap_or(0),
            }),
            // A cancelled turn arrives here too — the engine reports what it
            // had, so the partial reply is still usable.
            Err(e) => tx.send(UiMsg::Failed(describe(&e))),
        };
    });

    rx
}

/// Turn an error into something worth putting on screen.
///
/// `code()` is the part to branch on — it's stable, unlike the message text.
fn describe(e: &Error) -> String {
    match e.code() {
        Some(code) if e.is_retryable() => format!("{code} (retrying may help): {e}"),
        Some(code) => format!("{code}: {e}"),
        None => e.to_string(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .ok_or("usage: chat_app <model.gguf>")?;

    // One engine, shared. Loading is the slow part; do it once at startup.
    let engine = Arc::new(Engine::load(&model)?);

    // ── A turn, streamed to the "UI" ────────────────────────────────────────
    println!("── turn 1 ──");
    let rx = send_turn(engine.clone(), "general", "Name two colours.");
    for msg in rx {
        match msg {
            UiMsg::Delta(t) => print!("{t}"),
            UiMsg::Done { tokens, .. } => println!("\n[done, {tokens} tokens]"),
            UiMsg::Failed(why) => eprintln!("\n[failed: {why}]"),
        }
    }

    // ── A follow-up on the same conversation ────────────────────────────────
    // Same chat id, so the engine keeps the history and its warm KV cache. The
    // app never resends the transcript.
    println!("\n── turn 2 (same conversation) ──");
    let rx = send_turn(engine.clone(), "general", "Now name one more.");
    for msg in rx {
        match msg {
            UiMsg::Delta(t) => print!("{t}"),
            UiMsg::Done { .. } => println!("\n[done]"),
            UiMsg::Failed(why) => eprintln!("\n[failed: {why}]"),
        }
    }

    // ── Cancellation ────────────────────────────────────────────────────────
    // The user hits stop. `stop` is by chat id and can be called from any
    // thread, which is the point: the worker is blocked draining the stream.
    println!("\n── cancellation ──");
    let rx = send_turn(
        engine.clone(),
        "long",
        "Write a very long essay about rust.",
    );
    let canceller = engine.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        let _ = canceller.stop("long");
    });

    let mut got = 0;
    for msg in rx {
        match msg {
            UiMsg::Delta(_) => got += 1,
            UiMsg::Done { text, .. } => {
                println!("stopped after {got} fragments; kept {} chars", text.len())
            }
            UiMsg::Failed(why) => eprintln!("[failed: {why}]"),
        }
    }

    // ── Concurrent conversations ────────────────────────────────────────────
    // Distinct chat ids run independently. The controller schedules them; how
    // many run at once is `ControllerConfig::max_active_chats`.
    println!("\n── two conversations at once ──");
    let a = send_turn(engine.clone(), "chat-a", "Say 'apple' and nothing else.");
    let b = send_turn(engine.clone(), "chat-b", "Say 'banana' and nothing else.");
    for (name, rx) in [("a", a), ("b", b)] {
        let text: String = rx
            .into_iter()
            .filter_map(|m| match m {
                UiMsg::Done { text, .. } => Some(text),
                _ => None,
            })
            .collect();
        println!("{name}: {}", text.trim());
    }

    // Dropping the Arc's last reference shuts the engine down and waits for the
    // backend to be released.
    Ok(())
}
