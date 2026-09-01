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
use std::thread;
use std::time::Duration;

use pio_gen2::{Engine, Error, Update};

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
    // `spawn` runs the turn on a worker and hands back the updates. No thread,
    // no channel, no sender clone to write.
    println!("── turn 1 ──");
    let turn = engine
        .chat_owned("general")
        .user("Name two colours.")
        .max_tokens(256)
        .spawn();

    for update in turn {
        match update {
            Update::Delta(t) => print!("{t}"),
            Update::Done(done) => println!(
                "\n[done, {} tokens]",
                done.stats.map(|s| s.decode_tokens).unwrap_or(0)
            ),
            Update::Failed(e) => eprintln!("\n[failed: {}]", describe(&e)),
            _ => {}
        }
    }

    // ── A follow-up on the same conversation ────────────────────────────────
    // Same chat id, so the engine keeps the history and its warm KV cache. The
    // app never resends the transcript.
    println!("\n── turn 2 (same conversation) ──");
    for update in engine
        .chat_owned("general")
        .user("Now name one more.")
        .max_tokens(256)
        .spawn()
    {
        match update {
            Update::Delta(t) => print!("{t}"),
            Update::Done(_) => println!("\n[done]"),
            Update::Failed(e) => eprintln!("\n[failed: {}]", describe(&e)),
            _ => {}
        }
    }

    // ── Cancellation ────────────────────────────────────────────────────────
    // The user hits stop. `stop` is by chat id and can be called from any
    // thread, which is the point: the worker is blocked draining the stream.
    println!("\n── cancellation ──");
    let turn = engine
        .chat_owned("long")
        .user("Write a very long essay about rust.")
        .max_tokens(512)
        .spawn();

    // `canceller()` is movable and cheap to clone — the iterating thread below
    // is blocked, so cancelling has to come from somewhere else.
    let canceller = turn.canceller();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        let _ = canceller.cancel();
    });

    let mut got = 0;
    for update in turn {
        match update {
            Update::Delta(_) => got += 1,
            // A cancelled turn is Done, not Failed: the partial reply is real.
            Update::Done(done) => println!(
                "stopped after {got} fragments; kept {} chars ({:?})",
                done.text.len(),
                done.finish
            ),
            Update::Failed(e) => eprintln!("[failed: {}]", describe(&e)),
            _ => {}
        }
    }

    // ── Concurrent conversations ────────────────────────────────────────────
    // Distinct chat ids run independently. The controller schedules them; how
    // many run at once is `ControllerConfig::max_active_chats`.
    println!("\n── two conversations at once ──");
    let a = engine
        .chat_owned("chat-a")
        .user("Say 'apple' and nothing else.")
        .max_tokens(32)
        .spawn();
    let b = engine
        .chat_owned("chat-b")
        .user("Say 'banana' and nothing else.")
        .max_tokens(32)
        .spawn();

    for (name, turn) in [("a", a), ("b", b)] {
        let text: String = turn
            .filter_map(|u| match u {
                Update::Done(done) => Some(done.text),
                _ => None,
            })
            .collect();
        println!("{name}: {}", text.trim());
    }

    // Dropping the Arc's last reference shuts the engine down and waits for the
    // backend to be released.
    Ok(())
}
