//! The shape of a chat application.
//!
//! A UI can't block on a generation, so inference runs on a worker and the UI
//! reads a channel. That's the whole architecture; everything below is detail.
//!
//! Covers what a real app needs beyond "call the model": background
//! generation, cancellation, concurrent conversations, and owning the
//! transcript.
//!
//! ```sh
//! cargo run --example chat_app --no-default-features --features metal -- /path/model.gguf
//! ```

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use pio_gen2::{Engine, Error, Session, Update};

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

/// Run a turn on a worker, print deltas as they arrive, and hand the session
/// back. Your app would render instead of printing, and keep the session in
/// whatever holds its conversation list.
fn run_turn(engine: &Arc<Engine>, session: Session, prompt: &str) -> Session {
    let turn = engine
        .chat_owned(session)
        .user(prompt)
        .max_tokens(256)
        .spawn();

    let mut carried = None;
    for update in turn {
        match update {
            Update::Delta(t) => print!("{t}"),
            Update::Done {
                completion,
                session,
            } => {
                println!(
                    "\n[done, {} tokens]",
                    completion.stats.map(|s| s.decode_tokens).unwrap_or(0)
                );
                carried = Some(session);
            }
            Update::Failed { error, session } => {
                eprintln!("\n[failed: {}]", describe(&error));
                carried = Some(session);
            }
            _ => {}
        }
    }
    // Every terminal update returns the session, so this always resolves.
    carried.expect("a turn always ends with Done or Failed")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .ok_or("usage: chat_app <model.gguf>")?;

    // One engine, shared. Loading is the slow part; do it once at startup.
    let engine = Arc::new(Engine::load(&model)?);

    // ── A conversation, streamed to the "UI" ────────────────────────────────
    println!("── turn 1 ──");
    let session = Session::new().with_system("Answer briefly.");
    let session = run_turn(&engine, session, "Name two colours.");

    // ── A follow-up ─────────────────────────────────────────────────────────
    // The session carries the history, so nothing is resent and the engine's
    // warm cache is reused.
    println!("\n── turn 2 (same conversation) ──");
    let session = run_turn(&engine, session, "Now name one more.");

    println!("\ntranscript: {} messages", session.len());
    for m in session.messages() {
        println!("  {}", m.role);
    }

    // ── Cancellation ────────────────────────────────────────────────────────
    // The user hits stop. The thread iterating updates is blocked, so the stop
    // has to come from elsewhere — that's what `canceller()` is for.
    println!("\n── cancellation ──");
    let turn = engine
        .chat_owned(Session::new())
        .user("Write a very long essay about rust.")
        .max_tokens(512)
        .spawn();

    let canceller = turn.canceller();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        let _ = canceller.cancel();
    });

    let mut got = 0;
    for update in turn {
        match update {
            Update::Delta(_) => got += 1,
            // A cancelled turn is Done, not Failed: the partial reply is real,
            // and it is already in the session.
            Update::Done {
                completion,
                session,
            } => println!(
                "stopped after {got} fragments; kept {} chars ({:?}), session has {} messages",
                completion.text.len(),
                completion.finish,
                session.len()
            ),
            Update::Failed { error, .. } => eprintln!("[failed: {}]", describe(&error)),
            _ => {}
        }
    }

    // ── Concurrent conversations ────────────────────────────────────────────
    // Independent sessions run independently. How many at once is
    // `ControllerConfig::max_active_chats`.
    println!("\n── two conversations at once ──");
    let a = engine
        .chat_owned(Session::new())
        .user("Say 'apple' and nothing else.")
        .max_tokens(32)
        .spawn();
    let b = engine
        .chat_owned(Session::new())
        .user("Say 'banana' and nothing else.")
        .max_tokens(32)
        .spawn();

    for (name, turn) in [("a", a), ("b", b)] {
        let text: String = turn
            .filter_map(|u| match u {
                Update::Done { session, .. } => session.latest_text(),
                _ => None,
            })
            .collect();
        println!("{name}: {}", text.trim());
    }

    // Dropping the last Arc reference shuts the engine down and waits for the
    // backend to be released.
    Ok(())
}
