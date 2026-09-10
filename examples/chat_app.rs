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
//! cargo run --example chat_app -- /path/model.gguf
//! ```

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use gen2::event::Canceller;
use gen2::{Error, Event, Model, Response, Session};

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

/// What the worker sends the UI.
enum Ui {
    Delta(String),
    /// Boxed: a session is a large value next to a one-line delta.
    Done(Box<(gen2::Result<Response>, Session)>),
}

/// Run a turn on a worker, streaming deltas to the UI and handing the session
/// back at the end. A `Model` is `Clone + Send`, so the worker gets its own
/// handle; the session goes with it and comes back with the reply appended.
fn spawn_turn(
    model: &Model,
    mut session: Session,
    prompt: &str,
) -> (mpsc::Receiver<Ui>, mpsc::Receiver<Canceller>) {
    let (ui_tx, ui_rx) = mpsc::channel();
    let (cancel_tx, cancel_rx) = mpsc::channel();
    let model = model.clone();
    let prompt = prompt.to_string();
    thread::spawn(move || {
        let result = (|| {
            let mut stream = model
                .turn(&mut session)
                .user(prompt)
                .max_tokens(256)
                .stream()?;
            let _ = cancel_tx.send(stream.canceller());
            for event in &mut stream {
                if let Event::TextDelta(text) = event? {
                    let _ = ui_tx.send(Ui::Delta(text));
                }
            }
            stream.finish()
        })();
        let _ = ui_tx.send(Ui::Done(Box::new((result, session))));
    });
    (ui_rx, cancel_rx)
}

/// The UI side: print deltas, report the outcome, keep the session.
fn drain(rx: mpsc::Receiver<Ui>) -> Session {
    let mut fragments = 0;
    for update in rx {
        match update {
            Ui::Delta(t) => {
                fragments += 1;
                print!("{t}");
            }
            Ui::Done(done) => match *done {
                (Ok(response), session) => {
                    println!(
                        "\n[{}, {fragments} fragments, {} tokens, session has {} messages]",
                        response.finish_reason(),
                        response.usage().completion_tokens,
                        session.len()
                    );
                    return session;
                }
                (Err(e), session) => {
                    eprintln!("\n[failed: {}]", describe(&e));
                    return session;
                }
            },
        }
    }
    unreachable!("a turn always ends with Done")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: chat_app <model.gguf>")?;

    // One model, shared. Loading is the slow part; do it once at startup.
    let model = gen2::load(&path)?;

    // ── A conversation, streamed to the "UI" ────────────────────────────────
    println!("── turn 1 ──");
    let session = Session::new().with_system("Answer briefly.");
    let (rx, _) = spawn_turn(&model, session, "Name two colours.");
    let session = drain(rx);

    // ── A follow-up ─────────────────────────────────────────────────────────
    // The session carries the history, so nothing is resent and the model's
    // cached prefix is reused.
    println!("\n── turn 2 (same conversation) ──");
    let (rx, _) = spawn_turn(&model, session, "Now name one more.");
    let session = drain(rx);
    for m in session.messages() {
        println!("  {}", m.role);
    }

    // ── Cancellation ────────────────────────────────────────────────────────
    // The user hits stop. The worker is blocked in the stream, so the stop
    // comes from the UI thread through the `Canceller` the stream handed out.
    // A cancelled turn is a finish, not an error: `finish_reason()` is
    // `Cancelled`, and the partial reply is already in the session.
    println!("\n── cancellation ──");
    let (rx, cancel) = spawn_turn(
        &model,
        Session::new(),
        "Write a very long essay about rust.",
    );
    let canceller = cancel.recv()?;
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        canceller.cancel();
    });
    drain(rx);

    // ── Concurrent conversations ────────────────────────────────────────────
    // Independent sessions run independently on one model; turns queue on
    // its controller. How many stay warm at once is
    // `Runtime::builder().max_active_sessions(n)`.
    println!("\n── two conversations at once ──");
    let (a, _) = spawn_turn(&model, Session::new(), "Say 'apple' and nothing else.");
    let (b, _) = spawn_turn(&model, Session::new(), "Say 'banana' and nothing else.");
    for (name, rx) in [("a", a), ("b", b)] {
        let text: String = rx
            .iter()
            .filter_map(|u| match u {
                Ui::Done(done) => done.0.ok().map(|r| r.text()),
                _ => None,
            })
            .collect();
        println!("{name}: {}", text.trim());
    }

    // Dropping the last `Model` shuts its runtime down and waits for the
    // backend to be released.
    Ok(())
}
