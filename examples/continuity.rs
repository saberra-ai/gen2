//! An agent that survives you closing the terminal.
//!
//! The coding agent in `coding_agent.rs` runs once and exits. This one keeps a
//! conversation across restarts, accepts new instructions while it is
//! thinking, and picks up where it left off — the loop a terminal assistant
//! actually needs.
//!
//! ```sh
//! cargo run --example continuity -- /path/model.gguf "start indexing this repo"
//! # ctrl-c, then run it again with a follow-up:
//! cargo run --example continuity -- /path/model.gguf "what did you find?"
//! ```
//!
//! # The four kinds of continuity, and where each lives
//!
//! | Continuity across…      | Mechanism                                    |
//! |-------------------------|----------------------------------------------|
//! | tool calls in one run   | the agent loop — nothing to do                |
//! | user input mid-run      | `Steering::follow_up` / `interrupt`           |
//! | the context window      | automatic compaction; `Completion::compacted` |
//! | process restarts        | `Session` is `Serialize` — you persist it     |
//!
//! Only the last one is the application's job, and it is deliberately so:
//! gen2 does not choose your storage. What it does guarantee is that the thing
//! you have to store is a plain serialisable value handed back to you when a
//! run ends.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use gen2::schemars::JsonSchema;
use gen2::{
    AgentConfig, Completion, Engine, Finish, FunctionTool, Message, Session, ToolError, ToolOutput,
    ToolSet, Update,
};
use serde::Deserialize;

// ── A tool set, composed once ───────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
struct PathArg {
    /// Path relative to the working directory.
    path: String,
}

#[derive(Deserialize, JsonSchema)]
struct NoteArg {
    /// Something worth remembering across sessions.
    note: String,
}

/// The filesystem bundle, reusable anywhere.
///
/// A `ToolSet` exists so a bundle is stated once and registered by whoever
/// wants it — resident in a coding agent, deferred in a general assistant —
/// without either restating its contents.
fn filesystem_tools() -> ToolSet {
    ToolSet::new()
        .add(FunctionTool::new(
            "read_file",
            "Read a file's contents.",
            |_ctx, args: PathArg| async move {
                std::fs::read_to_string(&args.path)
                    .map(ToolOutput::from)
                    .map_err(|e| ToolError::Failed(format!("could not read {}: {e}", args.path)))
            },
        ))
        .add(FunctionTool::new(
            "list_files",
            "List a directory.",
            |_ctx, args: PathArg| async move {
                let mut names: Vec<String> = std::fs::read_dir(&args.path)
                    .map_err(|e| ToolError::Failed(format!("could not list: {e}")))?
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect();
                names.sort();
                Ok(ToolOutput::from(names.join("\n")))
            },
        ))
}

/// A note the agent keeps for its future self.
///
/// The transcript is continuity for the *conversation*; this is continuity for
/// *conclusions*. A long-running assistant that has to re-derive what it
/// already worked out is not continuous in any useful sense — and notes
/// survive compaction, which old messages do not.
fn memory_tools(notes: PathBuf) -> ToolSet {
    ToolSet::new().add(FunctionTool::new(
        "remember",
        "Save a durable note. Use for conclusions worth keeping after this conversation ends.",
        move |_ctx, args: NoteArg| {
            let notes = notes.clone();
            async move {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&notes)
                    .map_err(|e| ToolError::Unavailable(format!("notes unwritable: {e}")))?;
                writeln!(file, "- {}", args.note)
                    .map_err(|e| ToolError::Unavailable(format!("could not write note: {e}")))?;
                Ok(ToolOutput::from("noted"))
            }
        },
    ))
}

// ── Persistence: the one part that is the application's job ─────────────────

/// Load the conversation from last time, or start one.
fn restore(path: &PathBuf, notes: &PathBuf) -> Session {
    let restored: Option<Session> = std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok());

    match restored {
        Some(session) => {
            println!("resumed: {} messages", session.messages().len());
            repair(session)
        }
        None => {
            let remembered = std::fs::read_to_string(notes).unwrap_or_default();
            let mut prompt = String::from(
                "You are a long-running assistant. You keep working across sessions. \
                 Use `remember` for conclusions worth keeping.",
            );
            if !remembered.trim().is_empty() {
                prompt.push_str("\n\nWhat you noted previously:\n");
                prompt.push_str(&remembered);
            }
            Session::new().with_system(prompt)
        }
    }
}

/// Heal a transcript that was interrupted mid-tool.
///
/// The failure mode nobody plans for: the process died between the model
/// asking for a tool and the result being appended. On resume the model sees
/// its own call with no answer, which is a conversation state no chat template
/// can render honestly — most will either drop the call or hallucinate a
/// result.
///
/// This is only detectable because a tool result carries the id of the call it
/// answers (`Message::tool_call_id`). Without it, matching results to calls is
/// guesswork the moment a turn contains more than one.
fn repair(mut session: Session) -> Session {
    session.edit(|messages| {
        let answered: std::collections::HashSet<String> = messages
            .iter()
            .filter_map(|m| m.tool_call_id.clone())
            .collect();

        // Walk backwards so removing an entry cannot shift one not yet seen.
        for i in (0..messages.len()).rev() {
            let unanswered: Vec<String> = match &messages[i].body {
                gen2::MessageBody::Tool { tool_calls } => tool_calls
                    .iter()
                    .filter(|c| !answered.contains(&c.id))
                    .map(|c| c.id.clone())
                    .collect(),
                _ => continue,
            };
            if unanswered.is_empty() {
                continue;
            }
            // Answer it rather than delete it. Removing the call would leave
            // the model's own reasoning referring to something that is no
            // longer in the transcript; saying the run was cut is true, and it
            // is something the model can act on.
            println!("  repairing {} interrupted call(s)", unanswered.len());
            for id in unanswered.into_iter().rev() {
                messages.insert(
                    i + 1,
                    Message::tool_result_for(id, "interrupted: the session ended before this ran"),
                );
            }
        }
    });
    session
}

fn persist(path: &PathBuf, session: &Session) {
    match serde_json::to_string_pretty(session) {
        Ok(text) => {
            let _ = std::fs::write(path, text);
        }
        Err(e) => eprintln!("could not save the session: {e}"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or("usage: continuity <model.gguf> <task>")?;
    let task: String = args.collect::<Vec<_>>().join(" ");

    let state = PathBuf::from(".agent-session.json");
    let notes = PathBuf::from(".agent-notes.md");

    // `Arc` because a spawned run needs an engine that outlives this scope.
    let engine = Arc::new(Engine::builder().model(&model).context(8192).build()?);
    let session = restore(&state, &notes);

    // One config, reused for every run. This is what makes "the same agent"
    // across restarts mean something concrete rather than a description in a
    // README.
    let config = AgentConfig::new()
        .add_tools(filesystem_tools())
        .add_tools(memory_tools(notes.clone()))
        .max_steps(20)
        .deadline(Duration::from_secs(300));

    // Spawned, not borrowed. The run owns its session and an engine handle,
    // which is what lets `interrupt` cut a generation short instead of only
    // queueing behind it.
    let mut run = config.agent_owned(&engine, session).goal(task).spawn();

    // A steering handle is cloneable and `Send` — this is where a UI thread,
    // a socket, or a stdin reader would live. Anything it pushes is delivered
    // at the next step boundary without restarting the run.
    let steering = run.steering();
    std::thread::spawn(move || {
        let mut line = String::new();
        while std::io::stdin().read_line(&mut line).is_ok() && !line.trim().is_empty() {
            // `follow_up` waits for the current step; `interrupt` also stops
            // the generation in flight. Typing while it works is the whole
            // point of a terminal agent.
            steering.follow_up(line.trim());
            line.clear();
        }
    });

    // The run is an iterator of updates. Nothing here blocks the agent.
    let mut finished: Option<(Completion, Session)> = None;
    for update in &mut run {
        match update {
            Update::Delta(text) => {
                print!("{text}");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            Update::ToolCall { tool, args, .. } => println!("\n  → {tool} {args}"),
            Update::ToolResult { tool, result, .. } => match result {
                Ok(_) => println!("    {tool} ok"),
                Err(e) => println!("    {tool} failed — {e}"),
            },
            // The session comes back here, with the reply already appended.
            // That is the object worth storing; everything else is derived.
            Update::Done {
                completion,
                session,
            } => finished = Some((completion, session)),
            Update::Failed { error, session } => {
                // The session is returned intact on failure too, so a crash
                // mid-run still leaves something resumable rather than
                // discarding the work that did happen.
                eprintln!("\nrun failed: {error}");
                persist(&state, &session);
                return Err(error.into());
            }
            _ => {}
        }
    }

    let Some((completion, session)) = finished else {
        return Err("the run ended without reporting".into());
    };

    // Save before interpreting the outcome: a run that ran out of steps still
    // did work, and throwing it away because it did not finish is the bug this
    // whole example exists to avoid.
    persist(&state, &session);

    println!("\n\n── {} tool round(s)", completion.tool_rounds);
    if completion.compacted > 0 || completion.dropped > 0 {
        // Continuity across the context window. Compaction summarises rather
        // than truncates, which is why a long-running conversation stays
        // coherent instead of forgetting its own beginning.
        println!(
            "── context: {} message(s) summarised, {} dropped",
            completion.compacted, completion.dropped
        );
    }
    match completion.finish {
        Finish::Eos => println!("── done"),
        // Not an error, and not success either: there is more to do, and the
        // next invocation resumes with the transcript intact.
        other => println!("── paused ({other:?}) — run again to continue"),
    }
    println!("── session saved to {}", state.display());
    Ok(())
}
