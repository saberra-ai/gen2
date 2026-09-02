//! A coding agent, the way a terminal one actually works.
//!
//! Read, edit, list, search, run a command — a loop that keeps going until the
//! model answers or runs out of budget, printing what it does as it does it.
//! Everything here is gen2's public API; there is no privileged access.
//!
//! ```sh
//! cargo run --example coding_agent -- /path/model.gguf "add a docstring to src/lib.rs"
//! ```
//!
//! # What this is showing
//!
//! Five things a real agent needs that are easy to get wrong, and where gen2
//! puts each:
//!
//! | Need                          | Where it lives                        |
//! |-------------------------------|---------------------------------------|
//! | Tools with checked arguments  | `FunctionTool` + `JsonSchema`         |
//! | Not running forever           | `max_steps` / `max_tokens_total` / `deadline` |
//! | Noticing the model is stuck   | `Finish::GaveUp(Struggle::…)`         |
//! | Not clobbering things in parallel | `ExecutionPolicy::exclusive()`    |
//! | Showing work as it happens    | `run_streaming_text`                  |
//!
//! # What it deliberately does not do
//!
//! Sandboxing. `run_command` below executes what the model asks for, inside
//! whatever directory you point it at. That is fine for a demo you are
//! watching and wrong for anything else — see the note on that tool.

use std::path::{Path, PathBuf};
use std::time::Duration;

use gen2::schemars::JsonSchema;
use gen2::{
    AgentStep, ApprovalMode, Budget, Engine, ExecutionPolicy, Finish, FunctionTool, Session,
    Struggle, ToolError, ToolOutput,
};
use serde::Deserialize;

/// Everything the agent may touch. Passed to each tool by cloning it into the
/// handler's closure, which is why the tools below are built by functions
/// rather than declared as constants.
#[derive(Clone)]
struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Resolve a model-supplied path inside the workspace.
    ///
    /// The model writes paths; a path is an instruction to touch a file. This
    /// is the one check that keeps `../../.ssh/id_rsa` from being one of them.
    /// It is not a sandbox — see `run_command` — but a tool that takes a path
    /// from a model and joins it blindly is not defensible at any level of
    /// trust.
    fn resolve(&self, relative: &str) -> Result<PathBuf, ToolError> {
        let joined = self.root.join(relative);
        let canonical = joined
            .canonicalize()
            .or_else(|_| {
                // A file being created does not exist yet, so canonicalize its
                // parent instead and re-attach the name.
                let parent = joined.parent().ok_or(())?.canonicalize().map_err(|_| ())?;
                Ok::<PathBuf, ()>(parent.join(joined.file_name().ok_or(())?))
            })
            .map_err(|_| ToolError::Failed(format!("no such path: {relative}")))?;

        let root = self
            .root
            .canonicalize()
            .map_err(|e| ToolError::Unavailable(format!("workspace unreadable: {e}")))?;
        if !canonical.starts_with(&root) {
            return Err(ToolError::Failed(format!(
                "{relative} is outside the workspace"
            )));
        }
        Ok(canonical)
    }
}

// ── The tools ───────────────────────────────────────────────────────────────
//
// Each argument type derives `JsonSchema`, and that schema is what constrains
// decoding. The model cannot emit a call with a missing field or a number
// where a string belongs — not because it is asked nicely, but because the
// grammar will not let it.

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// Path to the file, relative to the workspace root.
    path: String,
}

fn read_file(ws: Workspace) -> FunctionTool<ReadArgs> {
    FunctionTool::new(
        "read_file",
        "Read a file's full contents. Use before editing anything.",
        move |_ctx, args: ReadArgs| {
            let ws = ws.clone();
            async move {
                let path = ws.resolve(&args.path)?;
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| ToolError::Failed(format!("could not read {}: {e}", args.path)))?;
                // Numbered, because the next thing the model does is talk
                // about a line.
                let numbered: String = text
                    .lines()
                    .enumerate()
                    .map(|(i, l)| format!("{:>4}  {l}\n", i + 1))
                    .collect();
                Ok(ToolOutput::from(numbered))
            }
        },
    )
}

#[derive(Deserialize, JsonSchema)]
struct ListArgs {
    /// Directory to list, relative to the workspace root. "." for the root.
    path: String,
}

fn list_files(ws: Workspace) -> FunctionTool<ListArgs> {
    FunctionTool::new(
        "list_files",
        "List the files and directories at a path.",
        move |_ctx, args: ListArgs| {
            let ws = ws.clone();
            async move {
                let path = ws.resolve(&args.path)?;
                let mut names: Vec<String> = std::fs::read_dir(&path)
                    .map_err(|e| ToolError::Failed(format!("could not list {}: {e}", args.path)))?
                    .flatten()
                    .map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if e.path().is_dir() {
                            format!("{name}/")
                        } else {
                            name
                        }
                    })
                    .collect();
                names.sort();
                Ok(ToolOutput::from(names.join("\n")))
            }
        },
    )
}

#[derive(Deserialize, JsonSchema)]
struct SearchArgs {
    /// Text to look for.
    query: String,
}

fn search(ws: Workspace) -> FunctionTool<SearchArgs> {
    FunctionTool::new(
        "search",
        "Find which files contain a piece of text, with line numbers.",
        move |_ctx, args: SearchArgs| {
            let ws = ws.clone();
            async move {
                let mut hits = Vec::new();
                walk(&ws.root, &mut |file| {
                    let Ok(text) = std::fs::read_to_string(file) else {
                        return;
                    };
                    for (i, line) in text.lines().enumerate() {
                        if line.contains(&args.query) {
                            let shown = file.strip_prefix(&ws.root).unwrap_or(file);
                            hits.push(format!("{}:{}: {}", shown.display(), i + 1, line.trim()));
                        }
                    }
                });
                // A tool that finds nothing has still worked. `Failed` would
                // tell the model to retry; an empty result tells it the truth.
                Ok(ToolOutput::from(if hits.is_empty() {
                    format!("no matches for {:?}", args.query)
                } else {
                    hits.into_iter().take(40).collect::<Vec<_>>().join("\n")
                }))
            }
        },
    )
}

#[derive(Deserialize, JsonSchema)]
struct EditArgs {
    /// Path to the file, relative to the workspace root.
    path: String,
    /// Exact text to replace. Must appear exactly once in the file.
    find: String,
    /// Text to put in its place.
    replace: String,
}

fn edit_file(ws: Workspace) -> FunctionTool<EditArgs> {
    FunctionTool::new(
        "edit_file",
        "Replace an exact snippet in a file. The snippet must be unique.",
        move |_ctx, args: EditArgs| {
            let ws = ws.clone();
            async move {
                let path = ws.resolve(&args.path)?;
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| ToolError::Failed(format!("could not read {}: {e}", args.path)))?;

                // Uniqueness is the whole safety property of a find/replace
                // edit. Two matches means the model does not know which one it
                // is changing — and neither does the person reading the diff.
                match text.matches(&args.find).count() {
                    0 => {
                        return Err(ToolError::Failed(
                            "that exact text is not in the file; read it again".into(),
                        ));
                    }
                    1 => {}
                    n => {
                        return Err(ToolError::Failed(format!(
                            "that text appears {n} times; include more context to \
                             pick the one you mean"
                        )));
                    }
                }

                std::fs::write(&path, text.replacen(&args.find, &args.replace, 1))
                    .map_err(|e| ToolError::Failed(format!("could not write: {e}")))?;
                Ok(ToolOutput::from(format!("edited {}", args.path)))
            }
        },
    )
    // Two edits racing on one file lose one of them. This is what
    // `parallel_safe: false` is for — the agent runs everything else
    // concurrently and serialises these.
    .with_policy(ExecutionPolicy::exclusive())
    // And it changes the caller's files, so `ApprovalMode::AskOnRisky` gates
    // it while `read_file` still runs freely.
    .risky()
}

#[derive(Deserialize, JsonSchema)]
struct CommandArgs {
    /// The shell command to run.
    command: String,
}

fn run_command(ws: Workspace) -> FunctionTool<CommandArgs> {
    FunctionTool::new(
        "run_command",
        "Run a shell command in the workspace and return its output.",
        move |_ctx, args: CommandArgs| {
            let ws = ws.clone();
            async move {
                let out = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(&args.command)
                    .current_dir(&ws.root)
                    .output()
                    .map_err(|e| ToolError::Unavailable(format!("could not spawn: {e}")))?;

                let mut text = String::new();
                text.push_str(&String::from_utf8_lossy(&out.stdout));
                if !out.stderr.is_empty() {
                    text.push_str("\n--- stderr ---\n");
                    text.push_str(&String::from_utf8_lossy(&out.stderr));
                }
                // A failing command is a *result*, not a tool failure: the
                // model asked what happens, and a non-zero exit is the answer.
                // Reporting it as an error would make the agent retry a
                // command that is working exactly as intended.
                if !out.status.success() {
                    text.push_str(&format!("\n(exit {})", out.status));
                }
                Ok(ToolOutput::from(text))
            }
        },
    )
    .with_policy(ExecutionPolicy::exclusive())
    .risky()
}

fn walk(dir: &Path, visit: &mut impl FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if path.is_dir() {
            walk(&path, visit);
        } else {
            visit(&path);
        }
    }
}

// ── The loop ────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: coding_agent <model.gguf> <task>")?;
    let task: String = args.collect::<Vec<_>>().join(" ");
    if task.is_empty() {
        return Err("give it something to do".into());
    }

    let workspace = Workspace {
        root: std::env::current_dir()?,
    };

    // One engine, loaded once. It owns the controller thread and shuts it down
    // when dropped.
    let engine = Engine::builder().model(&model).context(8192).build()?;

    // The conversation. It is the caller's, not the engine's — which is what
    // makes the run inspectable afterwards, and what an application would
    // serialise if it wanted the transcript to survive a restart.
    let mut session = Session::new().with_system(
        "You are a coding assistant working in a Rust repository. \
         Read files before you change them. Make the smallest edit that does \
         the job. When you are finished, say what you changed and why.",
    );

    println!("task: {task}\n");

    let done = engine
        .agent(&mut session)
        .add_tool(read_file(workspace.clone()))
        .add_tool(list_files(workspace.clone()))
        .add_tool(search(workspace.clone()))
        .add_tool(edit_file(workspace.clone()))
        .add_tool(run_command(workspace.clone()))
        // Three independent budgets, because runs fail in three different
        // ways: going round in circles, writing an essay, and taking all
        // afternoon. Whichever trips first ends the run.
        .max_steps(20)
        .max_tokens_total(30_000)
        .deadline(Duration::from_secs(300))
        // Yolo. Swap for `AskOnRisky` and the `edit_file` / `run_command`
        // tools above stop for a decision while reads keep flowing.
        .approval(ApprovalMode::Auto)
        .run_streaming_text(
            Some(task),
            // The model's prose, as it is produced.
            |delta| {
                print!("{delta}");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            },
            // And what it does between paragraphs.
            |step| match step {
                AgentStep::Calling { step, tool, args } => {
                    println!("\n  [{step}] {tool} {args}");
                }
                AgentStep::Called {
                    tool, result, took, ..
                } => match result {
                    Ok(out) => {
                        let text = out.to_model_text();
                        let first = text.lines().next().unwrap_or("");
                        println!(
                            "      {tool} ok in {took:?} — {first}{}",
                            if text.lines().count() > 1 { " …" } else { "" }
                        );
                    }
                    Err(e) => println!("      {tool} failed in {took:?} — {e}"),
                },
                _ => {}
            },
        )?;

    // How it ended matters as much as what it said. An agent that stopped
    // because it ran out of steps has not finished the job, and a caller that
    // treats that as success ships a half-done edit.
    println!("\n\n── {} tool round(s)", done.tool_rounds);
    match done.finish {
        Finish::Eos => println!("── finished"),
        Finish::OutOfBudget(Budget::Steps) => println!("── stopped: out of steps"),
        Finish::OutOfBudget(Budget::Tokens) => println!("── stopped: out of tokens"),
        Finish::OutOfBudget(Budget::Deadline) => println!("── stopped: out of time"),
        // The one a step budget alone never catches: the model is not
        // progressing, it is repeating itself. Ending here saves the rest of
        // the budget and tells the caller something a timeout would not.
        Finish::GaveUp(Struggle::RepeatingCall { tool, times }) => {
            println!("── gave up: called {tool} the same way {times} times");
        }
        Finish::GaveUp(Struggle::ToolKeepsFailing { tool, times }) => {
            println!("── gave up: {tool} failed {times} times");
        }
        other => println!("── {other:?}"),
    }

    if let Some(stats) = done.stats {
        println!(
            "── {} prompt + {} generated tokens",
            stats.prompt_tokens, stats.decode_tokens
        );
    }
    Ok(())
}
