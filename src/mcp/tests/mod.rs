//! What the MCP client owes a caller who points it at someone else's binary.
//!
//! The client's whole job is talking to a process this crate did not write, so
//! the tests that matter are the ones where that process misbehaves: it never
//! existed, it dies mid-sentence, it prints a log line down the JSON channel,
//! it answers a question nobody asked, it never answers at all. Every one of
//! those is driven against a real child process — `tests/fixtures/mcp/
//! mock_server.py`, whose role argument picks the misbehaviour — because the
//! bugs live in the pipe handling, and a hand-mocked pipe would test the mock.
//!
//! Two rules hold everywhere in here. Every child-process test runs inside
//! [`deadline`], so a client that wedges fails the suite instead of hanging it;
//! and every client is dropped on the way out, which kills the child
//! (`kill_on_drop`), including when an assertion panics.

mod client;
mod protocol;
mod tool;

use std::path::PathBuf;
use std::time::Duration;

use super::client::{McpClient, McpError};

/// Longest any test in this module may run. Far above the per-request timeouts
/// the tests themselves set, so tripping this means something hung, not that a
/// machine was slow.
const TEST_DEADLINE: Duration = Duration::from_secs(20);

/// The per-request budget for tests that expect an answer. Short enough that a
/// regression which breaks correlation shows up as a fast failure.
const FAST: Duration = Duration::from_millis(2_000);

/// The budget for tests that expect the timeout itself to fire.
const IMPATIENT: Duration = Duration::from_millis(250);

fn mock_server() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mcp/mock_server.py")
}

/// Whether a python3 that can run the mock is on PATH.
///
/// Printing is the point: a test that skips in silence reads as a test that
/// passed, and these are the only tests standing behind a documented feature.
fn python3_available() -> bool {
    match std::process::Command::new("python3")
        .arg("--version")
        .output()
    {
        Ok(out) if out.status.success() => true,
        _ => {
            eprintln!(
                "mcp: client tests SKIPPED — `python3` is not on PATH, so the mock MCP \
                 server at tests/fixtures/mcp/mock_server.py cannot be run"
            );
            false
        }
    }
}

/// Spawn the mock in `role` and hand back a connected (but un-handshaken)
/// client.
async fn spawn_role(role: &str, timeout: Duration) -> McpClient {
    spawn_recording(role, timeout, None).await
}

/// As [`spawn_role`], with every frame the client sends appended to `record`.
///
/// The path travels in argv rather than the environment: `set_var` is
/// process-wide, and these tests run in parallel with each other.
async fn spawn_recording(
    role: &str,
    timeout: Duration,
    record: Option<&std::path::Path>,
) -> McpClient {
    let script = mock_server();
    let mut args = vec![
        script.as_os_str().to_owned(),
        std::ffi::OsString::from(role),
    ];
    if let Some(p) = record {
        args.push(p.as_os_str().to_owned());
    }
    McpClient::spawn("python3", args, timeout)
        .await
        .expect("spawning python3 with the mock server script must succeed")
}

/// Run `fut` under a hard deadline. A hanging test is worse than a failing one:
/// it takes the whole suite with it and reports nothing.
async fn deadline<F: std::future::Future>(what: &str, fut: F) -> F::Output {
    match tokio::time::timeout(TEST_DEADLINE, fut).await {
        Ok(v) => v,
        Err(_) => panic!("{what}: exceeded the {TEST_DEADLINE:?} test deadline — the client hung"),
    }
}

/// A server that vanished can be noticed on either side of the pipe: the write
/// lands on a closed stdin, or it succeeds into the kernel buffer and the read
/// hits EOF. Both are the same fact about the world, so both are acceptable.
fn assert_reports_a_dead_server(err: &McpError, context: &str) {
    assert!(
        matches!(err, McpError::ServerClosed | McpError::Io(_)),
        "{context}: a server that is gone must surface as ServerClosed or an io error \
         (the write may lose the race to the read), got {err:?}"
    );
}
