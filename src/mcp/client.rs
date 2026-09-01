//! Minimal stdio JSON-RPC MCP **client**.
//!
//! Hand-rolled to match `pio-mcp-server`'s own hand-rolled server (no `rmcp` /
//! MCP SDK dependency, per the Slice B engineering fork). Speaks LF-delimited
//! JSON-RPC 2.0 over a child process's stdin/stdout — the exact transport
//! `pio-mcp-server/src/transport/stdio.rs` serves — so Pio's agent can consume
//! any external stdio MCP server.
//!
//! Framing: one JSON object per line, split on `\n` only (a JSON string may
//! contain an escaped `\\n`, which is not a real newline, so line-splitting is
//! safe). Stderr is left to the server for human-readable logs; only stdout
//! carries frames.
//!
//! Robustness contract (Slice B step 3/5): every read/write is bounded by a
//! timeout; malformed server output returns [`McpError::Protocol`] rather than
//! panicking; a server that closes mid-handshake returns
//! [`McpError::ServerClosed`]; a `tools/call` error response surfaces as
//! [`McpError::Rpc`]. The child is killed on drop.

use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::protocol::{
    CallToolResult, InitializeResult, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    ListToolsResult, PROTOCOL_VERSION, ToolDescriptor,
};

/// Default per-request timeout. An external MCP server that neither answers nor
/// closes within this window is treated as hung.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Errors the client surfaces. Every variant is a graceful return — the client
/// never panics on server output.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The child process could not be spawned.
    #[error("failed to spawn MCP server `{cmd}`: {source}")]
    Spawn {
        cmd: String,
        #[source]
        source: std::io::Error,
    },
    /// stdin/stdout pipe was not available on the child.
    #[error("MCP server pipe unavailable: {0}")]
    Pipe(String),
    /// Underlying I/O error talking to the child.
    #[error("MCP io error: {0}")]
    Io(String),
    /// A request exceeded its timeout with neither a response nor a close.
    #[error("timed out waiting for MCP server response")]
    Timeout,
    /// The server closed its stdout (EOF) before answering — e.g. it crashed
    /// mid-handshake.
    #[error("MCP server closed the connection before responding")]
    ServerClosed,
    /// The server emitted a line that is not a valid JSON-RPC response.
    #[error("MCP protocol error: {0}")]
    Protocol(String),
    /// The server answered with a JSON-RPC error object (e.g. `tools/call` on
    /// an unknown tool, or an invalid-argument error).
    #[error("MCP server returned error {code}: {message}")]
    Rpc { code: i32, message: String },
}

/// A live connection to a stdio MCP server subprocess.
pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    next_id: u64,
    timeout: Duration,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The pipes are not printable, and the pid is what a caller chasing a
        // stuck server actually needs.
        f.debug_struct("McpClient")
            .field("pid", &self.child.id())
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl McpClient {
    /// Spawn `program` with `args` as a child MCP server and wire up the stdio
    /// transport. Does **not** perform the handshake — call [`Self::initialize`]
    /// next. stderr is inherited so the server's logs reach the parent's
    /// terminal without polluting the JSONL channel.
    pub async fn spawn<S, I, A>(program: S, args: I, timeout: Duration) -> Result<Self, McpError>
    where
        S: AsRef<std::ffi::OsStr>,
        I: IntoIterator<Item = A>,
        A: AsRef<std::ffi::OsStr>,
    {
        let cmd_label = program.as_ref().to_string_lossy().into_owned();
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|source| McpError::Spawn {
                cmd: cmd_label.clone(),
                source,
            })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Pipe("child stdin not piped".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Pipe("child stdout not piped".into()))?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
            next_id: 1,
            timeout,
        })
    }

    /// Perform the MCP `initialize` handshake. Advertises the client's protocol
    /// version + identity; returns what the server advertised back.
    pub async fn initialize(&mut self) -> Result<InitializeResult, McpError> {
        let params = serde_json::json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "pio-agent", "version": env!("CARGO_PKG_VERSION") }
        });
        let result = self.request("initialize", params).await?;
        let parsed = serde_json::from_value(result)
            .map_err(|e| McpError::Protocol(format!("invalid initialize result: {e}")))?;

        // The lifecycle requires this notification before any other request:
        // an SDK-backed server (the Python `mcp` server behind `mcp-server-git`
        // among them) refuses `tools/list` until it arrives. Best-effort,
        // because a server that has already died must be reported by the next
        // real request — with `ServerClosed` — not by a broken-pipe `Io` from a
        // frame the caller never asked about.
        let _ = self
            .notify(
                "notifications/initialized",
                serde_json::Value::Object(Default::default()),
            )
            .await;

        Ok(parsed)
    }

    /// List the tools the server exposes (`tools/list`).
    pub async fn list_tools(&mut self) -> Result<Vec<ToolDescriptor>, McpError> {
        let result = self.request("tools/list", serde_json::json!({})).await?;
        let parsed: ListToolsResult = serde_json::from_value(result)
            .map_err(|e| McpError::Protocol(format!("invalid tools/list result: {e}")))?;
        Ok(parsed.tools)
    }

    /// Call one tool (`tools/call`). A JSON-RPC error from the server (unknown
    /// tool, bad args) is returned as [`McpError::Rpc`]; a successful call whose
    /// result carries `isError: true` is returned as a [`CallToolResult`] with
    /// `is_error == true` (an in-band tool failure, not a protocol failure).
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        let params = serde_json::json!({ "name": name, "arguments": arguments });
        let result = self.request("tools/call", params).await?;
        serde_json::from_value(result)
            .map_err(|e| McpError::Protocol(format!("invalid tools/call result: {e}")))
    }

    /// Send one notification. Carries no `id`, so there is nothing to await —
    /// only the write is bounded.
    async fn notify(&mut self, method: &str, params: serde_json::Value) -> Result<(), McpError> {
        let note = JsonRpcNotification {
            jsonrpc: "2.0",
            method,
            params,
        };
        self.write_frame(&note).await
    }

    /// Serialize one frame and put it on the child's stdin, LF-terminated.
    async fn write_frame<T: serde::Serialize>(&mut self, frame: &T) -> Result<(), McpError> {
        let mut line = serde_json::to_string(frame)
            .map_err(|e| McpError::Protocol(format!("failed to encode request: {e}")))?;
        line.push('\n');

        tokio::time::timeout(self.timeout, self.stdin.write_all(line.as_bytes()))
            .await
            .map_err(|_| McpError::Timeout)?
            .map_err(|e| McpError::Io(e.to_string()))?;
        tokio::time::timeout(self.timeout, self.stdin.flush())
            .await
            .map_err(|_| McpError::Timeout)?
            .map_err(|e| McpError::Io(e.to_string()))?;
        Ok(())
    }

    /// Send one request and return its `result` value, correlating by `id`.
    /// Notifications and any interleaved frame whose `id` does not match the
    /// one we sent are skipped. Every read/write is timeout-bounded.
    async fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, McpError> {
        let id = self.next_id;
        self.next_id += 1;

        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };
        self.write_frame(&req).await?;

        // Read until we see the response carrying our id (or the server closes,
        // or we time out). A single malformed line is a protocol violation —
        // the server's stdout is supposed to be pure JSONL — so we fail rather
        // than spin.
        loop {
            let next = tokio::time::timeout(self.timeout, self.stdout.next_line())
                .await
                .map_err(|_| McpError::Timeout)?
                .map_err(|e| McpError::Io(e.to_string()))?;

            let line = match next {
                Some(l) => l,
                None => return Err(McpError::ServerClosed),
            };
            if line.trim().is_empty() {
                continue;
            }

            let resp: JsonRpcResponse = serde_json::from_str(&line)
                .map_err(|e| McpError::Protocol(format!("malformed server frame: {e}")))?;

            // Correlate by id; skip notifications / mismatched frames.
            if resp.id != serde_json::json!(id) {
                continue;
            }
            if let Some(err) = resp.error {
                return Err(McpError::Rpc {
                    code: err.code,
                    message: err.message,
                });
            }
            return resp
                .result
                .ok_or_else(|| McpError::Protocol("response had neither result nor error".into()));
        }
    }

    /// Close stdin (signalling EOF to the server) and reap the child. Best-effort
    /// — a server that ignores EOF is killed via `kill_on_drop`.
    pub async fn shutdown(mut self) -> Result<(), McpError> {
        // Dropping stdin closes it; do it explicitly first so a well-behaved
        // server sees EOF and exits on its own.
        let _ = self.stdin.shutdown().await;
        let _ = self.child.start_kill();
        let _ = self.child.wait().await;
        Ok(())
    }
}
