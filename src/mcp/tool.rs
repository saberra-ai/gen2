//! Adapting MCP tools into this crate's [`Tool`].

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::client::{DEFAULT_TIMEOUT, McpClient, McpError};
use super::protocol::ToolDescriptor;
use crate::api::tools::{Tool, ToolContext, ToolError, ToolOutput, ToolSpec};

/// One tool exposed by an MCP server.
///
/// Holds a shared connection: a server is one subprocess speaking JSON-RPC over
/// stdio, so every tool from it takes a turn on the same pipe.
pub struct McpTool {
    spec: ToolSpec,
    client: Arc<Mutex<McpClient>>,
    /// The connection's per-request budget, carried so a timeout reports the
    /// deadline that actually elapsed rather than the default one.
    timeout: std::time::Duration,
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(
        &self,
        _ctx: &ToolContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let mut client = self.client.lock().await;
        let result = client
            .call_tool(&self.spec.name, args)
            .await
            .map_err(|e| map_error(e, self.timeout))?;

        // MCP returns content blocks; the model wants text. Concatenating is
        // lossy for images, which is honest until ToolOutput carries them.
        //
        // Through `CallToolResult::text`, so the two agree on what counts as
        // text. Reading `text` off every block regardless of type — which this
        // used to do — splices a non-text block's payload into the model's
        // transcript for any server that populates that field.
        let text = result.text();

        if result.is_error {
            // The server ran the tool and it failed — the model can react to
            // that, unlike a transport failure.
            return Err(ToolError::Failed(text));
        }
        Ok(ToolOutput::Text(text))
    }
}

/// A transport failure is not the model's to fix; a tool-level one is.
fn map_error(e: McpError, timeout: std::time::Duration) -> ToolError {
    match e {
        McpError::Timeout => ToolError::TimedOut(timeout),
        other => ToolError::Unavailable(other.to_string()),
    }
}

impl std::fmt::Debug for McpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpTool")
            .field("name", &self.spec.name)
            .finish_non_exhaustive()
    }
}

/// Every tool an MCP server offers, ready to register.
///
/// Iterating yields `Arc<dyn Tool>`, so it drops straight into
/// [`defer_tools`](crate::Agent::defer_tools).
pub struct McpToolSet {
    tools: Vec<Arc<dyn Tool>>,
    server: String,
}

impl McpToolSet {
    /// Spawn a server, handshake, and list its tools.
    pub async fn connect<S, I, A>(program: S, args: I) -> Result<Self, McpError>
    where
        S: AsRef<std::ffi::OsStr>,
        I: IntoIterator<Item = A>,
        A: AsRef<std::ffi::OsStr>,
    {
        Self::connect_with_timeout(program, args, DEFAULT_TIMEOUT).await
    }

    /// As [`McpToolSet::connect`], with a per-call timeout.
    pub async fn connect_with_timeout<S, I, A>(
        program: S,
        args: I,
        timeout: std::time::Duration,
    ) -> Result<Self, McpError>
    where
        S: AsRef<std::ffi::OsStr>,
        I: IntoIterator<Item = A>,
        A: AsRef<std::ffi::OsStr>,
    {
        let name = program.as_ref().to_string_lossy().into_owned();
        let mut client = McpClient::spawn(program, args, timeout).await?;
        client.initialize().await?;
        let descriptors = client.list_tools().await?;

        let client = Arc::new(Mutex::new(client));
        let tools = descriptors
            .into_iter()
            .map(|d| Arc::new(as_tool(d, Arc::clone(&client), timeout)) as Arc<dyn Tool>)
            .collect();

        Ok(Self {
            tools,
            server: name,
        })
    }

    /// How many tools the server offers.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the server offered none.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The server this came from.
    pub fn server(&self) -> &str {
        &self.server
    }

    /// The tool names, for logging or a settings UI.
    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.spec().name.clone()).collect()
    }
}

fn as_tool(
    d: ToolDescriptor,
    client: Arc<Mutex<McpClient>>,
    timeout: std::time::Duration,
) -> McpTool {
    McpTool {
        spec: ToolSpec::new(
            d.name,
            // A server may omit the description; search has nothing to match on
            // without one, so say plainly that it's missing rather than leave
            // it blank and fail the registry's validation.
            if d.description.trim().is_empty() {
                "(no description provided by the MCP server)".to_string()
            } else {
                d.description
            },
            d.input_schema,
        ),
        client,
        timeout,
    }
}

impl IntoIterator for McpToolSet {
    type Item = Arc<dyn Tool>;
    type IntoIter = std::vec::IntoIter<Arc<dyn Tool>>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.into_iter()
    }
}

impl std::fmt::Debug for McpToolSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolSet")
            .field("server", &self.server)
            .field("tools", &self.names())
            .finish()
    }
}
