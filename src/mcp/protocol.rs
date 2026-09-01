//! MCP wire types — the **client** side of the protocol.
//!
//! Self-contained mirror of `pio-mcp-server/src/protocol/{jsonrpc,initialize,
//! tools}.rs`. `pio-mcp-server` deliberately does not depend on `pio-core`, and
//! `pio-core` must not depend on it either (it stays a lean npm/brew binary),
//! so we re-declare the minimal shapes here rather than share a crate. Only the
//! wire *shape* is shared, pinned by the round-trip integration test against a
//! real server subprocess.
//!
//! Reference: <https://modelcontextprotocol.io/specification/2024-11-05> — the
//! same `2024-11-05` revision `pio-mcp-server` pins in its `protocol::mod`.

use serde::{Deserialize, Serialize};

/// MCP protocol revision Pio's client speaks in `initialize`. Matches
/// `pio_mcp_server::protocol::PROTOCOL_VERSION`.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Outbound JSON-RPC 2.0 request. Mirrors the server's `JsonRpcRequest`
/// (`jsonrpc`, `id`, `method`, `params`) from the write side. We always send a
/// numeric `id` — the client never emits notifications.
#[derive(Debug, Serialize)]
pub struct JsonRpcRequest<'a> {
    pub jsonrpc: &'static str,
    pub id: u64,
    pub method: &'a str,
    pub params: serde_json::Value,
}

/// Inbound JSON-RPC 2.0 response. Mirrors the server's `JsonRpcResponse`:
/// exactly one of `result` / `error` is present. `id` is kept as a raw
/// `Value` so we can correlate it against the numeric id we sent (and skip
/// any interleaved notification, which carries no matching id).
#[derive(Debug, Deserialize)]
pub struct JsonRpcResponse {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    #[serde(default)]
    pub id: serde_json::Value,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

/// Inbound JSON-RPC error object. Mirrors the server's `JsonRpcError`.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(default)]
    pub data: Option<serde_json::Value>,
}

/// `initialize` result. Mirrors the server's `InitializeResult`
/// (`protocolVersion`, `capabilities`, `serverInfo`). We keep `capabilities`
/// as a free-form `Value` — the client does not branch on individual
/// capability flags today, it just records what the server advertised.
#[derive(Debug, Deserialize, Clone)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    #[serde(default)]
    pub capabilities: serde_json::Value,
    #[serde(rename = "serverInfo", default)]
    pub server_info: Option<ServerInfo>,
}

/// Server identity from the `initialize` handshake. Mirrors `ServerInfoOut`.
#[derive(Debug, Deserialize, Clone)]
pub struct ServerInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: String,
}

/// One tool descriptor from `tools/list`. Mirrors the server's
/// `ToolDescriptor` (`name`, `description`, `inputSchema`). `input_schema` is a
/// free-form JSON Schema `Value` — the client surfaces it to the agent loop
/// verbatim rather than re-modelling every schema keyword.
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct ToolDescriptor {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: serde_json::Value,
}

/// `tools/list` result. Mirrors the server's `ListToolsResult`.
#[derive(Debug, Deserialize, Clone)]
pub struct ListToolsResult {
    #[serde(default)]
    pub tools: Vec<ToolDescriptor>,
}

/// One content block of a `tools/call` result. Mirrors the `{type,text}` shape
/// the server's `text_content` helper emits.
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct ContentBlock {
    #[serde(rename = "type", default)]
    pub block_type: String,
    #[serde(default)]
    pub text: String,
}

/// `tools/call` result. Mirrors the server's tool-call envelope
/// `{content:[{type:"text",text}], isError:bool}`.
#[derive(Debug, Deserialize, Clone)]
pub struct CallToolResult {
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

impl CallToolResult {
    /// Concatenate every text block — the usual way a host folds a tool result
    /// back into the model's context.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter(|b| b.block_type == "text")
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
