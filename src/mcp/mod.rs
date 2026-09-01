//! MCP client — consuming external stdio MCP servers.
//!
//! Ported from `pio-core` during the gen2 extraction, minus its host-specific
//! approval gate: this crate's [`Risk`](crate::Risk) and
//! [`ApprovalMode`](crate::ApprovalMode) cover that.
//!
//! Hand-rolled rather than depending on an MCP SDK — the wire surface a client
//! needs is `initialize`, `tools/list`, `tools/call`, and stdio framing.
//!
//! `McpToolSet` adapts a server's tools into [`Tool`](crate::Tool)s, so an
//! MCP server's whole surface registers as an iterator:
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! # use gen2::mcp::McpToolSet;
//! let mcp = McpToolSet::connect("mcp-server-git", ["--repo", "."]).await?;
//! # let engine: gen2::Engine = unimplemented!();
//! # let mut session = gen2::Session::new();
//! engine.agent(&mut session)
//!     .defer_tools(mcp)                       // forty tools, none in the prompt
//!     .tool_search(gen2::ToolSearch::Hybrid)
//!     .goal("What changed in the last commit?")?;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod protocol;
mod tool;

#[cfg(test)]
mod tests;

pub use client::{DEFAULT_TIMEOUT, McpClient, McpError};
pub use protocol::{
    CallToolResult, ContentBlock, InitializeResult, ListToolsResult, PROTOCOL_VERSION, ServerInfo,
    ToolDescriptor,
};
pub use tool::{McpTool, McpToolSet};
