//! An MCP server's tools, seen through this crate's [`Tool`] trait.
//!
//! The adaptation is where an external server stops being a subprocess and
//! starts being something the agent loop will call unattended, so what matters
//! is that the spec the model sees is the descriptor the server sent, that a
//! call reaches the right tool by name, and that the two kinds of failure stay
//! distinguishable: one the model can fix, one it cannot.

use std::sync::Arc;
use std::time::Duration;

use super::{
    FAST, IMPATIENT, assert_reports_a_dead_server, deadline, mock_server, python3_available,
};
use crate::api::tools::{Tool, ToolContext, ToolError, ToolOutput};
use crate::mcp::{McpError, McpToolSet};

async fn connect_role(role: &str, timeout: Duration) -> Result<McpToolSet, McpError> {
    connect_recording(role, timeout, None).await
}

async fn connect_recording(
    role: &str,
    timeout: Duration,
    record: Option<&std::path::Path>,
) -> Result<McpToolSet, McpError> {
    let script = mock_server();
    let mut args = vec![
        script.as_os_str().to_owned(),
        std::ffi::OsString::from(role),
    ];
    if let Some(p) = record {
        args.push(p.as_os_str().to_owned());
    }
    McpToolSet::connect_with_timeout("python3", args, timeout).await
}

/// Connect and keep only the tools — most tests here are about one tool's
/// behaviour, not the set's.
async fn tools_of(role: &str, timeout: Duration) -> Vec<Arc<dyn Tool>> {
    connect_role(role, timeout)
        .await
        .unwrap_or_else(|e| panic!("connecting to the `{role}` mock must succeed: {e}"))
        .into_iter()
        .collect()
}

fn tool_named(tools: &[Arc<dyn Tool>], name: &str) -> Arc<dyn Tool> {
    tools
        .iter()
        .find(|t| t.spec().name == name)
        .cloned()
        .unwrap_or_else(|| panic!("the server offered no tool called {name}"))
}

// ── What connecting produces ────────────────────────────────────────────────

#[tokio::test]
async fn connecting_yields_one_tool_per_descriptor_the_server_listed() {
    if !python3_available() {
        return;
    }
    deadline("connect", async {
        let set = connect_role("ok", FAST)
            .await
            .expect("connect must succeed");

        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());
        assert_eq!(set.names(), ["echo", "undescribed", "explode"]);
        assert_eq!(
            set.server(),
            "python3",
            "the set names the program it came from, which is what a log or a \
             settings UI has to show"
        );
    })
    .await;
}

#[tokio::test]
async fn a_tools_spec_is_the_descriptor_the_server_sent() {
    if !python3_available() {
        return;
    }
    deadline("spec derivation", async {
        let tools = tools_of("ok", FAST).await;
        let echo = tool_named(&tools, "echo");
        let spec = echo.spec();

        assert_eq!(spec.description, "Echo the arguments back as text");
        assert_eq!(
            spec.input_schema["required"],
            serde_json::json!(["text"]),
            "the server's JSON Schema is what the model is told to satisfy; \
             rewriting it here would let the model send arguments the server rejects"
        );
        assert!(
            spec.searchable_text().contains("what to echo"),
            "an MCP schema's argument docs must reach tool search, or a deferred \
             MCP tool is unfindable: {}",
            spec.searchable_text()
        );
    })
    .await;
}

#[tokio::test]
async fn a_server_that_omits_a_description_still_produces_a_registrable_tool() {
    if !python3_available() {
        return;
    }
    deadline("missing description", async {
        let tools = tools_of("ok", FAST).await;
        let spec = tool_named(&tools, "undescribed").spec().clone();

        assert!(
            !spec.description.trim().is_empty(),
            "the registry rejects an empty description, so a server that omits \
             one must not be able to take the whole tool set down with it"
        );
        assert!(
            spec.description.contains("no description"),
            "the placeholder must read as missing rather than as a real \
             description, got {:?}",
            spec.description
        );
    })
    .await;
}

#[tokio::test]
async fn a_server_offering_nothing_connects_to_an_empty_set() {
    if !python3_available() {
        return;
    }
    deadline("no tools", async {
        let set = connect_role("no-tools", FAST)
            .await
            .expect("a server with no tools is not a broken server");
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(set.names().is_empty());
    })
    .await;
}

#[tokio::test]
async fn a_server_that_cannot_handshake_fails_connect_rather_than_yielding_no_tools() {
    if !python3_available() {
        return;
    }
    deadline("failed connect", async {
        let err = connect_role("exit-immediately", FAST)
            .await
            .expect_err("there is nothing to connect to");
        assert_reports_a_dead_server(&err, "connecting to a server that exited");
    })
    .await;
}

// ── What calling one does ───────────────────────────────────────────────────

#[tokio::test]
async fn calling_a_tool_routes_to_the_server_under_that_tools_name() {
    if !python3_available() {
        return;
    }
    deadline("routing", async {
        let record = tempfile::NamedTempFile::new().expect("a temp file for the frame log");
        let tools: Vec<Arc<dyn Tool>> = connect_recording("ok", FAST, Some(record.path()))
            .await
            .expect("connect")
            .into_iter()
            .collect();
        let echo = tool_named(&tools, "echo");

        let out = echo
            .call(
                &ToolContext::new("session-1"),
                serde_json::json!({ "text": "hello" }),
            )
            .await
            .expect("the call must succeed");

        assert_eq!(
            out,
            ToolOutput::Text(r#"{"text": "hello"}"#.to_string()),
            "the server's text blocks are what the model reads back"
        );

        let call = std::fs::read_to_string(record.path())
            .expect("the mock records every frame")
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .find(|f| f["method"] == "tools/call")
            .expect("a tools/call frame must have reached the server");
        assert_eq!(
            call["params"],
            serde_json::json!({ "name": "echo", "arguments": { "text": "hello" } }),
            "the tool's own name is what routes the call; sending any other name \
             would silently run a different tool"
        );
    })
    .await;
}

#[tokio::test]
async fn a_tool_that_ran_and_failed_is_reported_back_to_the_model() {
    if !python3_available() {
        return;
    }
    deadline("in-band failure", async {
        let tools = tools_of("ok", FAST).await;
        let err = tool_named(&tools, "explode")
            .call(&ToolContext::new("s"), serde_json::json!({}))
            .await
            .expect_err("the server flagged isError");

        match &err {
            ToolError::Failed(msg) => assert_eq!(msg, "the tool ran and failed"),
            other => panic!(
                "an `isError` result is the tool's own answer, not a broken \
                 connection; collapsing the two costs the model the one failure \
                 it could have recovered from, got {other:?}"
            ),
        }
        assert!(err.is_model_actionable());
    })
    .await;
}

#[tokio::test]
async fn a_transport_failure_is_not_handed_to_the_model_to_fix() {
    if !python3_available() {
        return;
    }
    deadline("transport failure", async {
        let tools = tools_of("die-after-list", FAST).await;
        let err = tool_named(&tools, "echo")
            .call(&ToolContext::new("s"), serde_json::json!({}))
            .await
            .expect_err("the server exits instead of answering");

        assert!(
            matches!(err, ToolError::Unavailable(_)),
            "a dead subprocess is nothing the model can rephrase its way out of, \
             got {err:?}"
        );
        assert!(!err.is_model_actionable());
    })
    .await;
}

#[tokio::test]
async fn an_unknown_tool_error_from_the_server_reaches_the_caller_intact() {
    if !python3_available() {
        return;
    }
    deadline("rpc error through the tool", async {
        // `rpc-error` lists tools normally and then refuses every call.
        let tools = tools_of("rpc-error", FAST).await;
        let err = tool_named(&tools, "echo")
            .call(&ToolContext::new("s"), serde_json::json!({}))
            .await
            .expect_err("the server answers with a JSON-RPC error");

        match &err {
            ToolError::Unavailable(msg) => assert!(
                msg.contains("-32603") && msg.contains("bad day"),
                "the server's code and message are the only diagnosis a caller \
                 gets, got {msg}"
            ),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    })
    .await;
}

#[tokio::test]
async fn a_wedged_server_reports_the_deadline_that_actually_elapsed() {
    if !python3_available() {
        return;
    }
    deadline("tool timeout", async {
        let started = std::time::Instant::now();
        let tools = tools_of("silent-after-list", IMPATIENT).await;
        let err = tool_named(&tools, "echo")
            .call(&ToolContext::new("s"), serde_json::json!({}))
            .await
            .expect_err("the server never answers the call");

        match err {
            ToolError::TimedOut(d) => assert_eq!(
                d, IMPATIENT,
                "the reported deadline must be this connection's, not the crate \
                 default; a caller that set 250ms and is told `timed out after \
                 10s` will go looking for the wrong problem"
            ),
            other => panic!("expected TimedOut, got {other:?}"),
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
    })
    .await;
}

// ── Registration ────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_tool_set_iterates_into_things_the_registry_can_hold() {
    if !python3_available() {
        return;
    }
    deadline("iteration", async {
        let set = connect_role("ok", FAST).await.expect("connect");
        let tools: Vec<Arc<dyn Tool>> = set.into_iter().collect();
        assert_eq!(tools.len(), 3);
        assert!(
            tools.iter().all(|t| !t.spec().name.is_empty()),
            "every tool must arrive named, since the name is what the model calls"
        );
    })
    .await;
}

#[tokio::test]
async fn every_tool_from_one_server_shares_the_one_connection() {
    if !python3_available() {
        return;
    }
    deadline("shared pipe", async {
        // A server is one subprocess and one pipe, so two tools must take turns
        // on it rather than interleave frames.
        let tools = tools_of("ok", FAST).await;
        let echo = tool_named(&tools, "echo");
        let explode = tool_named(&tools, "explode");

        let ctx = ToolContext::new("s");
        let (a, b) = tokio::join!(
            echo.call(&ctx, serde_json::json!({ "text": "one" })),
            explode.call(&ctx, serde_json::json!({}))
        );

        assert_eq!(
            a.expect("echo must still get its own answer"),
            ToolOutput::Text(r#"{"text": "one"}"#.to_string()),
            "concurrent calls must not cross-deliver each other's results"
        );
        assert!(matches!(b, Err(ToolError::Failed(_))), "got {b:?}");
    })
    .await;
}

/// The two paths that turn content blocks into text must agree on which
/// blocks count. A server that puts a payload in a non-text block's `text`
/// field would otherwise have it spliced into the model's transcript.
#[test]
fn only_text_blocks_reach_the_model() {
    let result = crate::mcp::protocol::CallToolResult {
        content: vec![
            crate::mcp::protocol::ContentBlock {
                block_type: "text".into(),
                text: "the answer".into(),
            },
            crate::mcp::protocol::ContentBlock {
                block_type: "image".into(),
                text: "PAYLOAD THAT IS NOT TEXT".into(),
            },
        ],
        is_error: false,
    };

    let joined = result.text();
    assert!(joined.contains("the answer"));
    assert!(
        !joined.contains("PAYLOAD"),
        "a non-text block reached the model's transcript: {joined:?}"
    );
}
