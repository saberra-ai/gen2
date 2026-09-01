//! [`McpClient`] against a real subprocess that may or may not cooperate.

use std::time::{Duration, Instant};

use super::{
    FAST, IMPATIENT, assert_reports_a_dead_server, deadline, python3_available, spawn_recording,
    spawn_role,
};
use crate::mcp::client::{McpClient, McpError};

/// Every frame the mock recorded, in the order the client sent them.
fn frames_sent(path: &std::path::Path) -> Vec<serde_json::Value> {
    let raw = std::fs::read_to_string(path).unwrap_or_default();
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l)
                .unwrap_or_else(|e| panic!("the client sent a frame that is not JSON: {l:?} ({e})"))
        })
        .collect()
}

// ── The happy path, and the bytes it puts on the wire ───────────────────────

#[tokio::test]
async fn a_cooperative_server_completes_the_whole_conversation() {
    if !python3_available() {
        return;
    }
    deadline("happy path", async {
        let mut client = spawn_role("ok", FAST).await;

        let init = client
            .initialize()
            .await
            .expect("the handshake must succeed");
        assert_eq!(init.protocol_version, "2024-11-05");
        let info = init.server_info.expect("the mock advertises a serverInfo");
        assert_eq!(info.name, "mock-mcp");

        let tools = client.list_tools().await.expect("tools/list must succeed");
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            ["echo", "undescribed", "explode"],
            "tools/list must preserve the server's order and every descriptor"
        );
        assert_eq!(
            tools[0].input_schema["properties"]["text"]["type"], "string",
            "the JSON Schema reaches the caller verbatim, not re-modelled"
        );
        assert_eq!(
            tools[1].description, "",
            "a descriptor with no description parses as empty here; supplying a \
             placeholder is McpToolSet's job, not the wire layer's"
        );

        let result = client
            .call_tool("echo", serde_json::json!({ "text": "hi" }))
            .await
            .expect("tools/call must succeed");
        assert!(!result.is_error);
        assert_eq!(result.text(), r#"{"text": "hi"}"#);

        client
            .shutdown()
            .await
            .expect("shutdown is best-effort but must report Ok");
    })
    .await;
}

#[tokio::test]
async fn the_client_sends_exactly_the_frames_the_lifecycle_requires() {
    if !python3_available() {
        return;
    }
    deadline("wire capture", async {
        let record = tempfile::NamedTempFile::new().expect("a temp file for the frame log");
        let mut client = spawn_recording("ok", FAST, Some(record.path())).await;

        client.initialize().await.expect("handshake");
        client.list_tools().await.expect("tools/list");
        client
            .call_tool("echo", serde_json::json!({ "text": "hi" }))
            .await
            .expect("tools/call");
        client.shutdown().await.expect("shutdown");

        let sent = frames_sent(record.path());
        let methods: Vec<&str> = sent.iter().map(|f| f["method"].as_str().unwrap()).collect();
        assert_eq!(
            methods,
            [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call"
            ],
            "the MCP lifecycle requires the initialized notification between the \
             handshake and the first real request; an SDK-backed server refuses \
             everything until it arrives"
        );

        assert_eq!(
            sent[0],
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "pio-agent", "version": env!("CARGO_PKG_VERSION") }
                }
            }),
            "the handshake frame is the client's whole identity to a server it has \
             never met"
        );
        assert!(
            sent[1].get("id").is_none(),
            "the initialized notification must carry no id, or the server will \
             answer it and desynchronise correlation: {}",
            sent[1]
        );
        assert_eq!(sent[2]["params"], serde_json::json!({}));
        assert_eq!(
            sent[3]["params"],
            serde_json::json!({ "name": "echo", "arguments": { "text": "hi" } })
        );

        let ids: Vec<u64> = sent
            .iter()
            .filter_map(|f| f.get("id").and_then(|i| i.as_u64()))
            .collect();
        assert_eq!(
            ids,
            [1, 2, 3],
            "ids must be fresh and monotonic; a reused id would let a stale \
             response satisfy a later request"
        );
    })
    .await;
}

// ── The program itself ──────────────────────────────────────────────────────

#[tokio::test]
async fn a_program_that_does_not_exist_fails_at_spawn_naming_it() {
    let err = McpClient::spawn(
        "gen2-no-such-mcp-server-anywhere",
        Vec::<String>::new(),
        FAST,
    )
    .await
    .expect_err("spawning a program that is not on PATH cannot succeed");

    match err {
        McpError::Spawn { cmd, .. } => assert_eq!(
            cmd, "gen2-no-such-mcp-server-anywhere",
            "the message must name the command, because a typo'd server path is \
             the most likely cause and the only fixable one"
        ),
        other => panic!("expected a Spawn error, got {other:?}"),
    }
}

#[tokio::test]
async fn a_directory_is_not_an_executable_server() {
    // A caller pointing at the wrong path usually lands on a directory, and the
    // failure must arrive at spawn rather than as a mysterious silent server.
    let err = McpClient::spawn(env!("CARGO_MANIFEST_DIR"), Vec::<String>::new(), FAST)
        .await
        .expect_err("a directory cannot be executed");
    assert!(
        matches!(err, McpError::Spawn { .. }),
        "expected a Spawn error, got {err:?}"
    );
}

// ── Servers that stop talking ───────────────────────────────────────────────

#[tokio::test]
async fn a_server_that_exits_before_the_handshake_is_reported_not_awaited() {
    if !python3_available() {
        return;
    }
    deadline("exit before handshake", async {
        let mut client = spawn_role("exit-immediately", FAST).await;
        let err = client
            .initialize()
            .await
            .expect_err("a server that never spoke cannot have handshaken");
        assert_reports_a_dead_server(&err, "initialize against a server that exited");
    })
    .await;
}

#[tokio::test]
async fn a_server_that_dies_mid_conversation_is_reported_not_awaited() {
    if !python3_available() {
        return;
    }
    deadline("death mid-conversation", async {
        let mut client = spawn_role("die-after-initialize", FAST).await;
        client
            .initialize()
            .await
            .expect("the handshake happens before the server exits");

        let err = client
            .list_tools()
            .await
            .expect_err("the server is gone by now");
        assert_reports_a_dead_server(&err, "tools/list after the server exited");

        // The connection stays dead rather than recovering into something worse.
        let again = client
            .call_tool("echo", serde_json::json!({}))
            .await
            .expect_err("a dead pipe does not come back");
        assert_reports_a_dead_server(&again, "a second request on a dead pipe");
    })
    .await;
}

// ── Servers that talk nonsense ──────────────────────────────────────────────

#[tokio::test]
async fn a_line_that_is_not_json_is_a_protocol_error_not_a_panic() {
    if !python3_available() {
        return;
    }
    deadline("non-json line", async {
        let mut client = spawn_role("not-json", FAST).await;
        let err = client
            .initialize()
            .await
            .expect_err("a human-readable log line on stdout is not a response");
        assert!(
            matches!(err, McpError::Protocol(_)),
            "a server logging to stdout is the single most common MCP integration \
             mistake; it must be named as a protocol violation, got {err:?}"
        );
        assert!(
            err.to_string().contains("malformed server frame"),
            "the message must point at the frame, got {err}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_truncated_json_frame_is_a_protocol_error_not_a_panic() {
    if !python3_available() {
        return;
    }
    deadline("truncated frame", async {
        let mut client = spawn_role("malformed-json", FAST).await;
        let err = client
            .initialize()
            .await
            .expect_err("half a JSON object is not a response");
        assert!(
            matches!(err, McpError::Protocol(_)),
            "expected a protocol error, got {err:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_response_with_neither_result_nor_error_is_a_protocol_error() {
    if !python3_available() {
        return;
    }
    deadline("empty response", async {
        let mut client = spawn_role("no-result-no-error", FAST).await;
        let err = client
            .initialize()
            .await
            .expect_err("JSON-RPC requires exactly one of result/error");
        assert!(
            matches!(err, McpError::Protocol(_)),
            "expected a protocol error, got {err:?}"
        );
        assert!(
            err.to_string().contains("neither result nor error"),
            "the message must say what was missing, got {err}"
        );
    })
    .await;
}

// ── Servers that answer the wrong question, or none ─────────────────────────

#[tokio::test]
async fn a_frame_carrying_someone_elses_id_is_skipped_not_taken() {
    if !python3_available() {
        return;
    }
    deadline("id correlation", async {
        let mut client = spawn_role("noise-before-answer", FAST).await;
        let init = client
            .initialize()
            .await
            .expect("the correlated answer follows the noise");
        assert_eq!(
            init.protocol_version, "2024-11-05",
            "the mismatched-id frame advertises 0.0.0; taking it would mean the \
             client answered a question it never asked"
        );

        let tools = client
            .list_tools()
            .await
            .expect("correlation must keep working after skipping frames");
        assert_eq!(tools.len(), 3);
    })
    .await;
}

#[tokio::test]
async fn a_server_that_never_answers_trips_the_timeout_rather_than_hanging() {
    if !python3_available() {
        return;
    }
    deadline("silent server", async {
        let started = Instant::now();
        let mut client = spawn_role("silent", IMPATIENT).await;
        let err = client
            .initialize()
            .await
            .expect_err("a server that says nothing cannot have handshaken");

        assert!(
            matches!(err, McpError::Timeout),
            "the timeout argument is the only thing standing between a wedged \
             server and a wedged host, got {err:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the timeout must fire on its own deadline ({IMPATIENT:?}), not \
             whenever the server happens to give up; took {:?}",
            started.elapsed()
        );
    })
    .await;
}

#[tokio::test]
async fn a_server_that_only_ever_answers_the_wrong_id_times_out() {
    if !python3_available() {
        return;
    }
    deadline("uncorrelated forever", async {
        let mut client = spawn_role("wrong-id", IMPATIENT).await;
        let err = client
            .initialize()
            .await
            .expect_err("no frame ever carried our id");
        assert!(
            matches!(err, McpError::Timeout),
            "skipping uncorrelated frames must not become an unbounded wait, \
             got {err:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_json_rpc_error_object_surfaces_with_its_code_and_message() {
    if !python3_available() {
        return;
    }
    deadline("rpc error", async {
        let mut client = spawn_role("rpc-error", FAST).await;
        client
            .initialize()
            .await
            .expect("the handshake is answered normally");

        let err = client
            .call_tool("echo", serde_json::json!({}))
            .await
            .expect_err("the server refuses everything after the handshake");
        match err {
            McpError::Rpc { code, message } => {
                assert_eq!(code, -32603);
                assert_eq!(message, "the server is having a bad day");
            }
            other => panic!(
                "a JSON-RPC error object is the server's answer, not a transport \
                 failure; it must keep its code so a caller can tell -32602 (bad \
                 arguments) from -32603 (server fault), got {other:?}"
            ),
        }
    })
    .await;
}

#[tokio::test]
async fn calling_a_tool_the_server_does_not_have_is_an_rpc_error() {
    if !python3_available() {
        return;
    }
    deadline("unknown tool", async {
        let mut client = spawn_role("ok", FAST).await;
        client.initialize().await.expect("handshake");
        let err = client
            .call_tool("no_such_tool", serde_json::json!({}))
            .await
            .expect_err("the server rejects the call");
        match err {
            McpError::Rpc { code, message } => {
                assert_eq!(code, -32602);
                assert!(message.contains("no_such_tool"), "got {message}");
            }
            other => panic!("expected an Rpc error, got {other:?}"),
        }
    })
    .await;
}

// ── Servers that are merely rude ────────────────────────────────────────────

#[tokio::test]
async fn chatter_on_stderr_is_not_fatal() {
    if !python3_available() {
        return;
    }
    deadline("noisy stderr", async {
        // stderr is inherited, so the lines below land in this test's own
        // output. That is the contract: server logs are for humans, and only
        // stdout carries frames.
        let mut client = spawn_role("noisy-stderr", FAST).await;
        client
            .initialize()
            .await
            .expect("logging is not a protocol violation");
        let tools = client
            .list_tools()
            .await
            .expect("nor does it break the next request");
        assert_eq!(tools.len(), 3);
    })
    .await;
}

#[tokio::test]
async fn members_a_newer_server_adds_do_not_break_the_conversation() {
    if !python3_available() {
        return;
    }
    deadline("extra fields", async {
        let mut client = spawn_role("extra-fields", FAST).await;
        let init = client
            .initialize()
            .await
            .expect("unknown members must be ignored, not rejected");
        assert_eq!(init.protocol_version, "2024-11-05");

        let tools = client
            .list_tools()
            .await
            .expect("annotations on a descriptor are not this client's business");
        assert_eq!(tools.len(), 3);
    })
    .await;
}

#[tokio::test]
async fn a_multi_megabyte_result_arrives_intact() {
    if !python3_available() {
        return;
    }
    deadline("huge result", async {
        // A file read or a `git log` over a big repo is genuinely this size;
        // line-framing must not cap or truncate it.
        let mut client = spawn_role("huge", FAST).await;
        client.initialize().await.expect("handshake");
        let result = client
            .call_tool("echo", serde_json::json!({}))
            .await
            .expect("a large frame is still one frame");

        let text = result.text();
        assert_eq!(
            text.len(),
            2 * 1024 * 1024,
            "the whole payload must survive; a short read here would silently \
             hand the model a truncated file"
        );
        assert!(
            text.bytes().all(|b| b == b'z'),
            "the payload must be unmangled"
        );
    })
    .await;
}

// ── Teardown ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn shutdown_does_not_wait_on_a_server_that_ignores_eof() {
    if !python3_available() {
        return;
    }
    deadline("shutdown", async {
        // The mock's `silent` role sleeps for 30s regardless of stdin. Closing
        // stdin politely is only half of shutdown; the kill is the half that
        // has to be there.
        let started = Instant::now();
        let client = spawn_role("silent", FAST).await;
        client.shutdown().await.expect("shutdown must not fail");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "shutdown must kill a server that ignores EOF rather than wait it \
             out; took {:?}",
            started.elapsed()
        );
    })
    .await;
}
