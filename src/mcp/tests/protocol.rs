//! The wire shapes, checked without a child process.
//!
//! These types are one half of a contract whose other half lives in somebody
//! else's process, so the thing worth pinning is what goes on the wire
//! verbatim, and what the parser is willing to accept back.

use crate::mcp::protocol::{
    CallToolResult, ContentBlock, InitializeResult, JsonRpcNotification, JsonRpcRequest,
    JsonRpcResponse, ListToolsResult, PROTOCOL_VERSION, ToolDescriptor,
};

fn wire(v: &impl serde::Serialize) -> serde_json::Value {
    serde_json::to_value(v).expect("a request must always serialize")
}

// ── What the client puts on the wire ────────────────────────────────────────

#[test]
fn a_request_serializes_to_the_four_members_json_rpc_requires() {
    let req = JsonRpcRequest {
        jsonrpc: "2.0",
        id: 7,
        method: "tools/call",
        params: serde_json::json!({ "name": "echo", "arguments": { "text": "hi" } }),
    };

    assert_eq!(
        wire(&req),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": { "name": "echo", "arguments": { "text": "hi" } }
        }),
        "the request frame is read by a server this crate did not write; any \
         renamed or extra member is a wire break"
    );
}

#[test]
fn a_notification_carries_no_id_so_the_server_knows_not_to_answer() {
    let note = JsonRpcNotification {
        jsonrpc: "2.0",
        method: "notifications/initialized",
        params: serde_json::json!({}),
    };
    let frame = wire(&note);

    assert!(
        frame.get("id").is_none(),
        "an id would make this a request, and a server that answers it would \
         put an uncorrelated frame in front of the next real response: {frame}"
    );
    assert_eq!(frame["method"], "notifications/initialized");
}

#[test]
fn the_protocol_revision_is_the_one_the_handshake_advertises() {
    // Pinned rather than inferred: bumping it silently would leave servers
    // negotiating a revision nothing here was tested against.
    assert_eq!(PROTOCOL_VERSION, "2024-11-05");
}

// ── What the client accepts back ────────────────────────────────────────────

#[test]
fn a_result_response_parses_and_keeps_its_id_for_correlation() {
    let resp: JsonRpcResponse =
        serde_json::from_str(r#"{"jsonrpc":"2.0","id":3,"result":{"ok":true}}"#)
            .expect("a well-formed response must parse");

    assert_eq!(
        resp.id,
        serde_json::json!(3),
        "the id is kept as a raw value so it can be compared against the one we sent"
    );
    assert_eq!(resp.result, Some(serde_json::json!({ "ok": true })));
    assert!(resp.error.is_none());
}

#[test]
fn an_error_response_carries_its_code_message_and_optional_data() {
    let resp: JsonRpcResponse = serde_json::from_str(
        r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"no such method","data":{"hint":"typo"}}}"#,
    )
    .expect("a well-formed error response must parse");

    let err = resp.error.expect("the error object must survive parsing");
    assert_eq!(err.code, -32601);
    assert_eq!(err.message, "no such method");
    assert_eq!(err.data, Some(serde_json::json!({ "hint": "typo" })));
    assert!(
        resp.result.is_none(),
        "exactly one of result/error is present"
    );
}

#[test]
fn an_error_without_data_is_still_an_error() {
    let resp: JsonRpcResponse =
        serde_json::from_str(r#"{"id":1,"error":{"code":-1,"message":"nope"}}"#)
            .expect("`data` is optional in JSON-RPC");
    assert_eq!(resp.error.expect("still an error").data, None);
}

#[test]
fn members_the_client_does_not_model_are_ignored_rather_than_rejected() {
    // Forward compatibility is the whole reason to be lenient here: a server
    // speaking a later revision must not break a client that only needs the
    // members it already knows.
    let resp: JsonRpcResponse = serde_json::from_str(
        r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","instructions":"hi","_meta":{"x":1}},"trace":"abc"}"#,
    )
    .expect("unknown members must not fail the parse");

    let init: InitializeResult = serde_json::from_value(resp.result.expect("has a result"))
        .expect("an initialize result with unmodelled members must still parse");
    assert_eq!(init.protocol_version, "2025-06-18");
}

#[test]
fn an_initialize_result_needs_only_its_protocol_version() {
    let init: InitializeResult = serde_json::from_str(r#"{"protocolVersion":"2024-11-05"}"#)
        .expect("capabilities and serverInfo are optional");
    assert!(init.server_info.is_none());
    assert_eq!(init.capabilities, serde_json::Value::Null);

    let missing = serde_json::from_str::<InitializeResult>(r#"{"capabilities":{}}"#);
    assert!(
        missing.is_err(),
        "a handshake that names no protocol revision is not a handshake"
    );
}

#[test]
fn a_tool_descriptor_survives_a_server_that_omits_the_optional_halves() {
    let d: ToolDescriptor =
        serde_json::from_str(r#"{"name":"bare"}"#).expect("only `name` is required");
    assert_eq!(d.name, "bare");
    assert_eq!(d.description, "");
    assert_eq!(d.input_schema, serde_json::Value::Null);

    let nameless = serde_json::from_str::<ToolDescriptor>(r#"{"description":"no name"}"#);
    assert!(
        nameless.is_err(),
        "a tool with no name cannot be called, so it must not parse"
    );
}

#[test]
fn a_server_offering_no_tools_lists_none_rather_than_failing() {
    let empty: ListToolsResult =
        serde_json::from_str("{}").expect("an absent `tools` member means no tools");
    assert!(empty.tools.is_empty());
}

// ── Folding a tool result back into the model's context ─────────────────────

#[test]
fn a_call_result_defaults_to_success_with_no_content() {
    let r: CallToolResult = serde_json::from_str("{}").expect("both members are optional");
    assert!(r.content.is_empty());
    assert!(
        !r.is_error,
        "a result that says nothing about failure did not fail"
    );
    assert_eq!(r.text(), "");
}

#[test]
fn only_text_blocks_are_folded_into_the_model_text() {
    let r = CallToolResult {
        content: vec![
            ContentBlock {
                block_type: "text".into(),
                text: "first".into(),
            },
            ContentBlock {
                block_type: "image".into(),
                text: "base64-payload-the-model-must-not-read".into(),
            },
            ContentBlock {
                block_type: "text".into(),
                text: "second".into(),
            },
        ],
        is_error: false,
    };

    assert_eq!(
        r.text(),
        "first\nsecond",
        "a non-text block's `text` member is not prose; splicing it into the \
         transcript would feed the model an encoded blob"
    );
}

#[test]
fn an_in_band_tool_failure_is_flagged_without_losing_its_message() {
    let r: CallToolResult = serde_json::from_str(
        r#"{"content":[{"type":"text","text":"no such file"}],"isError":true}"#,
    )
    .expect("the error envelope is an ordinary result");

    assert!(r.is_error, "isError is renamed from camelCase on the wire");
    assert_eq!(r.text(), "no such file");
}
