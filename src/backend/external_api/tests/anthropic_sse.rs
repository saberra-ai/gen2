//! Parsing an Anthropic Messages SSE stream into `TokenEvent`s.
//!
//! Anthropic's stream is typed: the meaning of a `data:` line depends on the
//! `event:` line above it, and a handful of event types carry no text at all.
//! The cases worth pinning are the ones where the type is missing, stale, or
//! signals a failure the caller must not mistake for a finished answer.

use crate::generation::GenSpec;

use super::harness::Wire;

fn event(name: &str, data: serde_json::Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn text_delta(text: &str) -> String {
    event(
        "content_block_delta",
        serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text}
        }),
    )
}

fn message_stop() -> String {
    event("message_stop", serde_json::json!({"type": "message_stop"}))
}

#[test]
fn a_full_provider_stream_yields_its_tokens_in_order_then_eos() {
    let body = [
        event(
            "message_start",
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1", "role": "assistant"}}),
        ),
        event(
            "content_block_start",
            serde_json::json!({"type": "content_block_start", "index": 0,
                               "content_block": {"type": "text", "text": ""}}),
        ),
        text_delta("Hello"),
        text_delta(", world"),
        event(
            "content_block_stop",
            serde_json::json!({"type": "content_block_stop", "index": 0}),
        ),
        event(
            "message_delta",
            serde_json::json!({"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": null}}),
        ),
        message_stop(),
    ]
    .concat();
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:Hello", "token:, world", "eos"]);
}

#[test]
fn a_stream_with_no_event_lines_is_typed_from_the_payload_instead() {
    // Some proxies strip the `event:` line. The `type` field inside the JSON
    // carries the same information and has to be enough on its own.
    let body = "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"a\"}}\n\n\
                data: {\"type\":\"message_stop\"}\n\n";
    let x = Wire::anthropic().sse(body).run();

    assert_eq!(x.trace(), vec!["token:a", "eos"]);
}

#[test]
fn ping_and_unknown_event_types_are_ignored() {
    let body = [
        event("ping", serde_json::json!({"type": "ping"})),
        text_delta("a"),
        event(
            "something_new",
            serde_json::json!({"type": "something_new"}),
        ),
        text_delta("b"),
        message_stop(),
    ]
    .concat();
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(
        x.trace(),
        vec!["token:a", "token:b", "eos"],
        "an event type this parser has never heard of must be skipped, not fatal"
    );
}

#[test]
fn a_delta_with_no_text_field_produces_no_token() {
    // `thinking_delta` and `input_json_delta` share the `content_block_delta`
    // envelope but carry no `text`. Reading them as tokens would splice
    // reasoning and tool arguments into the visible answer.
    let body = [
        event(
            "content_block_delta",
            serde_json::json!({"type": "content_block_delta", "index": 0,
                               "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
        ),
        event(
            "content_block_delta",
            serde_json::json!({"type": "content_block_delta", "index": 1,
                               "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
        ),
        text_delta("visible"),
        message_stop(),
    ]
    .concat();
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:visible", "eos"]);
}

#[test]
fn an_empty_text_delta_produces_no_token() {
    let body = [text_delta(""), text_delta("real"), message_stop()].concat();
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:real", "eos"]);
}

#[test]
fn a_malformed_chunk_does_not_kill_the_stream() {
    let body = format!(
        "{}event: content_block_delta\ndata: {{\"delta\":{{\"text\":\n\n{}{}",
        text_delta("before"),
        text_delta("after"),
        message_stop()
    );
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:before", "token:after", "eos"]);
}

#[test]
fn blank_lines_and_comments_between_events_are_ignored() {
    let body = format!(
        "\n\n{}: keep-alive\n\n{}\n\n{}",
        text_delta("a"),
        text_delta("b"),
        message_stop()
    );
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "token:b", "eos"]);
}

#[test]
fn message_delta_with_a_stop_reason_ends_the_stream() {
    // The provider may close the block before `message_stop` arrives; the
    // stop reason is the real terminal signal.
    let body = [
        text_delta("a"),
        event(
            "message_delta",
            serde_json::json!({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}),
        ),
        text_delta("never"),
    ]
    .concat();
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "eos"]);
}

#[test]
fn message_delta_carrying_only_usage_does_not_end_the_stream() {
    let body = [
        text_delta("a"),
        event(
            "message_delta",
            serde_json::json!({"type": "message_delta", "usage": {"output_tokens": 1}}),
        ),
        text_delta("b"),
        message_stop(),
    ]
    .concat();
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(
        x.trace(),
        vec!["token:a", "token:b", "eos"],
        "a usage-only message_delta is bookkeeping, not a terminal event"
    );
}

#[test]
fn an_error_event_surfaces_as_an_error_carrying_the_providers_message() {
    // Anthropic reports overload mid-stream rather than as an HTTP status.
    // Swallowing it would hand the caller a truncated answer marked complete.
    let body = [
        text_delta("partial"),
        event(
            "error",
            serde_json::json!({"type": "error",
                               "error": {"type": "overloaded_error", "message": "Overloaded"}}),
        ),
    ]
    .concat();
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace().len(), 2, "trace was {:?}", x.trace());
    assert_eq!(x.trace()[0], "token:partial");
    assert!(
        x.trace()[1].contains("Overloaded"),
        "the provider's own wording is the only useful diagnostic here, got {:?}",
        x.trace()[1]
    );
}

#[test]
fn an_error_event_with_no_message_still_reports_something_actionable() {
    let body = event("error", serde_json::json!({"type": "error"}));
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace().len(), 1);
    assert!(
        x.trace()[0].contains("unknown Anthropic API error"),
        "got {:?}",
        x.trace()[0]
    );
}

#[test]
fn a_stream_that_ends_without_message_stop_still_ends_in_eos() {
    let body = [text_delta("a"), text_delta("b")].concat();
    let x = Wire::anthropic().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "token:b", "eos"]);
}

#[test]
fn an_empty_body_ends_immediately_in_eos() {
    let x = Wire::anthropic().sse("").run();

    assert_eq!(x.trace(), vec!["eos"]);
}

#[test]
fn multi_byte_text_split_across_chunk_boundaries_arrives_intact() {
    // Boundaries fall inside 你, inside 好, and inside the emoji.
    let line: &[u8] = b"event: content_block_delta\ndata: {\"delta\":{\"text\":\"\xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x91\x8b\"}}\n\n";
    let x = Wire::anthropic()
        .sse_chunks(&[
            &line[..52],
            &line[52..55],
            &line[55..60],
            &line[60..],
            message_stop().as_bytes(),
        ])
        .run();

    assert_eq!(x.text(), "你好 👋");
    assert_eq!(x.trace(), vec!["token:你好 👋", "eos"]);
}

#[test]
fn max_tokens_caps_the_number_of_tokens_the_caller_sees() {
    let spec = GenSpec {
        max_tokens: Some(2),
        ..Default::default()
    };
    let body = [
        text_delta("a"),
        text_delta("b"),
        text_delta("c"),
        message_stop(),
    ]
    .concat();
    let x = Wire::anthropic().gen_spec(spec).sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "token:b", "eos"]);
}
