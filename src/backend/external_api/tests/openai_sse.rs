//! Parsing an OpenAI-format SSE stream into `TokenEvent`s.
//!
//! The happy path is one line of code; the value is in the ugly cases. Real
//! providers interleave comments, `event:`/`id:` fields and keep-alives,
//! truncate chunks mid-object, and split multi-byte characters across TCP
//! reads. Every one of those has to leave the stream running and the decoded
//! text byte-identical.

use crate::generation::GenSpec;

use super::harness::Wire;

fn chunk(content: &str) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({
            "id": "1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
        })
    )
}

fn finish(reason: &str) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({
            "id": "1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]
        })
    )
}

fn stream(parts: &[&str]) -> String {
    parts.concat()
}

#[test]
fn a_plain_stream_yields_its_tokens_in_order_then_eos() {
    let body = stream(&[
        &chunk("Hello"),
        &chunk(", "),
        &chunk("world"),
        "data: [DONE]\n\n",
    ]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(
        x.trace(),
        vec!["token:Hello", "token:, ", "token:world", "eos"]
    );
}

#[test]
fn the_role_only_opening_chunk_produces_no_token() {
    // Every OpenAI stream opens with `delta: {"role": "assistant"}`. Emitting
    // it as a token would put an empty string at the head of every reply.
    let body = stream(&[
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        &chunk("hi"),
        "data: [DONE]\n\n",
    ]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:hi", "eos"]);
}

#[test]
fn an_empty_content_delta_produces_no_token() {
    let body = stream(&[&chunk(""), &chunk("real"), "data: [DONE]\n\n"]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:real", "eos"]);
}

#[test]
fn extra_blank_lines_between_events_are_ignored() {
    let body = format!(
        "\n\n{}\n\n\n{}\n\ndata: [DONE]\n\n",
        chunk("a").trim_end(),
        chunk("b").trim_end()
    );
    let x = Wire::openai().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "token:b", "eos"]);
}

#[test]
fn event_and_id_fields_are_skipped_without_disturbing_the_tokens() {
    // Gateways (Azure, Cloudflare, nginx' SSE proxying) add these; a parser
    // that treats an unrecognised field as data would try to JSON-parse it.
    let body = format!(
        "event: message\nid: 42\nretry: 3000\n{}\n: this is an SSE comment\n{}\ndata: [DONE]\n\n",
        chunk("a"),
        chunk("b")
    );
    let x = Wire::openai().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "token:b", "eos"]);
}

#[test]
fn a_data_field_with_no_space_after_the_colon_is_still_parsed() {
    // `data:{...}` is legal SSE — the single optional space is stripped by the
    // spec, not required by it.
    let body = "data:{\"choices\":[{\"delta\":{\"content\":\"tight\"}}]}\n\ndata:[DONE]\n\n";
    let x = Wire::openai().sse(body).run();

    assert_eq!(x.trace(), vec!["token:tight", "eos"]);
}

#[test]
fn a_malformed_chunk_does_not_kill_the_stream() {
    // One truncated object mid-stream is a provider hiccup, not a reason to
    // throw away the tokens on either side of it.
    let body = stream(&[
        &chunk("before"),
        "data: {\"choices\":[{\"delta\":{\"content\":\n\n",
        &chunk("after"),
        "data: [DONE]\n\n",
    ]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(
        x.trace(),
        vec!["token:before", "token:after", "eos"],
        "an unparseable chunk is dropped and the stream continues"
    );
}

#[test]
fn a_data_line_that_is_not_json_at_all_is_dropped() {
    let body = stream(&[
        "data: OK\n\n",
        "data: keep-alive\n\n",
        &chunk("real"),
        "data: [DONE]\n\n",
    ]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:real", "eos"]);
}

#[test]
fn unknown_fields_on_a_chunk_are_ignored() {
    let body = stream(&[
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\",\"refusal\":null,\
         \"tool_calls\":[]},\"logprobs\":null,\"finish_reason\":null}],\
         \"usage\":null,\"service_tier\":\"default\",\"system_fingerprint\":\"fp_1\"}\n\n",
        "data: [DONE]\n\n",
    ]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(
        x.trace(),
        vec!["token:x", "eos"],
        "a chunk is read for the two fields that matter and nothing else"
    );
}

#[test]
fn finish_reason_stop_ends_the_stream() {
    let body = stream(&[
        &chunk("a"),
        &finish("stop"),
        &chunk("never"),
        "data: [DONE]\n\n",
    ]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(
        x.trace(),
        vec!["token:a", "eos"],
        "nothing after the terminal chunk may reach the caller"
    );
}

#[test]
fn finish_reason_length_ends_the_stream_the_same_way() {
    // A truncated answer is still a finished stream; the caller learns the
    // reason from the provider, not from a different event type.
    let body = stream(&[&chunk("a"), &finish("length")]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "eos"]);
}

#[test]
fn content_and_finish_reason_in_the_same_chunk_yield_the_token_and_then_eos() {
    // llama.cpp's server, vLLM and Together all put the last token and
    // `finish_reason` in one chunk. Ending the iterator right after the token
    // makes the controller read the generation as cancelled-by-user rather
    // than completed, so the final token must be followed by a real Eos.
    let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\
                \"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let x = Wire::openai().sse(body).run();

    assert_eq!(x.trace(), vec!["token:!", "eos"]);
}

#[test]
fn an_unrecognised_finish_reason_does_not_end_the_stream_early() {
    // Only `stop` and `length` are terminal to the parser. `tool_calls` and
    // `content_filter` fall through to the normal end-of-stream signal, which
    // is what keeps a tool-call chunk's content from being swallowed.
    let body = stream(&[
        &chunk("a"),
        &finish("tool_calls"),
        &chunk("b"),
        "data: [DONE]\n\n",
    ]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "token:b", "eos"]);
}

#[test]
fn a_stream_that_ends_without_done_still_ends_in_eos() {
    // Servers drop the connection instead of sending `[DONE]` more often than
    // the spec suggests. EOF is a completion, not an error.
    let body = stream(&[&chunk("a"), &chunk("b")]);
    let x = Wire::openai().sse(&body).run();

    assert_eq!(x.trace(), vec!["token:a", "token:b", "eos"]);
}

#[test]
fn a_final_chunk_with_no_trailing_newline_is_still_parsed() {
    let body = "data: {\"choices\":[{\"delta\":{\"content\":\"last\"}}]}";
    let x = Wire::openai().sse(body).run();

    assert_eq!(
        x.trace(),
        vec!["token:last", "eos"],
        "the last line of a stream cut short must not be discarded"
    );
}

#[test]
fn an_empty_body_ends_immediately_in_eos() {
    let x = Wire::openai().sse("").run();

    assert_eq!(x.trace(), vec!["eos"]);
}

#[test]
fn multi_byte_text_split_across_chunk_boundaries_arrives_intact() {
    // Every chunk boundary below falls inside a character: after the first
    // byte of 你, inside 好, and between the second and third bytes of the
    // emoji. A reader that decoded each socket read on its own would produce
    // replacement characters here.
    let line: &[u8] =
        b"data: {\"choices\":[{\"delta\":{\"content\":\"\xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x91\x8b\"}}]}\n\n";
    let x = Wire::openai()
        .sse_chunks(&[
            &line[..40],
            &line[40..43],
            &line[43..48],
            &line[48..],
            b"data: [DONE]\n\n",
        ])
        .run();

    assert_eq!(
        x.text(),
        "你好 👋",
        "text reassembled across chunk boundaries must be byte-identical"
    );
    assert_eq!(x.trace(), vec!["token:你好 👋", "eos"]);
}

#[test]
fn a_single_token_split_into_two_http_chunks_arrives_as_one_token() {
    // SSE events are delimited by the blank line, not by the transport's
    // framing, so one event delivered in two chunks is still one event.
    let x = Wire::openai()
        .sse_chunks(&[
            b"data: {\"choices\":[{\"delta\":{\"con",
            b"tent\":\"whole\"}}]}\n\ndata: [DONE]\n\n",
        ])
        .run();

    assert_eq!(x.trace(), vec!["token:whole", "eos"]);
}

#[test]
fn max_tokens_caps_the_number_of_tokens_the_caller_sees() {
    let spec = GenSpec {
        max_tokens: Some(2),
        ..Default::default()
    };
    let body = stream(&[
        &chunk("a"),
        &chunk("b"),
        &chunk("c"),
        &chunk("d"),
        "data: [DONE]\n\n",
    ]);
    let x = Wire::openai().gen_spec(spec).sse(&body).run();

    assert_eq!(
        x.trace(),
        vec!["token:a", "token:b", "eos"],
        "the budget is enforced locally; a server that ignores max_tokens must not overrun it"
    );
}

#[test]
fn a_mid_stream_error_object_ends_the_stream_as_an_error() {
    // A provider that aborts with `data: {"error": ...}` must not leave the
    // caller holding a truncated answer marked complete. The Anthropic parser
    // in this same directory always surfaced its `event: error`; this one used
    // to skip the object and reach `[DONE]` or EOF as though nothing had gone
    // wrong.
    let body = stream(&[
        &chunk("partial"),
        "data: {\"error\":{\"message\":\"upstream exploded\",\"type\":\"server_error\"}}\n\n",
    ]);
    let x = Wire::openai().sse(&body).run();

    let trace = x.trace();
    assert_eq!(
        trace.first().map(String::as_str),
        Some("token:partial"),
        "what arrived before the error still belongs to the caller: {trace:?}"
    );
    assert!(
        trace.last().is_some_and(|t| t.starts_with("err")),
        "the stream must end as an error, not as a clean finish: {trace:?}"
    );
    assert!(
        !trace.iter().any(|t| t == "eos"),
        "an aborted stream is not an end-of-sequence: {trace:?}"
    );
}
