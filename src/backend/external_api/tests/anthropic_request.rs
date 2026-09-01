//! What the backend puts on the wire in Anthropic Messages format.
//!
//! Anthropic differs from OpenAI in three ways that are easy to get wrong and
//! silent when wrong: the system prompt is a top-level field rather than a
//! message, `max_tokens` is required, and auth is a header pair rather than a
//! bearer token.

use crate::engine::{PromptSettings, SamplingSettings, Settings, StoppingSettings};
use crate::generation::GenSpec;

use super::harness::{Wire, assert_number, assistant, chunked, system, tool_call, user};

const STOP: &str = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

#[test]
fn a_completion_is_posted_to_messages_under_the_configured_base_url() {
    let x = Wire::anthropic().sse(STOP).run();

    assert_eq!(
        x.request().path,
        "/v1/messages",
        "the Messages API lives at /messages, not /chat/completions"
    );
}

#[test]
fn the_api_key_travels_as_x_api_key_with_a_pinned_version() {
    let x = Wire::anthropic().sse(STOP).run();

    assert_eq!(
        x.request().header("x-api-key"),
        Some("test-key"),
        "Anthropic auth is `x-api-key`, unprefixed"
    );
    assert_eq!(
        x.request().header("anthropic-version"),
        Some("2023-06-01"),
        "the version header is mandatory; without it every request 400s"
    );
    assert_eq!(
        x.request().header("authorization"),
        None,
        "a bearer token would be ignored and leak the key to a second header"
    );
}

#[test]
fn a_session_with_no_api_key_sends_neither_the_key_nor_the_version_header() {
    // The version header rides along with the key in `Session::pull`, so
    // dropping the key drops both. Anthropic-compatible proxies that need no
    // key generally need no version either.
    let x = Wire::anthropic().no_key().sse(STOP).run();

    assert_eq!(x.request().header("x-api-key"), None);
    assert_eq!(x.request().header("anthropic-version"), None);
}

#[test]
fn the_system_prompt_is_a_top_level_field_and_not_a_message() {
    let settings = Settings {
        prompt: PromptSettings {
            system_prompt: Some("You are terse.".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Wire::anthropic()
        .settings(settings)
        .messages(vec![user("hi")])
        .sse(STOP)
        .run();

    assert_eq!(x.request().body["system"], "You are terse.");
    assert_eq!(
        x.request().roles(),
        vec!["user"],
        "Anthropic rejects `role: system` inside the messages array"
    );
}

#[test]
fn a_system_message_in_the_history_is_lifted_out_into_the_system_field() {
    let x = Wire::anthropic()
        .messages(vec![system("from history"), user("hi")])
        .sse(STOP)
        .run();

    assert_eq!(x.request().body["system"], "from history");
    assert_eq!(x.request().roles(), vec!["user"]);
}

#[test]
fn the_settings_system_prompt_wins_over_a_system_message_in_the_history() {
    // The reverse of the OpenAI builder's precedence, and worth pinning
    // because only one of the two can be sent: `build_anthropic_request`
    // seeds `system_text` from settings first and a history system message
    // only fills an empty slot.
    let settings = Settings {
        prompt: PromptSettings {
            system_prompt: Some("from settings".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Wire::anthropic()
        .settings(settings)
        .messages(vec![system("from history"), user("hi")])
        .sse(STOP)
        .run();

    assert_eq!(x.request().body["system"], "from settings");
}

#[test]
fn no_system_prompt_anywhere_means_no_system_field_at_all() {
    let x = Wire::anthropic().messages(vec![user("hi")]).sse(STOP).run();

    assert!(
        x.request().body.get("system").is_none(),
        "an absent system prompt must be omitted, not sent as an empty string"
    );
}

#[test]
fn max_tokens_is_always_present_and_defaults_when_unset() {
    let x = Wire::anthropic().sse(STOP).run();

    assert_eq!(
        x.request().body["max_tokens"],
        4096,
        "max_tokens is required by the Messages API, so the builder supplies a default"
    );
}

#[test]
fn an_explicit_max_tokens_replaces_the_default() {
    let spec = GenSpec {
        max_tokens: Some(32),
        ..Default::default()
    };
    let x = Wire::anthropic().gen_spec(spec).sse(STOP).run();

    assert_eq!(x.request().body["max_tokens"], 32);
}

#[test]
fn max_tokens_falls_back_to_the_engine_stopping_setting() {
    let settings = Settings {
        stopping: StoppingSettings {
            max_tokens: Some(11),
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Wire::anthropic().settings(settings).sse(STOP).run();

    assert_eq!(x.request().body["max_tokens"], 11);
}

#[test]
fn stop_words_are_serialized_as_stop_sequences() {
    let settings = Settings {
        stopping: StoppingSettings {
            stopwords: vec!["\n\nHuman:".into()],
            max_tokens: None,
        },
        ..Default::default()
    };
    let x = Wire::anthropic().settings(settings).sse(STOP).run();

    assert_eq!(
        x.request().body["stop_sequences"],
        serde_json::json!(["\n\nHuman:"]),
        "Anthropic names the field `stop_sequences`; `stop` is silently ignored"
    );
}

#[test]
fn sampling_fields_are_serialized_when_set() {
    let settings = Settings {
        sampling: SamplingSettings {
            temperature: Some(0.3),
            top_p: Some(0.7),
            seed: Some(1234),
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Wire::anthropic().settings(settings).sse(STOP).run();
    let body = &x.request().body;

    assert_number(body, "temperature", 0.3);
    assert_number(body, "top_p", 0.7);
    assert!(
        body.get("seed").is_none(),
        "the Messages API has no `seed` parameter; sending one is a 400, so it is dropped"
    );
}

#[test]
fn message_order_and_roles_survive_serialization() {
    let x = Wire::anthropic()
        .messages(vec![user("first"), assistant("second"), user("third")])
        .sse(STOP)
        .run();

    assert_eq!(x.request().contents(), vec!["first", "second", "third"]);
    assert_eq!(x.request().roles(), vec!["user", "assistant", "user"]);
}

#[test]
fn multi_chunk_text_is_flattened_into_one_string() {
    let x = Wire::anthropic()
        .messages(vec![chunked("user", &["alpha", " ", "beta"])])
        .sse(STOP)
        .run();

    assert_eq!(x.request().contents(), vec!["alpha beta"]);
}

#[test]
fn a_tool_call_turn_is_dropped() {
    let x = Wire::anthropic()
        .messages(vec![user("q"), tool_call("search"), user("q2")])
        .sse(STOP)
        .run();

    assert_eq!(x.request().contents(), vec!["q", "q2"]);
}

#[test]
fn the_request_asks_for_a_stream_of_the_configured_model() {
    let x = Wire::anthropic().model("claude-sonnet-4").sse(STOP).run();

    assert_eq!(x.request().body["model"], "claude-sonnet-4");
    assert_eq!(x.request().body["stream"], true);
}
