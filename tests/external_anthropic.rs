//! The Anthropic Messages path, driven the way a consumer drives it.
//!
//! Same job as `external_openai.rs` for the other wire format, plus the two
//! things only this format has: loading that never probes `/models`, and
//! failures the provider reports inside a 200 stream rather than as a status.

#![cfg(feature = "backend-external-api")]

use std::sync::{Arc, Mutex};

use gen2::{Engine, Finish, PromptSettings, Settings};

fn text_delta(text: &str) -> String {
    format!(
        "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\
         \"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{text}\"}}}}\n\n"
    )
}

const MESSAGE_STOP: &str = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

fn provider(status: usize, body: &str) -> mockito::ServerGuard {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/messages")
        .with_status(status)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .expect_at_least(0)
        .create();
    server
}

#[test]
fn an_anthropic_endpoint_loads_without_probing_models_and_generates_text() {
    // Nothing serves `/models` here. Anthropic has no such endpoint, so
    // probing it would make every load fail.
    let body = format!(
        "{}{}{}",
        text_delta("Hello"),
        text_delta(", world"),
        MESSAGE_STOP
    );
    let server = provider(200, &body);

    let engine = Engine::builder()
        .anthropic(format!("{}/v1", server.url()), "sk-ant-test")
        .build()
        .expect("an Anthropic endpoint should load without a /models probe");

    let done = engine
        .infer("say hello")
        .max_tokens(16)
        .run()
        .expect("generation over the Anthropic backend should succeed");

    assert_eq!(done.text, "Hello, world");
    assert_eq!(done.finish, Finish::Eos);
}

#[test]
fn the_public_builder_really_configures_the_anthropic_wire_format() {
    // The end-to-end check that `.anthropic(..)` selects a format rather than
    // just a URL: auth header, endpoint, and the system prompt lifted out of
    // the messages array all have to be the Anthropic shapes.
    let seen: Arc<Mutex<Option<(String, serde_json::Value)>>> = Arc::new(Mutex::new(None));
    let sink = seen.clone();

    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/messages")
        .match_request(move |req| {
            let key = req
                .header("x-api-key")
                .first()
                .map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
                .unwrap_or_default();
            let body = serde_json::from_slice(&req.body().cloned().unwrap_or_default())
                .unwrap_or(serde_json::Value::Null);
            *sink.lock().unwrap() = Some((key, body));
            true
        })
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(MESSAGE_STOP)
        .expect_at_least(0)
        .create();

    let settings = Settings {
        prompt: PromptSettings {
            system_prompt: Some("You are terse.".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = Engine::builder()
        .anthropic(format!("{}/v1", server.url()), "sk-ant-test")
        .settings(settings)
        .build()
        .expect("load should succeed");
    let _ = engine.infer("say hello").max_tokens(8).run();

    let (key, body) = seen
        .lock()
        .unwrap()
        .clone()
        .expect("the generation should have reached the server");

    assert_eq!(key, "sk-ant-test", "the key must ride on x-api-key");
    assert_eq!(
        body["system"], "You are terse.",
        "the system prompt belongs in the top-level field, got body {body}"
    );
    let roles: Vec<&str> = body["messages"]
        .as_array()
        .expect("a messages array")
        .iter()
        .map(|m| m["role"].as_str().unwrap_or("<missing>"))
        .collect();
    assert!(
        !roles.contains(&"system"),
        "Anthropic rejects a system role inside messages, got {roles:?}"
    );
    assert!(
        body["max_tokens"].is_number(),
        "max_tokens is mandatory on this API, got body {body}"
    );
}

#[test]
fn a_provider_error_status_reaches_the_caller_as_an_error() {
    let server = provider(
        429,
        r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
    );
    let engine = Engine::builder()
        .anthropic(format!("{}/v1", server.url()), "sk-ant-test")
        .build()
        .expect("load should succeed");

    let err = engine
        .infer("say hello")
        .text()
        .expect_err("a 429 must not come back as an empty successful answer");

    assert!(
        err.to_string().contains("429"),
        "the status has to survive into the caller's error, got: {err}"
    );
}

#[test]
fn an_error_reported_inside_a_200_stream_still_reaches_the_caller() {
    // Anthropic signals overload mid-stream rather than with a status code.
    // Treating that as a clean end would hand the caller a truncated answer
    // with no indication anything went wrong.
    let body = format!(
        "{}event: error\ndata: {{\"type\":\"error\",\"error\":\
         {{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}}}\n\n",
        text_delta("partial")
    );
    let server = provider(200, &body);
    let engine = Engine::builder()
        .anthropic(format!("{}/v1", server.url()), "sk-ant-test")
        .build()
        .expect("load should succeed");

    let err = engine
        .infer("say hello")
        .text()
        .expect_err("a mid-stream error must not be reported as a finished answer");

    assert!(
        err.to_string().contains("Overloaded"),
        "the provider's own wording is the only useful diagnostic, got: {err}"
    );
}
