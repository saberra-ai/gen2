//! The OpenAI-compatible path, driven the way a consumer drives it.
//!
//! The unit tests under `src/backend/external_api/` prove the wire format in
//! isolation. This target proves the whole seam holds through the public API —
//! builder, controller loop, completion — against a loopback server standing in
//! for a provider. It compiles against exactly what any other crate can reach,
//! so a gap in the public surface fails here first.

#![cfg(feature = "backend-external-api")]

use gen2::{Engine, Finish};

fn token(text: &str) -> String {
    format!("{{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{text}\"}}}}]}}")
}

fn sse(events: &[String]) -> String {
    let mut body = String::new();
    for e in events {
        body.push_str("data: ");
        body.push_str(e);
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// A server answering `/v1/models` (which loading probes) and
/// `/v1/chat/completions` with `body`.
fn provider(status: usize, body: &str) -> mockito::ServerGuard {
    let mut server = mockito::Server::new();
    server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(r#"{"data":[]}"#)
        .expect_at_least(0)
        .create();
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(status)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .expect_at_least(0)
        .create();
    server
}

#[test]
fn an_openai_endpoint_can_be_loaded_and_generates_text() {
    let server = provider(200, &sse(&[token("Hello"), token(", world")]));
    let engine = Engine::builder()
        .openai(format!("{}/v1", server.url()), "sk-test")
        .build()
        .expect("a reachable OpenAI-compatible endpoint should load");

    let text = engine
        .infer("say hello")
        .max_tokens(16)
        .text()
        .expect("generation over the external backend should succeed");

    assert_eq!(
        text, "Hello, world",
        "the fragments must arrive concatenated and in order"
    );
}

#[test]
fn a_generation_that_ends_at_finish_reason_reports_eos_not_a_user_stop() {
    // The provider puts its last token and `finish_reason` in one chunk, which
    // is what llama.cpp's server, vLLM and Together all do. Reporting
    // `Finish::Stopped` here would tell the caller a completed answer was
    // cancelled, and a host that retries on Stopped would retry forever.
    let last =
        "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}";
    let server = provider(200, &sse(&[token("done"), last.to_string()]));
    let engine = Engine::builder()
        .openai(format!("{}/v1", server.url()), "sk-test")
        .build()
        .expect("load should succeed");

    let done = engine
        .infer("say hello")
        .max_tokens(16)
        .run()
        .expect("generation should succeed");

    assert_eq!(done.text, "done!", "the final token must not be swallowed");
    assert_eq!(
        done.finish,
        Finish::Eos,
        "a stream the provider itself finished ends in Eos"
    );
}

#[test]
fn a_provider_error_status_reaches_the_caller_as_an_error() {
    let server = provider(401, r#"{"error":{"message":"Incorrect API key provided"}}"#);
    let engine = Engine::builder()
        .openai(format!("{}/v1", server.url()), "sk-wrong")
        .build()
        .expect("loading only probes /models, which still answers");

    let err = engine
        .infer("say hello")
        .text()
        .expect_err("a 401 must not come back as an empty successful answer");

    assert!(
        err.to_string().contains("401"),
        "the status has to survive into the caller's error, got: {err}"
    );
}

#[test]
fn an_unreachable_endpoint_fails_at_build_time_rather_than_at_first_token() {
    // Loading probes `/models` precisely so a typo'd URL or a down provider is
    // a load failure the caller can report, not a mystery on first generation.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("loopback bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let err = Engine::builder()
        .openai(format!("http://127.0.0.1:{port}/v1"), "sk-test")
        .build()
        .expect_err("a closed port cannot be loaded");

    assert!(
        err.to_string()
            .contains("cannot connect to external server"),
        "the error should name the connectivity problem, got: {err}"
    );
}
