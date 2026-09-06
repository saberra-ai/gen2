//! The remote path of the inference-first facade, driven the way a consumer
//! drives it: `Runtime::openai()…connect()` against a loopback server
//! standing in for a provider (api_spec.md §4.4, §28.12).
//!
//! `tests/external_openai.rs` proves the engine-level seam; this proves the
//! facade routes the model name onto the wire and maps the reply back, and
//! that it compiles against exactly what any other crate can reach.

#![cfg(feature = "backend-external-api")]

use gen2::event::Event;
use gen2::model::ModelSourceKind;
use gen2::output::FinishReason;
use gen2::{Runtime, Session};

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

/// A server answering `/v1/models` (which connecting probes) and
/// `/v1/chat/completions` — the latter only for a request naming `model`.
fn provider(model: &str, body: &str) -> (mockito::ServerGuard, mockito::Mock) {
    let mut server = mockito::Server::new();
    server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(r#"{"data":[]}"#)
        .expect_at_least(0)
        .create();
    let completions = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::PartialJsonString(format!(
            r#"{{"model":"{model}"}}"#
        )))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .expect(1)
        .create();
    (server, completions)
}

#[test]
fn a_remote_model_generates_and_names_itself_on_the_wire() {
    let (server, completions) = provider("m", &sse(&[token("Hello"), token(", world")]));
    let runtime = Runtime::new().expect("a runtime builds");
    let model = runtime
        .openai()
        .base_url(format!("{}/v1", server.url()))
        .model("m")
        .connect()
        .expect("a reachable OpenAI-compatible endpoint should connect");

    let text = model
        .generate("hi")
        .max_tokens(16)
        .text()
        .expect("generation over the remote model should succeed");

    assert_eq!(text, "Hello, world");
    completions.assert();

    let info = model.info();
    assert_eq!(info.name.as_deref(), Some("m"));
    assert_eq!(info.source, ModelSourceKind::Remote);
    assert!(!info.local);
    assert_eq!(
        info.context_window, None,
        "a provider that advertised nothing has no window to report"
    );
    let caps = model.capabilities();
    assert!(caps.text);
    assert!(
        !caps.structured_output,
        "no grammar reaches a remote endpoint"
    );
    assert_eq!(runtime.models(), vec![model.id()]);
}

#[test]
fn a_finished_remote_reply_reports_stop() {
    let last =
        "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}";
    let (server, _completions) = provider("qwen", &sse(&[token("done"), last.to_string()]));
    let model = Runtime::new()
        .expect("builds")
        .openai()
        .base_url(format!("{}/v1", server.url()))
        .model("qwen")
        .connect()
        .expect("connects");

    let response = model.generate("hi").run().expect("generates");
    assert_eq!(response.text(), "done!");
    assert_eq!(*response.finish_reason(), FinishReason::Stop);
}

#[test]
fn a_key_is_sent_when_given_and_nothing_when_not() {
    let mut server = mockito::Server::new();
    server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(r#"{"data":[]}"#)
        .expect_at_least(0)
        .create();
    let with_key = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer sk-test")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[token("keyed")]))
        .expect(1)
        .create();
    let without_key = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", mockito::Matcher::Missing)
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[token("open")]))
        .expect(1)
        .create();

    let runtime = Runtime::new().expect("builds");
    let keyed = runtime
        .openai()
        .base_url(format!("{}/v1", server.url()))
        .api_key("sk-test")
        .model("m")
        .connect()
        .expect("connects");
    let open = runtime
        .openai()
        .base_url(format!("{}/v1", server.url()))
        .model("m")
        .connect()
        .expect("a local server needs no key");

    assert_eq!(keyed.generate("hi").text().expect("generates"), "keyed");
    assert_eq!(open.generate("hi").text().expect("generates"), "open");
    with_key.assert();
    without_key.assert();
}

#[test]
fn an_unreachable_endpoint_fails_at_connect_not_at_first_token() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("loopback bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let err = Runtime::new()
        .expect("builds")
        .openai()
        .base_url(format!("http://127.0.0.1:{port}/v1"))
        .model("m")
        .connect()
        .expect_err("a closed port cannot be connected to");

    assert!(
        err.to_string()
            .contains("cannot connect to external server"),
        "the error should name the connectivity problem, got: {err}"
    );
}

/// A turn over a remote model streams the same semantic events a local one
/// does (api_spec.md §15, §28.10), and the session ends up holding the reply.
#[test]
fn a_remote_turn_streams_semantic_events_and_records_the_reply() {
    let last =
        "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}";
    let (server, completions) = provider(
        "m",
        &sse(&[token("Hello"), token(", world"), last.to_string()]),
    );
    let model = Runtime::new()
        .expect("builds")
        .openai()
        .base_url(format!("{}/v1", server.url()))
        .model("m")
        .connect()
        .expect("connects");

    let mut session = Session::new().with_system("Be brief.");
    let mut stream = model
        .turn(&mut session)
        .user("hi")
        .max_tokens(16)
        .stream()
        .expect("a remote turn streams");
    let mut deltas = Vec::new();
    let mut finished = None;
    for event in stream.by_ref() {
        match event.expect("no event fails") {
            Event::TextDelta(t) => deltas.push(t),
            Event::Finished(reason) => finished = Some(reason),
            _ => {}
        }
    }
    assert_eq!(deltas, ["Hello", ", world", "!"]);
    assert_eq!(finished, Some(FinishReason::Stop));

    let response = stream.finish().expect("the outcome");
    assert_eq!(response.text(), "Hello, world!");
    assert_eq!(*response.finish_reason(), FinishReason::Stop);
    completions.assert();

    let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant"]);
    assert_eq!(
        response.message_id(),
        session.active_ids().last().copied(),
        "the response names the message it appended"
    );
    assert_eq!(session.latest_text().as_deref(), Some("Hello, world!"));
}

/// Structured output over a remote model has no grammar to lean on: the
/// schema is asked for in the prompt and the reply is parsed, and a reply
/// that does not decode is a typed error rather than a panic or a guess.
#[test]
fn remote_structured_output_parses_and_reports_a_bad_reply_as_extraction() {
    #[derive(Debug, PartialEq, serde::Deserialize, gen2::schemars::JsonSchema)]
    struct Invoice {
        vendor: String,
        total: f64,
    }

    let mut server = mockito::Server::new();
    server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(r#"{"data":[]}"#)
        .expect_at_least(0)
        .create();
    let good = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::Regex("JSON Schema".into()))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[token(
            "{\\\"vendor\\\":\\\"Acme\\\",\\\"total\\\":12.5}",
        )]))
        .expect(1)
        .create();
    let model = Runtime::new()
        .expect("builds")
        .openai()
        .base_url(format!("{}/v1", server.url()))
        .model("m")
        .connect()
        .expect("connects");
    assert!(!model.capabilities().structured_output);

    let invoice: Invoice = model
        .generate("Acme — 12.50")
        .structured()
        .expect("valid JSON decodes");
    assert_eq!(
        invoice,
        Invoice {
            vendor: "Acme".into(),
            total: 12.5
        }
    );
    good.assert();

    let (server, _) = provider("m", &sse(&[token("no invoice here")]));
    let model = Runtime::new()
        .expect("builds")
        .openai()
        .base_url(format!("{}/v1", server.url()))
        .model("m")
        .connect()
        .expect("connects");
    let err = model
        .generate("nothing")
        .structured::<Invoice>()
        .expect_err("prose is not an invoice");
    assert_eq!(err.code(), Some("extraction_failed"));
}
