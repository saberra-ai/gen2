//! The wire bench every test in this directory runs on.
//!
//! A test states what the session is configured with and what the server will
//! say back; the bench puts a real `Session` in front of a real socket
//! (mockito on loopback) and hands back both halves of the exchange — the
//! bytes reqwest actually sent, and the `TokenEvent`s the puller produced.
//!
//! Going over a socket rather than calling `build_openai_request` directly is
//! deliberate: header casing, JSON encoding and chunk boundaries are part of
//! the contract with a real provider, and none of them are observable from
//! inside the request builder.

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use crate::backend::external_api::session::Session;
use crate::engine::{ExecError, HookBus, Settings};
use crate::generation::{GenSpec, TokenEvent};
use crate::types::message::{Message, MessageBody, MessageChunk, MessageContent};

/// No stream under test is infinite; a puller that keeps yielding past this is
/// looping, and the cap turns a hung suite into a failed assertion.
const MAX_EVENTS: usize = 256;

/// One HTTP request as it arrived at the server.
pub(super) struct Recorded {
    pub path: String,
    headers: Vec<(String, String)>,
    pub body: Value,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn messages(&self) -> &[Value] {
        match self.body.get("messages").and_then(Value::as_array) {
            Some(m) => m,
            None => panic!("request carried no `messages` array: {}", self.body),
        }
    }

    /// Roles in wire order — the shape most ordering assertions are about.
    pub fn roles(&self) -> Vec<&str> {
        self.messages()
            .iter()
            .map(|m| m["role"].as_str().unwrap_or("<non-string role>"))
            .collect()
    }

    /// Contents in wire order, paired positionally with [`Recorded::roles`].
    pub fn contents(&self) -> Vec<&str> {
        self.messages()
            .iter()
            .map(|m| m["content"].as_str().unwrap_or("<non-string content>"))
            .collect()
    }
}

/// The two halves of one `Session::pull`.
pub(super) struct Exchange {
    /// `None` when nothing ever reached the server.
    pub sent: Option<Recorded>,
    /// `Err` when the pull failed before a single event could be produced —
    /// a non-2xx status, or a transport failure.
    pub pull: Result<Vec<Result<TokenEvent, ExecError>>, ExecError>,
}

impl Exchange {
    pub fn request(&self) -> &Recorded {
        self.sent
            .as_ref()
            .expect("no request reached the server, so there is nothing to assert about it")
    }

    pub fn events(&self) -> &[Result<TokenEvent, ExecError>] {
        match &self.pull {
            Ok(events) => events,
            Err(e) => panic!("pull failed before any event was produced: {e}"),
        }
    }

    pub fn pull_error(&self) -> &ExecError {
        match &self.pull {
            Ok(events) => panic!("expected the pull to fail, but it produced {events:?} events"),
            Err(e) => e,
        }
    }

    /// The event sequence rendered as short strings, so a test can assert the
    /// whole sequence at once. `TokenEvent` has no `PartialEq`, and the exact
    /// ordering — not just the concatenated text — is what these tests are for.
    pub fn trace(&self) -> Vec<String> {
        self.events()
            .iter()
            .map(|e| match e {
                Ok(TokenEvent::Token(t)) => format!("token:{}", t.text),
                Ok(TokenEvent::Special(s)) => format!("special:{s}"),
                Ok(TokenEvent::MediaBoundary(_)) => "media".to_string(),
                Ok(TokenEvent::ToolCall(c)) => format!("tool:{}", c.name),
                Ok(TokenEvent::Paused) => "paused".to_string(),
                Ok(TokenEvent::Stopped) => "stopped".to_string(),
                Ok(TokenEvent::Eos) => "eos".to_string(),
                Err(e) => format!("error:{e}"),
            })
            .collect()
    }

    /// The decoded text, as a caller concatenating tokens would see it.
    pub fn text(&self) -> String {
        self.events()
            .iter()
            .filter_map(|e| match e {
                Ok(TokenEvent::Token(t)) => Some(t.text.as_str()),
                _ => None,
            })
            .collect()
    }
}

enum ResponseBody {
    Whole(String),
    /// Written one `Vec` per HTTP chunk, so a test can choose where the
    /// boundaries fall — including mid-character.
    Chunks(Vec<Vec<u8>>),
}

pub(super) struct Wire {
    format: &'static str,
    api_key: Option<String>,
    model_id: String,
    settings: Settings,
    messages: Vec<Message>,
    gen_spec: GenSpec,
    status: usize,
    content_type: &'static str,
    body: ResponseBody,
}

impl Wire {
    fn base(format: &'static str) -> Self {
        Self {
            format,
            api_key: Some("test-key".into()),
            model_id: "test-model".into(),
            settings: Settings::default(),
            messages: vec![user("Hi")],
            gen_spec: GenSpec::default(),
            status: 200,
            content_type: "text/event-stream",
            body: ResponseBody::Whole(String::new()),
        }
    }

    pub fn openai() -> Self {
        Self::base("openai")
    }

    pub fn anthropic() -> Self {
        Self::base("anthropic")
    }

    pub fn no_key(mut self) -> Self {
        self.api_key = None;
        self
    }

    pub fn model(mut self, id: &str) -> Self {
        self.model_id = id.into();
        self
    }

    pub fn settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn gen_spec(mut self, spec: GenSpec) -> Self {
        self.gen_spec = spec;
        self
    }

    pub fn sse(mut self, body: &str) -> Self {
        self.body = ResponseBody::Whole(body.into());
        self
    }

    pub fn sse_chunks(mut self, chunks: &[&[u8]]) -> Self {
        self.body = ResponseBody::Chunks(chunks.iter().map(|c| c.to_vec()).collect());
        self
    }

    pub fn status(mut self, code: usize, body: &str) -> Self {
        self.status = code;
        self.content_type = "application/json";
        self.body = ResponseBody::Whole(body.into());
        self
    }

    pub fn run(self) -> Exchange {
        let mut server = mockito::Server::new();
        let seen: Arc<Mutex<Vec<Recorded>>> = Arc::new(Mutex::new(Vec::new()));

        // Matched on `Any` rather than a fixed path so that a request sent to
        // the WRONG path is still recorded and fails on a path assertion,
        // instead of vanishing into mockito's 501.
        let sink = seen.clone();
        let mock = server
            .mock("POST", mockito::Matcher::Any)
            .match_request(move |req| {
                let raw = req.body().cloned().unwrap_or_default();
                sink.lock().unwrap().push(Recorded {
                    path: req.path().to_string(),
                    headers: req
                        .headers()
                        .iter()
                        .map(|(k, v)| {
                            (
                                k.as_str().to_string(),
                                String::from_utf8_lossy(v.as_bytes()).into_owned(),
                            )
                        })
                        .collect(),
                    body: serde_json::from_slice(&raw).unwrap_or(Value::Null),
                });
                true
            })
            .with_status(self.status)
            .with_header("content-type", self.content_type);

        let mock = match self.body {
            ResponseBody::Whole(ref body) => mock.with_body(body.clone()),
            ResponseBody::Chunks(ref chunks) => {
                let chunks = chunks.clone();
                mock.with_chunked_body(move |w| {
                    for chunk in &chunks {
                        w.write_all(chunk)?;
                        w.flush()?;
                    }
                    Ok(())
                })
            }
        };
        let _mock = mock.create();

        let session = session_at(
            &format!("{}/v1", server.url()),
            self.format,
            self.api_key.clone(),
            &self.model_id,
            self.settings.clone(),
            self.messages.clone(),
            fast_client(),
        );

        let pull = session.pull(self.gen_spec.clone()).map(collect);
        let sent = seen.lock().unwrap().pop();
        Exchange { sent, pull }
    }
}

/// Drains a puller to its terminal event.
fn collect(
    mut puller: crate::backend::external_api::session::RemotePuller,
) -> Vec<Result<TokenEvent, ExecError>> {
    let mut out = Vec::new();
    for event in puller.by_ref().take(MAX_EVENTS) {
        let terminal = matches!(
            event,
            Ok(TokenEvent::Eos) | Ok(TokenEvent::Stopped) | Err(_)
        );
        out.push(event);
        if terminal {
            break;
        }
    }
    out
}

/// A client whose timeouts are short enough that a stalled server fails the
/// test in milliseconds rather than at the engine's 300s production timeout.
pub(super) fn fast_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("building a blocking client with only a timeout set cannot fail")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn session_at(
    server_url: &str,
    format: &str,
    api_key: Option<String>,
    model_id: &str,
    settings: Settings,
    messages: Vec<Message>,
    client: reqwest::blocking::Client,
) -> Session {
    Session::new(
        7,
        server_url.to_string(),
        model_id.to_string(),
        api_key,
        format.to_string(),
        client,
        Arc::new(HookBus::new()),
        settings,
        messages,
    )
}

/// `Settings` holds sampling values as `f32` and serde widens them to `f64`
/// on the way into JSON, so `0.3f32` reaches the wire as 0.30000001192092896.
/// Providers accept that; an exact comparison would only be asserting on the
/// widening.
pub(super) fn assert_number(body: &Value, field: &str, expected: f64) {
    let actual = &body[field];
    let got = match actual.as_f64() {
        Some(n) => n,
        None => panic!("`{field}` should be a number, body was {body}"),
    };
    assert!(
        (got - expected).abs() < 1e-6,
        "`{field}` should have been {expected}, got {got}"
    );
}

// ── Message builders ────────────────────────────────────────────────────────

pub(super) fn message(role: &str, text: &str) -> Message {
    Message {
        role: role.into(),
        body: MessageBody::Content {
            content: MessageContent::SingleText(text.into()),
        },
        name: None,
        tool_call_id: None,
    }
}

pub(super) fn user(text: &str) -> Message {
    message("user", text)
}

pub(super) fn assistant(text: &str) -> Message {
    message("assistant", text)
}

pub(super) fn system(text: &str) -> Message {
    message("system", text)
}

pub(super) fn chunked(role: &str, parts: &[&str]) -> Message {
    Message {
        role: role.into(),
        body: MessageBody::Content {
            content: MessageContent::MultipleChunks(
                parts
                    .iter()
                    .map(|p| MessageChunk::Text { text: (*p).into() })
                    .collect(),
            ),
        },
        name: None,
        tool_call_id: None,
    }
}

pub(super) fn structured_assistant(content: &str, reasoning: &str) -> Message {
    Message {
        role: "assistant".into(),
        body: MessageBody::Content {
            content: MessageContent::StructuredAssistant {
                content: content.into(),
                reasoning: Some(reasoning.into()),
            },
        },
        name: None,
        tool_call_id: None,
    }
}

/// A message whose body is a tool-call record rather than text.
pub(super) fn tool_call(name: &str) -> Message {
    Message {
        role: "assistant".into(),
        body: MessageBody::Tool {
            tool_calls: vec![crate::types::message::ToolCall {
                id: "call_1".into(),
                r#type: "function".into(),
                function: crate::types::message::FunctionDefinition {
                    description: None,
                    name: name.into(),
                    arguments: serde_json::json!({}),
                },
            }],
        },
        name: None,
        tool_call_id: None,
    }
}
