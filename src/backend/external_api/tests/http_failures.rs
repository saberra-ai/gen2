//! How transport and status failures reach the caller.
//!
//! Everything here funnels into `ExecError::Other`, so the *message* is the
//! only thing a human or a host can route on. These pin that the status code
//! and the provider's own explanation both survive into it — and that no
//! failure mode leaves the puller hanging, which matters more than the
//! wording: the controller's pump loop calls `next()` inline, so a puller that
//! never returns freezes every other chat in the process.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::engine::{ExecError, Settings};
use crate::generation::{GenSpec, TokenEvent};

use super::harness::{Wire, session_at, user};

/// Every assertion in this file must fail rather than hang; nothing here is
/// allowed to outlive a stalled socket.
const DEADLINE: Duration = Duration::from_secs(20);

fn assert_mentions(err: &ExecError, needle: &str) {
    let rendered = err.to_string();
    assert!(
        rendered.contains(needle),
        "the error must carry {needle:?} so the failure can be told apart from any other, \
         got: {rendered}"
    );
}

#[test]
fn a_400_reports_the_status_and_the_providers_explanation() {
    let x = Wire::openai()
        .status(
            400,
            r#"{"error":{"message":"model is required","type":"invalid_request_error"}}"#,
        )
        .run();

    assert_mentions(x.pull_error(), "400");
    assert_mentions(x.pull_error(), "model is required");
}

#[test]
fn a_401_is_distinguishable_from_every_other_failure() {
    // Auth is the single most common external-API failure and the only one
    // the user can fix, so the code has to survive into the message.
    let x = Wire::openai()
        .status(401, r#"{"error":{"message":"Incorrect API key provided"}}"#)
        .run();

    assert_mentions(x.pull_error(), "401");
    assert_mentions(x.pull_error(), "Incorrect API key provided");
}

#[test]
fn a_429_is_distinguishable_from_every_other_failure() {
    let x = Wire::openai()
        .status(429, r#"{"error":{"message":"Rate limit reached"}}"#)
        .run();

    assert_mentions(x.pull_error(), "429");
    assert_mentions(x.pull_error(), "Rate limit reached");
}

#[test]
fn a_500_is_distinguishable_from_every_other_failure() {
    let x = Wire::openai().status(500, "upstream exploded").run();

    assert_mentions(x.pull_error(), "500");
    assert_mentions(x.pull_error(), "upstream exploded");
}

#[test]
fn an_empty_error_body_still_reports_the_status() {
    let x = Wire::openai().status(503, "").run();

    assert_mentions(x.pull_error(), "503");
}

#[test]
fn an_anthropic_status_failure_maps_the_same_way() {
    // Both formats share one status check, and both must keep reporting the
    // code rather than collapsing to a generic "request failed".
    let x = Wire::anthropic()
        .status(
            429,
            r#"{"type":"error","error":{"type":"rate_limit_error"}}"#,
        )
        .run();

    assert_mentions(x.pull_error(), "429");
    assert_mentions(x.pull_error(), "rate_limit_error");
}

#[test]
fn every_http_failure_reaches_the_caller_before_a_single_token_does() {
    // A non-2xx body is never fed to the SSE parser, so a caller can rely on
    // "an error means nothing was streamed".
    for status in [400, 401, 403, 404, 429, 500, 502, 503] {
        let x = Wire::openai().status(status, "nope").run();
        assert!(
            x.pull.is_err(),
            "status {status} produced a stream instead of an error"
        );
    }
}

#[test]
fn a_refused_connection_names_the_server_it_could_not_reach() {
    // Binding then dropping a listener leaves a port nothing is listening on,
    // which is the closest reproducible thing to "the provider is down".
    let listener = TcpListener::bind("127.0.0.1:0").expect("loopback bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let session = session_at(
        &format!("http://127.0.0.1:{port}/v1"),
        "openai",
        Some("k".into()),
        "m",
        Settings::default(),
        vec![user("hi")],
        super::harness::fast_client(),
    );
    let err = session
        .pull(GenSpec::default())
        .err()
        .expect("a closed port cannot produce a stream");

    assert_mentions(&err, "failed to connect to external server");
}

#[test]
fn a_server_that_never_answers_fails_instead_of_hanging() {
    let stall = StalledServer::spawn(Respond::Nothing);
    let session = session_at(
        &stall.base_url(),
        "openai",
        Some("k".into()),
        "m",
        Settings::default(),
        vec![user("hi")],
        impatient_client(),
    );

    let err = with_deadline(move || {
        session
            .pull(GenSpec::default())
            .err()
            .expect("a server that sends no response cannot produce a stream")
    });

    assert_mentions(&err, "failed to connect to external server");
}

#[test]
fn a_server_that_stalls_mid_stream_ends_the_stream_with_an_error() {
    // The puller treats `TimedOut`/`WouldBlock` reads as recoverable so it can
    // re-poll the stop flag, which would be a livelock if reqwest reported a
    // blown request deadline that way. It does not: the deadline surfaces as a
    // decode failure, which is fatal. That distinction is what keeps the
    // controller's pump loop — which calls `next()` inline, on the thread
    // every other chat shares — from wedging on one stalled provider.
    let stall = StalledServer::spawn(Respond::HeadersAndOneToken);
    let session = session_at(
        &stall.base_url(),
        "openai",
        Some("k".into()),
        "m",
        Settings::default(),
        vec![user("hi")],
        impatient_client(),
    );

    let events = with_deadline(move || {
        let mut puller = session
            .pull(GenSpec::default())
            .expect("headers arrived, so the pull starts");
        let mut out = Vec::new();
        for event in puller.by_ref().take(8) {
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
    });

    assert!(
        matches!(events.first(), Some(Ok(TokenEvent::Token(t))) if t.text == "hi"),
        "the token that did arrive must still be delivered, got {events:?}"
    );
    match events.last() {
        Some(Err(e)) => assert_mentions(e, "error reading SSE stream"),
        other => panic!("a stalled stream must end in an error, got {other:?}"),
    }
}

#[test]
fn an_anthropic_stream_that_stalls_mid_stream_also_ends_in_an_error() {
    let stall = StalledServer::spawn(Respond::HeadersAndOneAnthropicToken);
    let session = session_at(
        &stall.base_url(),
        "anthropic",
        Some("k".into()),
        "m",
        Settings::default(),
        vec![user("hi")],
        impatient_client(),
    );

    let events = with_deadline(move || {
        let mut puller = session
            .pull(GenSpec::default())
            .expect("headers arrived, so the pull starts");
        let mut out = Vec::new();
        for event in puller.by_ref().take(8) {
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
    });

    assert!(
        matches!(events.last(), Some(Err(_))),
        "a stalled Anthropic stream must end in an error, got {events:?}"
    );
}

/// A client whose whole-request deadline is short enough to make a stall a
/// test-length event rather than a five-minute one.
fn impatient_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(250))
        .build()
        .expect("building a blocking client with only a timeout set cannot fail")
}

/// Runs `f` on its own thread and fails the test if it does not return in
/// time, so a regression to the spinning-puller bug reports as a failure
/// rather than wedging the suite.
fn with_deadline<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(DEADLINE)
        .expect("the puller never returned — a stalled stream must fail, not hang")
}

/// What a stalled server says before it goes quiet: nothing at all, or a
/// 200 with one SSE event and no terminating chunk.
enum Respond {
    Nothing,
    HeadersAndOneToken,
    HeadersAndOneAnthropicToken,
}

impl Respond {
    fn reply(&self) -> Vec<u8> {
        let event = match self {
            Respond::Nothing => return Vec::new(),
            Respond::HeadersAndOneToken => {
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
            }
            Respond::HeadersAndOneAnthropicToken => {
                "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n"
            }
        };
        // Chunked, and deliberately missing its terminating `0\r\n\r\n`: the
        // client must see a stalled stream, not a clean end of body.
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
             Transfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n",
            event.len(),
            event
        )
        .into_bytes()
    }
}

/// A socket that accepts, says whatever `Respond` dictates, then goes silent
/// forever without closing the connection. Its thread outlives the test on
/// purpose — there is no way to unblock `accept()` portably, and the test
/// binary reclaims it on exit.
struct StalledServer {
    port: u16,
}

impl StalledServer {
    fn spawn(behaviour: Respond) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback bind");
        let port = listener.local_addr().unwrap().port();

        thread::spawn(move || {
            let reply = behaviour.reply();
            let mut held: Vec<TcpStream> = Vec::new();
            while let Ok((mut sock, _)) = listener.accept() {
                let _ = sock.set_read_timeout(Some(Duration::from_millis(200)));
                let mut scratch = [0u8; 4096];
                let _ = sock.read(&mut scratch);
                let _ = sock.write_all(&reply);
                let _ = sock.flush();
                held.push(sock);
            }
        });

        Self { port }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }
}
