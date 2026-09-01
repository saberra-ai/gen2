//! What the backend puts on the wire in OpenAI chat-completions format.
//!
//! A provider rejects a request for reasons it never explains well, so these
//! pin the request shape field by field: the wrong path 404s, a duplicated
//! system prompt silently doubles the instructions the model follows, and a
//! reordered history changes the answer without changing anything visible.

use crate::engine::{PromptSettings, SamplingSettings, Settings, StoppingSettings};
use crate::generation::GenSpec;

use super::harness::{
    Wire, assert_number, assistant, chunked, structured_assistant, system, tool_call, user,
};

const DONE: &str = "data: [DONE]\n\n";

#[test]
fn a_completion_is_posted_to_chat_completions_under_the_configured_base_url() {
    let x = Wire::openai().sse(DONE).run();

    assert_eq!(
        x.request().path,
        "/v1/chat/completions",
        "the base URL is joined with /chat/completions verbatim; any other path is a 404 \
         at every OpenAI-compatible provider"
    );
}

#[test]
fn the_api_key_travels_as_a_bearer_token() {
    let x = Wire::openai().sse(DONE).run();

    assert_eq!(
        x.request().header("authorization"),
        Some("Bearer test-key"),
        "OpenAI-format auth is `Authorization: Bearer <key>`; anything else is a 401"
    );
}

#[test]
fn a_session_with_no_api_key_sends_no_authorization_header() {
    // Local servers (ollama, llama.cpp, LM Studio) take no key, and some
    // reject a request that carries an empty bearer token.
    let x = Wire::openai().no_key().sse(DONE).run();

    assert_eq!(
        x.request().header("authorization"),
        None,
        "an unset key must omit the header, not send an empty bearer token"
    );
}

#[test]
fn the_request_asks_for_a_stream_of_the_configured_model() {
    let x = Wire::openai().model("gpt-4o-mini").sse(DONE).run();
    let body = &x.request().body;

    assert_eq!(body["model"], "gpt-4o-mini", "model id must round-trip");
    assert_eq!(
        body["stream"], true,
        "the puller only knows how to read SSE; a non-streaming response would parse as nothing"
    );
}

#[test]
fn an_unset_model_id_falls_back_to_the_literal_default() {
    // Documented fallback in `Session::pull`. Model-specific servers reject
    // it, which is the intended loud failure — better than an empty string.
    let x = Wire::openai().model("").sse(DONE).run();

    assert_eq!(x.request().body["model"], "default");
}

#[test]
fn the_system_prompt_is_prepended_exactly_once() {
    let settings = Settings {
        prompt: PromptSettings {
            system_prompt: Some("You are terse.".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Wire::openai()
        .settings(settings)
        .messages(vec![user("one"), assistant("two"), user("three")])
        .sse(DONE)
        .run();

    assert_eq!(
        x.request().roles(),
        vec!["system", "user", "assistant", "user"],
        "the system prompt goes at the head of the array, once"
    );
    assert_eq!(
        x.request().contents()[0],
        "You are terse.",
        "the system slot must carry the configured prompt"
    );
}

#[test]
fn a_system_message_in_the_history_suppresses_the_settings_system_prompt() {
    // Both being sent is the failure that matters: two system messages give
    // the model two sets of instructions and no way to rank them.
    let settings = Settings {
        prompt: PromptSettings {
            system_prompt: Some("from settings".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Wire::openai()
        .settings(settings)
        .messages(vec![system("from history"), user("hi")])
        .sse(DONE)
        .run();

    assert_eq!(
        x.request().roles(),
        vec!["system", "user"],
        "exactly one system message must survive"
    );
    assert_eq!(
        x.request().contents()[0],
        "from history",
        "the caller's own system message wins over the engine default"
    );
}

#[test]
fn an_empty_system_prompt_adds_no_system_message() {
    let settings = Settings {
        prompt: PromptSettings {
            system_prompt: Some(String::new()),
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Wire::openai().settings(settings).sse(DONE).run();

    assert_eq!(
        x.request().roles(),
        vec!["user"],
        "an empty configured prompt must not become an empty system turn"
    );
}

#[test]
fn message_order_survives_serialization() {
    let x = Wire::openai()
        .messages(vec![
            user("first"),
            assistant("second"),
            user("third"),
            assistant("fourth"),
            user("fifth"),
        ])
        .sse(DONE)
        .run();

    assert_eq!(
        x.request().contents(),
        vec!["first", "second", "third", "fourth", "fifth"],
        "reordering history silently changes what the model is answering"
    );
    assert_eq!(
        x.request().roles(),
        vec!["user", "assistant", "user", "assistant", "user"],
        "roles must stay pinned to their own turns"
    );
}

#[test]
fn assistant_history_is_sent_as_plain_content() {
    let x = Wire::openai()
        .messages(vec![user("q"), assistant("a"), user("q2")])
        .sse(DONE)
        .run();

    let msgs = x.request().messages();
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(
        msgs[1]["content"], "a",
        "a prior assistant turn is a string, not a content-part array"
    );
}

#[test]
fn a_structured_assistant_turn_sends_only_its_visible_content() {
    // Reasoning is deliberately dropped on replay — same rule the local chat
    // templates apply to prior turns.
    let x = Wire::openai()
        .messages(vec![
            user("q"),
            structured_assistant("the answer", "the private reasoning"),
        ])
        .sse(DONE)
        .run();

    assert_eq!(x.request().contents(), vec!["q", "the answer"]);
    assert!(
        !x.request().body.to_string().contains("private reasoning"),
        "reasoning must never leave the process on a replayed turn"
    );
}

#[test]
fn multi_chunk_text_is_flattened_into_one_string() {
    let x = Wire::openai()
        .messages(vec![chunked("user", &["alpha", " ", "beta"])])
        .sse(DONE)
        .run();

    assert_eq!(
        x.request().contents(),
        vec!["alpha beta"],
        "text chunks are concatenated with no separator inserted"
    );
}

#[test]
fn a_tool_call_turn_is_dropped_rather_than_sent_as_an_empty_message() {
    // The OpenAI body builder has no encoding for a tool-call turn, so it
    // skips it. Sending it as `content: ""` would be worse: providers reject
    // an assistant turn with neither content nor tool_calls.
    let x = Wire::openai()
        .messages(vec![user("q"), tool_call("search"), user("q2")])
        .sse(DONE)
        .run();

    assert_eq!(
        x.request().contents(),
        vec!["q", "q2"],
        "the tool-call turn is omitted entirely"
    );
}

#[test]
fn sampling_fields_are_serialized_when_set() {
    let settings = Settings {
        sampling: SamplingSettings {
            temperature: Some(0.25),
            top_p: Some(0.8),
            seed: Some(1234),
            ..Default::default()
        },
        ..Default::default()
    };
    let spec = GenSpec {
        max_tokens: Some(64),
        ..Default::default()
    };
    let x = Wire::openai()
        .settings(settings)
        .gen_spec(spec)
        .sse(DONE)
        .run();
    let body = &x.request().body;

    assert_number(body, "temperature", 0.25);
    assert_number(body, "top_p", 0.8);
    assert_eq!(body["seed"], 1234);
    assert_eq!(body["max_tokens"], 64);
}

#[test]
fn unset_sampling_fields_are_omitted_rather_than_sent_as_null() {
    // A literal `null` is not the same as absent to every provider; several
    // reject `"temperature": null` outright.
    let x = Wire::openai().sse(DONE).run();
    let body = &x.request().body;

    for field in ["temperature", "top_p", "seed", "max_tokens", "stop"] {
        assert!(
            body.get(field).is_none(),
            "{field} was left unset but still appeared in the body: {body}"
        );
    }
}

#[test]
fn a_per_pull_gen_spec_overrides_the_engine_temperature_and_seed() {
    let settings = Settings {
        sampling: SamplingSettings {
            temperature: Some(0.9),
            seed: Some(1),
            ..Default::default()
        },
        ..Default::default()
    };
    let spec = GenSpec {
        temperature: Some(0.1),
        seed: Some(99),
        ..Default::default()
    };
    let x = Wire::openai()
        .settings(settings)
        .gen_spec(spec)
        .sse(DONE)
        .run();

    // The engine defaults are 0.9 / 1, so anything but the GenSpec values
    // here means a per-generation override was silently discarded.
    assert_number(&x.request().body, "temperature", 0.1);
    assert_eq!(x.request().body["seed"], 99, "same for the seed");
}

#[test]
fn max_tokens_falls_back_to_the_engine_stopping_setting() {
    let settings = Settings {
        stopping: StoppingSettings {
            max_tokens: Some(17),
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Wire::openai().settings(settings).sse(DONE).run();

    assert_eq!(
        x.request().body["max_tokens"],
        17,
        "a GenSpec with no max_tokens inherits the engine's stopping settings"
    );
}

#[test]
fn stop_words_are_serialized_as_the_stop_array() {
    let settings = Settings {
        stopping: StoppingSettings {
            stopwords: vec!["\n\n".into(), "END".into()],
            max_tokens: None,
        },
        ..Default::default()
    };
    let x = Wire::openai().settings(settings).sse(DONE).run();

    assert_eq!(
        x.request().body["stop"],
        serde_json::json!(["\n\n", "END"]),
        "stop words go in `stop`, in order, as-is"
    );
}

#[test]
fn the_request_announces_json_in_and_event_stream_out() {
    let x = Wire::openai().sse(DONE).run();

    assert_eq!(x.request().header("content-type"), Some("application/json"));
    assert_eq!(
        x.request().header("accept"),
        Some("text/event-stream"),
        "some gateways buffer the whole response unless the client asks for SSE"
    );
}

#[test]
fn a_base_url_with_a_trailing_slash_does_not_produce_a_double_slash() {
    // `Engine::load_model` trims the trailing slash before it reaches the
    // session, so the session itself is entitled to assume it is gone. This
    // pins that the join is a plain concatenation, which is what makes the
    // trim in `load_model` load-bearing.
    let x = Wire::openai().sse(DONE).run();

    assert!(
        !x.request().path.contains("//"),
        "path had a doubled slash: {}",
        x.request().path
    );
}
