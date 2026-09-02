//! gen2's types, in the shapes LiteRT-LM's C API takes.
//!
//! Everything here is a pure function over gen2 types. That is deliberate:
//! this is where a backend integration is most likely to be quietly wrong —
//! a dropped tool call, a role the template never sees, a sampler field
//! silently ignored — and none of those need a model in the room to catch.
//!
//! LiteRT-LM takes conversations as JSON. The shape is the OpenAI chat
//! message shape, which is what its `tools_json` array and `messages_json`
//! array are built around.

use serde_json::{Value, json};

use crate::backend::common::grammar::GrammarSpec;
use crate::engine::{ExecError, Settings};
use crate::generation::GenSpec;
use crate::types::message::{Message, MessageBody, ToolSpec};

use super::ffi::constraint_type;

/// The role LiteRT-LM should see for a gen2 message.
///
/// Anything unrecognised becomes a user turn rather than being dropped. A
/// message the model never sees is worse than one it sees under the wrong
/// label: the transcript the caller holds would stop describing the
/// conversation the model actually had.
fn role_of(message: &Message) -> &str {
    match message.role.as_str() {
        "system" => "system",
        "assistant" => "assistant",
        "tool" => "tool",
        _ => "user",
    }
}

/// The text a message contributes.
///
/// [`Message::text`] is the crate's own answer to "what does a transcript
/// render", and it already leaves out stored reasoning — gen2 keeps a
/// reasoning channel separate from visible content, and replaying a model's
/// own working back to it is not what the transcript means. A tool-call turn
/// has no text at all; the calls travel separately.
fn text_of(message: &Message) -> String {
    message.text()
}

/// One message, as LiteRT-LM's JSON.
///
/// An assistant turn that asked for tools carries `tool_calls` rather than
/// text. Flattening it to prose is the failure mode that made the mistral.rs
/// backend replay a tool call as an empty assistant message: the transcript
/// said the model called a tool and the prompt did not.
pub(super) fn message_json(message: &Message, call_id: &str) -> Value {
    if let MessageBody::Tool { tool_calls } = &message.body {
        let calls: Vec<Value> = tool_calls
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        // Arguments travel as a JSON *string*, which is the
                        // shape every OpenAI-compatible template expects.
                        "arguments": c.function.arguments.to_string(),
                    },
                })
            })
            .collect();
        return json!({ "role": "assistant", "tool_calls": calls });
    }

    // Content is an array of typed chunks, which is the shape LiteRT-LM's own
    // streamed messages use. A bare string is also accepted, but matching what
    // the runtime emits is the form with evidence behind it.
    let mut out = json!({
        "role": role_of(message),
        "content": [{ "type": "text", "text": text_of(message) }],
    });
    if message.role == "tool" {
        // The call this result answers. gen2's `tool_result` does not carry
        // one, so it comes from the nearest preceding call — an empty id is
        // better than a wrong one.
        out["tool_call_id"] = json!(call_id);
        if let Some(name) = &message.name {
            out["name"] = json!(name);
        }
    }
    out
}

/// The id of the call a tool result at `index` is answering.
///
/// The message's own `tool_call_id` when it has one, which is the only answer
/// that stays right when the model asked for several tools at once. The
/// backward search is the fallback for results recorded before ids were
/// carried: it takes the nearest preceding call, which is correct whenever
/// there was only one.
fn call_id_for(messages: &[Message], index: usize) -> String {
    if let Some(id) = messages[index].tool_call_id.as_ref() {
        return id.clone();
    }
    messages[..index]
        .iter()
        .rev()
        .find_map(|m| match &m.body {
            MessageBody::Tool { tool_calls } => tool_calls.last().map(|c| c.id.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// A whole transcript, as the JSON array `set_messages` takes.
///
/// Order is preserved exactly. A conversation is the one thing this layer must
/// not reinterpret.
pub(super) fn messages_json(messages: &[Message]) -> String {
    let array: Vec<Value> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| message_json(m, &call_id_for(messages, i)))
        .collect();
    Value::Array(array).to_string()
}

/// A leading system turn, if the transcript opens with one.
///
/// LiteRT-LM has a dedicated setter for it, and a chat template generally
/// treats the system turn differently from an ordinary message. Only a
/// *leading* one is taken: a system message appearing mid-conversation is
/// something the caller put there on purpose, and moving it to the front would
/// change what the model was told and when.
pub(super) fn leading_system(messages: &[Message]) -> (Option<String>, &[Message]) {
    match messages.first() {
        Some(first) if first.role == "system" => (Some(text_of(first)), &messages[1..]),
        _ => (None, messages),
    }
}

/// Tool declarations, as the JSON array `set_tools` takes.
///
/// Schemas travel verbatim. The model is told what it may call and gen2
/// executes what it asks for — LiteRT-LM is never handed a callback, so
/// approvals, deferred tools and sub-agents keep working unchanged.
pub(super) fn tools_json(specs: &[ToolSpec]) -> String {
    let array: Vec<Value> = specs
        .iter()
        .map(|spec| {
            json!({
                "type": "function",
                "function": {
                    "name": spec.function.name,
                    "description": spec.function.description.clone().unwrap_or_default(),
                    "parameters": spec.function.arguments,
                },
            })
        })
        .collect();
    Value::Array(array).to_string()
}

/// One piece of a streamed reply, once the JSON around it is gone.
#[derive(Debug, PartialEq)]
pub(super) enum Part {
    Text(String),
    Call(crate::generation::ToolCall),
}

/// What a streamed chunk actually says.
///
/// LiteRT-LM does not stream plain text. Every chunk is a whole JSON message —
/// `{"role":"assistant","content":[{"type":"text","text":"Okay"}]}` for a
/// single token — so a backend that forwards `stream_chunk_get_text` verbatim
/// hands the caller a wall of JSON as the model's reply. That is what this
/// undoes.
///
/// Robust about three things, each because getting them wrong loses output:
/// a chunk may carry several concatenated JSON values; `content` may be a bare
/// string rather than an array; and anything that is not JSON at all is passed
/// through unchanged rather than dropped, because text the runtime meant for
/// the caller is worth more than a tidy parser.
pub(super) fn decode_chunk(raw: &str) -> Vec<Part> {
    let mut parts = Vec::new();
    let stream = serde_json::Deserializer::from_str(raw).into_iter::<Value>();
    let mut consumed = 0usize;

    for value in stream {
        match value {
            Ok(value) => {
                consumed = raw.len();
                absorb_message(&value, &mut parts);
            }
            Err(_) => break,
        }
    }

    if consumed == 0 && !raw.is_empty() {
        // Not JSON. Some other runtime build, or a plain-text stream: either
        // way it is the model's output and belongs to the caller.
        parts.push(Part::Text(raw.to_string()));
    }
    parts
}

/// Pull the text and any tool calls out of one decoded message.
fn absorb_message(value: &Value, parts: &mut Vec<Part>) {
    match &value["content"] {
        Value::String(text) if !text.is_empty() => parts.push(Part::Text(text.clone())),
        Value::Array(chunks) => {
            for chunk in chunks {
                if let Some(text) = chunk["text"].as_str().filter(|t| !t.is_empty()) {
                    parts.push(Part::Text(text.to_string()));
                }
            }
        }
        _ => {}
    }

    // A native tool call beats parsing one back out of prose, so it is taken
    // whenever the runtime provides one.
    if let Some(calls) = value["tool_calls"].as_array() {
        for call in calls {
            let Some(name) = call["function"]["name"].as_str() else {
                continue;
            };
            let arguments = match &call["function"]["arguments"] {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            parts.push(Part::Call(crate::generation::ToolCall {
                id: call["id"].as_str().map(str::to_string),
                name: name.to_string(),
                arguments,
            }));
        }
    }
}

/// A grammar, as a LiteRT-LM constraint.
///
/// Two of gen2's four map exactly. Lark does not: LiteRT-LM constrains with
/// regex and JSON schema only, and there is no honest translation from a Lark
/// grammar to either. Refusing is the right answer — silently dropping the
/// constraint would hand the caller unconstrained output while their code
/// believed it was parsing a guaranteed shape.
pub(super) fn constraint_of(grammar: &GrammarSpec) -> Result<(i32, String), ExecError> {
    Ok(match grammar {
        GrammarSpec::JsonObject => (
            constraint_type::JSON_SCHEMA,
            json!({"type": "object"}).to_string(),
        ),
        GrammarSpec::JsonSchema(schema) => (constraint_type::JSON_SCHEMA, schema.to_string()),
        GrammarSpec::Regex(rx) => (constraint_type::REGEX, rx.clone()),
        GrammarSpec::Lark(_) => {
            return Err(ExecError::FeatureUnsupported(
                "Lark grammar: LiteRT-LM constrains with regex and JSON schema only. \
                 Use GrammarSpec::JsonSchema or GrammarSpec::Regex, or a backend with \
                 a general grammar engine.",
            ));
        }
    })
}

/// Which sampler to ask for, and with what.
///
/// LiteRT-LM picks one sampler rather than composing a chain, so the choice
/// has to be made here: temperature zero is greedy, an explicit top-k is
/// top-k, and anything else is top-p. Deriving it from what the caller
/// actually set means a request that says nothing gets the runtime's own
/// default rather than a chain gen2 invented.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Sampler {
    pub kind: i32,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub seed: Option<i32>,
}

/// A `top_k` that truncates nothing.
///
/// LiteRT-LM samples top-p from the top-k candidates and rejects `k <= 0`, so
/// "the caller asked for no top-k" has to be said as a k larger than any
/// vocabulary. Not an invented sampling default: it is how this runtime spells
/// the absence of one. `i32::MAX` was accepted by the shipped runtime, and
/// 2^20 is still about eight times the largest vocabulary in use — big enough
/// to mean nothing, small enough not to bet on the runtime never allocating
/// against it.
const UNTRUNCATED_TOP_K: i32 = 1 << 20;

/// A `top_k` that keeps only the single most likely token — argmax.
///
/// This is what `.greedy()` becomes. It is exact rather than an approximation:
/// sampling from a one-token candidate set is argmax however the sampler is
/// implemented.
const GREEDY_TOP_K: i32 = 1;

/// A `top_p` that truncates nothing, for the same reason as
/// [`UNTRUNCATED_TOP_K`] — a caller who set only `top_k` must not silently
/// acquire the runtime's own nucleus threshold on top of it.
const UNTRUNCATED_TOP_P: f32 = 1.0;

/// What this backend cannot honour from a merged settings block.
///
/// min-p, DRY and XTC have no LiteRT-LM equivalent. Reporting them rather than
/// ignoring them is the difference between a caller knowing their sampling did
/// not apply and quietly getting different output than they asked for.
///
/// Merged first, deliberately: a caller who sets `min_p` on the request rather
/// than on the engine is asking for exactly the same thing, and checking only
/// the engine's settings let the per-request form through unreported.
pub(super) fn unsupported_sampling(settings: &Settings, spec: &GenSpec) -> Option<&'static str> {
    let merged = settings.with_gen_spec_overrides(spec);
    if merged.sampling.min_p.is_some_and(|v| v > 0.0) {
        return Some(
            "min_p: LiteRT-LM samples with top-k and top-p only; use top_p, or \
             a backend with min-p",
        );
    }
    if spec.dry_multiplier.is_some_and(|v| v > 0.0) {
        return Some("dry: LiteRT-LM has no DRY repetition penalty; use penalty_repeat instead");
    }
    if spec.xtc_probability.is_some_and(|v| v > 0.0) {
        return Some("xtc: LiteRT-LM has no XTC sampler");
    }
    None
}

/// The sampler a turn asks for, or `None` to leave the runtime's own alone.
///
/// # Why everything is top-p
///
/// The C ABI names three sampler types, and the shipped runtime implements
/// one. Asked directly, it answers:
///
/// ```text
/// greedy (type 3) → UNIMPLEMENTED: Sampler type: 3 not implemented yet.
/// top-k  (type 1) → UNIMPLEMENTED: Sampler type: 1 not implemented yet.
/// top-p  (type 2) → INVALID_ARGUMENT: k must be positive.   (k unset)
/// top-p  (type 2) with k > 0 → works
/// ```
///
/// So top-p with an explicit k is the only shape that runs, and the other two
/// modes are expressed through it exactly rather than approximated: `k = 1` is
/// argmax, and a k past the vocabulary is no truncation at all. Selecting
/// `GREEDY` or `TOP_K` because the names match would have made `.greedy()` and
/// `.top_k()` fail outright on every model.
pub(super) fn sampler_of(settings: &Settings, spec: &GenSpec) -> Option<Sampler> {
    let merged = settings.with_gen_spec_overrides(spec);
    let s = &merged.sampling;

    // Nobody expressed a preference, so the runtime keeps the sampler the
    // bundle shipped with. Checking every field rather than temperature alone:
    // a caller who sets only `.seed()` or only `.top_k()` has expressed one,
    // and reading temperature first silently dropped both.
    if s.temperature.is_none() && s.top_k.is_none() && s.top_p.is_none() && s.seed.is_none() {
        return None;
    }

    let seed = s.seed.map(|v| v as i32);
    if s.temperature.is_some_and(|t| t <= 0.0) {
        // Temperature is left unset: with one candidate it cannot matter, and
        // handing a runtime a zero to divide by is a bad trade for nothing.
        return Some(Sampler {
            kind: super::ffi::sampler_type::TOP_P,
            temperature: None,
            top_k: Some(GREEDY_TOP_K),
            top_p: Some(UNTRUNCATED_TOP_P),
            seed,
        });
    }

    Some(Sampler {
        kind: super::ffi::sampler_type::TOP_P,
        temperature: s.temperature,
        // Each unset field becomes the value that does nothing, so a caller who
        // set one knob does not silently acquire the runtime's default for the
        // other.
        top_k: Some(s.top_k.filter(|k| *k > 0).unwrap_or(UNTRUNCATED_TOP_K)),
        top_p: Some(s.top_p.unwrap_or(UNTRUNCATED_TOP_P)),
        // Narrowed to the C API's `int`. Deterministic either way; two seeds
        // differing only above bit 31 land on the same stream, which is a far
        // smaller surprise than a seed that does nothing.
        seed,
    })
}

/// The repetition-penalty window, if the caller set one.
///
/// `penalty_last_n` is how many recent tokens the penalties look back over,
/// which LiteRT-LM calls the window size.
pub(super) fn penalty_window(settings: &Settings, spec: &GenSpec) -> Option<i32> {
    settings
        .with_gen_spec_overrides(spec)
        .sampling
        .penalty_last_n
        .filter(|n| *n > 0)
}

/// The three penalties LiteRT-LM's repetition config takes, if any were asked
/// for. `None` means leave the runtime's own defaults alone.
pub(super) fn penalties_of(settings: &Settings, spec: &GenSpec) -> Option<(f32, f32, f32)> {
    let merged = settings.with_gen_spec_overrides(spec);
    let s = &merged.sampling;
    let any = s.penalty_repeat.is_some() || s.penalty_freq.is_some() || s.penalty_present.is_some();
    any.then(|| {
        (
            s.penalty_repeat.unwrap_or(1.0),
            s.penalty_freq.unwrap_or(0.0),
            s.penalty_present.unwrap_or(0.0),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::ToolCall as GenToolCall;
    use crate::types::message::{FunctionDefinition, MessageContent, ToolCall};

    /// The text a rendered message shows, whichever content shape it used.
    fn visible(message: &Value) -> String {
        match &message["content"] {
            Value::String(s) => s.clone(),
            Value::Array(chunks) => chunks
                .iter()
                .filter_map(|c| c["text"].as_str())
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        }
    }

    fn user(text: &str) -> Message {
        Message {
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.into()),
            },
            name: None,
            tool_call_id: None,
        }
    }

    fn call_turn(id: &str, name: &str, args: Value) -> Message {
        Message {
            role: "assistant".into(),
            body: MessageBody::Tool {
                tool_calls: vec![ToolCall {
                    id: id.into(),
                    r#type: "function".into(),
                    function: FunctionDefinition {
                        description: None,
                        name: name.into(),
                        arguments: args,
                    },
                }],
            },
            name: None,
            tool_call_id: None,
        }
    }

    fn tool_result(text: &str) -> Message {
        Message {
            role: "tool".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.into()),
            },
            name: Some("get_weather".into()),
            tool_call_id: None,
        }
    }

    #[test]
    fn a_tool_call_replays_as_a_call_and_not_as_an_empty_message() {
        // The half a puller test cannot see. A tool-call turn holds no text,
        // so a converter that only reads text sends the model a tool result
        // with nothing that asked for it.
        let turn = call_turn("call_1", "get_weather", json!({"city": "Paris"}));
        let out = message_json(&turn, "");

        assert_eq!(out["role"], "assistant");
        let calls = out["tool_calls"]
            .as_array()
            .expect("a tool-call turn must carry tool_calls, not prose");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(
            calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}",
            "arguments travel as a JSON string, which is what the template reads"
        );
    }

    #[test]
    fn a_tool_result_is_tied_to_the_call_it_answers() {
        let transcript = vec![
            user("weather in Paris?"),
            call_turn("call_7", "get_weather", json!({"city": "Paris"})),
            tool_result("18C and sunny"),
        ];
        let rendered: Value = serde_json::from_str(&messages_json(&transcript)).unwrap();
        let result = &rendered[2];

        assert_eq!(result["role"], "tool");
        assert_eq!(
            result["tool_call_id"], "call_7",
            "a result with no call id leaves the model guessing which call it answers"
        );
        assert_eq!(visible(result), "18C and sunny");
    }

    /// Two calls in one turn, two results, each tied to its own call.
    ///
    /// The reason `tool_call_id` exists on `Message`. Reconstructing the id by
    /// searching backward can only ever return one call, so both results were
    /// attributed to the same one — the model was told the calendar lookup
    /// answered the weather call, and nothing in the transcript disagreed.
    #[test]
    fn parallel_tool_results_each_answer_their_own_call() {
        let calls = Message {
            role: "assistant".into(),
            body: MessageBody::Tool {
                tool_calls: vec![
                    ToolCall {
                        id: "call_weather".into(),
                        r#type: "function".into(),
                        function: FunctionDefinition {
                            description: None,
                            name: "get_weather".into(),
                            arguments: json!({}),
                        },
                    },
                    ToolCall {
                        id: "call_calendar".into(),
                        r#type: "function".into(),
                        function: FunctionDefinition {
                            description: None,
                            name: "get_calendar".into(),
                            arguments: json!({}),
                        },
                    },
                ],
            },
            name: None,
            tool_call_id: None,
        };
        let transcript = vec![
            user("weather and calendar?"),
            calls,
            Message::tool_result_for("call_weather", "18C and sunny"),
            Message::tool_result_for("call_calendar", "one meeting at 3"),
        ];

        let rendered: Value = serde_json::from_str(&messages_json(&transcript)).unwrap();
        assert_eq!(
            rendered[2]["tool_call_id"], "call_weather",
            "the weather result was attributed to the wrong call"
        );
        assert_eq!(
            rendered[3]["tool_call_id"], "call_calendar",
            "the calendar result was attributed to the wrong call — this is \
             exactly what the backward search got wrong"
        );
    }

    /// A result with no id still lands on the nearest call.
    ///
    /// Models that emit calls without ids are real, and one call with one
    /// result is the overwhelmingly common shape. The fallback has to keep
    /// working.
    #[test]
    fn a_result_with_no_id_still_finds_the_call_before_it() {
        let transcript = vec![
            user("weather in Paris?"),
            call_turn("call_7", "get_weather", json!({"city": "Paris"})),
            Message::tool_result("18C and sunny"),
        ];
        let rendered: Value = serde_json::from_str(&messages_json(&transcript)).unwrap();
        assert_eq!(rendered[2]["tool_call_id"], "call_7");
    }

    #[test]
    fn transcript_order_is_never_reinterpreted() {
        let transcript = vec![user("one"), user("two"), user("three")];
        let rendered: Value = serde_json::from_str(&messages_json(&transcript)).unwrap();
        let texts: Vec<String> = rendered.as_array().unwrap().iter().map(visible).collect();
        assert_eq!(texts, ["one", "two", "three"]);
    }

    #[test]
    fn only_a_leading_system_turn_is_lifted_out() {
        let system = Message {
            role: "system".into(),
            body: MessageBody::Content {
                content: MessageContent::SingleText("be terse".into()),
            },
            name: None,
            tool_call_id: None,
        };
        let opens_with_one = [system.clone(), user("hi")];
        let (lifted, rest) = leading_system(&opens_with_one);
        assert_eq!(lifted.as_deref(), Some("be terse"));
        assert_eq!(rest.len(), 1);

        // One in the middle stays where the caller put it: moving it would
        // change what the model was told and when.
        let has_one_in_the_middle = [user("hi"), system];
        let (lifted, rest) = leading_system(&has_one_in_the_middle);
        assert_eq!(lifted, None);
        assert_eq!(rest.len(), 2);
    }

    #[test]
    fn a_lark_grammar_is_refused_rather_than_silently_dropped() {
        // The caller's code is parsing output it believes is guaranteed. If
        // the constraint quietly vanishes, they get unconstrained text and a
        // parse failure a long way from the cause.
        let err = constraint_of(&GrammarSpec::Lark("start: \"a\"".into()))
            .expect_err("Lark has no LiteRT-LM equivalent");
        assert!(
            matches!(err, ExecError::FeatureUnsupported(m) if m.contains("Lark")),
            "the refusal should name what was unsupported, got {err:?}"
        );
    }

    #[test]
    fn json_and_regex_constraints_map_to_their_native_kinds() {
        let (kind, body) = constraint_of(&GrammarSpec::JsonObject).unwrap();
        assert_eq!(kind, constraint_type::JSON_SCHEMA);
        assert!(body.contains("object"));

        let (kind, body) = constraint_of(&GrammarSpec::Regex("[0-9]+".into())).unwrap();
        assert_eq!(kind, constraint_type::REGEX);
        assert_eq!(body, "[0-9]+");
    }

    /// `.greedy()` has to reach the runtime as argmax.
    ///
    /// The obvious mapping — the sampler type literally named "greedy" — is
    /// the one that does not work: the shipped runtime answers
    /// `UNIMPLEMENTED: Sampler type: 3`. A single-candidate top-p is argmax,
    /// and it runs.
    #[test]
    fn greedy_becomes_a_single_candidate_rather_than_an_unimplemented_sampler() {
        let spec = GenSpec {
            temperature: Some(0.0),
            ..GenSpec::default()
        };
        let sampler = sampler_of(&Settings::default(), &spec).expect("a temperature was set");
        assert_eq!(sampler.kind, super::super::ffi::sampler_type::TOP_P);
        assert_eq!(
            sampler.top_k,
            Some(1),
            "greedy is one candidate; anything else is sampling"
        );
        assert_eq!(sampler.top_p, Some(1.0), "and no nucleus truncation on top");
    }

    #[test]
    fn a_seed_the_caller_set_reaches_the_sampler() {
        let spec = GenSpec {
            temperature: Some(0.8),
            seed: Some(42),
            ..GenSpec::default()
        };
        let sampler = sampler_of(&Settings::default(), &spec).expect("a temperature was set");
        assert_eq!(sampler.seed, Some(42));
    }

    /// A caller who sets only a seed has expressed a sampling preference.
    ///
    /// Reading temperature first and bailing meant `.seed(42)` produced no
    /// sampler at all, so the runtime kept its own — the seed was accepted,
    /// documented, and dropped. The same held for `.top_k()` and `.top_p()`
    /// alone.
    #[test]
    fn any_sampling_field_on_its_own_still_builds_a_sampler() {
        for (label, spec) in [
            (
                "seed",
                GenSpec {
                    seed: Some(42),
                    ..GenSpec::default()
                },
            ),
            (
                "top_k",
                GenSpec {
                    top_k: Some(40),
                    ..GenSpec::default()
                },
            ),
            (
                "top_p",
                GenSpec {
                    top_p: Some(0.9),
                    ..GenSpec::default()
                },
            ),
        ] {
            assert!(
                sampler_of(&Settings::default(), &spec).is_some(),
                "setting only {label} must still configure the sampler"
            );
        }
    }

    #[test]
    fn a_request_that_asks_for_nothing_leaves_the_runtime_its_own_sampler() {
        assert_eq!(
            sampler_of(&Settings::default(), &GenSpec::default()),
            None,
            "the bundle ships with a sampler; replacing it with one gen2 \
             invented would change output nobody asked to change"
        );
    }

    /// An explicit top-k must not silently acquire a nucleus threshold too.
    #[test]
    fn top_k_alone_truncates_by_k_and_by_nothing_else() {
        let spec = GenSpec {
            temperature: Some(0.8),
            top_k: Some(40),
            ..GenSpec::default()
        };
        let sampler = sampler_of(&Settings::default(), &spec).expect("top_k was set");
        assert_eq!(sampler.top_k, Some(40));
        assert_eq!(sampler.top_p, Some(1.0), "no top-p was asked for");
    }

    /// And top-p alone must not silently acquire a top-k cut.
    ///
    /// The runtime rejects `k <= 0`, so "no top-k" has to be spelled as a k
    /// past the vocabulary rather than left unset.
    #[test]
    fn top_p_alone_is_not_quietly_turned_into_top_k_as_well() {
        let spec = GenSpec {
            temperature: Some(0.8),
            top_p: Some(0.9),
            ..GenSpec::default()
        };
        let sampler = sampler_of(&Settings::default(), &spec).expect("top_p was set");
        assert_eq!(sampler.top_p, Some(0.9));
        assert!(
            sampler.top_k.is_some_and(|k| k >= 1 << 20),
            "expected a k that truncates nothing, got {:?}",
            sampler.top_k
        );
    }

    #[test]
    fn sampling_this_backend_cannot_honour_is_reported_rather_than_ignored() {
        let mut settings = Settings::default();
        settings.sampling.min_p = Some(0.05);
        let refusal = unsupported_sampling(&settings, &GenSpec::default())
            .expect("min-p has no LiteRT-LM equivalent");
        assert!(refusal.contains("min_p"));

        let spec = GenSpec {
            dry_multiplier: Some(0.8),
            ..GenSpec::default()
        };
        assert!(unsupported_sampling(&Settings::default(), &spec).is_some());
    }

    /// The same refusal has to fire for a per-request `min_p`.
    ///
    /// Checking the engine's settings before merging the request let the
    /// per-request form through silently, which is the shape a caller is most
    /// likely to use.
    #[test]
    fn min_p_set_on_the_request_is_refused_just_like_min_p_on_the_engine() {
        let spec = GenSpec {
            min_p: Some(0.1),
            ..GenSpec::default()
        };
        assert!(
            unsupported_sampling(&Settings::default(), &spec).is_some(),
            "a request-level min_p was accepted and then ignored"
        );
    }

    #[test]
    fn the_penalty_window_is_only_sent_when_the_caller_set_one() {
        assert_eq!(
            penalty_window(&Settings::default(), &GenSpec::default()),
            None
        );

        let mut settings = Settings::default();
        settings.sampling.penalty_last_n = Some(64);
        assert_eq!(penalty_window(&settings, &GenSpec::default()), Some(64));
    }

    #[test]
    fn penalties_are_left_alone_when_nobody_asked_for_any() {
        assert_eq!(
            penalties_of(&Settings::default(), &GenSpec::default()),
            None,
            "sending zeroed penalties would override the runtime's own defaults \
             with values the caller never chose"
        );

        let spec = GenSpec {
            penalty_repeat: Some(1.1),
            ..GenSpec::default()
        };
        let (repeat, freq, present) = penalties_of(&Settings::default(), &spec).unwrap();
        assert_eq!((repeat, freq, present), (1.1, 0.0, 0.0));
    }

    /// The bug a live run found: chunks are JSON messages, not text.
    ///
    /// Forwarding `stream_chunk_get_text` verbatim showed the caller
    /// `{"role":"assistant","content":[{"type":"text","text":"Okay"}]}` as the
    /// model's reply, one such object per token.
    #[test]
    fn a_streamed_chunk_yields_the_text_and_not_the_json_around_it() {
        let raw = r#"{"role":"assistant","content":[{"type":"text","text":"Okay"}]}"#;
        assert_eq!(decode_chunk(raw), vec![Part::Text("Okay".into())]);
    }

    #[test]
    fn several_messages_in_one_chunk_all_come_through() {
        // Nothing promises one message per callback, and a decoder that reads
        // only the first would drop tokens silently.
        let raw = concat!(
            r#"{"role":"assistant","content":[{"type":"text","text":"Hello"}]}"#,
            r#"{"role":"assistant","content":[{"type":"text","text":" there"}]}"#,
        );
        assert_eq!(
            decode_chunk(raw),
            vec![Part::Text("Hello".into()), Part::Text(" there".into())]
        );
    }

    #[test]
    fn a_bare_string_content_is_read_too() {
        let raw = r#"{"role":"assistant","content":"Hello"}"#;
        assert_eq!(decode_chunk(raw), vec![Part::Text("Hello".into())]);
    }

    #[test]
    fn text_that_is_not_json_reaches_the_caller_rather_than_vanishing() {
        // A different runtime build, or a plain-text stream. Either way it is
        // the model's output, and dropping it would be a silent truncation.
        assert_eq!(
            decode_chunk("just words"),
            vec![Part::Text("just words".into())]
        );
    }

    #[test]
    fn a_native_tool_call_is_taken_as_a_call_rather_than_as_prose() {
        let raw = r#"{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}"#;
        let parts = decode_chunk(raw);
        let Some(Part::Call(call)) = parts.into_iter().next() else {
            panic!("expected a tool call");
        };
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.id.as_deref(), Some("c1"));
        assert!(call.arguments.contains("Paris"));
    }

    #[test]
    fn an_empty_chunk_produces_nothing() {
        assert!(decode_chunk("").is_empty());
    }

    #[test]
    fn every_call_the_model_makes_survives_a_round_trip() {
        // Guards the shape gen2's own tool machinery reads back.
        let call = GenToolCall {
            id: Some("call_1".into()),
            name: "get_weather".into(),
            arguments: "{\"city\":\"Paris\"}".into(),
        };
        assert_eq!(call.name, "get_weather");
    }
}
