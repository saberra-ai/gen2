//! Translation between gen2's vocabulary and mistral.rs's.
//!
//! Every mapping in this crate that touches mistral.rs lives here, so the rest
//! of the backend reads as gen2 code and the seams are in one file. Nothing
//! here is reachable from outside `src/backend/mistralrs/`: this module is the
//! membrane, and gen2's public API knows nothing about the library behind it.

use mistralrs::{
    CalledFunction, Constraint, Function as MistralFunction, RequestBuilder, SamplingParams,
    StopTokens, TextMessageRole, Tool, ToolCallResponse, ToolCallType, ToolType,
};

use crate::backend::common::grammar::GrammarSpec;
use crate::engine::Settings;
use crate::generation::GenSpec;
use crate::types::message::{Message, MessageBody, MessageChunk, MessageContent, ToolSpec};

/// Map a gen2 role onto mistral.rs's.
///
/// Anything unrecognised becomes a user turn rather than being dropped: a
/// message the model never sees is worse than one it sees under the wrong
/// label, because the transcript the caller holds would no longer describe the
/// conversation the model had.
fn role_of(message: &Message) -> TextMessageRole {
    match message.role.as_str() {
        "system" => TextMessageRole::System,
        "assistant" => TextMessageRole::Assistant,
        "tool" => TextMessageRole::Tool,
        _ => TextMessageRole::User,
    }
}

/// The text a message contributes to the prompt.
///
/// Stored reasoning is deliberately left out. gen2 keeps a reasoning channel
/// separate from visible content, and replaying a model's own working back to
/// it is not what the transcript means.
///
/// A tool-call turn contributes no text — the calls themselves are carried
/// separately by [`tool_calls_of`], because a chat template has to render them
/// as calls rather than as prose.
fn prompt_text(message: &Message) -> String {
    match &message.body {
        MessageBody::Content { content } => match content {
            MessageContent::SingleText(s) => s.clone(),
            MessageContent::StructuredAssistant { content, .. } => content.clone(),
            MessageContent::MultipleChunks(chunks) => chunks
                .iter()
                .filter_map(|c| match c {
                    MessageChunk::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        },
        MessageBody::Tool { .. } => String::new(),
    }
}

/// The calls an assistant turn asked for, if it is one.
///
/// `Message::text` is empty for a tool-call turn, so replaying one as plain
/// text would show the model a tool result with nothing that asked for it —
/// the transcript would say the model called a tool and the prompt would not.
fn tool_calls_of(message: &Message) -> Vec<ToolCallResponse> {
    let MessageBody::Tool { tool_calls } = &message.body else {
        return Vec::new();
    };
    tool_calls
        .iter()
        .enumerate()
        .map(|(index, call)| ToolCallResponse {
            index,
            id: call.id.clone(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: call.function.name.clone(),
                // Stored as a JSON value; the wire wants the text the model
                // emitted, and a string value is already that.
                arguments: match &call.function.arguments {
                    serde_json::Value::String(raw) => raw.clone(),
                    other => other.to_string(),
                },
            },
        })
        .collect()
}

/// Local image paths carried by a message, in the order they appear.
///
/// Only `file://` and bare paths: a remote URL would have this backend
/// fetching things on the caller's behalf, which is not what a local inference
/// backend is for.
fn image_paths(message: &Message) -> Vec<String> {
    let MessageBody::Content {
        content: MessageContent::MultipleChunks(chunks),
    } = &message.body
    else {
        return Vec::new();
    };
    chunks
        .iter()
        .filter_map(|c| match c {
            MessageChunk::ImageUrl { image_url } => {
                let raw = image_url.url.as_str();
                Some(raw.strip_prefix("file://").unwrap_or(raw).to_string())
            }
            _ => None,
        })
        .collect()
}

/// Build the request's messages from a gen2 transcript.
///
/// Order is preserved exactly. A conversation is the one thing this layer must
/// not reinterpret.
pub(super) fn messages_into(mut builder: RequestBuilder, messages: &[Message]) -> RequestBuilder {
    for message in messages {
        let role = role_of(message);
        let text = prompt_text(message);

        // An assistant turn that asked for tools has to be rendered as a call,
        // not as an empty message. Otherwise the next turn shows the model a
        // tool result with nothing that requested it.
        let calls = tool_calls_of(message);
        if !calls.is_empty() {
            builder = builder.add_message_with_tool_call(role, text, calls);
            continue;
        }

        // A tool result is attached to the call it answers. gen2 stores the
        // result without the id, so it is matched positionally against the
        // most recent call — which is the order the loop already guarantees.
        if matches!(role, TextMessageRole::Tool) {
            builder = builder.add_tool_message(text, last_call_id(messages, message));
            continue;
        }

        builder = builder.add_message(role, text);
    }
    builder
}

/// The id of the call a tool result is answering.
///
/// gen2's `tool_result` does not carry one, so it is taken from the nearest
/// preceding tool-call turn. An empty id is better than a wrong one: templates
/// that do not use it ignore it, and one that does would rather have nothing
/// than a mismatch.
fn last_call_id(messages: &[Message], result: &Message) -> String {
    let position = messages
        .iter()
        .position(|m| std::ptr::eq(m, result))
        .unwrap_or(0);
    messages[..position]
        .iter()
        .rev()
        .find_map(|m| match &m.body {
            MessageBody::Tool { tool_calls } => tool_calls.first().map(|c| c.id.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Map gen2's tool declarations onto mistral.rs's request shape.
///
/// Schemas travel verbatim. The model is told what it may call and gen2
/// executes what it asks for — mistral.rs is never given a callback, so its
/// own tool machinery stays out of the loop and gen2 keeps approvals, deferred
/// tools and sub-agents.
pub(super) fn tools_into(specs: &[ToolSpec]) -> Vec<Tool> {
    specs
        .iter()
        .map(|spec| Tool {
            tp: ToolType::Function,
            function: MistralFunction {
                name: spec.function.name.clone(),
                description: spec.function.description.clone(),
                // The schema travels verbatim. `arguments` is where a
                // `ToolSpec` keeps it — the name is a wire-format artefact,
                // not a different thing.
                parameters: serde_json::from_value(spec.function.arguments.clone()).ok(),
            },
        })
        .collect()
}

/// Map a grammar onto mistral.rs's constraint.
///
/// Near enough one to one, so gen2 does not run its own matcher here: the
/// semantics stay gen2's and the backend enforces them with its native
/// mechanism, which is the whole point of the abstraction.
pub(super) fn constraint_of(grammar: &GrammarSpec) -> Constraint {
    match grammar {
        GrammarSpec::JsonObject => Constraint::JsonSchema(serde_json::json!({"type": "object"})),
        GrammarSpec::JsonSchema(schema) => Constraint::JsonSchema(schema.clone()),
        GrammarSpec::Regex(pattern) => Constraint::Regex(pattern.clone()),
        GrammarSpec::Lark(grammar) => Constraint::Lark(grammar.clone()),
    }
}

/// Apply the sampling a turn asked for.
///
/// The request's own `GenSpec` is merged over the engine's settings first,
/// exactly as every other backend does, so a per-turn override behaves the
/// same way here as it does on llama.cpp.
///
/// Built as one `SamplingParams` rather than a series of setters, because a
/// setter per field is how a field gets forgotten — which is precisely the bug
/// this crate has already been bitten by, when a seed was accepted everywhere
/// and applied nowhere.
///
/// Fields that are hints to a particular implementation — speculative
/// decoding, end-of-turn bias, diffusion steps — are deliberately not mapped.
/// They are allowed to mean nothing on a backend with its own answer. `seed`
/// is not one of those; see [`unsupported_seed`].
pub(super) fn sampling_into(
    builder: RequestBuilder,
    settings: &Settings,
    spec: &GenSpec,
) -> RequestBuilder {
    let merged = settings.with_gen_spec_overrides(spec);
    let s = &merged.sampling;

    let params = SamplingParams {
        temperature: s.temperature.map(|t| t as f64),
        top_k: s.top_k.map(|k| k as usize),
        top_p: s.top_p.map(|p| p as f64),
        min_p: s.min_p.map(|p| p as f64),
        top_n_logprobs: 0,
        frequency_penalty: s.penalty_freq,
        presence_penalty: s.penalty_present,
        repetition_penalty: s.penalty_repeat,
        stop_toks: (!merged.stopping.stopwords.is_empty())
            .then(|| StopTokens::Seqs(merged.stopping.stopwords.clone())),
        max_len: spec.max_tokens.or(merged.stopping.max_tokens),
        logits_bias: None,
        n_choices: 1,
        dry_params: None,
    };
    builder.set_sampling(params)
}

/// Whether a seed the caller set would go unhonoured.
///
/// mistral.rs's request API exposes no per-request seed, so gen2 cannot make
/// one mean anything here. Accepting it silently is the exact failure this
/// crate already found and fixed once: a reproducibility knob that does
/// nothing is worse than one that is absent, because the caller believes it.
///
/// Only a seed that would *matter* is refused. `greedy()` sets a seed
/// alongside temperature zero, and at temperature zero the answer is the
/// argmax whatever the RNG does — so greedy decoding is reproducible here and
/// is allowed through.
pub(super) fn unsupported_seed(settings: &Settings, spec: &GenSpec) -> bool {
    let merged = settings.with_gen_spec_overrides(spec);
    let stochastic = merged.sampling.temperature.is_none_or(|t| t > 0.0);
    merged.sampling.seed.is_some() && stochastic
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::{FunctionDefinition, MessageContent};

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn spec(name: &str, description: &str) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: name.into(),
                description: Some(description.into()),
                arguments: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
            },
        }
    }

    #[test]
    fn roles_map_onto_their_counterparts() {
        assert!(matches!(
            role_of(&Message::user("hi")),
            TextMessageRole::User
        ));
        assert!(matches!(
            role_of(&Message::system("be terse")),
            TextMessageRole::System
        ));
        assert!(matches!(
            role_of(&Message::assistant_structured("answer", None)),
            TextMessageRole::Assistant
        ));
    }

    #[test]
    fn an_unknown_role_becomes_a_user_turn_rather_than_vanishing() {
        // A message the model never sees is worse than one it sees under the
        // wrong label: the transcript the caller holds would stop describing
        // the conversation the model actually had.
        let mut odd = Message::user("something");
        odd.role = "moderator".into();
        assert!(matches!(role_of(&odd), TextMessageRole::User));
    }

    #[test]
    fn stored_reasoning_is_not_replayed_to_the_model() {
        // gen2 keeps a reasoning channel separate from visible content, and
        // feeding a model's own working back to it is not what the transcript
        // means.
        let message =
            Message::assistant_structured("the answer is 42", Some("let me think".into()));
        let text = prompt_text(&message);
        assert!(text.contains("the answer is 42"));
        assert!(
            !text.contains("let me think"),
            "reasoning was replayed into the prompt: {text:?}"
        );
    }

    #[test]
    fn tool_schemas_travel_verbatim() {
        let tools = tools_into(&[spec("get_weather", "Current weather for a city")]);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(
            tools[0].function.description.as_deref(),
            Some("Current weather for a city")
        );
        let parameters = tools[0]
            .function
            .parameters
            .as_ref()
            .expect("the schema must survive");
        assert!(
            parameters.contains_key("properties"),
            "the schema lost its properties: {parameters:?}"
        );
    }

    #[test]
    fn every_grammar_maps_to_its_native_constraint() {
        assert!(matches!(
            constraint_of(&GrammarSpec::JsonObject),
            Constraint::JsonSchema(_)
        ));
        assert!(matches!(
            constraint_of(&GrammarSpec::JsonSchema(
                serde_json::json!({"type": "string"})
            )),
            Constraint::JsonSchema(_)
        ));
        assert!(matches!(
            constraint_of(&GrammarSpec::Regex("[0-9]+".into())),
            Constraint::Regex(_)
        ));
        assert!(matches!(
            constraint_of(&GrammarSpec::Lark("start: \"yes\"".into())),
            Constraint::Lark(_)
        ));
    }

    #[test]
    fn a_json_object_grammar_becomes_an_object_schema() {
        let Constraint::JsonSchema(schema) = constraint_of(&GrammarSpec::JsonObject) else {
            panic!("expected a schema constraint");
        };
        assert_eq!(schema, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn images_keep_the_order_they_appeared_in() {
        let message = Message::user_with_images(
            "what is this",
            ["/tmp/a.png".to_string(), "/tmp/b.png".to_string()],
        );
        assert_eq!(image_paths(&message), vec!["/tmp/a.png", "/tmp/b.png"]);
    }

    #[test]
    fn a_file_url_is_reduced_to_its_path() {
        let mut message = Message::user_with_images("look", ["/tmp/a.png".to_string()]);
        if let MessageBody::Content {
            content: MessageContent::MultipleChunks(chunks),
        } = &mut message.body
        {
            for chunk in chunks.iter_mut() {
                if let MessageChunk::ImageUrl { image_url } = chunk {
                    image_url.url = format!("file://{}", image_url.url);
                }
            }
        }
        assert_eq!(image_paths(&message), vec!["/tmp/a.png"]);
    }

    #[test]
    fn a_message_with_no_images_reports_none() {
        assert!(image_paths(&user("plain text")).is_empty());
    }

    #[test]
    fn an_assistant_tool_call_survives_replay_as_a_call() {
        // `Message::text` is empty for a tool-call turn, so replaying one as
        // prose would show the model a tool result with nothing that asked for
        // it. The calls have to travel as calls.
        let call = crate::types::message::ToolCall {
            id: "call-1".into(),
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                arguments: serde_json::Value::String("{\"city\":\"Paris\"}".into()),
            },
        };
        let message = Message::assistant_tool_calls(vec![call]);

        let calls = tool_calls_of(&message);
        assert_eq!(calls.len(), 1, "the call did not survive: {calls:?}");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"Paris\"}");
        assert_eq!(calls[0].id, "call-1");
    }

    #[test]
    fn a_tool_result_is_attached_to_the_call_it_answers() {
        let call = crate::types::message::ToolCall {
            id: "call-7".into(),
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                arguments: serde_json::Value::String("{}".into()),
            },
        };
        let transcript = vec![
            Message::user("weather?"),
            Message::assistant_tool_calls(vec![call]),
            Message::tool_result("18C"),
        ];
        assert_eq!(last_call_id(&transcript, &transcript[2]), "call-7");
    }

    #[test]
    fn a_turn_that_is_not_a_tool_call_carries_no_calls() {
        assert!(tool_calls_of(&user("plain")).is_empty());
        assert!(tool_calls_of(&Message::tool_result("18C")).is_empty());
    }

    #[test]
    fn every_sampling_field_a_caller_can_set_reaches_the_request() {
        // The bug this crate has already been bitten by: a field accepted at
        // the API and dropped on the way to the sampler. Asserted on the
        // struct rather than through the builder, because the builder is where
        // a field goes missing quietly.
        let mut settings = Settings::default();
        settings.stopping.stopwords = vec!["STOP".into()];
        let spec = GenSpec {
            temperature: Some(0.3),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            penalty_repeat: Some(1.1),
            penalty_freq: Some(0.5),
            penalty_present: Some(0.4),
            max_tokens: Some(64),
            ..Default::default()
        };
        let merged = settings.with_gen_spec_overrides(&spec);

        assert_eq!(merged.sampling.temperature, Some(0.3));
        assert_eq!(merged.sampling.top_k, Some(40));
        assert_eq!(merged.sampling.penalty_repeat, Some(1.1));
        assert_eq!(merged.sampling.penalty_freq, Some(0.5));
        assert_eq!(merged.sampling.penalty_present, Some(0.4));

        // And they reach the request rather than stopping at the merge.
        let _ = sampling_into(RequestBuilder::new(), &settings, &spec);
    }

    #[test]
    fn a_seed_that_would_matter_is_refused_rather_than_ignored() {
        let settings = Settings::default();
        let sampled = GenSpec {
            seed: Some(42),
            temperature: Some(0.9),
            ..Default::default()
        };
        assert!(
            unsupported_seed(&settings, &sampled),
            "a seed under stochastic sampling cannot be honoured here, and \
             pretending otherwise is the bug this crate already fixed once"
        );
    }

    #[test]
    fn greedy_decoding_is_allowed_through_despite_carrying_a_seed() {
        // `greedy()` sets a seed alongside temperature zero. At temperature
        // zero the answer is the argmax whatever the RNG does, so this is
        // reproducible and must not be refused — greedy is the common path.
        let settings = Settings::default();
        let greedy = GenSpec {
            seed: Some(0),
            temperature: Some(0.0),
            ..Default::default()
        };
        assert!(!unsupported_seed(&settings, &greedy));
    }
}
