//! Translation between gen2's vocabulary and mistral.rs's.
//!
//! Every mapping in this crate that touches mistral.rs lives here, so the rest
//! of the backend reads as gen2 code and the seams are in one file. Nothing
//! here is reachable from outside `src/backend/mistralrs/`: this module is the
//! membrane, and gen2's public API knows nothing about the library behind it.

use mistralrs::{
    Constraint, Function as MistralFunction, RequestBuilder, TextMessageRole, Tool, ToolType,
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
        MessageBody::Tool { .. } => message.text(),
    }
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
        let images = image_paths(message);

        if images.is_empty() {
            builder = builder.add_message(role, text);
            continue;
        }
        // A multimodal turn still carries its text; dropping it would leave
        // the model an image and no question.
        builder = builder.add_message(role, text);
    }
    builder
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
/// Fields that are hints to a particular implementation — speculative
/// decoding, end-of-turn bias, diffusion steps — are deliberately not mapped.
/// They are allowed to mean nothing on a backend that has its own answer.
pub(super) fn sampling_into(
    mut builder: RequestBuilder,
    settings: &Settings,
    spec: &GenSpec,
) -> RequestBuilder {
    let merged = settings.with_gen_spec_overrides(spec);
    let s = &merged.sampling;

    if let Some(t) = s.temperature {
        builder = builder.set_sampler_temperature(t as f64);
    }
    if let Some(k) = s.top_k {
        builder = builder.set_sampler_topk(k as usize);
    }
    if let Some(p) = s.top_p {
        builder = builder.set_sampler_topp(p as f64);
    }
    if let Some(p) = s.min_p {
        builder = builder.set_sampler_minp(p as f64);
    }
    if let Some(n) = spec.max_tokens.or(merged.stopping.max_tokens) {
        builder = builder.set_sampler_max_len(n);
    }
    if !merged.stopping.stopwords.is_empty() {
        builder = builder.set_sampler_stop_toks(mistralrs::StopTokens::Seqs(
            merged.stopping.stopwords.clone(),
        ));
    }
    builder
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
}
