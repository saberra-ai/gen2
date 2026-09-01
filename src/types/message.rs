//! LLM-facing message types.
//!
//! These are the wire types used by inference backends (gen2, legacy generation)
//! and the Sabra search engine for prompt construction. They are intentionally
//! separate from the storage `Message` in `crate::types` (which carries id,
//! chat_id, timestamps, etc.).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(untagged)]
pub enum TokenizerConfigToken {
    String(String),
    Object { content: String },
}

impl TokenizerConfigToken {
    pub fn as_str(&self) -> &str {
        match self {
            TokenizerConfigToken::String(s) => s,
            TokenizerConfigToken::Object { content } => content,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialEq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct FunctionDefinition {
    #[serde(default)]
    pub description: Option<String>,
    pub name: String,
    #[serde(alias = "parameters", serialize_with = "serialize_as_string")]
    pub arguments: serde_json::Value,
}

fn serialize_as_string<S>(value: &serde_json::Value, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&value.to_string())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(test, derive(PartialEq))]
pub struct Tool {
    // The type of the tool. Currently, only 'function' is supported.
    pub r#type: String,
    // Grab the tool as generic JSON for debugging purposes.
    pub function: FunctionDefinition,
}

#[derive(Clone, Deserialize, Serialize, Default, Debug, PartialEq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Url {
    pub url: String,
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum MessageChunk {
    Text { text: String },
    ImageUrl { image_url: Url },
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Message {
    pub role: String,
    #[serde(flatten)]
    pub body: MessageBody,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(untagged)]
pub enum MessageBody {
    // When a regular text message is provided.
    Content {
        #[serde(rename = "content")]
        content: MessageContent,
    },
    // When tool calls are provided.
    Tool {
        #[serde(rename = "tool_calls")]
        tool_calls: Vec<ToolCall>,
    },
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(untagged)]
pub enum MessageContent {
    SingleText(String),
    MultipleChunks(Vec<MessageChunk>),
    /// Structured assistant reply with the reasoning channel split out
    /// from the visible content. Produced by Phase 2a's state machine
    /// (see `pio_gen2::generation::ReplyStateMachine`) and by
    /// OAI-compat clients that POST back a prior turn with separate
    /// `content` and `reasoning_content` fields.
    ///
    /// When rendered via `TextMessage` for the chat template, only
    /// `content` flows through — matching the template's `strip_thinking`
    /// behaviour on prior turns. `reasoning` is preserved for UI display,
    /// Phase-0 telemetry, and the Phase-1 template contract test suite.
    ///
    /// The variant sits at the tail of the `untagged` enum so serde
    /// prefers the simpler `SingleText` / `MultipleChunks` shapes for
    /// legacy JSON; only bodies with an explicit `content`+`reasoning`
    /// object land here.
    StructuredAssistant {
        content: String,
        #[serde(default)]
        reasoning: Option<String>,
    },
}

impl Message {
    /// Convenience constructor for a single-text user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.into()),
            },
            name: None,
        }
    }

    /// Convenience constructor for a multimodal user message: the prompt text
    /// followed by one `ImageUrl` chunk per attached image. `image_urls` are
    /// already-resolved URLs (`file://…` for local files, or `http(s)://…`);
    /// the llama VLM session strips the `file://` prefix and loads each bitmap.
    /// With no images this is identical to [`Message::user`] (a single-text
    /// message) so callers can pass an empty slice unconditionally.
    pub fn user_with_images(
        text: impl Into<String>,
        image_urls: impl IntoIterator<Item = String>,
    ) -> Self {
        let text = text.into();
        let mut chunks: Vec<MessageChunk> = image_urls
            .into_iter()
            .map(|url| MessageChunk::ImageUrl {
                image_url: Url { url },
            })
            .collect();
        if chunks.is_empty() {
            return Self::user(text);
        }
        // Text chunk first, then the images — matches the chunk order the
        // src-tauri chat path builds (`api/message.rs`).
        chunks.insert(0, MessageChunk::Text { text });
        Self {
            role: "user".to_string(),
            body: MessageBody::Content {
                content: MessageContent::MultipleChunks(chunks),
            },
            name: None,
        }
    }

    /// Convenience constructor for a single-text system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            body: MessageBody::Content {
                content: MessageContent::SingleText(text.into()),
            },
            name: None,
        }
    }

    /// Structured assistant reply preserving the thinking/visible split
    /// produced by `ReplyStateMachine`.
    ///
    /// When the message flows back through the chat template (next
    /// turn's replay), only `content` is rendered. `reasoning` is
    /// preserved for storage, telemetry, and UI display.
    pub fn assistant_structured(content: impl Into<String>, reasoning: Option<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            body: MessageBody::Content {
                content: MessageContent::StructuredAssistant {
                    content: content.into(),
                    reasoning,
                },
            },
            name: None,
        }
    }
}

/// Normalize an attached image/document reference into a URL the inference
/// backends can resolve: an already-formed `http(s)://` / `file://` URL passes
/// through unchanged; a local path becomes a `file://` URL. The gen2 llama VLM
/// session strips the `file://` prefix and loads the bitmap from disk.
///
/// The single source of truth for the chat image paths — the desktop chat
/// (`src-tauri/api/message.rs`), the prompt builder, and the Birds chat all
/// route attachments through here so they agree on the URL shape the session
/// expects.
pub fn to_file_url(uri: &str) -> String {
    if uri.starts_with("http://") || uri.starts_with("https://") || uri.starts_with("file://") {
        uri.to_string()
    } else {
        url::Url::from_file_path(uri)
            .map(|u| u.to_string())
            .unwrap_or_else(|_| format!("file://{uri}"))
    }
}

// The `From<pio_core::types::Message>` conversions that used to sit here moved
// to the host: they translate a stored *chat* record into this wire type, and
// the chat record is the host's, not the engine's. The orphan rule allows them
// there (the host owns the source type).

// Pushing a chunk to a single text message will convert it to a multiple chunks message
impl MessageContent {
    pub fn push(&mut self, chunk: MessageChunk) {
        match self {
            MessageContent::SingleText(text) => {
                *self = MessageContent::MultipleChunks(vec![
                    MessageChunk::Text { text: text.clone() },
                    chunk,
                ]);
            }
            MessageContent::MultipleChunks(chunks) => {
                chunks.push(chunk);
            }
            MessageContent::StructuredAssistant { content, reasoning } => {
                // Mixed-modality structured assistant messages are rare
                // (models don't emit images); collapse to chunks with the
                // visible content only. Reasoning is preserved by any
                // upstream persistence (stored separately); dropping it
                // here is consistent with the template's replay contract.
                let _ = reasoning;
                *self = MessageContent::MultipleChunks(vec![
                    MessageChunk::Text {
                        text: content.clone(),
                    },
                    chunk,
                ]);
            }
        }
    }

    /// Visible text the chat template renders for this content. For
    /// `StructuredAssistant`, returns only `content` — the reasoning
    /// channel is intentionally dropped on replay (Gemma-4's
    /// `strip_thinking`; Qwen3's equivalent). Callers that need the
    /// reasoning text must match on the variant directly.
    pub fn as_visible_text(&self) -> String {
        match self {
            MessageContent::SingleText(text) => text.clone(),
            MessageContent::MultipleChunks(chunks) => chunks
                .iter()
                .map(|chunk| match chunk {
                    MessageChunk::Text { text } => text.clone(),
                    MessageChunk::ImageUrl { image_url } => format!("![]({})", image_url.url),
                })
                .collect::<Vec<_>>()
                .join(""),
            MessageContent::StructuredAssistant { content, .. } => content.clone(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq, Default)]

pub struct TextMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl From<Message> for TextMessage {
    fn from(value: Message) -> Self {
        let content = match value.body {
            MessageBody::Content { content } => content,
            MessageBody::Tool { tool_calls } => {
                let content = serde_json::to_string(&tool_calls).unwrap_or_default();
                MessageContent::SingleText(content)
            }
        };
        TextMessage {
            role: value.role,
            content: content.as_visible_text(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]

pub struct ChatTemplateInputs<'a> {
    pub messages: Vec<TextMessage>,
    pub(crate) bos_token: Option<&'a str>,
    pub(crate) eos_token: Option<&'a str>,
    pub(crate) add_generation_prompt: bool,
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) tools: Option<Vec<Tool>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_assistant_flattens_to_content_only_for_template() {
        // The load-bearing invariant behind Phase 2b: replaying a
        // structured assistant message through the chat template must
        // emit ONLY the visible `content`. Reasoning stays in storage
        // for UI/telemetry — feeding it back into the model risks the
        // template-lossy regression the whole plan was built to avoid.
        let m = Message::assistant_structured("the answer is 42", Some("let me think".into()));
        let tm: TextMessage = m.into();
        assert_eq!(tm.content, "the answer is 42");
        assert!(!tm.content.contains("let me think"));
    }

    #[test]
    fn user_with_images_builds_text_then_image_chunks() {
        let m = Message::user_with_images(
            "what's in this image?",
            [
                "file:///tmp/a.png".to_string(),
                "file:///tmp/b.png".to_string(),
            ],
        );
        assert_eq!(m.role, "user");
        let MessageBody::Content {
            content: MessageContent::MultipleChunks(chunks),
        } = m.body
        else {
            panic!("expected multi-chunk content");
        };
        // text first, then one ImageUrl per attached image.
        assert_eq!(chunks.len(), 3);
        assert!(
            matches!(&chunks[0], MessageChunk::Text { text } if text == "what's in this image?")
        );
        assert!(
            matches!(&chunks[1], MessageChunk::ImageUrl { image_url } if image_url.url == "file:///tmp/a.png")
        );
        assert!(
            matches!(&chunks[2], MessageChunk::ImageUrl { image_url } if image_url.url == "file:///tmp/b.png")
        );
    }

    #[test]
    fn user_with_images_falls_back_to_single_text_when_empty() {
        let m = Message::user_with_images("hello", Vec::<String>::new());
        assert!(matches!(
            m.body,
            MessageBody::Content {
                content: MessageContent::SingleText(t)
            } if t == "hello"
        ));
    }

    #[test]
    fn structured_assistant_with_no_reasoning_still_works() {
        let m = Message::assistant_structured("hi", None);
        let tm: TextMessage = m.into();
        assert_eq!(tm.content, "hi");
    }

    #[test]
    fn as_visible_text_strips_reasoning() {
        let c = MessageContent::StructuredAssistant {
            content: "visible".into(),
            reasoning: Some("hidden reasoning".into()),
        };
        assert_eq!(c.as_visible_text(), "visible");
    }

    #[test]
    fn structured_assistant_serde_untagged_roundtrip() {
        // The `untagged` shape means `{ "content": "...", "reasoning": "..." }`
        // deserialises as StructuredAssistant. A plain string still picks
        // SingleText (legacy).
        let js = r#"{"content":"answer","reasoning":"thoughts"}"#;
        let parsed: MessageContent = serde_json::from_str(js).unwrap();
        match parsed {
            MessageContent::StructuredAssistant { content, reasoning } => {
                assert_eq!(content, "answer");
                assert_eq!(reasoning.as_deref(), Some("thoughts"));
            }
            other => panic!("expected StructuredAssistant, got {other:?}"),
        }

        let legacy: MessageContent = serde_json::from_str("\"just a string\"").unwrap();
        assert!(matches!(legacy, MessageContent::SingleText(ref s) if s == "just a string"));
    }

    #[test]
    fn structured_assistant_without_reasoning_field_parses() {
        // `reasoning` is `#[serde(default)]` so clients that omit it
        // still land in the StructuredAssistant arm when they post an
        // object body with just `content`.
        //
        // Note: due to `untagged` enum resolution, we have to
        // explicitly construct — a bare `{"content": "x"}` could also
        // match other variants. This test locks the serialization
        // side.
        let m = MessageContent::StructuredAssistant {
            content: "hi".into(),
            reasoning: None,
        };
        let js = serde_json::to_string(&m).unwrap();
        // `reasoning` serializes as `null` (serde default), which is
        // what the OAI wire receives.
        assert!(js.contains("\"content\":\"hi\""));
    }
}
