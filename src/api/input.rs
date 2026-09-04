//! What a model is given: text, and images for models that read them.
//!
//! One [`Input`] is one user turn. Modalities are parts of it rather than
//! separate methods, so a new modality is a new [`InputPart`] variant and not
//! a new combination of method names.

use std::path::Path;

/// One user turn's worth of input.
///
/// A `&str` or `String` converts directly, which is what
/// [`Model::generate`](crate::Model::generate) takes for the plain case:
///
/// ```no_run
/// # let model = gen2::load("m.gguf")?;
/// let text = model.generate("hello").text()?;
/// # Ok::<(), gen2::Error>(())
/// ```
///
/// Build one explicitly to attach images:
///
/// ```no_run
/// # let model = gen2::load("m.gguf")?;
/// use gen2::Input;
///
/// let response = model
///     .generate(Input::new().text("Compare these").image("a.png").image("b.png"))
///     .run()?;
/// # Ok::<(), gen2::Error>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Input {
    parts: Vec<InputPart>,
}

impl Input {
    /// An empty input. Add parts with [`Input::text`] and [`Input::image`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Append text.
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(InputPart::Text(text.into()));
        self
    }

    /// Append an image by path or URL.
    ///
    /// The model must accept images — see
    /// [`ModelCapabilities::images`](crate::model::ModelCapabilities::images).
    /// A text-only model refuses before generating anything.
    pub fn image(mut self, source: impl AsRef<Path>) -> Self {
        self.parts.push(InputPart::Image(Image::from_path(source)));
        self
    }

    /// Append an already-built part.
    pub fn part(mut self, part: InputPart) -> Self {
        self.parts.push(part);
        self
    }

    /// The parts, in order.
    pub fn parts(&self) -> &[InputPart] {
        &self.parts
    }

    /// The text parts joined with newlines. The prompt a text-only view
    /// of this input would see.
    pub fn text_only(&self) -> String {
        let mut out = String::new();
        for p in &self.parts {
            if let InputPart::Text(t) = p {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            }
        }
        out
    }

    /// Whether any part is an image.
    pub fn has_images(&self) -> bool {
        self.parts.iter().any(|p| matches!(p, InputPart::Image(_)))
    }

    /// The image sources, in order.
    pub(crate) fn images(&self) -> impl Iterator<Item = &Image> {
        self.parts.iter().filter_map(|p| match p {
            InputPart::Image(i) => Some(i),
            _ => None,
        })
    }

    /// The transcript message this input becomes.
    pub(crate) fn into_message(self) -> crate::types::message::Message {
        use crate::types::message::{Message, to_file_url};
        let text = self.text_only();
        let images: Vec<String> = self.images().map(|i| to_file_url(i.source())).collect();
        Message::user_with_images(text, images)
    }
}

impl From<&str> for Input {
    fn from(text: &str) -> Self {
        Self::new().text(text)
    }
}

impl From<String> for Input {
    fn from(text: String) -> Self {
        Self::new().text(text)
    }
}

impl From<&String> for Input {
    fn from(text: &String) -> Self {
        Self::new().text(text.as_str())
    }
}

/// One piece of an [`Input`].
///
/// Audio is not here yet: no backend accepts an audio part on a message
/// today, and a variant nothing can consume would be a promise.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum InputPart {
    /// Text.
    Text(String),
    /// An image, for a model that reads them.
    Image(Image),
}

impl InputPart {
    /// A text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// An image part, by path or URL.
    pub fn image(source: impl AsRef<Path>) -> Self {
        Self::Image(Image::from_path(source))
    }
}

/// An image the model is shown, referenced by path or URL.
///
/// The bytes are read by the backend at generation time. A local path
/// becomes a `file://` URL; an `http(s)://` or `file://` URL passes through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    source: String,
}

impl Image {
    /// From a path or URL.
    pub fn from_path(source: impl AsRef<Path>) -> Self {
        Self {
            source: source.as_ref().to_string_lossy().into_owned(),
        }
    }

    /// From a URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self { source: url.into() }
    }

    /// The path or URL as given.
    pub fn source(&self) -> &str {
        &self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::{MessageBody, MessageChunk, MessageContent};

    #[test]
    fn a_string_is_a_single_text_part() {
        let input: Input = "hi".into();
        assert_eq!(input.parts(), &[InputPart::Text("hi".into())]);
        assert!(!input.has_images());
        let input: Input = String::from("hi").into();
        assert_eq!(input.text_only(), "hi");
    }

    #[test]
    fn text_only_input_becomes_a_plain_user_message() {
        let msg = Input::new().text("a").text("b").into_message();
        assert_eq!(msg.role, "user");
        assert!(matches!(
            msg.body,
            MessageBody::Content {
                content: MessageContent::SingleText(ref t)
            } if t == "a\nb"
        ));
    }

    #[test]
    fn images_become_file_url_chunks_after_the_text() {
        let msg = Input::new()
            .text("Compare")
            .image("/tmp/a.png")
            .image("https://x/b.png")
            .into_message();
        let MessageBody::Content {
            content: MessageContent::MultipleChunks(chunks),
        } = msg.body
        else {
            panic!("an input with images must be a chunked message");
        };
        assert_eq!(chunks.len(), 3);
        assert!(matches!(&chunks[0], MessageChunk::Text { text } if text == "Compare"));
        assert!(
            matches!(&chunks[1], MessageChunk::ImageUrl { image_url } if image_url.url == "file:///tmp/a.png")
        );
        assert!(
            matches!(&chunks[2], MessageChunk::ImageUrl { image_url } if image_url.url == "https://x/b.png")
        );
    }
}
