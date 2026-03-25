use crate::gen2::{Message, MessageBody, MessageChunk, MessageContent};
pub(crate) fn messages_have_images(messages: &Vec<Message>) -> bool {
    for msg in messages {
        match &msg.body {
            MessageBody::Content { content } => match content {
                MessageContent::SingleText(_) => {}
                MessageContent::MultipleChunks(chunks) => {
                    if chunks
                        .iter()
                        .any(|c| matches!(c, MessageChunk::ImageUrl { .. }))
                    {
                        return true;
                    }
                }
            },
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::model_runner::types::{MessageBody, Url};

    #[test]
    fn detect_images() {
        let msgs = vec![Message {
            name: None,
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::MultipleChunks(vec![
                    MessageChunk::Text { text: "hi".into() },
                    MessageChunk::ImageUrl {
                        image_url: Url {
                            url: "file:///tmp/x.png".into(),
                        },
                    },
                ]),
            },
        }];
        assert!(messages_have_images(&msgs));
    }
}
