pub trait MediaEncoder: Send + Sync {
    fn encode(&self, _att: &Attachment) -> anyhow::Result<EncodedMedia>;
}

#[derive(Debug, Clone)]
pub enum Attachment {
    Image(Vec<u8>),
    Audio(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct EncodedMedia {
    pub tokens: Vec<u32>,
}
