use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub id: u32,
    pub text: String,
    pub logprob: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenEvent {
    Token(Token),
    Special(&'static str),
    MediaBoundary(MediaBoundary),
    Paused,
    Stopped,
    Eos,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediaBoundary {
    BeginImage { idx: usize },
    EndImage { idx: usize },
    BeginAudio { idx: usize },
    EndAudio { idx: usize },
}
