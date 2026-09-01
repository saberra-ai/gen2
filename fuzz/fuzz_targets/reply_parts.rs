//! The reply-part scanner over adversarial model output.
//!
//! Invariant: any sequence of chunks, under any marker set, terminates
//! without panicking — and the scanner never invents bytes. Structure-aware:
//! the input is a marker set plus a chunk list, so the generator produces
//! streams that actually reach the marker-matching paths instead of noise
//! that never matches anything.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use gen2::{ChannelMarkers, ReplyStateMachine, StreamEmission};

/// A chunk is either free-form text or one of the marker literals the real
/// families use — the latter so the generator hits marker boundaries far
/// more often than random bytes would.
#[derive(Arbitrary, Debug)]
enum Chunk {
    Text(String),
    GemmaOpen,
    GemmaOpenNoNewline,
    GemmaClose,
    GemmaCloseNoNewline,
    ThinkOpen,
    ThinkClose,
    Newline,
    Fragment(u8),
}

impl Chunk {
    fn as_str(&self) -> &str {
        match self {
            Chunk::Text(s) => s,
            Chunk::GemmaOpen => "<|channel>thought\n",
            Chunk::GemmaOpenNoNewline => "<|channel>thought",
            Chunk::GemmaClose => "\n<channel|>",
            Chunk::GemmaCloseNoNewline => "<channel|>",
            Chunk::ThinkOpen => "<think>",
            Chunk::ThinkClose => "</think>",
            Chunk::Newline => "\n",
            Chunk::Fragment(b) => match b % 6 {
                0 => "<",
                1 => "<|",
                2 => "<|c",
                3 => "</",
                4 => "|>",
                _ => ">",
            },
        }
    }
}

#[derive(Arbitrary, Debug)]
enum Markers {
    None,
    Gemma4,
    Qwen3DeepSeek,
    /// Caller-supplied markers: `ChannelMarkers` has public fields, so a
    /// host can put anything here — including the empty string, which
    /// matches everywhere and consumes nothing.
    Custom {
        open: Vec<String>,
        close: Vec<String>,
    },
}

impl Markers {
    fn build(self) -> ChannelMarkers {
        match self {
            Markers::None => ChannelMarkers::none(),
            Markers::Gemma4 => ChannelMarkers::gemma4(),
            Markers::Qwen3DeepSeek => ChannelMarkers::qwen3_deepseek(),
            Markers::Custom { open, close } => ChannelMarkers { open, close },
        }
    }
}

#[derive(Arbitrary, Debug)]
struct Input {
    markers: Markers,
    chunks: Vec<Chunk>,
}

fuzz_target!(|input: Input| {
    // Bound the work per case: libFuzzer inputs are small, but a `Vec<Chunk>`
    // of huge `String`s would time out for reasons that are not bugs.
    let total: usize = input.chunks.iter().map(|c| c.as_str().len()).sum();
    if total > 1 << 16 {
        return;
    }
    let markers = input.markers.build();
    let joined: String = input.chunks.iter().map(Chunk::as_str).collect();

    // Batch path.
    let mut batch = ReplyStateMachine::new(markers.clone());
    for c in &input.chunks {
        batch.push(c.as_str());
    }
    let batch = batch.finish();

    // Streaming path, accumulating what a client would have seen.
    let mut sm = ReplyStateMachine::new(markers);
    let mut streamed_content = String::new();
    let mut streamed_reasoning = String::new();
    for c in &input.chunks {
        for e in sm.push_emit(c.as_str()) {
            match e {
                StreamEmission::Content(s) => streamed_content.push_str(&s),
                StreamEmission::Reasoning(s) => streamed_reasoning.push_str(&s),
            }
        }
    }
    for e in sm.flush_pending() {
        match e {
            StreamEmission::Content(s) => streamed_content.push_str(&s),
            StreamEmission::Reasoning(s) => streamed_reasoning.push_str(&s),
        }
    }
    let streamed = sm.finish();

    // The streaming client and the stored reply must agree.
    assert_eq!(streamed_content, streamed.content);
    assert_eq!(
        streamed_reasoning,
        streamed.reasoning.clone().unwrap_or_default()
    );
    // Batch and streaming are two spellings of one contract.
    assert_eq!(batch, streamed);
    // The scanner may drop markers; it may never invent bytes.
    let kept = batch.content.len() + batch.reasoning.map_or(0, |r| r.len());
    assert!(kept <= joined.len(), "scanner produced more bytes than it read");
});
