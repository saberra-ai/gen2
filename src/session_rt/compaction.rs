//! Algorithmic context compaction — Tier 1, instant, no LLM call.
//!
//! Vendored from `pio-core`'s `app::chat::compaction` when gen2 became its own
//! crate. It operates purely on the wire `Message` list, so it belongs with the
//! session runtime that needs it: [`WarmStart::apply`] compacts here first and
//! only falls back to dropping whole messages when this can't free enough
//! context. The host's Tier 2 (LLM summarisation) stayed behind — it needs a
//! generation, which is the caller's to schedule.

use tracing::{debug, info};

use crate::types::message::{Message, MessageBody, MessageContent};

/// Recent messages kept verbatim by default (≈ 3 exchange pairs).
pub const DEFAULT_KEEP_RECENT: usize = 6;

/// Which compaction strategy produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// No compaction needed/performed.
    None,
    /// Tier 1: algorithmic extraction.
    Algorithmic,
}

impl CompactionStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Algorithmic => "algorithmic",
        }
    }
}

/// Outcome of a compaction pass.
#[derive(Debug, Clone)]
pub struct CompactResult {
    /// The (possibly compacted) message list.
    pub messages: Vec<Message>,
    /// How many original messages were replaced by the summary.
    pub compacted_count: usize,
    /// Which strategy ran.
    pub strategy: CompactionStrategy,
}

/// Extract the plain-text content of a message.
pub fn text_of(msg: &Message) -> String {
    match &msg.body {
        MessageBody::Content { content } => match content {
            MessageContent::SingleText(s) => s.clone(),
            MessageContent::StructuredAssistant { content, .. } => content.clone(),
            MessageContent::MultipleChunks(chunks) => chunks
                .iter()
                .filter_map(|c| match c {
                    crate::types::message::MessageChunk::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        },
        MessageBody::Tool { .. } => String::new(),
    }
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        let boundary = text
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        format!("{}...", &text[..boundary])
    }
}

/// Compact messages algorithmically (instant, no LLM).
///
/// Strategy:
///   1. Preserve the system message (index 0 if role == "system").
///   2. Split remaining messages into `head` (to compact) and `tail` (recent).
///   3. From `head`, keep: first user message (original intent) and last
///      assistant message (most recent context before the window).
///   4. Build a `<compact-summary>` user message combining these excerpts.
///   5. Return `[system?] + [summary] + [tail]`.
pub fn compact_algorithmic(messages: Vec<Message>, keep_recent: usize) -> CompactResult {
    let total = messages.len();
    let min_to_compact = keep_recent + 3; // need at least system + 1 compactable pair + tail

    if total <= min_to_compact {
        debug!(total, "Too few messages to compact — keeping all");
        return CompactResult {
            messages,
            compacted_count: 0,
            strategy: CompactionStrategy::None,
        };
    }

    // Separate system message if present.
    let has_system = messages
        .first()
        .map(|m| m.role == "system")
        .unwrap_or(false);
    let body_start = if has_system { 1 } else { 0 };

    // Split into head (to compact) and tail (to keep).
    let split_at = total.saturating_sub(keep_recent);
    if split_at <= body_start {
        return CompactResult {
            messages,
            compacted_count: 0,
            strategy: CompactionStrategy::None,
        };
    }

    let head = &messages[body_start..split_at];
    let tail = &messages[split_at..];

    if head.is_empty() {
        return CompactResult {
            messages,
            compacted_count: 0,
            strategy: CompactionStrategy::None,
        };
    }

    let compacted_count = head.len();

    // Build a structured digest from the head messages.
    let summary_text = build_algorithmic_digest(head, compacted_count);

    let summary_msg = Message {
        role: "system".into(),
        body: MessageBody::Content {
            content: MessageContent::SingleText(summary_text),
        },
        name: None,
    };

    // Assemble: [system?] + [summary] + [tail]
    let mut result = Vec::with_capacity(2 + tail.len());
    if has_system {
        result.push(messages[0].clone());
    }
    result.push(summary_msg);
    result.extend_from_slice(tail);

    info!(
        original = total,
        compacted = compacted_count,
        remaining = result.len(),
        "Algorithmic compaction complete"
    );

    CompactResult {
        messages: result,
        compacted_count,
        strategy: CompactionStrategy::Algorithmic,
    }
}

/// Build a structured digest from messages being compacted.
///
/// Instead of keeping only first-user + last-assistant (which loses the
/// middle of the conversation), this samples topic shifts at regular
/// intervals so the model retains awareness of what was discussed.
///
/// Structure:
///   - Original question (first user message, truncated)
///   - Key exchanges: up to 5 evenly-spaced user→assistant pairs from the
///     middle, each truncated to a short summary line
///   - Last context (final assistant message before the window, longer)
fn build_algorithmic_digest(head: &[Message], compacted_count: usize) -> String {
    const MAX_TOPIC_SAMPLES: usize = 5;
    const TOPIC_CHARS: usize = 120;
    const INTENT_CHARS: usize = 300;
    const CONTEXT_CHARS: usize = 400;

    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("[{compacted_count} earlier messages compacted]"));

    // 1. Original intent — first user message.
    if let Some(m) = head.iter().find(|m| m.role == "user") {
        parts.push(format!(
            "Original question: {}",
            truncate_text(&text_of(m), INTENT_CHARS)
        ));
    }

    // 2. Key exchanges — sample user messages at even intervals from the
    //    middle, paired with the following assistant response.
    let pairs: Vec<(&Message, Option<&Message>)> = head
        .windows(2)
        .filter_map(|w| {
            if w[0].role == "user" && w[1].role == "assistant" {
                Some((&w[0], Some(&w[1])))
            } else {
                None
            }
        })
        .collect();

    if pairs.len() > 2 {
        // Skip first (already captured as intent) and last (captured below).
        let middle = &pairs[1..pairs.len().saturating_sub(1)];
        let step = middle.len().max(1) / MAX_TOPIC_SAMPLES.min(middle.len()).max(1);
        let step = step.max(1);

        let mut topics: Vec<String> = Vec::new();
        let mut i = 0;
        while i < middle.len() && topics.len() < MAX_TOPIC_SAMPLES {
            let (user_msg, asst_opt) = &middle[i];
            let user_text = truncate_text(&text_of(user_msg), TOPIC_CHARS);
            let line = if let Some(asst) = asst_opt {
                let asst_text = truncate_text(&text_of(asst), TOPIC_CHARS);
                format!("- Q: {user_text}\n  A: {asst_text}")
            } else {
                format!("- Q: {user_text}")
            };
            topics.push(line);
            i += step;
        }

        if !topics.is_empty() {
            parts.push(format!("Key exchanges:\n{}", topics.join("\n")));
        }
    }

    // 3. Last context — final assistant message before the recent window.
    if let Some(m) = head.iter().rev().find(|m| m.role == "assistant") {
        parts.push(format!(
            "Last context before this window: {}",
            truncate_text(&text_of(m), CONTEXT_CHARS)
        ));
    }

    format!(
        "<compact-summary>\n{}\n</compact-summary>",
        parts.join("\n\n")
    )
}
