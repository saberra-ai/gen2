use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Token {
    pub id: u32,
    pub text: String,
    pub logprob: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum TokenEvent {
    Token(Token),
    Special(&'static str),
    MediaBoundary(MediaBoundary),
    Paused,
    Stopped,
    Eos,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum MediaBoundary {
    BeginImage { idx: usize },
    EndImage { idx: usize },
    BeginAudio { idx: usize },
    EndAudio { idx: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Step 3c: TokenEvent serde tests ─────────────────────────────

    // TokenEvent contains `Special(&'static str)`, so Deserialize only
    // implements for `'static` lifetime. We use static string literals
    // for deserialization and `to_string` for serialization verification.

    /// Serialize Token variant to JSON and deserialize from a static literal.
    #[test]
    fn token_event_serde_roundtrip_token() {
        let event = TokenEvent::Token(Token {
            id: 42,
            text: "hello".to_string(),
            logprob: Some(-1.5),
        });
        let json = serde_json::to_string(&event).expect("serialize Token");
        assert!(
            json.contains("\"Token\""),
            "JSON should contain Token variant tag"
        );
        assert!(json.contains("\"hello\""), "JSON should contain text value");

        // Deserialize from a static string literal (satisfies 'static lifetime).
        let back: TokenEvent =
            serde_json::from_str(r#"{"Token":{"id":42,"text":"hello","logprob":-1.5}}"#)
                .expect("deserialize Token");
        match back {
            TokenEvent::Token(t) => {
                assert_eq!(t.id, 42);
                assert_eq!(t.text, "hello");
                assert!((t.logprob.unwrap() - (-1.5)).abs() < f32::EPSILON);
            }
            other => panic!("expected Token, got {:?}", other),
        }
    }

    /// Serialize/deserialize Eos variant.
    #[test]
    fn token_event_serde_roundtrip_eos() {
        let event = TokenEvent::Eos;
        let json = serde_json::to_string(&event).expect("serialize Eos");
        assert!(json.contains("Eos"));
        let back: TokenEvent = serde_json::from_str("\"Eos\"").expect("deserialize Eos");
        assert!(matches!(back, TokenEvent::Eos));
    }

    /// Serialize/deserialize Stopped variant.
    #[test]
    fn token_event_serde_roundtrip_stopped() {
        let event = TokenEvent::Stopped;
        let json = serde_json::to_string(&event).expect("serialize Stopped");
        assert!(json.contains("Stopped"));
        let back: TokenEvent = serde_json::from_str("\"Stopped\"").expect("deserialize Stopped");
        assert!(matches!(back, TokenEvent::Stopped));
    }

    /// Serialize/deserialize Paused variant.
    #[test]
    fn token_event_serde_roundtrip_paused() {
        let event = TokenEvent::Paused;
        let json = serde_json::to_string(&event).expect("serialize Paused");
        assert!(json.contains("Paused"));
        let back: TokenEvent = serde_json::from_str("\"Paused\"").expect("deserialize Paused");
        assert!(matches!(back, TokenEvent::Paused));
    }

    /// MediaBoundary variants serialize/deserialize correctly.
    /// MediaBoundary has no `&'static str` fields, so from_value works fine.
    #[test]
    fn media_boundary_serde_roundtrip() {
        let variants = vec![
            MediaBoundary::BeginImage { idx: 0 },
            MediaBoundary::EndImage { idx: 1 },
            MediaBoundary::BeginAudio { idx: 2 },
            MediaBoundary::EndAudio { idx: 3 },
        ];
        for variant in variants {
            let val = serde_json::to_value(&variant).expect("serialize MediaBoundary");
            let back: MediaBoundary =
                serde_json::from_value(val).expect("deserialize MediaBoundary");
            match (&variant, &back) {
                (MediaBoundary::BeginImage { idx: a }, MediaBoundary::BeginImage { idx: b }) => {
                    assert_eq!(a, b)
                }
                (MediaBoundary::EndImage { idx: a }, MediaBoundary::EndImage { idx: b }) => {
                    assert_eq!(a, b)
                }
                (MediaBoundary::BeginAudio { idx: a }, MediaBoundary::BeginAudio { idx: b }) => {
                    assert_eq!(a, b)
                }
                (MediaBoundary::EndAudio { idx: a }, MediaBoundary::EndAudio { idx: b }) => {
                    assert_eq!(a, b)
                }
                _ => panic!("variant mismatch after roundtrip"),
            }
        }
    }

    /// Token with None logprob deserializes correctly.
    #[test]
    fn token_event_serde_token_no_logprob() {
        let event = TokenEvent::Token(Token {
            id: 0,
            text: String::new(),
            logprob: None,
        });
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(
            json.contains("\"logprob\":null"),
            "None logprob should serialize as null"
        );

        let back: TokenEvent =
            serde_json::from_str(r#"{"Token":{"id":0,"text":"","logprob":null}}"#)
                .expect("deserialize");
        match back {
            TokenEvent::Token(t) => {
                assert_eq!(t.id, 0);
                assert!(t.text.is_empty());
                assert!(t.logprob.is_none());
            }
            other => panic!("expected Token, got {:?}", other),
        }
    }
}
