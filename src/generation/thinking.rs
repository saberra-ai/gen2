//! Thinking-mode toggle for models that expose a reasoning channel.
//!
//! Gemma-4 (`<|channel>thought…<channel|>` via `enable_thinking`),
//! DeepSeek-R1 (`<think>…</think>`), GPT-oss (`reasoning` channel),
//! Qwen3-Thinking (`<think>`), and the Claude 3.5/4 family all expose
//! a per-turn reasoning channel whose visibility the caller can toggle.
//! This enum is the canonical Pio-side representation.
//!
//! Why an enum vs `Option<bool>`:
//! - `None` was ambiguous ("unset", "default", or "off"?). `Auto` is
//!   explicit: don't override, let the model's chat template decide.
//! - Leaves room for effort-graded variants (OpenAI's `reasoning_effort:
//!   low | medium | high`, Anthropic's `thinking.budget_tokens`) as
//!   additive variants without a breaking type change.
//!
//! Mapping to the chat template's `enable_thinking`:
//!   `Off`  → `Some(false)`
//!   `On`   → `Some(true)`
//!   `Auto` → `None` (template's own default applies)

use serde::{Deserialize, Serialize};

/// Per-session thinking-channel policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    /// Don't override the model's chat-template default. When callers
    /// don't set anything, this is what lands.
    #[default]
    Auto,
    /// Force the reasoning channel off. Model answers directly.
    Off,
    /// Force the reasoning channel on. Useful on models that support
    /// toggling and default to off (some Gemma-4 setups, Qwen3 base).
    On,
}

impl ThinkingMode {
    /// Map to the chat template's `enable_thinking: Option<bool>`.
    ///
    /// `Auto` maps to `Some(true)` — NOT `None`. Reason: most chat
    /// templates (Gemma-4, Qwen3-Thinking, DeepSeek-R1) treat
    /// `undefined` as "thinking off" and these models were trained
    /// expecting the channel active. Passing `None` makes them emit
    /// the opening `<think>` / `<|channel>thought` tokens but fail
    /// to close cleanly, leaving the stream stuck before any visible
    /// answer lands — the exact "skeleton loader never resolves"
    /// regression we shipped for a day on Qwen3-0.6B. So the enum's
    /// "let the template decide" semantically collapses to
    /// "thinking on" for every model we bundle. Explicit `Off`
    /// remains the escape hatch.
    ///
    /// If a future model's template is genuinely coherent with
    /// `enable_thinking` undefined, add an `AutoTrue` variant
    /// rather than flipping this mapping — the default must never
    /// silently disable reasoning on a thinking-trained model.
    pub fn as_enable_thinking(self) -> Option<bool> {
        match self {
            Self::Auto | Self::On => Some(true),
            Self::Off => Some(false),
        }
    }

    /// Parse OpenAI's `reasoning_effort` convention (`"none" | "low" |
    /// "medium" | "high"`) into `ThinkingMode`. `low`/`medium`/`high`
    /// all collapse to `On` for now — when we grow graded variants,
    /// this mapping gets refined without touching call sites.
    pub fn from_reasoning_effort(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Self::Off),
            "low" | "medium" | "high" => Some(Self::On),
            _ => None,
        }
    }

    /// Parse the bool that vLLM / openai-python convention sends inside
    /// `chat_template_kwargs.enable_thinking`.
    pub fn from_enable_thinking_bool(b: bool) -> Self {
        if b { Self::On } else { Self::Off }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_auto() {
        assert_eq!(ThinkingMode::default(), ThinkingMode::Auto);
    }

    #[test]
    fn as_enable_thinking_maps_correctly() {
        // Auto preserves the legacy default (thinking on) to avoid
        // regressing models trained with the reasoning channel
        // expected. Explicit Off is the only way to disable.
        assert_eq!(ThinkingMode::Auto.as_enable_thinking(), Some(true));
        assert_eq!(ThinkingMode::Off.as_enable_thinking(), Some(false));
        assert_eq!(ThinkingMode::On.as_enable_thinking(), Some(true));
    }

    #[test]
    fn from_reasoning_effort_parses_openai_levels() {
        assert_eq!(
            ThinkingMode::from_reasoning_effort("none"),
            Some(ThinkingMode::Off)
        );
        assert_eq!(
            ThinkingMode::from_reasoning_effort("low"),
            Some(ThinkingMode::On)
        );
        assert_eq!(
            ThinkingMode::from_reasoning_effort("Medium"),
            Some(ThinkingMode::On)
        );
        assert_eq!(
            ThinkingMode::from_reasoning_effort("HIGH"),
            Some(ThinkingMode::On)
        );
        assert_eq!(ThinkingMode::from_reasoning_effort("goblin"), None);
        assert_eq!(
            ThinkingMode::from_reasoning_effort(" none "),
            Some(ThinkingMode::Off)
        );
    }

    #[test]
    fn from_enable_thinking_bool() {
        assert_eq!(
            ThinkingMode::from_enable_thinking_bool(true),
            ThinkingMode::On
        );
        assert_eq!(
            ThinkingMode::from_enable_thinking_bool(false),
            ThinkingMode::Off
        );
    }

    #[test]
    fn serde_wire_format_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&ThinkingMode::Auto).unwrap(),
            "\"auto\""
        );
        assert_eq!(
            serde_json::to_string(&ThinkingMode::Off).unwrap(),
            "\"off\""
        );
        assert_eq!(serde_json::to_string(&ThinkingMode::On).unwrap(), "\"on\"");
        assert_eq!(
            serde_json::from_str::<ThinkingMode>("\"off\"").unwrap(),
            ThinkingMode::Off
        );
    }
}
