use crate::types::Persona;
use chrono::Local;
use std::env;

pub struct PromptContext {
    pub meta_prompt: String,
    pub persona: Option<Persona>,
}

/// Assemble the context a session's system prompt is built from.
///
/// The caller resolves the persona (a host app reads its own selected-persona
/// record); the engine only decides how it merges with the meta prompt.
pub fn build_prompt_context(persona: Option<Persona>, include_meta: bool) -> PromptContext {
    let meta_prompt = if include_meta {
        build_meta_prompt()
    } else {
        String::new()
    };

    PromptContext {
        meta_prompt,
        persona,
    }
}

pub fn build_meta_prompt() -> String {
    let device = env::consts::OS;
    let arch = env::consts::ARCH;
    let now = Local::now();
    let date = now.format("%B %e, %Y");

    format!(
        "You are Pio Chat, a local AI assistant running on device.\n\
         Device: {device}-{arch}\n\
         Current date: {date}\n\
         ",
        device = device,
        arch = arch,
        date = date,
    )
}

pub fn merge_prompts(
    meta: &str,
    system_prompt: Option<&str>,
    persona_opt: Option<&Persona>,
) -> String {
    let mut sections = Vec::new();

    sections.push(meta.trim().to_string());

    if let Some(sys) = system_prompt.and_then(|s| {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }) {
        sections.push(format!("# User instructions\n{}", sys));
    }

    if let Some(persona) = persona_opt {
        let instructions = persona.instructions.trim();
        let name = persona.name.trim();
        if !instructions.is_empty() || !name.is_empty() {
            sections.push(format!(
                "# Persona\nName: {}\nInstructions:\n{}",
                name, instructions
            ));
        }
    }

    sections.join("\n\n")
}

/// Compute how many tokens to reserve for generation output.
/// Uses `max_tokens` when set, otherwise 25% of context. Clamped to [1, ctx_size/2].
/// For very small contexts (< 128 tokens), the minimum is lowered to avoid
/// a `clamp` panic where min > max.
pub fn generation_reserve(ctx_size: usize, max_tokens: Option<usize>) -> usize {
    let half = ctx_size / 2;
    let min_reserve = half.min(64); // never panic: min ≤ max
    let default = ctx_size / 4;
    max_tokens
        .unwrap_or(default)
        .clamp(min_reserve.max(1), half.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_persona(name: &str, instructions: &str) -> Persona {
        Persona {
            id: String::new(),
            name: name.to_string(),
            instructions: instructions.to_string(),
            is_selected: false,
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn merge_prompts_all_empty() {
        let result = merge_prompts("", None, None);
        assert_eq!(result.trim(), "");
    }

    #[test]
    fn merge_prompts_meta_only() {
        let result = merge_prompts("You are Pio", None, None);
        assert_eq!(result, "You are Pio");
    }

    #[test]
    fn merge_prompts_meta_and_system() {
        let result = merge_prompts("You are Pio", Some("Be helpful"), None);
        assert!(result.contains("You are Pio"));
        assert!(result.contains("# User instructions\nBe helpful"));
    }

    #[test]
    fn merge_prompts_persona_formatting() {
        let persona = make_persona("Socrates", "Ask questions, never give answers directly.");
        let result = merge_prompts("", None, Some(&persona));
        assert!(result.contains("# Persona\nName: Socrates\nInstructions:\nAsk questions"));
    }

    #[test]
    fn merge_prompts_all_three() {
        let persona = make_persona("Teacher", "Be patient");
        let result = merge_prompts("meta", Some("sys"), Some(&persona));
        assert!(result.contains("meta"));
        assert!(result.contains("# User instructions\nsys"));
        assert!(result.contains("# Persona\nName: Teacher\nInstructions:\nBe patient"));
    }

    #[test]
    fn merge_prompts_empty_persona_skipped() {
        let persona = make_persona("", "");
        let result = merge_prompts("meta", None, Some(&persona));
        assert!(!result.contains("Persona"));
    }

    #[test]
    fn merge_prompts_empty_system_skipped() {
        let result = merge_prompts("meta", Some("  "), None);
        assert!(!result.contains("User instructions"));
    }

    #[test]
    fn generation_reserve_with_max_tokens() {
        assert_eq!(generation_reserve(4096, Some(512)), 512);
    }

    #[test]
    fn generation_reserve_default_25_percent() {
        assert_eq!(generation_reserve(4096, None), 1024);
    }

    #[test]
    fn generation_reserve_clamp_min() {
        // max_tokens = 10, but min is 64
        assert_eq!(generation_reserve(4096, Some(10)), 64);
    }

    #[test]
    fn generation_reserve_clamp_max() {
        // max_tokens = 99999, but max is ctx_size/2
        assert_eq!(generation_reserve(4096, Some(99999)), 2048);
    }

    #[test]
    fn generation_reserve_small_context() {
        // 256 ctx: default would be 64, which is ctx/4
        assert_eq!(generation_reserve(256, None), 64);
    }

    #[test]
    fn generation_reserve_tiny_context() {
        // 128 ctx: default 32, clamped to 64 (min)
        assert_eq!(generation_reserve(128, None), 64);
    }

    #[test]
    fn generation_reserve_very_tiny_context() {
        // 64 ctx: this used to panic with clamp(64, 32) where min > max.
        // Now: half=32, min_reserve=min(32,64)=32, default=16, 16.clamp(32,32)=32
        assert_eq!(generation_reserve(64, None), 32);
        assert_eq!(generation_reserve(64, Some(10)), 32);
        assert_eq!(generation_reserve(64, Some(100)), 32);
    }

    #[test]
    fn generation_reserve_zero_context() {
        // Edge case: ctx_size=0 should not panic
        assert_eq!(generation_reserve(0, None), 1);
        assert_eq!(generation_reserve(0, Some(10)), 1);
    }

    #[test]
    fn generation_reserve_one_context() {
        assert_eq!(generation_reserve(1, None), 1);
    }

    #[test]
    fn build_meta_prompt_contains_date_and_device() {
        let meta = build_meta_prompt();
        assert!(meta.contains("Pio Chat"));
        assert!(meta.contains("Device:"));
        assert!(meta.contains("Current date:"));
    }
}
