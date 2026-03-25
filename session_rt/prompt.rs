use crate::store::AppStore;
use crate::types::Persona;
use chrono::Local;
use std::env;

pub struct PromptContext {
    pub meta_prompt: String,
    pub persona: Option<Persona>,
}

pub async fn build_prompt_context(store: Option<&AppStore>) -> PromptContext {
    // let meta_prompt = build_meta_prompt();
    let persona = if let Some(app_store) = store {
        app_store.persona_store.get_selected_persona().await.ok().flatten()
    } else {
        None
    };

    PromptContext {
        meta_prompt: String::new(),
        persona,
    }
}

fn build_meta_prompt() -> String {
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
        sections.push(format!(
            "# Persona\n name: {} \n instructions {}",
            name, instructions
        ));
    }

    sections.join("\n\n")
}
