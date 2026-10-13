//! A real field-filling request, usable with either a GGUF or ONNX GenAI bundle.
//! cargo run --features backend-onnx --example text_filler -- /path/woof-genai

use gen2::model::ThinkingMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, gen2::schemars::JsonSchema)]
struct FieldValue {
    text: String,
}

// Typed serialization makes prompt order stable even when another Cargo
// dependency enables serde_json's preserve_order feature. Context comes first;
// the field to fill is the final, focused part of the request.
#[derive(Serialize)]
struct Request<'a> {
    goal: &'a str,
    page_context: Page<'a>,
    selected_field: Field<'a>,
}
#[derive(Serialize)]
struct Page<'a> {
    fields: [&'a str; 3],
    title: &'a str,
}
#[derive(Serialize)]
struct Field<'a> {
    current_value: &'a str,
    label: &'a str,
    tag: &'a str,
}

fn main() -> gen2::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("pass a model path or GenAI bundle directory");
    let model = gen2::load(&path)?;
    println!("model: {:?}", model.info());
    let cases = [
        ("First name", "Fill this form for Alice Chen.", "Alice"),
        (
            "Email address",
            "Sign up using alice@example.com.",
            "alice@example.com",
        ),
        (
            "City",
            "Complete my shipping address in Montréal, Canada.",
            "Montréal",
        ),
    ];
    for (label, goal, expected) in cases {
        let prompt = serde_json::to_string(&Request {
            goal,
            page_context: Page {
                fields: ["First name", "Email address", "City"],
                title: "Contact details",
            },
            selected_field: Field {
                current_value: "",
                label,
                tag: "input",
            },
        })
        .unwrap();
        let started = std::time::Instant::now();
        let value: FieldValue = model.generate(prompt)
            .system("You fill the selected browser field from the user's goal and page context. Return only a JSON object with a text property containing the value to type. Use only the requested component: a first-name field needs the first name, not the full name. Never invent personal details.")
            .reasoning(ThinkingMode::Off).greedy().max_tokens(64)
            .structured()?;
        println!(
            "{label}: {:?} (expected {:?}; correct={}; {:.3} s)",
            value.text,
            expected,
            value.text == expected,
            started.elapsed().as_secs_f32()
        );
    }
    Ok(())
}
