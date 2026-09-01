//! An agent: tools registered, dispatch owned by the crate, hydration on demand.
//!
//! ```sh
//! cargo run --example agent --no-default-features --features metal -- /path/model.gguf
//! ```

use gen2::schemars::JsonSchema;
use gen2::{AgentStep, Engine, ExecutionPolicy, FunctionTool, Session, ToolOutput, ToolSearch};
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City to look up.
    city: String,
}

#[derive(Deserialize, JsonSchema)]
struct ResizeArgs {
    /// Path to the image.
    path: String,
    /// Target width in pixels.
    width: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).ok_or("usage: agent <model.gguf>")?;
    let engine = Engine::load(&model)?;
    let mut session = Session::new();

    let weather = FunctionTool::new(
        "get_weather",
        "Current weather for a city",
        |_ctx, a: WeatherArgs| async move {
            Ok(ToolOutput::Json(serde_json::json!({
                "city": a.city, "temp_c": 18, "sky": "clear"
            })))
        },
    );

    // Deferred: absent from the prompt until the model goes looking. Marked
    // GPU-bound because resizing contends with the model for the accelerator.
    let resize = FunctionTool::new(
        "resize_image",
        "Shrink a picture to a smaller width",
        |_ctx, a: ResizeArgs| async move {
            Ok(ToolOutput::from(format!(
                "resized {} to {}px",
                a.path, a.width
            )))
        },
    )
    .with_policy(ExecutionPolicy::gpu_bound());

    let done = engine
        .agent(&mut session)
        .add_tool(weather)
        .defer_tool(resize)
        .tool_search(ToolSearch::Hybrid)
        .max_steps(6)
        .run_streaming(
            Some(
                std::env::args()
                    .nth(2)
                    .unwrap_or_else(|| "What is the weather in Paris? Use the tool.".into()),
            ),
            |step| match step {
                AgentStep::Calling { tool, args, .. } => println!("  → {tool}({args})"),
                AgentStep::Called {
                    tool, result, took, ..
                } => match result {
                    Ok(out) => println!("  ← {tool} ok in {took:?}: {}", out.to_model_text()),
                    Err(e) => println!("  ← {tool} failed: {e}"),
                },
                _ => {}
            },
        )?;

    println!("\nanswer: {}", done.text.trim());
    println!("steps: {}, finish: {:?}", done.tool_rounds, done.finish);
    println!(
        "transcript: {:?}",
        session
            .messages()
            .iter()
            .map(|m| m.role.as_str())
            .collect::<Vec<_>>()
    );
    Ok(())
}
