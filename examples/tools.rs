//! Tool calling: the model asks, the crate dispatches, the loop continues.
//!
//! ```sh
//! cargo run --example tools --no-default-features --features metal -- /path/model.gguf
//! ```

use pio_gen2::{Engine, FunctionDefinition, Session, ToolSpec};

fn weather_tool() -> ToolSpec {
    ToolSpec {
        r#type: "function".into(),
        function: FunctionDefinition {
            name: "get_weather".into(),
            description: Some("Current weather for a city".into()),
            arguments: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).ok_or("usage: tools <model.gguf>")?;
    let engine = Engine::load(&model)?;
    let mut session = Session::new();

    let mut calls = 0;
    let done = engine
        .chat(&mut session)
        .user("What is the weather in Paris? Use the tool.")
        .tools(vec![weather_tool()], "Call a tool when you need data.")
        // With a handler the turn becomes a loop: generate → dispatch →
        // feed results back → generate, until the model stops asking.
        .on_tool(|call| {
            calls += 1;
            println!("  → tool: {}({})", call.name, call.arguments);
            match call.name.as_str() {
                "get_weather" => r#"{"temp_c":18,"sky":"clear"}"#.to_string(),
                other => format!("no such tool: {other}"),
            }
        })
        .tool_depth(4)
        .send()?;

    println!("\nanswer: {}", done.text.trim());
    println!(
        "tool rounds: {}, calls dispatched: {calls}",
        done.tool_rounds
    );
    println!("finish: {:?}", done.finish);

    println!("\ntranscript:");
    for m in session.messages() {
        println!("  {}", m.role);
    }

    Ok(())
}
