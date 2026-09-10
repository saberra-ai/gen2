//! Tool calling: the model asks, you dispatch, the conversation continues.
//!
//! gen2 renders the definitions and parses the calls; it never executes one.
//! The loop is yours, which is what lets it be a harness's loop (api_spec.md
//! §10, §28.5).
//!
//! ```sh
//! cargo run --example tools -- /path/model.gguf
//! ```

use gen2::{Session, ToolDefinition, ToolSet};

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct WeatherArgs {
    /// City to look up.
    city: String,
}

fn execute(name: &str, args: &serde_json::Value) -> String {
    match name {
        "get_weather" => match serde_json::from_value::<WeatherArgs>(args.clone()) {
            Ok(a) => format!(r#"{{"city":"{}","temp_c":18,"sky":"clear"}}"#, a.city),
            Err(e) => format!("bad arguments: {e}"),
        },
        other => format!("no such tool: {other}"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: tools <model.gguf>")?;
    let model = gen2::load(&path)?;

    // Tools are session state, like the system prompt.
    let tools = ToolSet::new().with(
        ToolDefinition::new("get_weather")
            .description("Current weather for a city")
            .input_schema::<WeatherArgs>(),
    );
    let mut session = Session::new().with_tools(tools);
    session.push_user("What is the weather in Paris? Use the tool.");

    let mut dispatched = 0;
    let answer = loop {
        // A turn with no new user message is how tool results reach the model.
        let response = model.turn(&mut session).run()?;
        if response.tool_calls().is_empty() {
            break response;
        }
        for call in response.tool_calls() {
            dispatched += 1;
            println!("  → {}({})", call.name(), call.arguments());
            let result = execute(call.name(), call.arguments());
            session.push_tool_result(call.id().as_str(), result);
        }
    };

    println!("\nanswer: {}", answer.text().trim());
    println!(
        "calls dispatched: {dispatched}, finish: {}",
        answer.finish_reason()
    );
    println!("\ntranscript:");
    for m in session.messages() {
        println!("  {}", m.role);
    }
    Ok(())
}
