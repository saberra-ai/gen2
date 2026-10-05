//! cargo run --no-default-features --features laya-dynamic --example laya_decisions -- BUNDLE ORT_LIBRARY
use gen2::{
    Runtime,
    decision::{DecisionOptions, DecisionRequest, LoadOptions, Question},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let bundle = args
        .next()
        .ok_or("usage: laya_decisions BUNDLE [ORT_LIBRARY]")?;
    let native_library = args.next().map(Into::into);
    let runtime = match args.next() {
        Some(mb) => Runtime::builder()
            .resident_memory_budget_mb(mb.parse()?)
            .build()?,
        None => Runtime::new()?,
    };
    let model = runtime.load_decider(
        bundle,
        LoadOptions {
            native_library,
            ..Default::default()
        },
    )?;
    let request = DecisionRequest::text("Our invoice was charged twice. Please refund it.")
        .question(
            "team",
            Question::choice(
                "Who handles this?",
                [("billing", "Payments"), ("support", "Product issues")],
            ),
        )
        .question(
            "urgency",
            Question::score("How urgent?", ["Routine", "Soon", "Blocking"]),
        )
        .question("refund", Question::yes_no("Is a refund requested?"));
    println!(
        "{}",
        serde_json::to_string_pretty(&model.decide(request, DecisionOptions::default())?)?
    );
    model.shutdown();
    Ok(())
}
