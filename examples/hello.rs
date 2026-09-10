//! The five-minute first run. No arguments:
//!
//! ```sh
//! cargo run --example hello
//! ```
//!
//! Downloads Qwen3-0.6B (Q4_K_M, ~400 MB) from Hugging Face on the first run
//! and reads it from the cache after that, asks one question, and prints the
//! answer with the timings and where the weights went. Pass a path or another
//! `hf:` reference to run it on a different model:
//!
//! ```sh
//! cargo run --example hello -- /path/model.gguf
//! cargo run --example hello -- hf:unsloth/Qwen3-1.7B-GGUF
//! ```

use std::time::Instant;

use gen2::model::ThinkingMode;

/// The smoke model: small enough to arrive inside the five minutes on an
/// ordinary connection. Qwen3-1.7B (`hf:unsloth/Qwen3-1.7B-GGUF`) is the
/// first size worth building on.
const DEFAULT_MODEL: &str = "hf:unsloth/Qwen3-0.6B-GGUF";

fn main() -> gen2::Result<()> {
    let reference = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    if reference.starts_with("hf:") {
        println!(
            "loading {reference} (the first run downloads it into {})",
            gen2::hf::cache_dir().display()
        );
    } else {
        println!("loading {reference}");
    }
    let started = Instant::now();
    let model = gen2::load(&reference)?;
    println!("loaded in {:.1} s", started.elapsed().as_secs_f32());

    let info = model.info();
    match &info.offload {
        // "29/29 layers on the GPU (MTL, Apple M4 Pro)" — or the CPU. MTL is
        // ggml's name for its Metal backend.
        Some(offload) => println!("weights: {offload}"),
        None => println!("weights: placement not reported"),
    }

    let response = model
        .generate("Why is the sky blue? Answer in two sentences.")
        // Qwen3 thinks before it answers; the smoke test wants the answer.
        .reasoning(ThinkingMode::Off)
        .max_tokens(128)
        .run()?;
    println!("\n{}\n", response.text().trim());

    let stats = response.stats();
    if stats.reported() {
        let ttft = stats
            .time_to_first_token()
            .map(|d| d.as_millis())
            .unwrap_or_default();
        println!(
            "{} tokens · first token in {ttft} ms · {:.1} tok/s",
            response.usage().completion_tokens,
            stats.tokens_per_second()
        );
    }
    Ok(())
}
