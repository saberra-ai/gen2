//! Run with --no-default-features --features llama-dflash2,metal.
//! Arguments: target.gguf draft.gguf|- [trials=2] [output_tokens=128].
use gen2::{
    Runtime,
    advanced::generation::{DFlash2Config, Settings, ThinkingMode},
    event::Event,
};
use serde_json::json;
use std::time::Instant;

const PROMPT: &str = "Write a complete Python implementation of a thread-safe LRU cache with get, put, eviction, capacity validation and unit tests. Explain the design briefly, then provide the full code.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let target = args.get(1).ok_or("target GGUF required")?;
    let draft = args.get(2).ok_or("draft GGUF or '-' required")?;
    let trials: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(2);
    let limit: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(128);
    if trials == 0 || limit < 2 {
        return Err("trials >= 1 and output_tokens >= 2 required".into());
    }
    let mut settings = Settings::default();
    settings.system.ctx_size = Some(8192);
    settings.system.batch_size = Some(512);
    settings.system.threads = Some(4);
    settings.system.threads_batch = Some(4);
    settings.system.gpu_layers = Some(99);
    settings.system.flash_attn = Some(true);
    settings.system.kv_cache_type = Some("q8_0".into());
    settings.prompt.include_meta = Some(false);
    if draft != "-" {
        settings.system.dflash2 = Some(DFlash2Config {
            model_path: draft.into(),
            max_tokens: 5,
        });
    }
    let runtime = Runtime::builder().settings(settings).build()?;
    let start = Instant::now();
    let model = runtime.load(target)?;
    println!(
        "BENCH_JSON {}",
        json!({"phase":"load", "load_s":start.elapsed().as_secs_f64(),
        "backend":"gen2-embedded-847f447", "model":target, "draft_model":draft,
        "load_definition":"Runtime.load weights only; context creation and prefill belong to TTFT"})
    );
    for trial in -2..trials as i32 {
        let start = Instant::now();
        let mut first = None;
        let mut last = 0.0;
        let mut arrival = Vec::new();
        // One-shot generate owns a new session: no prompt/KV reuse across trials.
        let mut stream = model
            .generate(PROMPT)
            .greedy()
            .seed(42)
            .reasoning(ThinkingMode::Off)
            .max_tokens(limit)
            .stream()?;
        for event in stream.by_ref() {
            if let Event::TextDelta(text) = event?
                && !text.is_empty()
            {
                last = start.elapsed().as_secs_f64();
                first.get_or_insert(last);
                arrival.push(last);
            }
        }
        let reply = stream.finish()?;
        let wall = start.elapsed().as_secs_f64();
        let stats = reply.stats.execution();
        let first = first.ok_or("no visible output")?;
        let n = reply.usage.completion_tokens;
        if n != limit as u32 || reply.reasoning().is_some_and(|s| !s.is_empty()) {
            return Err("unexpected token count or thinking channel".into());
        }
        println!(
            "BENCH_JSON {}",
            json!({"phase":if trial == -2 {"first_request"} else if trial == -1 {"warmup_short"} else {"short"},
            "trial":trial, "wall_s":wall, "ttft_visible_ms":first*1000.0,
            "prompt_tokens":reply.usage.prompt_tokens, "generated_tokens":n,
            "decode_tps":(n-1) as f64/(last-first), "end_to_end_tps":n as f64/wall,
            "timings":{"draft_n":stats.spec_drafted,"draft_n_accepted":stats.spec_accepted},
            "thinking_requested":false,"thinking_actual":false,"output_text":reply.text(),
            "chunk_intervals_ms":arrival.windows(2).map(|w|(w[1]-w[0])*1000.0).collect::<Vec<_>>() })
        );
    }
    Ok(())
}
