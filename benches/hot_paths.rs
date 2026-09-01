//! What costs time on every turn, measured so a regression is visible.
//!
//! Not a gate. A benchmark that fails a pull request because one CI machine
//! was fourteen percent slower teaches people to ignore benchmarks, and these
//! are worth more tracked over time than enforced per commit.
//!
//! What is here is the work that happens per turn and does not involve the
//! model: parsing a GGUF header before deciding whether to load it, searching
//! a tool catalogue, scanning generated text for channel markers, encoding a
//! KV blob. Decoding itself is dominated by the backend and belongs in a
//! hardware benchmark, not this one.
//!
//! Run with `cargo bench`. Criterion is deliberately not a dependency: it
//! pulls a large tree for something that runs by hand, and `std::time` is
//! enough to see a change of the size that matters here.

use std::hint::black_box;
use std::time::{Duration, Instant};

use gen2::{GrammarSpec, HardwareProfile, ModelInfo};

/// Run `f` enough times to be worth timing, and report the per-iteration cost.
fn bench(name: &str, iterations: u32, mut f: impl FnMut()) {
    // One untimed pass, so a lazily-built cache is not charged to the first
    // measured iteration.
    f();

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let total = start.elapsed();
    let each = total / iterations;

    println!(
        "{name:<44} {:>10} per iteration  ({iterations} iterations in {:.2?})",
        format_duration(each),
        total
    );
}

fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns} ns")
    } else if ns < 1_000_000 {
        format!("{:.1} us", ns as f64 / 1_000.0)
    } else {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    }
}

fn main() {
    println!("gen2 hot paths — per-turn work that does not involve the model\n");

    bench_hardware();
    bench_model_header();
    bench_grammar();

    println!(
        "\nNot a gate: these are for comparing against a previous run on the \
         same machine, not for passing or failing one."
    );
}

/// Detecting the machine, which a host does once at startup and again on every
/// fit check.
fn bench_hardware() {
    bench("HardwareProfile::detect", 200, || {
        black_box(HardwareProfile::detect());
    });

    let hw = HardwareProfile::detect();
    bench("budget and context sizing", 100_000, || {
        black_box(hw.inference_budget_bytes());
    });
}

/// Reading a model's header, which happens before every load and for every
/// candidate in a directory the host is offering the user.
///
/// Expect single-digit milliseconds, and expect it not to move with file size
/// — 378 MB and 2.3 GB both measured around 5 ms, while a 697 MB model came in
/// at 2 ms. What it tracks is metadata volume, which is dominated by the
/// tokenizer vocabulary. If this ever starts scaling with the file, the
/// "header only, no weights" promise has been broken.
fn bench_model_header() {
    let Some(path) = std::env::var("PIO_TEST_MODEL")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        println!("{:<44} skipped — set PIO_TEST_MODEL", "ModelInfo::read");
        return;
    };

    bench("ModelInfo::read (header only, no weights)", 200, || {
        black_box(ModelInfo::read(&path).expect("the model should parse"));
    });

    let info = ModelInfo::read(&path).expect("the model should parse");
    let hw = HardwareProfile::detect();
    bench("ModelInfo::fits", 100_000, || {
        black_box(info.fits(&hw, Some(8192)));
    });
}

/// Compiling a grammar, which is paid once per constrained generation and sits
/// directly between the caller's request and the first token.
fn bench_grammar() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature_c": {"type": "number"},
            "conditions": {"type": "string", "enum": ["sunny", "cloudy", "rain"]}
        },
        "required": ["city", "temperature_c"]
    });

    bench("GrammarSpec::JsonSchema construction", 100_000, || {
        black_box(GrammarSpec::JsonSchema(schema.clone()));
    });
}
