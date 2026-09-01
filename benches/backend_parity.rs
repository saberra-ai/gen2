//! What the wrapper costs, measured against the thing it wraps.
//!
//! gen2 is a layer over llama.cpp. The question this answers is the one
//! nothing else in the repo asks: does going through the controller, the
//! session runtime and the token puller cost throughput, and how much?
//!
//! The reference is `llama-bench`, llama.cpp's own benchmark, run on the same
//! file with the same shape of work. It reports two numbers and so does this:
//!
//! - **prefill** — how fast the prompt is consumed, `pp` in llama-bench.
//! - **decode** — how fast tokens come out afterwards, `tg` in llama-bench.
//!
//! Decode is the number that matters for parity. Prefill through gen2 also
//! pays session setup, template rendering and tokenisation, so it is expected
//! to be lower and is reported for information rather than as a target.
//!
//! ```sh
//! PIO_TEST_MODEL=/path/model.gguf cargo bench --bench backend_parity --features metal
//! ```
//!
//! With `llama-bench` on `PATH` it runs the reference itself and prints the
//! ratio. Without it, it prints gen2's numbers alone, because a benchmark that
//! refuses to run without a comparison is a benchmark nobody runs.
//!
//! # Reading the result
//!
//! Both sides must be given the same work or the ratio is meaningless. Same
//! file, same token counts, greedy decoding on both, and the same machine at
//! the same moment. What is *not* controlled is the llama.cpp version: gen2
//! pins a `llama-cpp-rs` revision while `llama-bench` is whatever is on
//! `PATH`, so a few percent either way says more about the two builds than
//! about the wrapper. A large gap is the finding; a small one is noise.

use std::time::Instant;

use gen2::{Engine, Session};

/// Tokens of prompt to consume, matching the `-p` given to llama-bench.
const PREFILL_TOKENS: usize = 128;
/// Tokens to generate, matching `-n`.
const DECODE_TOKENS: usize = 64;
/// How many times to run each side. Enough to see variance without turning a
/// benchmark into an errand.
const REPEATS: usize = 5;

/// Pause between the two sides so the second is not measured while the first
/// is still giving the GPU back.
///
/// Without it the reference measured 262 tok/s on one run and 89 on the next,
/// and the headline swung from 83% to 185% — a benchmark reporting a different
/// answer each time is worse than none.
const SETTLE: std::time::Duration = std::time::Duration::from_secs(2);

struct Measurement {
    prefill_tps: f64,
    decode_tps: f64,
    /// Decode samples, so the spread can be reported rather than hidden behind
    /// a single number.
    decode_samples: Vec<f64>,
}

impl Measurement {
    /// How far the decode samples spread, as a fraction of the median. A run
    /// that varies by more than a few percent was measuring the machine.
    fn decode_spread(&self) -> f64 {
        if self.decode_samples.len() < 2 {
            return 0.0;
        }
        let (lo, hi) = (
            self.decode_samples.iter().copied().fold(f64::MAX, f64::min),
            self.decode_samples.iter().copied().fold(0.0, f64::max),
        );
        let mid = self.decode_tps.max(f64::MIN_POSITIVE);
        (hi - lo) / mid
    }
}

fn main() {
    let Some(model) = std::env::var("PIO_TEST_MODEL")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        println!("backend parity: skipped — set PIO_TEST_MODEL to a GGUF");
        return;
    };

    println!("gen2 against llama.cpp, on {model}\n");

    let ours = match measure_gen2(&model) {
        Ok(m) => m,
        Err(e) => {
            println!("gen2 failed to run: {e}");
            return;
        }
    };
    println!(
        "gen2         prefill {:>8.1} tok/s    decode {:>7.1} tok/s  (±{:.0}%)",
        ours.prefill_tps,
        ours.decode_tps,
        ours.decode_spread() * 100.0
    );

    std::thread::sleep(SETTLE);

    let Some(theirs) = measure_llama_bench(&model) else {
        println!(
            "\nllama-bench not on PATH, so there is nothing to compare against. \
             `brew install llama.cpp` for the reference numbers."
        );
        return;
    };
    println!(
        "llama-bench  prefill {:>8.1} tok/s    decode {:>7.1} tok/s",
        theirs.prefill_tps, theirs.decode_tps
    );

    // Both sides are medians. A mean lets one disturbed run move the headline,
    // which is exactly what a shared GPU produces.
    let ratio = ours.decode_tps / theirs.decode_tps.max(f64::MIN_POSITIVE);
    println!("\ndecode parity: {:.0}% of the reference", ratio * 100.0);

    // Refuse to conclude from a measurement that was measuring the machine.
    // The threshold is generous: anything above it and the two sides were not
    // given comparable conditions, whatever the ratio says.
    const NOISE_CEILING: f64 = 0.25;
    if ours.decode_spread() > NOISE_CEILING {
        println!(
            "inconclusive: gen2's own runs varied by {:.0}%, so this ratio is \
             noise. Close what else is using the GPU and run it again.",
            ours.decode_spread() * 100.0
        );
        return;
    }
    println!(
        "{}",
        match ratio {
            r if r >= 1.05 =>
                "Faster than the reference, which means the two were \
                               not given the same work — check the settings before believing it.",
            r if r >= 0.9 => "The wrapper is not in the way.",
            r if r >= 0.75 => "A real gap, worth a profile.",
            _ => "Most of the model's throughput is being lost above llama.cpp.",
        }
    );
}

/// Time gen2 doing the same work llama-bench times.
///
/// Prefill and decode are separated with the token counts and first-token
/// timing the engine reports, so the split is the engine's own accounting
/// rather than a guess from wall clock.
fn measure_gen2(model: &str) -> Result<Measurement, gen2::Error> {
    let engine = Engine::builder().model(model).context(4096).build()?;

    // A prompt long enough to be worth measuring. Word count is a stand-in for
    // token count; the engine reports what it actually tokenised, and that is
    // what the rate is computed from.
    let prompt = "token ".repeat(PREFILL_TOKENS);

    let mut prefill = Vec::new();
    let mut decode = Vec::new();

    for _ in 0..REPEATS {
        let mut session = Session::new();
        let started = Instant::now();
        let done = engine
            .chat(&mut session)
            .user(&prompt)
            .max_tokens(DECODE_TOKENS)
            .greedy()
            .send()?;
        let wall = started.elapsed().as_secs_f64();

        let Some(stats) = done.stats else { continue };
        let ttft = stats.first_token_us as f64 / 1_000_000.0;
        if ttft > 0.0 && stats.prompt_tokens > 0 {
            prefill.push(stats.prompt_tokens as f64 / ttft);
        }
        let generating = (wall - ttft).max(f64::MIN_POSITIVE);
        if stats.decode_tokens > 1 {
            // The first token is charged to prefill, so it is not part of the
            // decode rate.
            decode.push((stats.decode_tokens - 1) as f64 / generating);
        }
    }

    Ok(Measurement {
        prefill_tps: median(&prefill),
        decode_tps: median(&decode),
        decode_samples: decode,
    })
}

/// Run llama.cpp's own benchmark and read its two rates back out.
fn measure_llama_bench(model: &str) -> Option<Measurement> {
    let out = std::process::Command::new("llama-bench")
        .args([
            "-m",
            model,
            "-p",
            &PREFILL_TOKENS.to_string(),
            "-n",
            &DECODE_TOKENS.to_string(),
            "-r",
            &REPEATS.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut prefill_tps = 0.0;
    let mut decode_tps = 0.0;
    for line in text.lines() {
        // Rows look like `| … | pp128 | 4573.44 ± 1.63 |`; the rate is the
        // last numeric cell.
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        let Some(test) = cells.iter().rev().nth(2) else {
            continue;
        };
        let Some(rate) = cells
            .iter()
            .rev()
            .nth(1)
            .and_then(|c| c.split_whitespace().next())
            .and_then(|n| n.parse::<f64>().ok())
        else {
            continue;
        };
        if test.starts_with("pp") {
            prefill_tps = rate;
        } else if test.starts_with("tg") {
            decode_tps = rate;
        }
    }

    (decode_tps > 0.0).then_some(Measurement {
        prefill_tps,
        decode_tps,
        decode_samples: vec![decode_tps],
    })
}

/// The median run.
///
/// Not the fastest, which flatters, and not the mean, which one disturbed run
/// on a shared GPU can drag anywhere. The spread is reported separately so a
/// noisy measurement announces itself instead of being averaged into
/// respectability.
fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}
