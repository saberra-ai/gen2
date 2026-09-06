//! What the wrapper costs, measured against the thing it wraps — and written
//! down so a skeptic can reproduce the row.
//!
//! gen2 is a layer over llama.cpp. The question this answers is the one
//! nothing else in the repo asks: does going through the controller, the
//! session runtime and the token puller cost throughput, and how much?
//!
//! The reference is `llama-bench`, llama.cpp's own benchmark, **built from the
//! same llama.cpp commit the crate links** (`benches/build-llama-bench.sh`)
//! and run on the same file with the same shape of work:
//!
//! - **pp512** — 512 prompt tokens consumed, in batches. Through gen2 this
//!   also pays chat-template rendering, tokenisation and session setup, so it
//!   is reported for information and labelled as such.
//! - **tg128** — 128 tokens generated one at a time at batch 1, on the
//!   engine's own clock (prefill start to last token, the span llama-bench
//!   times). This is the number the headline ratio is built from. What the
//!   turn costs around that span — context creation, template, teardown — is
//!   reported separately in milliseconds, not folded into the rate.
//! - **TTFT** — gen2 only: prefill start to first token, from the engine's own
//!   `first_token_us`. llama-bench has no such number and none is invented.
//!
//! Protocol (docs/plans/research/04-benchmark-method.md): one warmup, five
//! repetitions, median ± sample standard deviation from the raw samples on
//! both sides, greedy decoding, batch 1; the reference's tg128 is run first
//! and last so a thermal drop shows up as an invalid run rather than a slow
//! wrapper.
//!
//! ```sh
//! benches/build-llama-bench.sh
//! PIO_TEST_MODEL=/path/model.gguf cargo bench --bench backend_parity
//! ```
//!
//! Environment:
//! - `PIO_TEST_MODEL` — the GGUF (required; prints a skip without it).
//! - `GEN2_LLAMA_BENCH` — reference binary (default `target/llama-bench/llama-bench`).
//! - `GEN2_BENCH_MACHINE` — machine slug for the result directory (default derived).
//! - `GEN2_BENCH_NOTES` — free text recorded in the result.
//! - `GEN2_BENCH_NO_WRITE=1` — measure and print, write nothing.
//!
//! Without the reference it prints gen2's numbers alone and writes nothing:
//! a row without its comparison is not a row. With a reference whose
//! `build_commit` is not the pinned commit it stops before measuring.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use gen2::Engine;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// Prompt tokens, `-p` in llama-bench.
const N_PROMPT: usize = 512;
/// Tokens to generate, `-n`.
const N_GEN: usize = 128;
/// Untimed passes before the measured ones. llama-bench's own default is one.
const WARMUP: usize = 1;
/// Measured repetitions, `-r`.
const REPS: usize = 5;
/// gen2's context window for the run. llama-bench sizes its own to
/// `n_prompt + n_gen`; ≥1024 here so the prompt and reply fit without any
/// context management kicking in.
const N_CTX: u32 = 2048;
/// gen2's prefill batch: `DEFAULT_BATCH_SIZE` in `src/backend/llama/session.rs`,
/// applied to both `n_batch` and `n_ubatch`. Passed to the reference so both
/// sides prefill in the same pieces.
const N_BATCH: u32 = 512;
/// CPU threads. gen2 leaves `n_threads` at llama.cpp's context default
/// (`llama_config.rs: threads: None` → `GGML_DEFAULT_N_THREADS` = 4);
/// llama-bench would otherwise pick the machine's math-core count.
const N_THREADS: u32 = 4;
/// Layers offloaded. gen2 leaves `n_gpu_layers` at llama.cpp's model default,
/// which is every layer on a GPU build; `99` is llama-bench's spelling of
/// the same thing for models this size.
const N_GPU_LAYERS: u32 = 99;
/// Pause between the two sides so the second is not measured while the first
/// is still giving the GPU back.
const SETTLE: Duration = Duration::from_secs(5);
/// llama-bench `--delay` between its tests: 5 s on mains, 30 s on a machine
/// with a battery (llama.cpp discussion #4167's laptop protocol).
const DELAY_MAINS_S: u32 = 5;
const DELAY_BATTERY_S: u32 = 30;
/// Reference tg128 last-vs-first drop beyond which the run is throttled.
const THERMAL_DROP_LIMIT: f64 = 0.05;
/// stddev/median beyond which a side was measuring the machine, not the code.
const SPREAD_LIMIT: f64 = 0.10;
/// Result file schema; `benches/results/README.md` documents it.
const SCHEMA_VERSION: u64 = 1;

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("backend parity: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let Some(model_path) = env::var("PIO_TEST_MODEL")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
    else {
        println!("backend parity: skipped — set PIO_TEST_MODEL to a GGUF");
        return Ok(ExitCode::SUCCESS);
    };
    if !model_path.is_file() {
        return Err(format!("{} is not a file", model_path.display()));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let no_write = env::var("GEN2_BENCH_NO_WRITE").is_ok_and(|v| v == "1");
    let started_at = utc_now();

    println!("gen2 against llama.cpp, on {}\n", model_path.display());

    // Everything about the run that is not a number, gathered before anything
    // heats the machine up.
    let machine = Machine::fingerprint();
    let model = ModelFile::inspect(&model_path)?;
    let gen2 = Gen2Build::inspect(&root);
    let load = concurrent_load();
    if !load.is_empty() {
        println!("note: concurrent load present — {}", load.join("; "));
    }
    let therm_before = machine.therm();

    // The reference. Missing → measure gen2 alone and write nothing. Wrong
    // commit → stop; a comparison against a different llama.cpp is what this
    // harness exists to prevent.
    let reference = match Reference::locate(&root) {
        Ok(r) => Some(r),
        Err(missing) => {
            println!("{missing}\n");
            None
        }
    };
    let delay_s = if machine.battery {
        DELAY_BATTERY_S
    } else {
        DELAY_MAINS_S
    };

    // Reference, first pass: the pp512/tg128 row, and the first half of the
    // thermal bracket.
    let mut ref_first = None;
    if let Some(reference) = &reference {
        println!(
            "reference: {} (build {})",
            reference.binary.display(),
            reference.build_label()
        );
        let rows = reference.run(&model_path, N_PROMPT, N_GEN, delay_s)?;
        let pp = rows.test(N_PROMPT, 0)?;
        let tg = rows.test(0, N_GEN)?;
        println!(
            "llama-bench  pp512 {:>8.1} ± {:<6.1} tok/s   tg128 {:>7.1} ± {:<5.1} tok/s  (n={})",
            pp.median,
            pp.stddev,
            tg.median,
            tg.stddev,
            tg.samples.len()
        );
        ref_first = Some(rows);
        std::thread::sleep(SETTLE);
    }

    // gen2, through the public API.
    let ours = measure_gen2(&model_path)?;
    println!(
        "gen2         pp{:<3} {:>8.1} ± {:<6.1} tok/s   tg{:<3} {:>7.1} ± {:<5.1} tok/s  (n={})   ttft {:.1} ± {:.1} ms",
        ours.prompt_tokens,
        ours.pp.median,
        ours.pp.stddev,
        ours.decode_tokens_min(),
        ours.tg.median,
        ours.tg.stddev,
        ours.tg.samples.len(),
        ours.ttft_ms.median,
        ours.ttft_ms.stddev
    );
    println!(
        "             on the caller's clock tg {:>7.1} ± {:<5.1} tok/s; the turn adds {:.1} ± {:.1} ms around the engine's span",
        ours.tg_wall.median, ours.tg_wall.stddev, ours.overhead_ms.median, ours.overhead_ms.stddev
    );

    // Reference, last pass: tg128 only, closing the bracket.
    let mut ref_last = None;
    if let Some(reference) = &reference {
        std::thread::sleep(SETTLE);
        let rows = reference.run(&model_path, 0, N_GEN, delay_s)?;
        let tg = rows.test(0, N_GEN)?;
        println!(
            "llama-bench  tg128 again {:>7.1} ± {:<5.1} tok/s  (thermal bracket)",
            tg.median, tg.stddev
        );
        ref_last = Some(rows);
    }
    let therm_after = machine.therm();
    let finished_at = utc_now();

    let Some((reference, first, last)) = reference
        .as_ref()
        .zip(ref_first)
        .zip(ref_last)
        .map(|((r, f), l)| (r, f, l))
    else {
        println!("\nno reference, so no ratio and no result file.");
        return Ok(ExitCode::SUCCESS);
    };
    let ref_pp = first.test(N_PROMPT, 0)?;
    let ref_tg = first.test(0, N_GEN)?;
    let ref_tg_last = last.test(0, N_GEN)?;

    // Validity: a thermal drop or a noisy side makes the run unpublishable.
    // The ratio itself never does; a bad ratio is the finding.
    let mut invalid: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let drop = 1.0 - ref_tg_last.median / ref_tg.median.max(f64::MIN_POSITIVE);
    if drop > THERMAL_DROP_LIMIT {
        invalid.push(format!(
            "throttled: reference tg128 fell {:.1}% between the first and last pass",
            drop * 100.0
        ));
    }
    for (who, s) in [("gen2", &ours.tg), ("reference", &ref_tg)] {
        let spread = s.stddev / s.median.max(f64::MIN_POSITIVE);
        if spread > SPREAD_LIMIT {
            invalid.push(format!(
                "noisy: {who} tg128 stddev is {:.1}% of its median",
                spread * 100.0
            ));
        }
    }
    let tg_ratio = ours.tg.median / ref_tg.median.max(f64::MIN_POSITIVE);
    let pp_ratio = ours.pp.median / ref_pp.median.max(f64::MIN_POSITIVE);
    if tg_ratio >= 1.05 {
        warnings.push(
            "gen2 decoded faster than the reference: the two sides were not given the \
             same work — check the settings before believing it"
                .into(),
        );
    }
    if !load.is_empty() {
        warnings.push(format!("concurrent load present: {}", load.join("; ")));
    }

    println!(
        "\ntg128: gen2 is {:.1}% of llama-bench   pp512: {:.1}% (includes chat template + session setup)   thermal drop {:.1}%",
        tg_ratio * 100.0,
        pp_ratio * 100.0,
        drop * 100.0
    );
    for w in &warnings {
        println!("warning: {w}");
    }
    for i in &invalid {
        println!("INVALID: {i}");
    }

    let notes = env::var("GEN2_BENCH_NOTES").unwrap_or_default();
    let result = json!({
        "schema_version": SCHEMA_VERSION,
        "valid": invalid.is_empty(),
        "reason": if invalid.is_empty() { Value::Null } else { Value::String(invalid.join("; ")) },
        "warnings": warnings,
        "notes": notes,
        "date": started_at.get(..10).unwrap_or(""),
        "started_at": started_at,
        "finished_at": finished_at,
        "protocol": {
            "n_prompt": N_PROMPT,
            "n_gen": N_GEN,
            "warmup": WARMUP,
            "repetitions": REPS,
            "sampling": "greedy",
            "batch": 1,
            "statistic": "median ± sample stddev (n-1) over the raw per-repetition samples, both sides",
            "delay_s": delay_s,
            "settle_s": SETTLE.as_secs(),
            "thermal_drop_limit": THERMAL_DROP_LIMIT,
            "spread_limit": SPREAD_LIMIT,
        },
        "gen2": {
            "commit": gen2.commit,
            "short": gen2.short,
            "dirty": gen2.dirty,
            "version": env!("CARGO_PKG_VERSION"),
            "features": gen2_features(),
            "rustc": gen2.rustc,
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        },
        "llama_cpp": {
            "commit": reference.pinned_commit,
            "build_commit": reference.build_commit,
            "build_number": reference.build_number,
            "llama_cpp_sys_2": reference.sys_version,
            "backends": first.backends(),
            "reference_binary": reference.binary.strip_prefix(&root).unwrap_or(&reference.binary).display().to_string(),
        },
        "machine": machine.to_json(&first, &therm_before, &therm_after),
        "model": model.to_json(&first),
        "settings": {
            "n_ctx_gen2": N_CTX,
            "n_batch": N_BATCH,
            "n_ubatch": N_BATCH,
            "n_threads": N_THREADS,
            "n_gpu_layers": N_GPU_LAYERS,
            "n_gpu_layers_gen2": "default (all layers on a GPU build)",
            "flash_attn": "auto",
            "type_k": "f16",
            "type_v": "f16",
            "use_mmap": true,
            "use_mlock_gen2": true,
            "gen2_prompt_tokens": ours.prompt_tokens,
            "gen2_decode_tokens": ours.decode_tokens,
            "gen2_finish": ours.finish,
            "gen2_prompt_note": "text through the chat template; token count is the engine's own, so pp is the rate over what was actually prefilled",
        },
        "runs": {
            "gen2": {
                "pp": ours.pp.to_json(),
                "tg": ours.tg.to_json(),
                "tg_note": "engine clock: (decode_tokens - 1) / (decode_tokens / avg_tps - first_token); the span llama-bench times",
                "tg_wall": ours.tg_wall.to_json(),
                "tg_wall_note": "caller clock: the same tokens over run() wall time after the first token; context creation and teardown included",
                "overhead_ms": ours.overhead_ms.to_json(),
                "overhead_note": "run() wall time minus the engine's prefill+decode span: what the wrapper adds per turn",
                "ttft_ms": ours.ttft_ms.to_json(),
                "wall_s": ours.wall_s,
            },
            "reference": {
                "pp": ref_pp.to_json(),
                "tg": ref_tg.to_json(),
                "command": first.command,
                "raw": first.rows,
            },
        },
        "ratio": {
            "tg": tg_ratio,
            "pp": pp_ratio,
            "pp_note": "includes chat template + session setup",
        },
        "thermal": {
            "ref_tg_first": ref_tg.to_json(),
            "ref_tg_last": ref_tg_last.to_json(),
            "ref_tg_last_command": last.command,
            "drop": drop,
            "therm_before": therm_before,
            "therm_after": therm_after,
        },
    });

    if no_write {
        println!("\nGEN2_BENCH_NO_WRITE set; not writing a result file.");
        return Ok(ExitCode::SUCCESS);
    }
    let dir = root.join("benches").join("results").join(&machine.slug);
    fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let file = dir.join(format!(
        "{}-{}-{}.json",
        started_at.get(..10).unwrap_or("undated"),
        gen2.short,
        model.slug
    ));
    let mut text = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
    text.push('\n');
    fs::write(&file, text).map_err(|e| format!("write {}: {e}", file.display()))?;
    println!(
        "\nwrote {}{}",
        file.strip_prefix(&root).unwrap_or(&file).display(),
        if invalid.is_empty() {
            ""
        } else {
            "  (valid: false — it will not enter the table)"
        }
    );
    println!("regenerate the README table with: cargo bench --bench bench_table -- --write");
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------
// gen2 side

struct Gen2Run {
    pp: Stat,
    /// Decode rate on the engine's clock: comparable to llama-bench's tg.
    tg: Stat,
    /// Decode rate on the caller's clock: what a host application sees.
    tg_wall: Stat,
    /// What `run()` took beyond the engine's own prefill+decode span.
    overhead_ms: Stat,
    ttft_ms: Stat,
    wall_s: Vec<f64>,
    prompt_tokens: u32,
    decode_tokens: Vec<u32>,
    finish: Vec<String>,
}

impl Gen2Run {
    fn decode_tokens_min(&self) -> u32 {
        self.decode_tokens.iter().copied().min().unwrap_or(0)
    }
}

/// Time gen2 doing the same work llama-bench times, through the public API
/// only: `Engine::load`, `infer`, greedy, a token cap.
///
/// Prefill and decode are separated with the token counts, first-token time
/// and average rate the engine reports, so the split is the engine's own
/// accounting over the same span llama-bench times (prefill start to last
/// token). The harness's wall clock around the whole call is kept beside it:
/// the difference is what the wrapper adds per turn — context creation,
/// template, tokenisation, teardown — and is reported as such rather than
/// smeared into a per-token rate.
fn measure_gen2(model: &Path) -> Result<Gen2Run, String> {
    let engine = Engine::builder()
        .model(model)
        .context(N_CTX)
        .build()
        .map_err(|e| format!("gen2 failed to load the model: {e}"))?;

    // Calibrate the prompt so that what the engine prefills — template
    // included — is N_PROMPT tokens, using the engine's own count. One-token
    // generations, so the calibration costs a prefill each.
    let mut words = N_PROMPT;
    let mut prompt_tokens = 0u32;
    for _ in 0..4 {
        let done = engine
            .infer(prompt_text(words, 0))
            .max_tokens(1)
            .greedy()
            .run()
            .map_err(|e| format!("gen2 calibration run failed: {e}"))?;
        let stats = done
            .stats
            .ok_or("gen2 reported no stats; the backend did not account for the run")?;
        prompt_tokens = stats.prompt_tokens;
        if prompt_tokens.abs_diff(N_PROMPT as u32) <= 2 || prompt_tokens == 0 {
            break;
        }
        let next = (words as f64 * N_PROMPT as f64 / prompt_tokens as f64).round() as usize;
        if next == words {
            break;
        }
        words = next.max(1);
    }

    let mut pp = Vec::new();
    let mut tg = Vec::new();
    let mut tg_wall = Vec::new();
    let mut overhead_ms = Vec::new();
    let mut ttft = Vec::new();
    let mut wall_s = Vec::new();
    let mut decode_tokens = Vec::new();
    let mut finish = Vec::new();
    let mut counted_prompt = prompt_tokens;

    for rep in 0..(WARMUP + REPS) {
        // A different first word per repetition, so no prefix cache can hand
        // a later run the earlier run's prefill.
        let prompt = prompt_text(words, rep);
        let started = Instant::now();
        let done = engine
            .infer(prompt)
            .max_tokens(N_GEN)
            .greedy()
            .run()
            .map_err(|e| format!("gen2 run {rep} failed: {e}"))?;
        let wall = started.elapsed().as_secs_f64();
        if rep < WARMUP {
            continue;
        }
        let stats = done.stats.ok_or("gen2 reported no stats")?;
        let ttft_s = stats.first_token_us as f64 / 1_000_000.0;
        if ttft_s <= 0.0 || stats.prompt_tokens == 0 {
            return Err("gen2 reported no first-token time; TTFT is required".into());
        }
        if stats.decode_tokens < 2 {
            return Err(format!(
                "gen2 generated {} token(s); the decode rate needs more than one",
                stats.decode_tokens
            ));
        }
        if stats.avg_tps <= 0.0 {
            return Err("gen2 reported no average rate; the engine's clock is required".into());
        }
        counted_prompt = stats.prompt_tokens;
        pp.push(stats.prompt_tokens as f64 / ttft_s);
        // The engine's own clock: `avg_tps` is `decode_tokens` over the time
        // from prefill start to the final stats, so this is prefill + decode
        // and nothing the wrapper does before or after — the same span
        // llama-bench times. The first token is charged to prefill.
        let engine_s = stats.decode_tokens as f64 / stats.avg_tps as f64;
        let generating = (engine_s - ttft_s).max(f64::MIN_POSITIVE);
        tg.push((stats.decode_tokens - 1) as f64 / generating);
        // The caller's clock: the same tokens over everything `run()` took
        // after the first token, context creation and teardown included.
        tg_wall.push((stats.decode_tokens - 1) as f64 / (wall - ttft_s).max(f64::MIN_POSITIVE));
        overhead_ms.push((wall - engine_s) * 1000.0);
        ttft.push(ttft_s * 1000.0);
        wall_s.push(wall);
        decode_tokens.push(stats.decode_tokens);
        finish.push(format!("{:?}", done.finish));
    }

    Ok(Gen2Run {
        pp: Stat::of(pp),
        tg: Stat::of(tg),
        tg_wall: Stat::of(tg_wall),
        overhead_ms: Stat::of(overhead_ms),
        ttft_ms: Stat::of(ttft),
        wall_s,
        prompt_tokens: counted_prompt,
        decode_tokens,
        finish,
    })
}

/// A deterministic prompt of roughly `words` words: plain English drawn by a
/// fixed-seed generator, sentence-shaped so the tokeniser sees prose rather
/// than one repeated token. `salt` changes only the opening word.
fn prompt_text(words: usize, salt: usize) -> String {
    const VOCAB: [&str; 64] = [
        "the",
        "river",
        "carried",
        "a",
        "small",
        "boat",
        "past",
        "old",
        "stone",
        "houses",
        "where",
        "children",
        "watched",
        "from",
        "windows",
        "and",
        "waved",
        "at",
        "every",
        "traveller",
        "who",
        "passed",
        "under",
        "bridge",
        "morning",
        "light",
        "fell",
        "across",
        "market",
        "square",
        "while",
        "vendors",
        "arranged",
        "baskets",
        "of",
        "apples",
        "bread",
        "fish",
        "on",
        "wooden",
        "tables",
        "nobody",
        "hurried",
        "because",
        "town",
        "kept",
        "its",
        "own",
        "slow",
        "time",
        "measured",
        "by",
        "bells",
        "tide",
        "rather",
        "than",
        "clocks",
        "later",
        "rain",
        "came",
        "softly",
        "over",
        "hills",
        "fields",
    ];
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut out = String::with_capacity(words * 7);
    out.push_str(&format!("Passage {}.", salt + 1));
    for i in 0..words {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let pick = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 58) as usize;
        let word = VOCAB[pick % VOCAB.len()];
        out.push(' ');
        if i % 13 == 0 {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                out.extend(first.to_uppercase());
                out.push_str(chars.as_str());
            }
        } else {
            out.push_str(word);
        }
        if i % 13 == 12 {
            out.push('.');
        }
    }
    out.push('.');
    out
}

fn gen2_features() -> Vec<&'static str> {
    let mut f = Vec::new();
    if cfg!(feature = "backend-llamacpp") {
        f.push("backend-llamacpp");
    }
    if cfg!(feature = "metal") {
        f.push("metal");
    }
    if cfg!(feature = "cuda") {
        f.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        f.push("vulkan");
    }
    f
}

// ---------------------------------------------------------------------------
// reference side

struct Reference {
    binary: PathBuf,
    sys_version: String,
    pinned_commit: String,
    build_commit: String,
    build_number: u64,
}

struct ReferenceRows {
    rows: Vec<Value>,
    command: String,
}

impl Reference {
    /// Find the reference binary and the pin the build script wrote, and
    /// refuse anything that does not match `Cargo.toml`. The `build_commit`
    /// the binary reports is checked on its first run.
    fn locate(root: &Path) -> Result<Self, String> {
        let binary = env::var("GEN2_LLAMA_BENCH")
            .ok()
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("target").join("llama-bench").join("llama-bench"));
        if !binary.is_file() {
            return Err(format!(
                "no reference at {}: run benches/build-llama-bench.sh (or set GEN2_LLAMA_BENCH) \
                 for the comparison",
                binary.display()
            ));
        }
        let pin_path = root
            .join("benches")
            .join("results")
            .join("llama-cpp-pin.txt");
        let pin = fs::read_to_string(&pin_path).map_err(|e| {
            format!(
                "no pin at {}: {e}; run benches/build-llama-bench.sh",
                pin_path.display()
            )
        })?;
        let field = |key: &str| -> Result<String, String> {
            pin.lines()
                .filter_map(|l| l.split_once('='))
                .find(|(k, _)| k.trim() == key)
                .map(|(_, v)| v.trim().to_string())
                .ok_or_else(|| format!("{} has no `{key}`", pin_path.display()))
        };
        let sys_version = field("llama_cpp_sys_2")?;
        let pinned_commit = field("llama_cpp_commit")?;
        let build_number = field("build_number")?
            .parse::<u64>()
            .map_err(|e| format!("bad build_number in pin: {e}"))?;
        let manifest = sys_version_in_manifest(root)?;
        if manifest != sys_version {
            return Err(format!(
                "Cargo.toml pins llama-cpp-sys-2 {manifest} but the reference was built for \
                 {sys_version}; rerun benches/build-llama-bench.sh"
            ));
        }
        Ok(Self {
            binary,
            sys_version,
            pinned_commit,
            build_commit: String::new(),
            build_number,
        })
    }

    fn build_label(&self) -> String {
        format!(
            "b{} {}",
            self.build_number,
            &self.pinned_commit[..8.min(self.pinned_commit.len())]
        )
    }

    /// Run llama-bench with the work gen2 is given, and read its JSON.
    fn run(
        &self,
        model: &Path,
        n_prompt: usize,
        n_gen: usize,
        delay_s: u32,
    ) -> Result<ReferenceRows, String> {
        let args: Vec<String> = [
            "-m",
            &model.display().to_string(),
            "-p",
            &n_prompt.to_string(),
            "-n",
            &n_gen.to_string(),
            "-r",
            &REPS.to_string(),
            "-b",
            &N_BATCH.to_string(),
            "-ub",
            &N_BATCH.to_string(),
            "-t",
            &N_THREADS.to_string(),
            "-ngl",
            &N_GPU_LAYERS.to_string(),
            "-fa",
            "auto",
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "--delay",
            &delay_s.to_string(),
            "-o",
            "json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let command = format!("{} {}", self.binary.display(), args.join(" "));
        let out = Command::new(&self.binary)
            .args(&args)
            .output()
            .map_err(|e| format!("could not run {}: {e}", self.binary.display()))?;
        if !out.status.success() {
            return Err(format!(
                "llama-bench exited with {}\n{}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        let rows: Vec<Value> = serde_json::from_slice(&out.stdout)
            .map_err(|e| format!("llama-bench -o json was not a JSON array: {e}"))?;
        if rows.is_empty() {
            return Err("llama-bench produced no rows".into());
        }
        for row in &rows {
            let commit = row["build_commit"].as_str().unwrap_or("");
            if commit.len() < 7 || !self.pinned_commit.starts_with(commit) {
                return Err(format!(
                    "reference build_commit `{commit}` is not the pinned llama.cpp commit \
                     {}; the comparison would be against a different llama.cpp. \
                     Rebuild with benches/build-llama-bench.sh",
                    self.pinned_commit
                ));
            }
            // llama-bench abbreviates backend names in this field (`MTL`
            // for Metal at b10405; older builds spelled it out).
            let expected_gpu: &[&str] = if cfg!(target_os = "macos") {
                &["Metal", "MTL"]
            } else if cfg!(feature = "cuda") {
                &["CUDA"]
            } else {
                &[]
            };
            let backends = row["backends"].as_str().unwrap_or("");
            if !expected_gpu.is_empty() && !expected_gpu.iter().any(|b| backends.contains(b)) {
                return Err(format!(
                    "reference backends `{backends}` lack {}, which gen2 links; \
                     the two sides would not run on the same device",
                    expected_gpu[0]
                ));
            }
        }
        Ok(ReferenceRows { rows, command })
    }
}

impl ReferenceRows {
    fn test(&self, n_prompt: usize, n_gen: usize) -> Result<Stat, String> {
        let row = self
            .rows
            .iter()
            .find(|r| {
                r["n_prompt"].as_u64() == Some(n_prompt as u64)
                    && r["n_gen"].as_u64() == Some(n_gen as u64)
            })
            .ok_or_else(|| format!("llama-bench has no pp{n_prompt}/tg{n_gen} row"))?;
        let samples: Vec<f64> = row["samples_ts"]
            .as_array()
            .ok_or("llama-bench row has no samples_ts")?
            .iter()
            .filter_map(Value::as_f64)
            .collect();
        if samples.len() != REPS {
            return Err(format!(
                "llama-bench reported {} samples, expected {REPS}",
                samples.len()
            ));
        }
        Ok(Stat::of(samples))
    }

    fn backends(&self) -> String {
        self.rows
            .first()
            .and_then(|r| r["backends"].as_str())
            .unwrap_or("")
            .to_string()
    }

    fn field(&self, key: &str) -> Value {
        self.rows
            .first()
            .map(|r| r[key].clone())
            .unwrap_or(Value::Null)
    }
}

/// The `llama-cpp-sys-2` version the manifest pins.
fn sys_version_in_manifest(root: &Path) -> Result<String, String> {
    let manifest = fs::read_to_string(root.join("Cargo.toml")).map_err(|e| e.to_string())?;
    manifest
        .lines()
        .find(|l| l.trim_start().starts_with("llama-cpp-sys-2"))
        .and_then(|l| l.split("version").nth(1))
        .and_then(|rest| rest.split('"').nth(1))
        .map(|v| v.trim_start_matches('=').to_string())
        .ok_or_else(|| "Cargo.toml has no llama-cpp-sys-2 version".into())
}

// ---------------------------------------------------------------------------
// statistics

struct Stat {
    samples: Vec<f64>,
    median: f64,
    stddev: f64,
}

impl Stat {
    /// Median and sample standard deviation (n−1), the way llama-bench's
    /// `stdev()` computes its spread, so the two sides are summarised alike.
    fn of(samples: Vec<f64>) -> Self {
        let median = median(&samples);
        let stddev = stddev(&samples);
        Self {
            samples,
            median,
            stddev,
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "samples_ts": self.samples,
            "median": self.median,
            "stddev": self.stddev,
            "n": self.samples.len(),
        })
    }
}

/// The median run: not the fastest, which flatters, and not the mean, which
/// one disturbed run on a shared GPU can drag anywhere.
fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn stddev(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n <= 1 {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt()
}

// ---------------------------------------------------------------------------
// fingerprints

struct Machine {
    slug: String,
    label: String,
    chip: String,
    cpu_cores: u64,
    perf_cores: Option<u64>,
    eff_cores: Option<u64>,
    ram_bytes: u64,
    os: String,
    power_source: String,
    battery: bool,
}

impl Machine {
    fn fingerprint() -> Self {
        let (chip, cpu_cores, perf_cores, eff_cores, ram_bytes, os) = if cfg!(target_os = "macos") {
            (
                sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into()),
                sysctl("hw.ncpu").and_then(|v| v.parse().ok()).unwrap_or(0),
                sysctl("hw.perflevel0.physicalcpu").and_then(|v| v.parse().ok()),
                sysctl("hw.perflevel1.physicalcpu").and_then(|v| v.parse().ok()),
                sysctl("hw.memsize")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                format!(
                    "macOS {} ({})",
                    sh("sw_vers", &["-productVersion"])
                        .unwrap_or_default()
                        .trim(),
                    sh("sw_vers", &["-buildVersion"]).unwrap_or_default().trim()
                ),
            )
        } else {
            let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            let chip = cpuinfo
                .lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "unknown".into());
            let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
            let ram_kb: u64 = meminfo
                .lines()
                .find(|l| l.starts_with("MemTotal"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let os = fs::read_to_string("/etc/os-release")
                .ok()
                .and_then(|s| {
                    s.lines().find(|l| l.starts_with("PRETTY_NAME=")).map(|l| {
                        l.trim_start_matches("PRETTY_NAME=")
                            .trim_matches('"')
                            .to_string()
                    })
                })
                .unwrap_or_else(|| env::consts::OS.into());
            (
                chip,
                std::thread::available_parallelism()
                    .map(|n| n.get() as u64)
                    .unwrap_or(0),
                None,
                None,
                ram_kb * 1024,
                os,
            )
        };
        let batt = sh("pmset", &["-g", "batt"]).unwrap_or_default();
        let battery = batt.contains("InternalBattery");
        let power_source = batt
            .lines()
            .find(|l| l.contains("drawing from"))
            .and_then(|l| l.split('\'').nth(1))
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                if cfg!(target_os = "macos") {
                    "unknown".into()
                } else {
                    "n/a".into()
                }
            });
        let ram_gb = ram_bytes / (1 << 30);
        let slug = env::var("GEN2_BENCH_MACHINE")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                let chip_slug: String = chip
                    .to_lowercase()
                    .replace("apple ", "")
                    .replace("(r)", "")
                    .replace("(tm)", "")
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric())
                    .collect();
                format!("{}-{}c-{}g", chip_slug, cpu_cores, ram_gb)
            });
        let label = format!("{chip}, {ram_gb} GB, {os}");
        Self {
            slug,
            label,
            chip,
            cpu_cores,
            perf_cores,
            eff_cores,
            ram_bytes,
            os,
            power_source,
            battery,
        }
    }

    /// `pmset -g therm` on macOS: the CPU speed limit is the throttling
    /// signal. Raw text, trimmed, so nothing is interpreted.
    fn therm(&self) -> Value {
        match sh("pmset", &["-g", "therm"]) {
            Some(t) => Value::String(t.trim().to_string()),
            None => Value::Null,
        }
    }

    fn to_json(
        &self,
        reference: &ReferenceRows,
        therm_before: &Value,
        therm_after: &Value,
    ) -> Value {
        json!({
            "slug": self.slug,
            "label": self.label,
            "chip": self.chip,
            "cpu_cores": self.cpu_cores,
            "perf_cores": self.perf_cores,
            "eff_cores": self.eff_cores,
            "ram_bytes": self.ram_bytes,
            "os": self.os,
            "cpu_info": reference.field("cpu_info"),
            "gpu_info": reference.field("gpu_info"),
            "power_source": self.power_source,
            "battery": self.battery,
            "therm_before": therm_before,
            "therm_after": therm_after,
        })
    }
}

struct ModelFile {
    path: PathBuf,
    file: String,
    slug: String,
    name: String,
    quant: String,
    bytes: u64,
    sha256: String,
}

impl ModelFile {
    fn inspect(path: &Path) -> Result<Self, String> {
        let file = path
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or("model path has no file name")?
            .to_string();
        let stem = file.strip_suffix(".gguf").unwrap_or(&file).to_string();
        // `Name-Q4_K_M` → the quant is the last dash-separated piece that
        // looks like one; everything before it is the model name.
        let (name, quant) = match stem.rsplit_once('-') {
            Some((n, q))
                if q.starts_with(|c: char| c.is_ascii_uppercase() || c.is_ascii_digit())
                    && q.contains(|c: char| c.is_ascii_digit()) =>
            {
                (n.to_string(), q.to_string())
            }
            _ => (stem.clone(), "unknown".to_string()),
        };
        let slug: String = stem
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        let bytes = fs::metadata(path).map_err(|e| e.to_string())?.len();
        let mut hasher = Sha256::new();
        let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = std::io::Read::read(&mut f, &mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let sha256 = hex::encode(hasher.finalize());
        Ok(Self {
            path: path.to_path_buf(),
            file,
            slug,
            name,
            quant,
            bytes,
            sha256,
        })
    }

    fn to_json(&self, reference: &ReferenceRows) -> Value {
        json!({
            "path": self.path.display().to_string(),
            "file": self.file,
            "name": self.name,
            "quant": self.quant,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "type": reference.field("model_type"),
            "n_params": reference.field("model_n_params"),
            "size_reported": reference.field("model_size"),
            "hf_repo": Value::Null,
        })
    }
}

struct Gen2Build {
    commit: String,
    short: String,
    dirty: bool,
    rustc: String,
}

impl Gen2Build {
    fn inspect(root: &Path) -> Self {
        let commit = git(root, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
        let short = commit.get(..7).unwrap_or("unknown").to_string();
        let dirty = git(root, &["status", "--porcelain", "--untracked-files=no"])
            .is_some_and(|s| !s.trim().is_empty());
        let rustc = sh("rustc", &["-V"]).unwrap_or_default().trim().to_string();
        Self {
            commit,
            short,
            dirty,
            rustc,
        }
    }
}

/// Other compilers or benchmarks on the machine while this ran. Recorded, not
/// acted on: the operator decides, the file says.
fn concurrent_load() -> Vec<String> {
    let Some(list) = sh("ps", &["-axo", "pid=,ppid=,command="]) else {
        return Vec::new();
    };
    let procs: Vec<(u32, u32, &str)> = list
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let pid = it.next()?.parse().ok()?;
            let ppid = it.next()?.parse().ok()?;
            let cmd = l.splitn(3, char::is_whitespace).nth(2).unwrap_or("").trim();
            Some((pid, ppid, cmd))
        })
        .collect();
    // This process, its ancestors (the shell and the `cargo bench` that
    // launched it) and its children are not "other" load.
    let me = std::process::id();
    let mut family = vec![me];
    let mut cur = me;
    while let Some(&(_, ppid, _)) = procs.iter().find(|(pid, _, _)| *pid == cur) {
        if ppid <= 1 || family.contains(&ppid) {
            break;
        }
        family.push(ppid);
        cur = ppid;
    }
    let interesting = [
        "cargo build",
        "cargo test",
        "cargo bench",
        "rustc ",
        "clang",
        "cmake",
        "llama-bench",
        "ollama run",
    ];
    procs
        .iter()
        .filter(|(pid, ppid, cmd)| {
            !family.contains(pid)
                && !family.contains(ppid)
                && interesting.iter().any(|k| cmd.contains(k))
        })
        .map(|(_, _, cmd)| cmd.chars().take(120).collect::<String>())
        .collect()
}

fn utc_now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn sysctl(key: &str) -> Option<String> {
    sh("sysctl", &["-n", key]).map(|s| s.trim().to_string())
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn sh(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}
