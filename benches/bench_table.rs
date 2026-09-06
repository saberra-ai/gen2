//! The README benchmark table, generated from `benches/results/**/*.json`.
//!
//! Humans never edit the table. Every row comes from a result file written by
//! `benches/backend_parity.rs`, so a number in the README can be traced to
//! the raw samples, the machine, the model hash and the llama.cpp commit that
//! produced it. Output is byte-deterministic — rows sorted, fixed formatting
//! — so a CI job can regenerate it and diff.
//!
//! ```sh
//! cargo bench --bench bench_table                    # print the table
//! cargo bench --bench bench_table -- --write         # rewrite README between the markers
//! cargo bench --bench bench_table -- --check         # exit 1 if the README differs
//! cargo bench --bench bench_table -- --check --max-age-days 90
//! ```
//!
//! It refuses to render when `benches/results/llama-cpp-pin.txt` disagrees
//! with the `llama-cpp-sys-2` version in `Cargo.toml`, when a valid result was
//! measured against a different llama.cpp commit than the pin, or when a
//! result's `schema_version` is one it does not know — each of those is the
//! table rotting, and rot is a failure, not a footnote.
//!
//! No dependency beyond what the crate already has (`serde_json`); same
//! stance as `hot_paths.rs` on criterion.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::Value;

const BEGIN: &str = "<!-- bench:begin -->";
const END: &str = "<!-- bench:end -->";
const KNOWN_SCHEMA: u64 = 1;

fn main() -> ExitCode {
    let mut write = false;
    let mut check = false;
    let mut max_age_days: Option<i64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--write" => write = true,
            "--check" => check = true,
            "--max-age-days" => {
                max_age_days = args.next().and_then(|v| v.parse().ok());
                if max_age_days.is_none() {
                    eprintln!("bench-table: --max-age-days needs a number");
                    return ExitCode::FAILURE;
                }
            }
            // `cargo bench` passes `--bench`; anything else libtest-shaped is
            // ignored the same way.
            _ => {}
        }
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    match run(&root, write, check, max_age_days) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bench-table: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(root: &Path, write: bool, check: bool, max_age_days: Option<i64>) -> Result<(), String> {
    let results_dir = root.join("benches").join("results");
    let pin = read_pin(&results_dir.join("llama-cpp-pin.txt"))?;
    let manifest_version = sys_version_in_manifest(root)?;
    if pin.sys_version != manifest_version {
        return Err(format!(
            "llama-cpp-pin.txt says llama-cpp-sys-2 {} but Cargo.toml pins {}; rerun \
             benches/build-llama-bench.sh and the benchmarks",
            pin.sys_version, manifest_version
        ));
    }

    let results = load_results(&results_dir)?;
    let mut rows = Vec::new();
    let mut invalid = 0usize;
    for r in &results {
        if !r.valid {
            invalid += 1;
            continue;
        }
        if r.llama_cpp_commit != pin.commit {
            return Err(format!(
                "{} is valid but was measured against llama.cpp {} while the pin is {}; \
                 the pin moved without a rerun",
                r.file.display(),
                r.llama_cpp_commit,
                pin.commit
            ));
        }
        rows.push(r);
    }
    rows.sort_by_key(|a| a.sort_key());

    if let Some(limit) = max_age_days {
        let today = today_days()?;
        let newest = rows.iter().filter_map(|r| days_from_civil(&r.date)).max();
        match newest {
            Some(d) if today - d <= limit => {}
            Some(d) => {
                return Err(format!(
                    "the newest valid result is {} days old; the limit is {limit}",
                    today - d
                ));
            }
            None => return Err("no valid result to check the age of".into()),
        }
    }

    let table = render(&rows, invalid);
    let readme_path = root.join("README.md");
    let readme = fs::read_to_string(&readme_path).map_err(|e| format!("README.md: {e}"))?;
    let updated = splice(&readme, &table)?;

    if check {
        if updated != readme {
            return Err(
                "README.md benchmark table differs from the generated one; run \
                 `cargo bench --bench bench_table -- --write` and commit"
                    .into(),
            );
        }
        println!("bench-table: README.md is current ({} rows)", rows.len());
        return Ok(());
    }
    if write {
        if updated != readme {
            fs::write(&readme_path, updated).map_err(|e| format!("write README.md: {e}"))?;
            println!("bench-table: README.md rewritten ({} rows)", rows.len());
        } else {
            println!("bench-table: README.md unchanged ({} rows)", rows.len());
        }
        return Ok(());
    }
    print!("{table}");
    Ok(())
}

// ---------------------------------------------------------------------------
// inputs

struct Pin {
    sys_version: String,
    commit: String,
}

fn read_pin(path: &Path) -> Result<Pin, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("{}: {e}; run benches/build-llama-bench.sh", path.display()))?;
    let field = |key: &str| {
        text.lines()
            .filter_map(|l| l.split_once('='))
            .find(|(k, _)| k.trim() == key)
            .map(|(_, v)| v.trim().to_string())
            .ok_or_else(|| format!("{} has no `{key}`", path.display()))
    };
    Ok(Pin {
        sys_version: field("llama_cpp_sys_2")?,
        commit: field("llama_cpp_commit")?,
    })
}

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

struct ResultRow {
    file: PathBuf,
    valid: bool,
    date: String,
    gen2_short: String,
    machine_slug: String,
    machine_label: String,
    model_name: String,
    quant: String,
    llama_cpp_commit: String,
    build_number: u64,
    gen2_tg: (f64, f64),
    ref_tg: (f64, f64),
    gen2_pp: (f64, f64),
    ref_pp: (f64, f64),
    ttft_ms: (f64, f64),
    n: u64,
}

impl ResultRow {
    fn sort_key(&self) -> (String, String, String, String, String) {
        (
            self.machine_slug.clone(),
            self.model_name.to_lowercase(),
            self.quant.clone(),
            self.date.clone(),
            self.gen2_short.clone(),
        )
    }
}

fn load_results(dir: &Path) -> Result<Vec<ResultRow>, String> {
    let mut files = Vec::new();
    collect_json(dir, &mut files)?;
    files.sort();
    files.into_iter().map(|f| parse_result(&f)).collect()
}

fn collect_json(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            collect_json(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "json") {
            out.push(path);
        }
    }
    Ok(())
}

fn parse_result(path: &Path) -> Result<ResultRow, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let schema = v["schema_version"].as_u64().unwrap_or(0);
    if schema != KNOWN_SCHEMA {
        return Err(format!(
            "{}: schema_version {schema} is unknown to this generator (knows {KNOWN_SCHEMA})",
            path.display()
        ));
    }
    let stat = |v: &Value| -> Result<(f64, f64), String> {
        Ok((
            v["median"].as_f64().ok_or("missing median")?,
            v["stddev"].as_f64().ok_or("missing stddev")?,
        ))
    };
    let str_at = |v: &Value, what: &str| -> Result<String, String> {
        v.as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{}: missing {what}", path.display()))
    };
    Ok(ResultRow {
        file: path.to_path_buf(),
        valid: v["valid"].as_bool().unwrap_or(false),
        date: str_at(&v["date"], "date")?,
        gen2_short: str_at(&v["gen2"]["short"], "gen2.short")?,
        machine_slug: str_at(&v["machine"]["slug"], "machine.slug")?,
        machine_label: str_at(&v["machine"]["label"], "machine.label")?,
        model_name: str_at(&v["model"]["name"], "model.name")?,
        quant: str_at(&v["model"]["quant"], "model.quant")?,
        llama_cpp_commit: str_at(&v["llama_cpp"]["commit"], "llama_cpp.commit")?,
        build_number: v["llama_cpp"]["build_number"].as_u64().unwrap_or(0),
        gen2_tg: stat(&v["runs"]["gen2"]["tg"])
            .map_err(|e| format!("{}: gen2.tg {e}", path.display()))?,
        ref_tg: stat(&v["runs"]["reference"]["tg"])
            .map_err(|e| format!("{}: reference.tg {e}", path.display()))?,
        gen2_pp: stat(&v["runs"]["gen2"]["pp"])
            .map_err(|e| format!("{}: gen2.pp {e}", path.display()))?,
        ref_pp: stat(&v["runs"]["reference"]["pp"])
            .map_err(|e| format!("{}: reference.pp {e}", path.display()))?,
        ttft_ms: stat(&v["runs"]["gen2"]["ttft_ms"])
            .map_err(|e| format!("{}: gen2.ttft_ms {e}", path.display()))?,
        n: v["runs"]["gen2"]["tg"]["n"].as_u64().unwrap_or(0),
    })
}

// ---------------------------------------------------------------------------
// output

fn render(rows: &[&ResultRow], invalid: usize) -> String {
    let mut out = String::new();
    if rows.is_empty() {
        out.push_str("_No valid results yet. Run `benches/build-llama-bench.sh`, then `PIO_TEST_MODEL=… cargo bench --bench backend_parity`._\n");
    } else {
        out.push_str(
            "| Model | Machine | tg128 gen2 (tok/s) | tg128 llama-bench (tok/s) | tg128 ratio | pp512 gen2 (tok/s) † | pp512 llama-bench (tok/s) | pp512 ratio † | TTFT (ms) | llama.cpp | Date |\n",
        );
        out.push_str(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n",
        );
        // The same machine label once per slug, so a renamed machine cannot
        // produce two spellings in one table.
        let mut labels: BTreeMap<&str, &str> = BTreeMap::new();
        for r in rows {
            labels.entry(&r.machine_slug).or_insert(&r.machine_label);
        }
        for r in rows {
            let tg_ratio = r.gen2_tg.0 / r.ref_tg.0.max(f64::MIN_POSITIVE);
            let pp_ratio = r.gen2_pp.0 / r.ref_pp.0.max(f64::MIN_POSITIVE);
            out.push_str(&format!(
                "| {} {} | {} | {} | {} | {:.2} | {} | {} | {:.2} | {} | b{} `{}` | {} |\n",
                r.model_name,
                r.quant,
                labels[r.machine_slug.as_str()],
                pm(r.gen2_tg, 1),
                pm(r.ref_tg, 1),
                tg_ratio,
                pm(r.gen2_pp, 0),
                pm(r.ref_pp, 0),
                pp_ratio,
                pm(r.ttft_ms, 1),
                r.build_number,
                &r.llama_cpp_commit[..8.min(r.llama_cpp_commit.len())],
                r.date,
            ));
        }
        let n = rows.iter().map(|r| r.n).max().unwrap_or(0);
        out.push('\n');
        out.push_str(&format!(
            "Median ± sample standard deviation over n={n} repetitions after one warmup, greedy, batch 1, \
             both sides on the same GGUF and the same llama.cpp commit. Ratio is gen2 ÷ llama-bench. \
             † pp512 through gen2 includes chat template + session setup, so it is reported, not targeted. \
             TTFT is gen2's prefill-start-to-first-token; llama-bench has no equivalent. \
             Raw samples, machine fingerprint and model hash: `benches/results/`.\n"
        ));
    }
    if invalid > 0 {
        out.push_str(&format!(
            "\n{invalid} result file{} marked `valid: false` (throttled or noisy) {} not shown.\n",
            if invalid == 1 { "" } else { "s" },
            if invalid == 1 { "is" } else { "are" }
        ));
    }
    out
}

/// `median ± stddev` with a fixed number of decimals, so the table is stable
/// byte-for-byte across runs of the generator.
fn pm((median, stddev): (f64, f64), decimals: usize) -> String {
    format!("{median:.decimals$} ± {stddev:.decimals$}")
}

/// Replace what lies between the markers; add the section when there is none.
fn splice(readme: &str, table: &str) -> Result<String, String> {
    let block = format!("{BEGIN}\n{table}{END}");
    if let (Some(b), Some(e)) = (readme.find(BEGIN), readme.find(END)) {
        if e < b {
            return Err("README.md has bench:end before bench:begin".into());
        }
        let after = e + END.len();
        return Ok(format!("{}{}{}", &readme[..b], block, &readme[after..]));
    }
    if readme.contains(BEGIN) || readme.contains(END) {
        return Err("README.md has one bench marker but not the other".into());
    }
    let anchor = "\n## Backends\n";
    let at = readme
        .find(anchor)
        .ok_or("README.md has no `## Backends` section to put the benchmarks before")?;
    let section = format!(
        "\n## Benchmarks\n\ngen2 against `llama-bench` built from the same llama.cpp commit the crate links, \
         on the same file; the method, the result schema and how to add a machine are in \
         [benches/results/README.md](benches/results/README.md).\n\n{block}\n"
    );
    Ok(format!("{}{}{}", &readme[..at], section, &readme[at..]))
}

// ---------------------------------------------------------------------------
// dates, without a dependency

fn today_days() -> Result<i64, String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs() as i64;
    Ok(secs.div_euclid(86_400))
}

/// Days since 1970-01-01 for a `YYYY-MM-DD` string (Howard Hinnant's civil
/// algorithm), `None` if it is not one.
fn days_from_civil(date: &str) -> Option<i64> {
    let mut parts = date.split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}
