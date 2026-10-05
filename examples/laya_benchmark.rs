//! BUNDLE ORT_LIBRARY PLAN_JSON OUTPUT_JSONL; use a release build for qualification.
use gen2::{Runtime, decision::*};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs::OpenOptions, io::Write, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Plan {
    schema: String,
    warmups: usize,
    runs: usize,
    threads: usize,
    host: serde_json::Value,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    request: DecisionRequest,
    encode: EncodeOptions,
}
fn peak_bytes() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::{
            ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
            Threading::GetCurrentProcess,
        };
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        counters.cb = std::mem::size_of_val(&counters) as u32;
        // The current process pseudo-handle and initialized writable structure
        // remain valid throughout this synchronous Win32 call.
        let size = counters.cb;
        if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) } != 0 {
            return Some(counters.PeakWorkingSetSize as u64);
        }
        None
    }
    #[cfg(unix)]
    {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0 {
            return Some(
                usage.ru_maxrss as u64
                    * if cfg!(target_vendor = "apple") {
                        1
                    } else {
                        1024
                    },
            );
        }
        None
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        None
    }
}
fn emit(file: &mut std::fs::File, value: serde_json::Value) -> std::io::Result<()> {
    serde_json::to_writer(&mut *file, &value)?;
    writeln!(file)?;
    file.flush()
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 4 {
        return Err("usage: laya_benchmark BUNDLE ORT_LIBRARY PLAN_JSON OUTPUT_JSONL".into());
    }
    let bytes = std::fs::read(&args[2])?;
    let plan: Plan = serde_json::from_slice(&bytes)?;
    if plan.schema != "gen2-laya-benchmark/v1"
        || plan.runs == 0
        || plan.threads == 0
        || plan.cases.is_empty()
    {
        return Err("invalid benchmark plan".into());
    }
    let manifest: BundleManifest =
        serde_json::from_slice(&std::fs::read(args[0].join("manifest.json"))?)?;
    // create_new prevents accidentally overwriting an earlier experiment.
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(manifest.envelope.reservation_mb)
        .build()?;
    let start = Instant::now();
    let model = runtime.load_decider(
        &args[0],
        LoadOptions {
            native_library: Some(args[1].clone()),
            intra_threads: plan.threads,
            ..Default::default()
        },
    )?;
    let load_seconds = start.elapsed().as_secs_f64();
    emit(
        &mut output,
        json!({"type":"host", "schema":plan.schema, "engine":"rust",
        "release_build":!cfg!(debug_assertions), "host":plan.host,
        "plan_sha256":hex::encode(Sha256::digest(&bytes)),
        "manifest_sha256":LayaBundle::open(&args[0])?.manifest_sha256(),
        "native_library_sha256":hex::encode(Sha256::digest(std::fs::read(&args[1])?)),
        "runtime_build":ort::info(), "checkpoint":manifest.checkpoint,
        "threads":plan.threads, "warmups":plan.warmups, "runs":plan.runs,
        "load_seconds":load_seconds, "peak_process_bytes":peak_bytes(),
        "reservation_mb":manifest.envelope.reservation_mb}),
    )?;
    for case in plan.cases {
        let options = DecisionOptions {
            encode: Some(case.encode.clone()),
            ..Default::default()
        };
        let mut durations = Vec::with_capacity(plan.runs);
        let mut result = None;
        for run in 0..plan.warmups + plan.runs {
            let start = Instant::now();
            let answer = model.decide(case.request.clone(), options.clone())?;
            if run >= plan.warmups {
                durations.push(start.elapsed().as_secs_f64());
            }
            result = Some(answer);
            if (run + 1) % 10 == 0 {
                eprintln!(
                    "{}: {}/{} calls",
                    case.name,
                    run + 1,
                    plan.warmups + plan.runs
                );
            }
        }
        let result = result.ok_or("empty experiment")?;
        emit(
            &mut output,
            json!({"type":"case", "name":case.name,
            "request":case.request, "encode":case.encode, "result":result,
            "durations_seconds":durations, "peak_process_bytes":peak_bytes()}),
        )?;
        eprintln!("completed {}", case.name);
    }
    model.shutdown();
    emit(
        &mut output,
        json!({"type":"complete", "peak_process_bytes":peak_bytes(),
        "decision_reserved_mb_after_shutdown":runtime.decision_reserved_mb()}),
    )?;
    Ok(())
}
