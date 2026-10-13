//! Matched PCM duration sweep and repeated-request residency measurements.
use gen2::{Runtime, speech::*};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() == 4,
        "usage: speech_benchmark bark|whistle BUNDLE BASE.f32 OUTPUT.json"
    );
    let source = match args[0].as_str() {
        "bark" => SpeechSource::bark(&args[1]),
        "whistle" => SpeechSource::whistle(&args[1]),
        _ => anyhow::bail!("choose bark or whistle"),
    };
    let bytes = std::fs::read(&args[2])?;
    anyhow::ensure!(bytes.len().is_multiple_of(4), "incomplete float32 PCM");
    let base = Audio::mono_16khz(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    )?;
    let pid = sysinfo::get_current_pid().map_err(|e| anyhow::anyhow!(e))?;
    let memory = || {
        let mut system = sysinfo::System::new();
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        system.process(pid).map_or(0, |p| p.memory())
    };
    let baseline = memory();
    let peak = Arc::new(AtomicU64::new(baseline));
    let stop = Arc::new(AtomicBool::new(false));
    let (sample_peak, sample_stop) = (peak.clone(), stop.clone());
    let sampler = std::thread::spawn(move || {
        let mut system = sysinfo::System::new();
        while !sample_stop.load(Ordering::Acquire) {
            system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
            if let Some(p) = system.process(pid) {
                sample_peak.fetch_max(p.memory(), Ordering::Relaxed);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    });
    let runtime = Runtime::builder().resident_memory_budget_mb(8192).build()?;
    let load_start = Instant::now();
    let speech = runtime.load_speech(
        source,
        LoadOptions {
            native_library: Some(std::env::var("GEN2_ORT_LIBRARY")?.into()),
            reservation_mb: if args[0] == "whistle" { 2048 } else { 4096 },
            ..Default::default()
        },
    )?;
    let cold_load_micros = load_start.elapsed().as_micros();
    let loaded_rss = memory();
    let infer = |audio: Audio| {
        let mut request = TranscriptionRequest::new(audio);
        if let Some(limit) = speech.profile().maximum_text_tokens {
            request = request.max_tokens(limit);
        }
        speech.transcribe_request(request)
    };
    let mut sweep = Vec::new();
    for seconds in [1usize, 5, 15, 30, 60] {
        let audio = Audio::mono_16khz(
            base.samples()
                .iter()
                .copied()
                .cycle()
                .take(seconds * 16000)
                .collect(),
        )?;
        let expected = infer(audio.clone())?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            let result = infer(audio.clone())?;
            anyhow::ensure!(
                result.transcription.token_ids == expected.transcription.token_ids,
                "duration state contamination"
            );
            samples.push(serde_json::json!({"wall_micros":started.elapsed().as_micros(),"stages":result.transcription}));
        }
        sweep.push(serde_json::json!({"seconds":seconds,"language":expected.language,"text":expected.transcription.text,"token_ids":expected.transcription.token_ids,"samples":samples}));
    }
    let expected = infer(base.clone())?;
    let mut soak = Vec::new();
    let soak_start = memory();
    for _ in 0..100 {
        let actual = infer(base.clone())?;
        anyhow::ensure!(
            actual.transcription.token_ids == expected.transcription.token_ids,
            "soak state contamination"
        );
        soak.push(memory());
    }
    let before_shutdown = memory();
    speech.shutdown();
    anyhow::ensure!(runtime.speech_reserved_mb() == 0, "reservation leak");
    stop.store(true, Ordering::Release);
    sampler
        .join()
        .map_err(|_| anyhow::anyhow!("memory sampler failed"))?;
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()?;
    let system = sysinfo::System::new_all();
    let library = std::env::var("GEN2_ORT_LIBRARY")?;
    let native_hash = hex::encode(Sha256::digest(std::fs::read(&library)?));
    let receipt = serde_json::json!({"os":sysinfo::System::long_os_version(),"architecture":std::env::consts::ARCH,"cpu":system.cpus().first().map(|c|c.brand()),"native_library_sha256":native_hash,"native_library":library,"git_sha":String::from_utf8_lossy(&sha.stdout).trim(),"model":args[0],"model_revision":speech.bundle().revision(),"manifest_sha256":speech.bundle().manifest_sha256(),"provider":"CPUExecutionProvider","intra_threads":4,"inter_threads":1,"cold_load_micros":cold_load_micros,"baseline_rss":baseline,"loaded_rss":loaded_rss,"peak_rss":peak.load(Ordering::Relaxed),"soak_start_rss":soak_start,"before_shutdown_rss":before_shutdown,"after_shutdown_rss":memory(),"soak_rss":soak,"sweep":sweep});
    std::fs::write(&args[3], serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}
