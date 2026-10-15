//! Controlled warm duration sweep and a 100-request memory/state soak.
use gen2::{
    Runtime,
    speech::{Audio, LoadOptions, TranscribeOptions},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: bark_benchmark BUNDLE BASE_WAV OUTPUT_JSON".into());
    }
    let library = std::env::var("GEN2_ORT_LIBRARY")?;
    let pid = sysinfo::get_current_pid()?;
    let memory = || {
        let mut s = sysinfo::System::new();
        s.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        s.process(pid).map_or(0, |p| p.memory())
    };
    let baseline = memory();
    let peak = Arc::new(AtomicU64::new(baseline));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler_peak = peak.clone();
    let sampler_stop = stop.clone();
    let sampler = std::thread::spawn(move || {
        let mut system = sysinfo::System::new();
        while !sampler_stop.load(Ordering::Acquire) {
            system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
            if let Some(p) = system.process(pid) {
                sampler_peak.fetch_max(p.memory(), Ordering::Relaxed);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    });
    let mut settings = gen2::advanced::generation::Settings::default();
    settings.system.gpu_layers = Some(0);
    settings.system.threads = Some(4);
    settings.system.threads_batch = Some(4);
    settings.system.ctx_size = Some(2048);
    let runtime = Runtime::builder()
        .settings(settings)
        .resident_memory_budget_mb(8192)
        .build()?;
    let chat_path = std::env::var("GEN2_BARK_CHAT").ok();
    let chat = chat_path
        .as_ref()
        .map(|path| runtime.load(path))
        .transpose()?;
    let chat_before = chat
        .as_ref()
        .map(|m| {
            m.generate("Reply with one word: hello")
                .max_tokens(24)
                .temperature(0.)
                .text()
        })
        .transpose()?;
    let laya = if chat.is_some() {
        Some(runtime.load_decider(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke"),
            gen2::decision::LoadOptions {
                native_library: Some(library.clone().into()),
                ..Default::default()
            },
        )?)
    } else {
        None
    };
    let mixed_residency = runtime.stats();
    let started = Instant::now();
    let speech = runtime.load_transcriber(
        &args[0],
        LoadOptions {
            native_library: Some(library.clone().into()),
            ..Default::default()
        },
    )?;
    let cold_load_micros = started.elapsed().as_micros();
    if let Some(chat) = &chat {
        assert!(
            runtime.residency().is_resident(chat.id()),
            "speech admission evicted the mixed-residency chat"
        );
        assert_eq!(runtime.decision_reserved_mb(), 1);
    }
    let simultaneous_residency = runtime.stats();
    let base = Audio::from_wav(&args[1])?;
    let mut sweep = Vec::new();
    for seconds in [1, 5, 15, 30, 60] {
        let wave: Vec<f32> = base
            .samples()
            .iter()
            .copied()
            .cycle()
            .take(seconds * 16000)
            .collect();
        let audio = Audio::mono_16khz(wave)?;
        let expected = speech.transcribe(audio.clone(), TranscribeOptions::default())?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            let result = speech.transcribe(audio.clone(), TranscribeOptions::default())?;
            assert_eq!(
                result.token_ids, expected.token_ids,
                "duration state contamination"
            );
            samples.push(serde_json::json!({"wall_micros":started.elapsed().as_micros(),"frontend_micros":result.frontend_micros,"encoder_micros":result.encoder_micros,"decoder_micros":result.decoder_micros,"queue_micros":result.queue_micros}));
        }
        sweep.push(serde_json::json!({"seconds":seconds,"text":expected.fulltext,"token_ids":expected.token_ids,"samples":samples}));
    }
    let expected = speech.transcribe(base.clone(), TranscribeOptions::default())?;
    let soak_start = memory();
    let mut soak_memory = Vec::new();
    for _ in 0..100 {
        let result = speech.transcribe(base.clone(), TranscribeOptions::default())?;
        assert_eq!(
            result.token_ids, expected.token_ids,
            "soak state contamination"
        );
        soak_memory.push(memory());
    }
    let before_shutdown = memory();
    if let Some(chat) = &chat {
        assert!(
            runtime.residency().is_resident(chat.id()),
            "chat was not resident throughout the speech sweep/soak"
        );
    }
    let chat_after = chat
        .as_ref()
        .map(|m| {
            m.generate("Reply with one word: hello")
                .max_tokens(24)
                .temperature(0.)
                .text()
        })
        .transpose()?;
    assert_eq!(
        chat_before, chat_after,
        "chat generation changed across speech residency"
    );
    if chat.is_some() {
        assert!(matches!(
            runtime.load_transcriber(
                &args[0],
                LoadOptions {
                    native_library: Some(library.into()),
                    ..Default::default()
                }
            ),
            Err(gen2::Error::Speech(
                gen2::speech::SpeechError::ResourceLimit(_)
            ))
        ));
        assert_eq!(runtime.speech_reserved_mb(), 4096);
    }
    if let Some(laya) = &laya {
        laya.decide(
            gen2::decision::DecisionRequest::text("a b")
                .question("q", gen2::decision::Question::yes_no("a?")),
            gen2::decision::DecisionOptions::default(),
        )?;
        laya.shutdown();
    }
    if let Some(chat) = &chat {
        runtime.evict(chat)?;
    }
    speech.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
    assert_eq!(runtime.stats().estimated_resident_mb, 0);
    stop.store(true, Ordering::Release);
    sampler.join().unwrap();
    let receipt = serde_json::json!({"git_sha":String::from_utf8(std::process::Command::new("git").args(["rev-parse","HEAD"]).output()?.stdout)?.trim(),"cpu":sysinfo::System::new_all().cpus().first().map(|c|c.brand()),"cold_load_micros":cold_load_micros,"process_baseline_bytes":baseline,"sampled_peak_rss_bytes":peak.load(Ordering::Relaxed),"soak_start_bytes":soak_start,"soak_memory_bytes":soak_memory,"before_shutdown_bytes":before_shutdown,"after_shutdown_bytes":memory(),"provider":"CPUExecutionProvider","intra_threads":4,"inter_threads":1,"model_revision":gen2::speech::MODEL_REVISION,"sweep":sweep,"mixed_residency":{"chat_path":chat_path,"chat_before":chat_before,"chat_after":chat_after,"decision_bundle":"synthetic Laya smoke when chat is present","estimated_before_speech_mb":mixed_residency.estimated_resident_mb,"simultaneous_estimated_mb":simultaneous_residency.estimated_resident_mb,"simultaneous_chat_models":simultaneous_residency.resident_models,"simultaneous_speech_mb":simultaneous_residency.speech_reserved_mb,"simultaneous_decision_mb":simultaneous_residency.decision_reserved_mb},"final_reserved_mb":runtime.stats().estimated_resident_mb});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}
