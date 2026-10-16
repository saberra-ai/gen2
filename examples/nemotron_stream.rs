//! Stream normalized WAV repeatedly to exercise cache continuity and long sessions.
use gen2::{
    Runtime,
    diarization::{
        DiarizationError, DiarizeOptions, LoadOptions, MAX_PUSH_SAMPLES, MAX_SESSION_SAMPLES,
    },
};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().collect();
    let metal = args.get(1).is_some_and(|arg| arg == "--metal");
    if metal {
        args.remove(1);
    }
    if !(5..=6).contains(&args.len()) {
        return Err(
            "usage: nemotron_stream [--metal] MODEL_DIRECTORY NATIVE_C_LIBRARY AUDIO.wav REPEATS [PROBABILITIES.f32]"
                .into(),
        );
    }
    let repeats: u32 = args[4].parse()?;
    let reader = hound::WavReader::open(&args[3])?;
    let spec = reader.spec();
    if spec.sample_rate != 16000
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
        || repeats == 0
        || u64::from(reader.duration()) * u64::from(repeats) > MAX_SESSION_SAMPLES as u64
    {
        return Err(
            "use mono 16 kHz PCM16 WAV and positive repeats totaling at most one hour".into(),
        );
    }
    drop(reader);
    let runtime = Runtime::builder().resident_memory_budget_mb(2048).build()?;
    let load_started = Instant::now();
    let model = runtime.load_diarizer(
        &args[1],
        if metal {
            LoadOptions::metal(&args[2])
        } else {
            LoadOptions::cpu(&args[2])
        },
    )?;
    let cold_load_micros = load_started.elapsed().as_micros();
    let started = Instant::now();
    let mut stream = model.start_stream(DiarizeOptions {
        frame_probabilities: true,
        ..Default::default()
    })?;
    let mut backpressure_count = 0u64;
    let mut snapshots = Vec::new();
    let mut latest_revision = 0;
    for _ in 0..repeats {
        let mut reader = hound::WavReader::open(&args[3])?;
        let mut samples = reader.samples::<i16>();
        loop {
            // A Result iterator does not promise an exact allocation size;
            // collect can grow capacity past the public per-push admission cap.
            let mut block = Vec::with_capacity(MAX_PUSH_SAMPLES);
            for sample in samples.by_ref().take(MAX_PUSH_SAMPLES) {
                block.push(sample? as f32 / 32768.);
            }
            if block.is_empty() {
                break;
            }
            loop {
                match stream.try_push(block, 16000) {
                    Ok(()) => break,
                    Err(rejected) if matches!(rejected.error, DiarizationError::Busy) => {
                        block = rejected.samples;
                        backpressure_count += 1;
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(rejected) => return Err(rejected.error.into()),
                }
            }
            if let Some(snapshot) = stream.snapshot()
                && snapshot.revision > latest_revision
            {
                latest_revision = snapshot.revision;
                snapshots.push(serde_json::json!({"revision": snapshot.revision, "processed_audio_seconds": snapshot.processed_audio_seconds, "turn_count": snapshot.result.turns.len(), "wall_micros": started.elapsed().as_micros()}));
            }
        }
    }
    let before_finish = Instant::now();
    let result = stream.finish()?;
    let finish_micros = before_finish.elapsed().as_micros();
    let elapsed_micros = started.elapsed().as_micros();
    let retained = result
        .frame_probabilities
        .as_ref()
        .map(|f| serde_json::json!({"first_frame": f.first_frame, "rows": f.values.len()}));
    // Export the already bounded native tail separately from the JSON receipt.
    let probability_sha256 = if let Some(path) = args.get(5) {
        let frames = result
            .frame_probabilities
            .as_ref()
            .ok_or("missing diagnostic probabilities")?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut output = std::io::BufWriter::new(file);
        let mut digest = Sha256::new();
        for value in frames.values.iter().flatten() {
            let bytes = value.to_le_bytes();
            output.write_all(&bytes)?;
            digest.update(bytes);
        }
        output.flush()?;
        Some(hex::encode(digest.finalize()))
    } else {
        None
    };
    // Raw probabilities stay diagnostic and bounded; avoid megabytes on stdout.
    println!(
        "{}",
        serde_json::to_string(
            &serde_json::json!({"session_id": stream.session_id(), "complete": stream.snapshot().is_some_and(|s| s.complete), "accepted_samples": stream.accepted_samples(), "audio_duration_seconds": result.audio_duration_seconds, "native_frame_count": result.native_frame_count, "retained_probabilities": retained, "retained_probability_file": args.get(5), "retained_probability_sha256": probability_sha256, "model_revision": result.model_revision, "runtime_version": result.runtime_version, "backend": result.backend, "device": result.device, "geometry": result.geometry, "turns": result.turns, "cold_load_micros": cold_load_micros, "native_compute_micros": result.native_compute_micros, "finish_micros": finish_micros, "elapsed_micros": elapsed_micros, "backpressure_count": backpressure_count, "snapshots": snapshots})
        )?
    );
    model.shutdown();
    if runtime.diarization_reserved_mb() != 0 {
        return Err("native reservation leaked".into());
    }
    Ok(())
}
