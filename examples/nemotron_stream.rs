//! Stream normalized WAV repeatedly to exercise cache continuity and long sessions.
use gen2::{
    Runtime,
    diarization::{
        DiarizationError, DiarizeOptions, LoadOptions, MAX_PUSH_SAMPLES, MAX_SESSION_SAMPLES,
    },
};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: nemotron_stream MODEL_DIRECTORY NATIVE_C_LIBRARY AUDIO.wav REPEATS".into(),
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
    let model = runtime.load_diarizer(&args[1], LoadOptions::cpu(&args[2]))?;
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
            let mut block = samples
                .by_ref()
                .take(MAX_PUSH_SAMPLES)
                .map(|x| x.map(|x| x as f32 / 32768.))
                .collect::<Result<Vec<_>, _>>()?;
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
    // Raw probabilities stay diagnostic and bounded; avoid dumping megabytes to stdout.
    println!(
        "{}",
        serde_json::to_string(
            &serde_json::json!({"session_id": stream.session_id(), "complete": stream.snapshot().is_some_and(|s| s.complete), "accepted_samples": stream.accepted_samples(), "audio_duration_seconds": result.audio_duration_seconds, "native_frame_count": result.native_frame_count, "retained_probabilities": retained, "model_revision": result.model_revision, "runtime_version": result.runtime_version, "backend": result.backend, "geometry": result.geometry, "turns": result.turns, "native_compute_micros": result.native_compute_micros, "finish_micros": finish_micros, "elapsed_micros": elapsed_micros, "backpressure_count": backpressure_count, "snapshots": snapshots})
        )?
    );
    model.shutdown();
    if runtime.diarization_reserved_mb() != 0 {
        return Err("native reservation leaked".into());
    }
    Ok(())
}
