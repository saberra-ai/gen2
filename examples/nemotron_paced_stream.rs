//! Fixed 80.04s qualification replay. Measures public delivery, never GUI/acoustic latency.
use gen2::{
    Runtime,
    diarization::{DiarizationError, DiarizationStream, DiarizeOptions, LoadOptions},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

const SAMPLES: usize = 1_280_640;
const BLOCK: usize = 320;
type ProbeResult = Result<Value, Box<dyn std::error::Error>>;

fn saved_evidence(
    outcome: &ProbeResult,
    pushes: &[Value],
    observations: &[Value],
    checkpoints: &serde_json::Map<String, Value>,
) -> Value {
    match outcome {
        Ok(value) => value.clone(),
        Err(error) => json!({"complete":false,"error":error.to_string(),
            "pushes":pushes,"observations":observations,"checkpoints":checkpoints}),
    }
}

fn observe(
    stream: &DiarizationStream,
    started: Instant,
    observations: &mut Vec<Value>,
    previous: &mut u64,
    poll_count: &mut u64,
    poll_micros: &mut u128,
) -> Result<(), Box<dyn std::error::Error>> {
    let before = started.elapsed().as_micros();
    let snapshot = stream.snapshot();
    let after = started.elapsed().as_micros();
    *poll_count += 1;
    *poll_micros += after - before;
    if let Some(snapshot) = snapshot
        && snapshot.revision > *previous
    {
        if observations.len() >= 17 {
            return Err("too many observations for fixed five-second coalescing policy".into());
        }
        *previous = snapshot.revision;
        observations.push(json!({
            "session_id":snapshot.session_id,"revision":snapshot.revision,"complete":snapshot.complete,
            "poll_started_micros":before,"observed_micros":after,
            "accepted_samples":stream.accepted_samples(),"processed_audio_seconds":snapshot.processed_audio_seconds,
            "native_compute_micros":snapshot.result.native_compute_micros,
            "execution_micros":snapshot.result.execution_micros,"turns":snapshot.result.turns
        }));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: nemotron_paced_stream MODEL_DIRECTORY NATIVE_C_LIBRARY AUDIO_80_04s.wav OUTPUT.json".into());
    }
    let raw_path = format!("{}.f32", args[4]);
    if Path::new(&args[4]).exists() || Path::new(&raw_path).exists() {
        return Err("refusing to replace paced-stream evidence".into());
    }
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    let mut pushes = Vec::with_capacity(SAMPLES / BLOCK);
    let mut observations = Vec::with_capacity(17);
    let mut checkpoints = serde_json::Map::new();
    let outcome = replay(&args, &mut pushes, &mut observations, &mut checkpoints);
    let value = saved_evidence(&outcome, &pushes, &observations, &checkpoints);
    serde_json::to_writer_pretty(&mut output, &value)?;
    writeln!(output)?;
    output.sync_all()?;
    println!("{}", json!({"complete":outcome.is_ok(),"output":args[4]}));
    outcome.map(|_| ())
}

fn replay(
    args: &[String],
    pushes: &mut Vec<Value>,
    observations: &mut Vec<Value>,
    checkpoints: &mut serde_json::Map<String, Value>,
) -> ProbeResult {
    let raw_path = format!("{}.f32", args[4]);
    if std::fs::metadata(&args[3])?.len() > 20 * 1024 * 1024 {
        return Err("bounded fixed WAV required".into());
    }
    let mut reader = hound::WavReader::open(&args[3])?;
    let spec = reader.spec();
    if spec.sample_rate != 16000
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
        || reader.duration() as usize != SAMPLES
    {
        return Err("use exactly 80.04s mono16k PCM16".into());
    }
    let pcm: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;
    if pcm.len() != SAMPLES {
        return Err("incomplete fixed WAV".into());
    }
    let runtime = Runtime::builder().resident_memory_budget_mb(2048).build()?;
    let before_load = Instant::now();
    let model = runtime.load_diarizer(&args[1], LoadOptions::cpu(&args[2]))?;
    let cold_load_micros = before_load.elapsed().as_micros();
    checkpoints.insert("cold_load_micros".into(), json!(cold_load_micros));
    let before_open = Instant::now();
    let mut stream = model.start_stream(DiarizeOptions {
        frame_probabilities: true,
        deadline: Some(Instant::now() + Duration::from_secs(180)),
        ..Default::default()
    })?;
    let stream_open_micros = before_open.elapsed().as_micros();
    checkpoints.insert("stream_open_micros".into(), json!(stream_open_micros));
    checkpoints.insert("session_id".into(), json!(stream.session_id()));
    // Clock starts after model/session startup. A whole block becomes available
    // at its endpoint, so this replay never intentionally supplies future PCM.
    let started = Instant::now();
    let mut previous_revision = 0;
    let mut poll_count = 0;
    let mut poll_micros = 0;
    let mut digest = Sha256::new();
    for (index, chunk) in pcm.chunks(BLOCK).enumerate() {
        let endpoint_samples = index * BLOCK + chunk.len();
        let nominal_endpoint_micros = endpoint_samples as u64 * 1_000_000 / 16000;
        if let Some(wait) =
            Duration::from_micros(nominal_endpoint_micros).checked_sub(started.elapsed())
        {
            std::thread::sleep(wait);
        }
        let ready_micros = started.elapsed().as_micros();
        let mut block = Vec::with_capacity(BLOCK);
        for sample in chunk {
            let value = *sample as f32 / 32768.;
            digest.update(value.to_le_bytes());
            block.push(value);
        }
        let mut busy_retries = 0u64;
        loop {
            match stream.try_push(block, 16000) {
                Ok(()) => break,
                Err(rejected) if matches!(rejected.error, DiarizationError::Busy) => {
                    block = rejected.samples;
                    busy_retries += 1;
                    observe(
                        &stream,
                        started,
                        observations,
                        &mut previous_revision,
                        &mut poll_count,
                        &mut poll_micros,
                    )?;
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(rejected) => return Err(rejected.error.into()),
            }
        }
        let accepted_micros = started.elapsed().as_micros();
        pushes.push(json!({"endpoint_samples":endpoint_samples,"nominal_endpoint_micros":nominal_endpoint_micros,
            "ready_micros":ready_micros,"accepted_micros":accepted_micros,"busy_retries":busy_retries}));
        observe(
            &stream,
            started,
            observations,
            &mut previous_revision,
            &mut poll_count,
            &mut poll_micros,
        )?;
    }
    let finish_requested_micros = started.elapsed().as_micros();
    checkpoints.insert(
        "finish_requested_micros".into(),
        json!(finish_requested_micros),
    );
    let result = stream.finish()?;
    let finish_returned_micros = started.elapsed().as_micros();
    checkpoints.insert(
        "finish_returned_micros".into(),
        json!(finish_returned_micros),
    );
    checkpoints.insert(
        "native_compute_micros".into(),
        json!(result.native_compute_micros),
    );
    let cached = stream.finish()?;
    if !Arc::ptr_eq(&result, &cached) {
        return Err("repeat finish did not return the cached result".into());
    }
    observe(
        &stream,
        started,
        observations,
        &mut previous_revision,
        &mut poll_count,
        &mut poll_micros,
    )?;
    if !observations
        .last()
        .is_some_and(|snapshot| snapshot["complete"] == true)
    {
        return Err("missing final snapshot".into());
    }
    let frames = result
        .frame_probabilities
        .as_ref()
        .ok_or("missing final probabilities")?;
    let mut raw = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&raw_path)?,
    );
    let mut probability_digest = Sha256::new();
    for value in frames.values.iter().flatten() {
        let bytes = value.to_le_bytes();
        raw.write_all(&bytes)?;
        probability_digest.update(bytes);
    }
    raw.flush()?;
    let value = json!({
        "scope":"fixed public-API paced replay; no microphone, GUI, acoustic-label latency or finality watermark",
        "session_id":stream.session_id(),"complete":true,"accepted_samples":stream.accepted_samples(),
        "input_pcm_f32le_sha256":hex::encode(digest.finalize()),"input_repeats":1,
        "audio_duration_seconds":result.audio_duration_seconds,"native_frame_count":result.native_frame_count,
        "retained_probabilities":{"first_frame":frames.first_frame,"rows":frames.values.len()},
        "retained_probability_sha256":hex::encode(probability_digest.finalize()),"retained_probability_file":raw_path,
        "model_revision":result.model_revision,"runtime_version":result.runtime_version,"backend":result.backend,
        "device":result.device,"geometry":result.geometry,"turns":result.turns,
        "cold_load_micros":cold_load_micros,"stream_open_micros":stream_open_micros,
        "native_compute_micros":result.native_compute_micros,"elapsed_micros":finish_returned_micros,
        "finish_micros":finish_returned_micros-finish_requested_micros,
        "finish_requested_micros":finish_requested_micros,"finish_returned_micros":finish_returned_micros,
        "poll_count":poll_count,"poll_micros":poll_micros,"repeat_finish_cached":true,
        "pushes":&*pushes,"observations":&*observations
    });
    model.shutdown();
    if runtime.diarization_reserved_mb() != 0 {
        return Err("native reservation leaked".into());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_evidence_survives_replay_error() {
        let pushes = vec![json!({"endpoint_samples":320,"accepted_micros":20001})];
        let observations = vec![json!({"revision":1,"turns":[{"speaker":1}]})];
        let checkpoints =
            serde_json::Map::from_iter([("finish_requested_micros".into(), json!(80040000))]);
        let outcome: ProbeResult = Err("simulated finish or raw export failure".into());
        let bytes = serde_json::to_vec(&saved_evidence(
            &outcome,
            &pushes,
            &observations,
            &checkpoints,
        ))
        .unwrap();
        let restored: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored["complete"], false);
        assert_eq!(restored["error"], "simulated finish or raw export failure");
        assert_eq!(restored["pushes"], json!(pushes));
        assert_eq!(restored["observations"], json!(observations));
        assert_eq!(restored["checkpoints"], json!(checkpoints));
    }
}
