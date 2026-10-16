//! Qualification diagnostic; each load order runs in a fresh offline process.
use gen2::{
    Runtime,
    advanced::generation::{Settings, SystemSettings},
    diarization::{Audio, DiarizeOptions, Diarizer, LoadOptions, Status},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::Path,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const PROMPT: &str = "Reply with the single word hello.";

#[cfg(windows)]
#[path = "support/windows_modules.rs"]
mod windows_modules;

fn hash(path: impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes = [0; 65536];
    loop {
        let count = file.read(&mut bytes)?;
        if count == 0 {
            return Ok(hex::encode(digest.finalize()));
        }
        digest.update(&bytes[..count]);
    }
}

fn write_json(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(serde_json::to_string_pretty(value)?.as_bytes())?;
    Ok(())
}

struct Observation {
    value: Value,
    api_micros: [u128; 2],
}

fn chat(model: &gen2::Model, zero: Instant) -> Result<Observation> {
    let request = model.generate(PROMPT).max_tokens(8).temperature(0.).seed(0);
    let start = zero.elapsed().as_micros();
    let output = request.run()?;
    let end = zero.elapsed().as_micros();
    let text = output.text();
    if text.is_empty() {
        return Err("chat returned no text".into());
    }
    let offload = model
        .info()
        .offload
        .ok_or("chat placement was not reported")?;
    if offload.on_gpu() {
        return Err("CPU coexistence unexpectedly selected a GPU".into());
    }
    Ok(Observation {
        value: json!({"text": text, "offload": offload, "api_call_micros": [start, end]}),
        api_micros: [start, end],
    })
}

fn recording(model: &Diarizer, wav: &Path, output: &Path, zero: Instant) -> Result<Observation> {
    let mut reader = hound::WavReader::open(wav)?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != 16000
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
        || reader.duration() > 120 * 16000
    {
        return Err("use the locked mono 16 kHz PCM16 fixture".into());
    }
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|s| s as f32 / 32768.))
        .collect::<std::result::Result<_, _>>()?;
    let count = samples.len();
    let mut input = Sha256::new();
    for sample in &samples {
        input.update(sample.to_le_bytes());
    }
    let audio = Audio::mono_16khz(samples)?;
    let started = Instant::now();
    let call_start = zero.elapsed().as_micros();
    let result = model.diarize(
        audio,
        DiarizeOptions {
            frame_probabilities: true,
            ..Default::default()
        },
    )?;
    let call_end = zero.elapsed().as_micros();
    let elapsed = started.elapsed().as_micros();
    let frames = result
        .frame_probabilities
        .as_ref()
        .ok_or("missing native probabilities")?;
    let mut raw = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("f32"))?;
    let mut probabilities = Sha256::new();
    for value in frames.values.iter().flatten() {
        let bytes = value.to_le_bytes();
        raw.write_all(&bytes)?;
        probabilities.update(bytes);
    }
    let value = json!({"complete": true, "accepted_samples": count,
        "input_pcm_f32le_sha256": hex::encode(input.finalize()), "input_repeats": 1,
        "audio_duration_seconds": result.audio_duration_seconds,
        "native_frame_count": result.native_frame_count,
        "retained_probabilities": {"first_frame": frames.first_frame, "rows": frames.values.len()},
        "retained_probability_sha256": hex::encode(probabilities.finalize()),
        "model_revision": result.model_revision, "runtime_version": result.runtime_version,
        "backend": result.backend, "device": result.device, "geometry": result.geometry,
        "turns": result.turns, "native_compute_micros": result.native_compute_micros,
        "elapsed_micros": elapsed, "api_call_micros": [call_start, call_end]});
    write_json(output, &value)?;
    Ok(Observation {
        value,
        api_micros: [call_start, call_end],
    })
}

fn maps(output: &Path, stage: &str) -> Result<()> {
    #[cfg(target_os = "linux")]
    std::fs::write(
        output.join(format!("{stage}.maps")),
        std::fs::read("/proc/self/maps")?,
    )?;
    #[cfg(windows)]
    {
        let modules = windows_modules::stable_paths()?
            .into_iter()
            .map(|path| {
                let digest = hash(&path)?;
                Ok(json!({"path": path, "sha256": digest, "bytes": path.metadata()?.len()}))
            })
            .collect::<Result<Vec<_>>>()?;
        write_json(
            output.join(format!("{stage}.modules.json")),
            &json!(modules),
        )?;
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    let _ = (output, stage);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 || !["isolated", "chat-first", "diar-first"].contains(&args[1].as_str()) {
        return Err("usage: nemotron_chat_coexistence isolated|chat-first|diar-first CHAT_GGUF DIAR_ASSETS C_ABI_LIBRARY WAV NEW_OUTPUT_DIR".into());
    }
    let zero = Instant::now();
    let output = Path::new(&args[6]);
    std::fs::create_dir(output)?;
    let settings = Settings {
        system: SystemSettings {
            gpu_layers: Some(0),
            threads: Some(2),
            threads_batch: Some(2),
            ctx_size: Some(1024),
            ..Default::default()
        },
        ..Default::default()
    };
    let runtime = Runtime::builder()
        .settings(settings.clone())
        .resident_memory_budget_mb(4096)
        .build()?;
    let fingerprints = json!({"executable": hash(std::env::current_exe()?)?,
        "chat_model": hash(&args[2])?, "diarization_model": hash(Path::new(&args[3]).join("Nemotron-3-Diarization.q8_0.gguf"))?,
        "library": hash(&args[4])?, "wav": hash(&args[5])?});
    let mut summary = json!({"order": args[1], "settings": settings, "prompt": PROMPT,
        "generation": {"max_tokens": 8, "temperature": 0.0, "seed": 0},
        "fingerprints": fingerprints, "chat_outputs": {}});
    if args[1] == "isolated" {
        let model = runtime.load(&args[2])?;
        summary["chat_outputs"]["isolated"] = chat(&model, zero)?.value;
        maps(output, "isolated")?;
    } else {
        let (model, diarizer) = if args[1] == "chat-first" {
            let model = runtime.load(&args[2])?;
            summary["chat_outputs"]["before_diarizer"] = chat(&model, zero)?.value;
            (
                model,
                runtime.load_diarizer(&args[3], LoadOptions::cpu(&args[4]))?,
            )
        } else {
            let diarizer = runtime.load_diarizer(&args[3], LoadOptions::cpu(&args[4]))?;
            (runtime.load(&args[2])?, diarizer)
        };
        if runtime.diarization_reserved_mb() != 1024 || !runtime.residency().is_resident(model.id())
        {
            return Err("chat/diarizer did not remain admitted together".into());
        }
        summary["chat_outputs"]["both_loaded"] = chat(&model, zero)?.value;
        recording(
            &diarizer,
            Path::new(&args[5]),
            &output.join("first.json"),
            zero,
        )?;
        maps(output, "both-loaded")?;
        let concurrent = std::thread::scope(|scope| -> Result<Value> {
            let worker = scope.spawn(|| -> std::result::Result<_, String> {
                let value = recording(
                    &diarizer,
                    Path::new(&args[5]),
                    &output.join("concurrent.json"),
                    zero,
                )
                .map_err(|error| error.to_string())?;
                Ok(value)
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            while diarizer.status() != Status::Running {
                if worker.is_finished() || Instant::now() >= deadline {
                    return Err("diarization never entered Running for concurrent chat".into());
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            let value = chat(&model, zero)?;
            let observed = worker.join().map_err(|_| "concurrent worker panicked")??;
            let [start, end] = value.api_micros;
            let [diar_start, diar_end] = observed.api_micros;
            if start.max(diar_start) >= end.min(diar_end) {
                return Err("chat and diarization execution intervals did not overlap".into());
            }
            Ok(
                json!({"chat": value.value, "chat_micros": [start, end], "diarization_micros": [diar_start, diar_end]}),
            )
        })?;
        summary["chat_outputs"]["concurrent"] = concurrent["chat"].clone();
        summary["concurrent_intervals"] = concurrent;
        diarizer.shutdown();
        if runtime.diarization_reserved_mb() != 0 {
            return Err("native reservation leaked after shutdown".into());
        }
        summary["chat_outputs"]["after_shutdown"] = chat(&model, zero)?.value;
        let reloaded = runtime.load_diarizer(&args[3], LoadOptions::cpu(&args[4]))?;
        recording(
            &reloaded,
            Path::new(&args[5]),
            &output.join("reloaded.json"),
            zero,
        )?;
        summary["chat_outputs"]["after_reload"] = chat(&model, zero)?.value;
        maps(output, "reloaded")?;
        reloaded.shutdown();
        if runtime.diarization_reserved_mb() != 0 {
            return Err("reloaded native reservation leaked".into());
        }
        summary["reservation_after_shutdown_mb"] = json!(runtime.diarization_reserved_mb());
    }
    write_json(output.join("summary.json"), &summary)?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}
