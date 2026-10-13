#![cfg(feature = "asr-dynamic")]
use gen2::{Runtime, speech::*};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[test]
#[ignore = "requires GEN2_BARK_BUNDLE, GEN2_ORT_LIBRARY, GEN2_BARK_REFERENCE"]
fn real_bark_matches_reference_and_reuses_resident_weights() {
    let bundle = std::env::var("GEN2_BARK_BUNDLE").expect("GEN2_BARK_BUNDLE");
    let library: PathBuf = std::env::var("GEN2_ORT_LIBRARY")
        .expect("GEN2_ORT_LIBRARY")
        .into();
    let reference: PathBuf = std::env::var("GEN2_BARK_REFERENCE")
        .expect("GEN2_BARK_REFERENCE")
        .into();
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("reference.json")).unwrap()).unwrap();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(8192)
        .build()
        .unwrap();
    let speech = runtime
        .load_transcriber(
            &bundle,
            LoadOptions {
                native_library: Some(library.clone()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(runtime.speech_reserved_mb(), 4096);
    assert_eq!(runtime.speech_models().len(), 1);
    assert!(!speech.capabilities().streaming);
    let records = receipt["records"].as_array().unwrap();
    let mut mismatches = Vec::new();
    let mut results = Vec::new();
    for record in records {
        let bytes = std::fs::read(reference.join(record["pcm"].as_str().unwrap())).unwrap();
        let samples = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let result = speech
            .transcribe(
                Audio::mono_16khz(samples).unwrap(),
                TranscribeOptions::default(),
            )
            .unwrap();
        if result.text != record["text"].as_str().unwrap() {
            mismatches.push(record["id"].clone());
        }
        if serde_json::to_value(&result.token_ids).unwrap() != record["token_ids"]
            && !mismatches.contains(&record["id"])
        {
            mismatches.push(record["id"].clone());
        }
        eprintln!(
            "clip {}: {} us {:?}",
            record["id"], result.execution_micros, result.text
        );
        results.push(result);
    }
    if let Ok(path) = std::env::var("GEN2_BARK_RESULTS") {
        std::fs::write(path, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
    }
    assert!(
        mismatches.is_empty(),
        "raw transcript mismatches: {mismatches:?}"
    );
    let first = &records[0];
    let wav = reference.join(format!("{:03}.wav", first["id"].as_u64().unwrap()));
    for _ in 0..3 {
        assert_eq!(
            speech
                .transcribe(Audio::from_wav(&wav).unwrap(), TranscribeOptions::default())
                .unwrap()
                .text,
            first["text"].as_str().unwrap()
        );
    }
    let cancelled = Cancellation::default();
    cancelled.cancel();
    assert!(matches!(
        speech.transcribe(
            Audio::mono_16khz(vec![0.; 320]).unwrap(),
            TranscribeOptions {
                cancellation: cancelled,
                ..Default::default()
            }
        ),
        Err(SpeechError::Cancelled)
    ));
    assert!(matches!(
        speech.transcribe(
            Audio::mono_16khz(vec![0.; 320]).unwrap(),
            TranscribeOptions {
                deadline: Some(Instant::now() - Duration::from_secs(1)),
                ..Default::default()
            }
        ),
        Err(SpeechError::DeadlineExceeded)
    ));
    speech.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
    assert_eq!(speech.status(), SpeechStatus::Unloaded);
    assert!(matches!(
        speech.transcribe(
            Audio::mono_16khz(vec![0.; 320]).unwrap(),
            TranscribeOptions::default()
        ),
        Err(SpeechError::Unloaded)
    ));
    let reloaded = runtime
        .load_transcriber(
            &bundle,
            LoadOptions {
                native_library: Some(library),
                ..Default::default()
            },
        )
        .unwrap();
    reloaded.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
}

#[test]
fn wav_rates_and_channel_averaging_preserve_duration_and_signal() {
    let rates: &[u32] = if cfg!(feature = "asr-resample-soxr") {
        &[8000, 16000, 44100, 48000]
    } else {
        &[16000]
    };
    for &rate in rates {
        for channels in [1, 2, 8] {
            let samples = (0..rate as usize)
                .flat_map(|i| {
                    std::iter::repeat_n(
                        (std::f32::consts::TAU * 400. * i as f32 / rate as f32).sin() * 0.2,
                        channels,
                    )
                })
                .collect();
            let audio = Audio::from_interleaved(samples, rate, channels).unwrap();
            assert_eq!(audio.samples().len(), 16000);
            let signal = &audio.samples()[1000..15000];
            let rms = (signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32).sqrt();
            assert!(
                (rms - 0.14142136).abs() < 0.001,
                "rate={rate}, channels={channels}, RMS={rms}"
            );
            let crossings = signal
                .windows(2)
                .filter(|s| s[0] <= 0. && s[1] > 0.)
                .count();
            assert!(
                (349..=351).contains(&crossings),
                "resampled 400 Hz tone has {crossings} crossings"
            );
        }
    }
    assert!(Audio::from_interleaved(vec![0.; 48000 * 60 + 1], 48000, 1).is_err());
    assert!(Audio::from_interleaved(vec![0.; 16000], 22050, 1).is_err());
    let antiphase = (0..16000).flat_map(|_| [0.5, -0.5]).collect();
    assert!(
        Audio::from_interleaved(antiphase, 16000, 2)
            .unwrap()
            .samples()
            .iter()
            .all(|s| *s == 0.)
    );
}

#[test]
fn wav_files_validate_encodings_bounds_and_truncation() {
    let root = tempfile::tempdir().unwrap();
    let rates: &[u32] = if cfg!(feature = "asr-resample-soxr") {
        &[8000, 16000, 44100, 48000]
    } else {
        &[16000]
    };
    for &rate in rates {
        for channels in 1..=8 {
            for (format, bits) in [
                (hound::SampleFormat::Int, 16),
                (hound::SampleFormat::Float, 32),
            ] {
                let path = root.path().join("audio.wav");
                let mut writer = hound::WavWriter::create(
                    &path,
                    hound::WavSpec {
                        channels,
                        sample_rate: rate,
                        bits_per_sample: bits,
                        sample_format: format,
                    },
                )
                .unwrap();
                for _ in 0..rate * channels as u32 / 50 {
                    if format == hound::SampleFormat::Int {
                        writer.write_sample(3276i16).unwrap();
                    } else {
                        writer.write_sample(0.1f32).unwrap();
                    }
                }
                writer.finalize().unwrap();
                assert_eq!(Audio::from_wav(&path).unwrap().samples().len(), 320);
            }
        }
    }
    let path = root.path().join("invalid.wav");
    std::fs::write(&path, b"not a WAV").unwrap();
    assert!(matches!(
        Audio::from_wav(&path),
        Err(SpeechError::InvalidAudio(_))
    ));
    let mut writer = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..320 {
        writer.write_sample(0i32).unwrap();
    }
    writer.finalize().unwrap();
    assert!(Audio::from_wav(&path).is_err());
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(48)
        .unwrap();
    assert!(Audio::from_wav(&path).is_err());
}

#[test]
#[ignore = "requires the complete pinned GEN2_BARK_BUNDLE"]
fn real_bundle_integrity_and_warm_offline_cache() {
    let bundle = PathBuf::from(std::env::var("GEN2_BARK_BUNDLE").expect("bundle"));
    let cache = tempfile::tempdir().unwrap();
    let destination = cache.path().join(format!("bark-{MODEL_REVISION}"));
    std::fs::create_dir(&destination).unwrap();
    for entry in std::fs::read_dir(&bundle).unwrap() {
        let entry = entry.unwrap();
        std::fs::hard_link(entry.path(), destination.join(entry.file_name())).unwrap();
    }
    SpeechBundle::open(&destination).unwrap();
    // Works even without the hf feature: there is no network client on this path.
    assert_eq!(download_bark(cache.path()).unwrap(), destination);
    std::fs::remove_file(destination.join("encoder-model.onnx.data")).unwrap();
    assert!(matches!(
        SpeechBundle::open(&destination),
        Err(SpeechError::InvalidBundle(_))
    ));
    std::fs::remove_file(destination.join("manifest.json")).unwrap();
    std::fs::write(destination.join("manifest.json"), b"{\"files\":{}}").unwrap();
    assert!(matches!(
        SpeechBundle::open(&destination),
        Err(SpeechError::InvalidBundle(_))
    ));
}

#[cfg(feature = "hf")]
#[test]
#[ignore = "downloads the pinned 488 MB model; explicitly requested provisioning qualification"]
fn real_provisioner_recovers_incomplete_and_corrupt_entries() {
    let cache = tempfile::tempdir().unwrap();
    let incomplete = cache.path().join("bark-staging-interrupted");
    std::fs::create_dir(&incomplete).unwrap();
    std::fs::write(incomplete.join("partial"), b"interrupted download").unwrap();
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let bundle = host.block_on(async { download_bark(cache.path()).unwrap() });
    SpeechBundle::open(&bundle).unwrap();
    assert_eq!(download_bark(cache.path()).unwrap(), bundle);
    std::fs::write(bundle.join("vocab.txt"), b"corrupt").unwrap();
    assert_eq!(download_bark(cache.path()).unwrap(), bundle);
    SpeechBundle::open(&bundle).unwrap();
    assert!(incomplete.join("partial").exists());
    let quarantines: Vec<_> = std::fs::read_dir(cache.path())
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains(".corrupt-"))
        .collect();
    assert_eq!(quarantines.len(), 1);
    assert_eq!(
        std::fs::read(quarantines[0].path().join("vocab.txt")).unwrap(),
        b"corrupt"
    );
}

#[test]
#[ignore = "requires GEN2_BARK_BUNDLE and GEN2_ORT_LIBRARY"]
fn real_worker_bounds_queue_cancellation_shutdown_and_admission() {
    let bundle = std::env::var("GEN2_BARK_BUNDLE").expect("bundle");
    let library: PathBuf = std::env::var("GEN2_ORT_LIBRARY").expect("runtime").into();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(4096)
        .build()
        .unwrap();
    let speech = runtime
        .load_transcriber(
            &bundle,
            LoadOptions {
                native_library: Some(library.clone()),
                queue_capacity: 1,
                reservation_mb: 4096,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(matches!(
        runtime.load_transcriber(
            &bundle,
            LoadOptions {
                native_library: Some(library),
                reservation_mb: 4096,
                ..Default::default()
            }
        ),
        Err(gen2::Error::Speech(SpeechError::ResourceLimit(_)))
    ));
    let cancel = Cancellation::default();
    let running = speech.clone();
    let options = TranscribeOptions {
        cancellation: cancel.clone(),
        ..Default::default()
    };
    let active = std::thread::spawn(move || {
        running.transcribe(Audio::mono_16khz(vec![0.; 960000]).unwrap(), options)
    });
    let until = Instant::now() + Duration::from_secs(5);
    while speech.status() != SpeechStatus::Running {
        assert!(Instant::now() < until);
        std::thread::yield_now();
    }
    let competing: Vec<_> = (0..8)
        .map(|_| {
            let handle = speech.clone();
            std::thread::spawn(move || {
                handle.transcribe(
                    Audio::mono_16khz(vec![0.; 320]).unwrap(),
                    TranscribeOptions {
                        deadline: Some(Instant::now() + Duration::from_millis(150)),
                        ..Default::default()
                    },
                )
            })
        })
        .collect();
    let failures: Vec<_> = competing
        .into_iter()
        .map(|h| h.join().unwrap().unwrap_err())
        .collect();
    assert!(failures.iter().any(|e| matches!(e, SpeechError::Busy)));
    assert!(
        failures
            .iter()
            .any(|e| matches!(e, SpeechError::DeadlineExceeded))
    );
    cancel.cancel();
    assert!(matches!(
        active.join().unwrap(),
        Err(SpeechError::Cancelled)
    ));
    assert_eq!(runtime.speech_reserved_mb(), 4096);
    speech.unload();
    speech.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
    assert_eq!(speech.status(), SpeechStatus::Unloaded);
    assert_eq!(runtime.stats().estimated_resident_mb, 0);
}

#[test]
#[ignore = "requires GEN2_BARK_AUDIO_REFERENCE, GEN2_BARK_BUNDLE and GEN2_ORT_LIBRARY"]
fn real_audio_workflow_matches_qualified_reference() {
    let root = PathBuf::from(std::env::var("GEN2_BARK_AUDIO_REFERENCE").expect("audio reference"));
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("audio-reference.json")).unwrap()).unwrap();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(4096)
        .build()
        .unwrap();
    let speech = runtime
        .load_transcriber(
            std::env::var("GEN2_BARK_BUNDLE").unwrap(),
            LoadOptions {
                native_library: Some(std::env::var("GEN2_ORT_LIBRARY").unwrap().into()),
                ..Default::default()
            },
        )
        .unwrap();
    let mut results = Vec::new();
    let mut mismatches = Vec::new();
    for record in receipt["records"].as_array().unwrap() {
        let wav = record["wav"].as_str().unwrap();
        let result = speech
            .transcribe(
                Audio::from_wav(root.join(wav)).unwrap(),
                TranscribeOptions::default(),
            )
            .unwrap();
        if result.text != record["text"].as_str().unwrap() {
            mismatches.push(wav);
        }
        results.push(result);
    }
    std::fs::write(
        root.join("native-audio.json"),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .unwrap();
    speech.shutdown();
    assert!(
        mismatches.is_empty(),
        "audio transcript mismatches: {mismatches:?}"
    );
}

#[cfg(feature = "laya-dynamic")]
#[test]
#[ignore = "requires GEN2_BARK_BUNDLE and GEN2_ORT_LIBRARY; run in a fresh process"]
fn coexistence_laya_first() {
    coexistence(true);
}
#[cfg(feature = "laya-dynamic")]
#[test]
#[ignore = "requires GEN2_BARK_BUNDLE and GEN2_ORT_LIBRARY; run in a fresh process"]
fn coexistence_speech_first() {
    coexistence(false);
}
#[cfg(feature = "laya-dynamic")]
fn coexistence(laya_first: bool) {
    let bundle = std::env::var("GEN2_BARK_BUNDLE").expect("bundle");
    let library: PathBuf = std::env::var("GEN2_ORT_LIBRARY").expect("runtime").into();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(4097)
        .build()
        .unwrap();
    let load_speech = || {
        runtime
            .load_transcriber(
                &bundle,
                LoadOptions {
                    native_library: Some(library.clone()),
                    reservation_mb: 4096,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let load_laya = || {
        runtime
            .load_decider(
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke"),
                gen2::decision::LoadOptions {
                    native_library: Some(library.clone()),
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let (speech, laya) = if laya_first {
        let laya = load_laya();
        (load_speech(), laya)
    } else {
        (load_speech(), load_laya())
    };
    assert_eq!(runtime.stats().estimated_resident_mb, 4097);
    laya.decide(
        gen2::decision::DecisionRequest::text("a b")
            .question("q", gen2::decision::Question::yes_no("a?")),
        gen2::decision::DecisionOptions::default(),
    )
    .unwrap();
    speech
        .transcribe(
            Audio::mono_16khz(vec![0.; 320]).unwrap(),
            TranscribeOptions::default(),
        )
        .unwrap();
    let copied = tempfile::tempdir().unwrap();
    let other = copied.path().join(library.file_name().unwrap());
    std::fs::copy(&library, &other).unwrap();
    laya.shutdown();
    speech.shutdown();
    assert!(matches!(
        runtime.load_transcriber(
            &bundle,
            LoadOptions {
                native_library: Some(other),
                reservation_mb: 4096,
                ..Default::default()
            }
        ),
        Err(gen2::Error::Speech(SpeechError::BackendUnavailable(_)))
    ));
    let until = Instant::now() + Duration::from_secs(2);
    while runtime.speech_reserved_mb() != 0 {
        assert!(Instant::now() < until);
        std::thread::yield_now();
    }
    assert_eq!(runtime.stats().estimated_resident_mb, 0);
}
