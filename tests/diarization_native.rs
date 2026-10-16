//! Real-model gates; ignored in ordinary CI, fail explicitly if assets are absent.
#![cfg(feature = "backend-diar-nemo")]

use gen2::{
    Runtime,
    diarization::{Audio, DiarizeOptions, LoadOptions},
};
use std::path::PathBuf;

#[test]
#[ignore = "requires pinned N0 assets and SDK; checks scoped pipeline cancellation"]
fn dropping_a_diarization_child_keeps_the_parent_pipeline_active() {
    use gen2::diarization::{Cancellation, Status};
    use std::time::{Duration, Instant};
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(1024)
        .build()
        .unwrap();
    let model = runtime
        .load_diarizer(
            std::env::var("GEN2_NEMOTRON_ASSETS").unwrap(),
            LoadOptions::cpu(std::env::var("GEN2_NEMO_LIBRARY").unwrap()),
        )
        .unwrap();
    let parent = Cancellation::default();
    let stream = model
        .start_stream(DiarizeOptions {
            cancellation: parent.child_token(),
            ..Default::default()
        })
        .unwrap();
    drop(stream);
    let deadline = Instant::now() + Duration::from_secs(5);
    while model.status() != Status::Ready {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(!parent.is_cancelled());
    model
        .diarize(
            Audio::mono_16khz(vec![0.; 320]).unwrap(),
            DiarizeOptions {
                cancellation: parent,
                ..Default::default()
            },
        )
        .unwrap();
    model.shutdown();
    assert_eq!(runtime.diarization_reserved_mb(), 0);
}

#[test]
#[ignore = "requires pinned N0 assets and SDK; silence, short tail and idle deadline"]
fn silence_short_tail_and_idle_deadline_have_explicit_terminal_results() {
    use gen2::diarization::{DiarizationError, StreamStatus};
    use std::time::{Duration, Instant};
    let assets = std::env::var("GEN2_NEMOTRON_ASSETS").unwrap();
    let library = std::env::var("GEN2_NEMO_LIBRARY").unwrap();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(1024)
        .build()
        .unwrap();
    let model = runtime
        .load_diarizer(assets, LoadOptions::cpu(library))
        .unwrap();
    let result = model
        .diarize(
            Audio::mono_16khz(vec![0.; 60 * 16000 + 17]).unwrap(),
            Default::default(),
        )
        .unwrap();
    assert!(result.turns.is_empty());
    assert_eq!(result.native_frame_count, 6001);
    assert_eq!(result.audio_duration_seconds, 60. + 17. / 16000.);
    let mut stream = model
        .start_stream(DiarizeOptions {
            deadline: Some(Instant::now() + Duration::from_millis(50)),
            ..Default::default()
        })
        .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert!(matches!(
        stream.finish(),
        Err(DiarizationError::DeadlineExceeded)
    ));
    assert_eq!(stream.status(), StreamStatus::Failed);
    assert!(!stream.snapshot().is_some_and(|s| s.complete));
    assert!(matches!(
        stream.finish(),
        Err(DiarizationError::DeadlineExceeded)
    ));
    let mut too_short = model.start_stream(Default::default()).unwrap();
    too_short.try_push(vec![0.; 1], 16000).unwrap();
    assert!(matches!(
        too_short.finish(),
        Err(DiarizationError::InvalidAudio(_))
    ));
    assert_eq!(too_short.status(), StreamStatus::Failed);
    model.shutdown();
    assert_eq!(runtime.diarization_reserved_mb(), 0);
}

#[test]
#[ignore = "requires pinned N0 assets and SDK; 100 actual model reloads"]
fn repeated_reload_full_queue_cancel_and_dropped_stream_release_resources() {
    use gen2::diarization::{DiarizationError, Status};
    use std::time::{Duration, Instant};
    let assets = PathBuf::from(std::env::var("GEN2_NEMOTRON_ASSETS").unwrap());
    let library = PathBuf::from(std::env::var("GEN2_NEMO_LIBRARY").unwrap());
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(1024)
        .build()
        .unwrap();
    for index in 0..100 {
        let model = runtime
            .load_diarizer(&assets, LoadOptions::cpu(&library))
            .unwrap();
        let mut stream = model.start_stream(Default::default()).unwrap();
        if index == 0 {
            loop {
                match stream.try_push(vec![0.; 16_640], 16000) {
                    Ok(()) => {}
                    Err(rejected) if matches!(rejected.error, DiarizationError::Busy) => break,
                    Err(error) => panic!("{error:?}"),
                }
            }
            let started = Instant::now();
            stream.cancel();
            assert!(matches!(stream.finish(), Err(DiarizationError::Cancelled)));
            assert!(started.elapsed() < Duration::from_secs(5));
        } else if index % 2 == 0 {
            stream.try_push(vec![0.; 320], 16000).unwrap();
            assert_eq!(stream.finish().unwrap().native_frame_count, 3);
        }
        // Odd iterations abandon the consumer before finish.
        drop(stream);
        let deadline = Instant::now() + Duration::from_secs(5);
        while model.status() == Status::Running {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        model.shutdown();
        assert_eq!(runtime.diarization_reserved_mb(), 0, "reload {index}");
        drop(model);
        assert!(runtime.diarization_models().is_empty());
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires the pinned N0 assets and a C compiler for ABI failure fixtures"]
fn unsupported_abi_and_missing_symbols_fail_before_ready() {
    use gen2::diarization::DiarizationError;
    let assets = PathBuf::from(std::env::var("GEN2_NEMOTRON_ASSETS").unwrap());
    let directory = tempfile::tempdir().unwrap();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(2048)
        .build()
        .unwrap();
    for (name, version) in [
        ("unsupported", "nemo-speech-asr 9.0.0"),
        ("missing", "nemo-speech-asr 0.2.0"),
    ] {
        let source = directory.path().join(format!("{name}.c"));
        let library = directory.path().join(format!("{name}.dylib"));
        std::fs::write(
            &source,
            format!("const char *nemo_speech_asr_version(void) {{return \"{version}\";}}\n"),
        )
        .unwrap();
        assert!(
            std::process::Command::new("cc")
                .args(["-dynamiclib"])
                .arg(&source)
                .arg("-o")
                .arg(&library)
                .status()
                .unwrap()
                .success()
        );
        let error = runtime
            .load_diarizer(&assets, LoadOptions::cpu(library))
            .unwrap_err();
        assert!(matches!(
            error,
            gen2::Error::Diarization(DiarizationError::BackendUnavailable(_))
        ));
        let message = error.to_string();
        assert!(
            message.contains(if name == "unsupported" {
                "unqualified NeMo ABI"
            } else {
                "nemo_speech_diar_create"
            }),
            "{message}"
        );
        assert_eq!(runtime.diarization_reserved_mb(), 0);
        assert!(runtime.diarization_models().is_empty());
    }
}

#[test]
#[ignore = "requires the pinned N0 model and CPU SDK; tests streaming transport parity"]
fn streamed_uneven_pcm_matches_recording_and_finish_is_idempotent() {
    use gen2::diarization::{DiarizationError, StreamStatus};
    use std::time::Duration;
    let assets = PathBuf::from(std::env::var("GEN2_NEMOTRON_ASSETS").unwrap());
    let library = PathBuf::from(std::env::var("GEN2_NEMO_LIBRARY").unwrap());
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(2048)
        .build()
        .unwrap();
    let model = runtime
        .load_diarizer(&assets, LoadOptions::cpu(library))
        .unwrap();
    let wav = assets.join("ami_en2002d_2132.wav");
    let expected = model
        .diarize(
            Audio::from_wav(&wav).unwrap(),
            DiarizeOptions {
                frame_probabilities: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut stream = model
        .start_stream(DiarizeOptions {
            frame_probabilities: true,
            ..Default::default()
        })
        .unwrap();
    assert!(matches!(
        model.start_stream(Default::default()),
        Err(DiarizationError::Busy)
    ));
    let invalid = stream.try_push(vec![1.; 80], 8000).unwrap_err();
    assert!(matches!(invalid.error, DiarizationError::InvalidAudio(_)));
    assert_eq!(invalid.samples.len(), 80);
    let mut oversized = Vec::with_capacity(1_000_000);
    oversized.push(0.);
    let rejected = stream.try_push(oversized, 16000).unwrap_err();
    assert!(matches!(rejected.error, DiarizationError::ResourceLimit(_)));
    assert_eq!(rejected.samples.capacity(), 1_000_000);
    let mut reader = hound::WavReader::open(wav).unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.)
        .collect();
    let mut cursor = 0;
    let mut saturated = false;
    let mut latest_revision = 0;
    let mut previous_snapshot = None;
    for size in [1, 13, 2559, 16640].into_iter().cycle() {
        if cursor == pcm.len() {
            break;
        }
        let end = (cursor + size).min(pcm.len());
        let mut block = pcm[cursor..end].to_vec();
        loop {
            match stream.try_push(block, 16000) {
                Ok(()) => break,
                Err(rejected) if matches!(rejected.error, DiarizationError::Busy) => {
                    saturated = true;
                    block = rejected.samples;
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(e) => panic!("unexpected push failure: {e:?}"),
            }
        }
        cursor = end;
        if let Some(snapshot) = stream.snapshot() {
            assert!(snapshot.revision >= latest_revision);
            assert!(!snapshot.complete);
            assert_eq!(snapshot.session_id, stream.session_id());
            assert!(snapshot.processed_audio_seconds <= cursor as f64 / 16000.);
            latest_revision = snapshot.revision;
            previous_snapshot = Some(snapshot);
        }
    }
    assert!(saturated);
    assert!(previous_snapshot.is_some());
    let final_result = stream.finish().unwrap();
    assert_eq!(stream.status(), StreamStatus::Finished);
    assert_eq!(final_result.turns, expected.turns);
    assert_eq!(
        final_result.frame_probabilities,
        expected.frame_probabilities
    );
    assert_eq!(final_result.native_frame_count, 6001);
    assert!(std::sync::Arc::ptr_eq(
        &final_result,
        &stream.finish().unwrap()
    ));
    assert!(matches!(
        stream.try_push(vec![0.; 160], 16000).unwrap_err().error,
        DiarizationError::InvalidState(_)
    ));
    let final_snapshot = stream.snapshot().unwrap();
    assert!(final_snapshot.complete);
    assert!(final_snapshot.revision > latest_revision);
    let mut cancelled = model.start_stream(Default::default()).unwrap();
    cancelled.cancel();
    assert!(matches!(
        cancelled.finish(),
        Err(DiarizationError::Cancelled)
    ));
    assert_eq!(cancelled.status(), StreamStatus::Cancelled);
    model.shutdown();
    assert_eq!(runtime.diarization_reserved_mb(), 0);
}

#[test]
#[ignore = "requires the pinned N0 model and CPU SDK"]
fn native_admission_cancellation_and_failed_load_release_reservations() {
    use gen2::diarization::{Cancellation, DiarizationError, Status};
    use std::time::{Duration, Instant};
    let assets = PathBuf::from(std::env::var("GEN2_NEMOTRON_ASSETS").expect("assets required"));
    let library = PathBuf::from(std::env::var("GEN2_NEMO_LIBRARY").expect("library required"));
    let small = Runtime::builder()
        .resident_memory_budget_mb(1023)
        .build()
        .unwrap();
    assert!(matches!(
        small.load_diarizer(&assets, LoadOptions::cpu(&library)),
        Err(gen2::Error::Diarization(DiarizationError::ResourceLimit(_)))
    ));
    assert_eq!(small.diarization_reserved_mb(), 0);
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(2048)
        .build()
        .unwrap();
    let bad = tempfile::NamedTempFile::new().unwrap();
    assert!(matches!(
        runtime.load_diarizer(&assets, LoadOptions::cpu(bad.path())),
        Err(gen2::Error::Diarization(
            DiarizationError::BackendUnavailable(_)
        ))
    ));
    assert_eq!(runtime.diarization_reserved_mb(), 0);
    let model = runtime
        .load_diarizer(&assets, LoadOptions::cpu(&library))
        .unwrap();
    let minimum = || Audio::mono_16khz(vec![0.; 320]).unwrap();
    assert!(matches!(
        model.diarize(
            minimum(),
            DiarizeOptions {
                deadline: Some(Instant::now()),
                ..Default::default()
            }
        ),
        Err(DiarizationError::DeadlineExceeded)
    ));
    assert_eq!(
        model
            .diarize(minimum(), Default::default())
            .unwrap()
            .native_frame_count,
        3
    );
    let cancellation = Cancellation::default();
    let other = model.clone();
    let audio = Audio::from_wav(assets.join("ami_en2002d_2132.wav")).unwrap();
    let options = DiarizeOptions {
        cancellation: cancellation.clone(),
        ..Default::default()
    };
    let job = std::thread::spawn(move || other.diarize(audio, options));
    let deadline = Instant::now() + Duration::from_secs(5);
    while model.status() != Status::Running {
        assert!(Instant::now() < deadline, "worker never entered Running");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(matches!(
        model.diarize(minimum(), Default::default()),
        Err(DiarizationError::Busy)
    ));
    cancellation.cancel();
    assert!(matches!(
        job.join().unwrap(),
        Err(DiarizationError::Cancelled)
    ));
    assert_eq!(runtime.diarization_reserved_mb(), 1024);
    model.shutdown();
    assert_eq!(runtime.diarization_reserved_mb(), 0);
    assert!(matches!(
        model.diarize(minimum(), Default::default()),
        Err(DiarizationError::Unloaded)
    ));
}

#[test]
#[ignore = "requires the pinned N0 assets; checks warm cache without any network feature"]
fn verified_provisioned_cache_is_reused_offline() {
    let assets = PathBuf::from(std::env::var("GEN2_NEMOTRON_ASSETS").expect("assets required"));
    let temp = tempfile::tempdir().unwrap();
    let destination = temp
        .path()
        .join(format!("nemotron-{}", gen2::diarization::MODEL_REVISION));
    std::fs::create_dir(&destination).unwrap();
    for name in [gen2::diarization::MODEL_FILENAME, "README.md"] {
        std::fs::copy(assets.join(name), destination.join(name)).unwrap();
    }
    assert_eq!(
        gen2::diarization::download_nemotron(temp.path()).unwrap(),
        destination
    );
}

#[test]
#[ignore = "requires the pinned N0 model, CPU SDK and reference outputs"]
fn real_diarizer_matches_native_reference_and_resets_recordings() {
    let assets = PathBuf::from(
        std::env::var("GEN2_NEMOTRON_ASSETS").expect("GEN2_NEMOTRON_ASSETS required"),
    );
    let library =
        PathBuf::from(std::env::var("GEN2_NEMO_LIBRARY").expect("GEN2_NEMO_LIBRARY required"));
    let reference = PathBuf::from(
        std::env::var("GEN2_NEMOTRON_REFERENCE").expect("GEN2_NEMOTRON_REFERENCE required"),
    );
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(2048)
        .build()
        .unwrap();
    let model = runtime
        .load_diarizer(&assets, LoadOptions::cpu(library))
        .unwrap();
    let wav = assets.join("ami_en2002d_2132.wav");
    let options = DiarizeOptions {
        frame_probabilities: true,
        ..Default::default()
    };
    let first = model
        .diarize(Audio::from_wav(&wav).unwrap(), options.clone())
        .unwrap();
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("ami.json")).unwrap()).unwrap();
    let turns = expected["segments"].as_array().unwrap();
    assert_eq!(first.turns.len(), turns.len());
    for (actual, expected) in first.turns.iter().zip(turns) {
        assert!((actual.start_seconds - expected["start"].as_f64().unwrap().min(60.)).abs() < 1e-6);
        assert!((actual.end_seconds - expected["end"].as_f64().unwrap().min(60.)).abs() < 1e-6);
        assert_eq!(
            u64::from(actual.speaker),
            expected["speaker"].as_u64().unwrap()
        );
    }
    let raw = std::fs::read(reference.join("ami.json.f32")).unwrap();
    let expected: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let frames = first.frame_probabilities.as_ref().unwrap();
    assert_eq!(frames.values.len() * 8, expected.len());
    for (actual, expected) in frames.values.iter().flatten().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    let silent = model
        .diarize(
            Audio::mono_16khz(vec![0.; 32000]).unwrap(),
            DiarizeOptions::default(),
        )
        .unwrap();
    assert!(silent.turns.is_empty());
    let repeated = model
        .diarize(Audio::from_wav(&wav).unwrap(), options)
        .unwrap();
    assert_eq!(first.turns, repeated.turns);
    assert_eq!(first.frame_probabilities, repeated.frame_probabilities);
    assert_eq!(runtime.diarization_reserved_mb(), 1024);
    model.shutdown();
    assert_eq!(runtime.diarization_reserved_mb(), 0);
}
