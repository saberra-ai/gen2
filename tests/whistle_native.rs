#![cfg(feature = "asr-dynamic")]
use gen2::{Runtime, speech::*};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[test]
#[ignore = "requires GEN2_WHISTLE_BUNDLE, GEN2_ORT_LIBRARY, GEN2_WHISTLE_REFERENCE, GEN2_WHISTLE_CORPUS"]
fn real_whistle_matches_reference_and_reports_language_and_completion() {
    let bundle = std::env::var("GEN2_WHISTLE_BUNDLE").expect("GEN2_WHISTLE_BUNDLE");
    let corpus = PathBuf::from(std::env::var("GEN2_WHISTLE_CORPUS").expect("GEN2_WHISTLE_CORPUS"));
    let reference = std::env::var("GEN2_WHISTLE_REFERENCE").expect("GEN2_WHISTLE_REFERENCE");
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(8192)
        .build()
        .unwrap();
    let speech = runtime
        .load_speech(
            SpeechSource::whistle(bundle),
            LoadOptions {
                native_library: Some(
                    std::env::var("GEN2_ORT_LIBRARY")
                        .expect("GEN2_ORT_LIBRARY")
                        .into(),
                ),
                reservation_mb: 2048,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(speech.bundle().model(), SpeechModel::Whistle);
    assert!(!speech.capabilities().cleanup);
    assert_eq!(speech.supported_languages().len(), 7);
    let first = &receipt["records"][0];
    let bytes = std::fs::read(corpus.join(first["pcm"].as_str().unwrap())).unwrap();
    let first_audio = Audio::mono_16khz(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    )
    .unwrap();
    assert!(matches!(
        speech.transcribe_request(TranscriptionRequest {
            audio: first_audio.clone(),
            options: TranscribeOptions::default(),
            settings: TranscriptionSettings {
                max_tokens: Some(1),
                ..Default::default()
            }
        }),
        Err(SpeechError::TokenLimit)
    ));
    let partial = speech
        .transcribe_request(TranscriptionRequest {
            audio: first_audio.clone(),
            options: TranscribeOptions::default(),
            settings: TranscriptionSettings {
                max_tokens: Some(1),
                allow_partial: true,
                ..Default::default()
            },
        })
        .unwrap();
    assert_eq!(partial.completion, Completion::TokenLimit);
    assert_eq!(partial.transcription.token_ids.len(), 1);
    assert!(matches!(
        speech.transcribe_request(TranscriptionRequest {
            audio: first_audio.clone(),
            options: TranscribeOptions::default(),
            settings: TranscriptionSettings {
                max_tokens: Some(0),
                ..Default::default()
            }
        }),
        Err(SpeechError::InvalidRequest(_))
    ));
    assert!(matches!(
        speech.transcribe(
            Audio::mono_16khz(vec![0.; 320]).unwrap(),
            TranscribeOptions::default()
        ),
        Err(SpeechError::InvalidAudio(_))
    ));
    let mut results = Vec::new();
    let mut forced_results = Vec::new();
    for record in receipt["records"].as_array().unwrap() {
        let bytes = std::fs::read(corpus.join(record["pcm"].as_str().unwrap())).unwrap();
        let audio = Audio::mono_16khz(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        )
        .unwrap();
        let audio = if let Some(wav) = record["wav"].as_str() {
            #[cfg(feature = "asr-resample-soxr")]
            let supported = true;
            #[cfg(not(feature = "asr-resample-soxr"))]
            let supported = record["rate"] == 16000;
            if supported {
                let normalized = Audio::from_wav(corpus.join(wav)).unwrap();
                assert_eq!(normalized.samples().len(), audio.samples().len());
                let error = normalized
                    .samples()
                    .iter()
                    .zip(audio.samples())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(error <= 1e-5, "normalization error {error}: {wav}");
                normalized
            } else {
                audio
            }
        } else {
            audio
        };
        let result = speech
            .transcribe_request(TranscriptionRequest {
                audio: audio.clone(),
                options: TranscribeOptions::default(),
                settings: TranscriptionSettings {
                    allow_partial: true,
                    ..Default::default()
                },
            })
            .unwrap();
        assert_eq!(
            result.transcription.text,
            record["text"].as_str().unwrap(),
            "{}",
            record["id"]
        );
        assert_eq!(
            serde_json::to_value(&result.transcription.token_ids).unwrap(),
            record["token_ids"],
            "{}",
            record["id"]
        );
        assert_eq!(
            serde_json::to_value(result.language).unwrap(),
            record["detected_language"]
        );
        assert_eq!(
            result.completion == Completion::EndOfSpeech,
            record["complete"].as_bool().unwrap()
        );
        assert_eq!(
            serde_json::to_value(result.detected_language).unwrap(),
            record["detected_language"]
        );
        let forced = speech
            .transcribe_request(TranscriptionRequest {
                audio,
                options: TranscribeOptions::default(),
                settings: TranscriptionSettings {
                    language: Some(serde_json::from_value(record["language"].clone()).unwrap()),
                    allow_partial: true,
                    ..Default::default()
                },
            })
            .unwrap();
        assert_eq!(
            forced.transcription.text,
            record["forced_text"].as_str().unwrap()
        );
        assert_eq!(
            serde_json::to_value(&forced.transcription.token_ids).unwrap(),
            record["forced_token_ids"]
        );
        assert_eq!(
            serde_json::to_value(forced.language).unwrap(),
            record["language"]
        );
        assert_eq!(
            serde_json::to_value(forced.detected_language).unwrap(),
            record["detected_language"]
        );
        assert_eq!(
            forced.completion == Completion::EndOfSpeech,
            record["forced_complete"].as_bool().unwrap()
        );
        results.push(result);
        forced_results.push(forced);
    }
    for _ in 0..3 {
        assert_eq!(
            speech
                .transcribe(first_audio.clone(), TranscribeOptions::default())
                .unwrap()
                .text,
            first["text"].as_str().unwrap()
        );
    }
    if let Ok(path) = std::env::var("GEN2_WHISTLE_RESULTS") {
        std::fs::write(&path, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
        std::fs::write(
            PathBuf::from(path).with_extension("forced.json"),
            serde_json::to_vec_pretty(&forced_results).unwrap(),
        )
        .unwrap();
    }
    let cancelled = Cancellation::default();
    cancelled.cancel();
    assert!(matches!(
        speech.transcribe(
            Audio::mono_16khz(vec![0.; 640]).unwrap(),
            TranscribeOptions {
                cancellation: cancelled,
                ..Default::default()
            }
        ),
        Err(SpeechError::Cancelled)
    ));
    assert!(matches!(
        speech.transcribe(
            Audio::mono_16khz(vec![0.; 640]).unwrap(),
            TranscribeOptions {
                deadline: Some(Instant::now() - Duration::from_secs(1)),
                ..Default::default()
            }
        ),
        Err(SpeechError::DeadlineExceeded)
    ));
    speech.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
}

#[test]
#[ignore = "requires the pinned Whistle package and explicit native runtime"]
fn real_whistle_integrity_offline_cache_and_lifecycle() {
    let source = PathBuf::from(std::env::var("GEN2_WHISTLE_BUNDLE").expect("Whistle bundle"));
    let library = PathBuf::from(std::env::var("GEN2_ORT_LIBRARY").expect("native runtime"));
    let cache = tempfile::tempdir().unwrap();
    let destination = cache.path().join(format!("whistle-{WHISTLE_REVISION}"));
    for file in [
        "onnx/encoder.onnx",
        "onnx/encoder.onnx.data",
        "onnx/decoder.onnx",
        "onnx/decoder.onnx.data",
        "whistle.pack",
    ] {
        let p = destination.join(file);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::copy(source.join(file), p).unwrap();
    }
    assert_eq!(download_whistle(cache.path()).unwrap(), destination);
    assert_eq!(
        WhistleBundle::open(&destination).unwrap().revision(),
        WHISTLE_REVISION
    );
    let damaged = destination.join("whistle.pack");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&damaged)
        .unwrap()
        .set_len(100)
        .unwrap();
    assert!(matches!(
        WhistleBundle::open(&destination),
        Err(SpeechError::InvalidBundle(_))
    ));
    std::fs::copy(source.join("whistle.pack"), &damaged).unwrap();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(2048)
        .build()
        .unwrap();
    let options = LoadOptions {
        native_library: Some(library.clone()),
        queue_capacity: 1,
        reservation_mb: 2048,
        ..Default::default()
    };
    let speech = runtime
        .load_speech(SpeechSource::whistle(&destination), options.clone())
        .unwrap();
    assert!(matches!(
        runtime.load_speech(SpeechSource::whistle(&destination), options.clone()),
        Err(gen2::Error::Speech(SpeechError::ResourceLimit(_)))
    ));
    let cancel = Cancellation::default();
    let running = speech.clone();
    let cancellation = cancel.clone();
    let active = std::thread::spawn(move || {
        running.transcribe(
            Audio::mono_16khz(vec![0.; 960000]).unwrap(),
            TranscribeOptions {
                cancellation,
                ..Default::default()
            },
        )
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
                    Audio::mono_16khz(vec![0.; 640]).unwrap(),
                    TranscribeOptions {
                        deadline: Some(Instant::now() + Duration::from_millis(50)),
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
    assert_eq!(runtime.speech_reserved_mb(), 2048);
    speech.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
    assert!(matches!(
        speech.transcribe(
            Audio::mono_16khz(vec![0.; 640]).unwrap(),
            TranscribeOptions::default()
        ),
        Err(SpeechError::Unloaded)
    ));
    let other = cache.path().join(library.file_name().unwrap());
    std::fs::copy(&library, &other).unwrap();
    assert!(matches!(
        runtime.load_speech(
            SpeechSource::whistle(&destination),
            LoadOptions {
                native_library: Some(other),
                ..options.clone()
            }
        ),
        Err(gen2::Error::Speech(SpeechError::BackendUnavailable(_)))
    ));
    assert_eq!(runtime.speech_reserved_mb(), 0);
    let reloaded = runtime
        .load_speech(SpeechSource::whistle(destination), options)
        .unwrap();
    reloaded
        .transcribe_request(TranscriptionRequest::new(
            Audio::mono_16khz(vec![0.; 640]).unwrap(),
        ))
        .unwrap();
    reloaded.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
}

#[test]
#[ignore = "requires Bark/Whistle bundles and native runtime"]
fn both_models_share_the_request_interface_and_admission() {
    let bark = std::env::var("GEN2_BARK_BUNDLE").expect("Bark bundle");
    let whistle = std::env::var("GEN2_WHISTLE_BUNDLE").expect("Whistle bundle");
    let library = PathBuf::from(std::env::var("GEN2_ORT_LIBRARY").expect("native runtime"));
    for whistle_first in [true, false] {
        let runtime = Runtime::builder()
            .resident_memory_budget_mb(6145)
            .build()
            .unwrap();
        let load = |source: SpeechSource, mb| {
            runtime
                .load_speech(
                    source,
                    LoadOptions {
                        native_library: Some(library.clone()),
                        reservation_mb: mb,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        let (b, w) = if whistle_first {
            let w = load(SpeechSource::whistle(&whistle), 2048);
            (load(SpeechSource::bark(&bark), 4096), w)
        } else {
            let b = load(SpeechSource::bark(&bark), 4096);
            (b, load(SpeechSource::whistle(&whistle), 2048))
        };
        assert_eq!(runtime.speech_reserved_mb(), 6144);
        for model in [&b, &w] {
            let result = model
                .transcribe_request(
                    TranscriptionRequest::new(Audio::mono_16khz(vec![0.; 640]).unwrap())
                        .language(SpeechLanguage::En),
                )
                .unwrap();
            assert_eq!(result.language, Some(SpeechLanguage::En));
            assert_eq!(result.completion, Completion::EndOfSpeech);
        }
        assert!(matches!(
            b.transcribe_request(
                TranscriptionRequest::new(Audio::mono_16khz(vec![0.; 640]).unwrap())
                    .language(SpeechLanguage::Es)
            ),
            Err(SpeechError::InvalidRequest(_))
        ));
        #[cfg(feature = "laya-dynamic")]
        {
            let laya = runtime
                .load_decider(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke"),
                    gen2::decision::LoadOptions {
                        native_library: Some(library.clone()),
                        ..Default::default()
                    },
                )
                .unwrap();
            laya.decide(
                gen2::decision::DecisionRequest::text("a b")
                    .question("q", gen2::decision::Question::yes_no("a?")),
                gen2::decision::DecisionOptions::default(),
            )
            .unwrap();
            assert_eq!(runtime.decision_reserved_mb(), 1);
            laya.shutdown();
        }
        b.shutdown();
        w.shutdown();
        assert_eq!(runtime.stats().estimated_resident_mb, 0);
    }
}

#[cfg(all(feature = "backend-llamacpp", feature = "laya-dynamic"))]
#[test]
#[ignore = "requires GEN2_WHISTLE_CHAT, both speech bundles and native runtime"]
fn chat_laya_and_both_speech_models_remain_resident_together() {
    let mut settings = gen2::advanced::generation::Settings::default();
    settings.system.gpu_layers = Some(0);
    settings.system.threads = Some(4);
    settings.system.threads_batch = Some(4);
    settings.system.ctx_size = Some(2048);
    let runtime = Runtime::builder()
        .settings(settings)
        .resident_memory_budget_mb(8192)
        .build()
        .unwrap();
    let chat = runtime
        .load(std::env::var("GEN2_WHISTLE_CHAT").expect("chat model"))
        .unwrap();
    let before = chat
        .generate("Reply with one word: hello")
        .max_tokens(24)
        .temperature(0.)
        .text()
        .unwrap();
    let library: PathBuf = std::env::var("GEN2_ORT_LIBRARY")
        .expect("native runtime")
        .into();
    let laya = runtime
        .load_decider(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke"),
            gen2::decision::LoadOptions {
                native_library: Some(library.clone()),
                ..Default::default()
            },
        )
        .unwrap();
    let load = |source, mb| {
        runtime
            .load_speech(
                source,
                LoadOptions {
                    native_library: Some(library.clone()),
                    reservation_mb: mb,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let bark = load(
        SpeechSource::bark(std::env::var("GEN2_BARK_BUNDLE").expect("Bark bundle")),
        4096,
    );
    let whistle = load(
        SpeechSource::whistle(std::env::var("GEN2_WHISTLE_BUNDLE").expect("Whistle bundle")),
        2048,
    );
    assert_eq!(runtime.speech_reserved_mb(), 6144);
    assert_eq!(runtime.decision_reserved_mb(), 1);
    assert!(runtime.residency().is_resident(chat.id()));
    for model in [&bark, &whistle] {
        model
            .transcribe_request(TranscriptionRequest::new(
                Audio::mono_16khz(vec![0.; 640]).unwrap(),
            ))
            .unwrap();
    }
    laya.decide(
        gen2::decision::DecisionRequest::text("a b")
            .question("q", gen2::decision::Question::yes_no("a?")),
        gen2::decision::DecisionOptions::default(),
    )
    .unwrap();
    assert!(runtime.residency().is_resident(chat.id()));
    let after = chat
        .generate("Reply with one word: hello")
        .max_tokens(24)
        .temperature(0.)
        .text()
        .unwrap();
    assert_eq!(before, after);
    assert!(matches!(
        runtime.load_speech(
            SpeechSource::whistle(std::env::var("GEN2_WHISTLE_BUNDLE").unwrap()),
            LoadOptions {
                native_library: Some(library),
                reservation_mb: 2048,
                ..Default::default()
            }
        ),
        Err(gen2::Error::Speech(SpeechError::ResourceLimit(_)))
    ));
    bark.shutdown();
    whistle.shutdown();
    laya.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
    assert_eq!(runtime.decision_reserved_mb(), 0);
}
