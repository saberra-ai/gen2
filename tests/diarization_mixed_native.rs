//! Public runtime admission across chat and all local inference operations.
#![cfg(all(
    feature = "backend-diar-nemo",
    feature = "asr-dynamic",
    feature = "laya-dynamic",
    feature = "backend-llamacpp"
))]
use gen2::{
    Runtime,
    diarization::{Audio, DiarizationError, LoadOptions},
};
use std::path::PathBuf;

fn load_options(library: impl Into<PathBuf>) -> LoadOptions {
    match std::env::var("GEN2_NEMO_EXECUTION").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("cpu") => LoadOptions::cpu(library),
        Ok("metal") => LoadOptions::metal(library),
        other => panic!("unsupported GEN2_NEMO_EXECUTION: {other:?}"),
    }
}

fn verify_chat_placement(chat: &gen2::Model, stage: &str) {
    let offload = chat.info().offload;
    println!(
        "chat-placement {stage}: {}",
        serde_json::to_string(&offload).unwrap()
    );
    if let Ok(expected) = std::env::var("GEN2_CHAT_EXPECT_BACKEND") {
        let offload = offload.expect("chat GPU placement must be reported");
        assert!(offload.on_gpu(), "chat unexpectedly fell back to CPU");
        assert_eq!(offload.backend.as_deref(), Some(expected.as_str()));
        assert!(offload.device.is_some_and(|device| !device.is_empty()));
    }
}

#[test]
#[ignore = "requires GEN2_NEMOTRON_ASSETS, GEN2_NEMO_LIBRARY, GEN2_BARK_BUNDLE, GEN2_ORT_LIBRARY, GEN2_CHAT_GGUF; fresh process"]
fn mixed_chat_first() {
    mixed(true, false);
}

#[test]
#[ignore = "requires GEN2_NEMOTRON_ASSETS, GEN2_NEMO_LIBRARY, GEN2_BARK_BUNDLE, GEN2_ORT_LIBRARY, GEN2_CHAT_GGUF; fresh process"]
fn mixed_diarization_first() {
    mixed(false, false);
}

#[test]
#[ignore = "same native variables plus GEN2_LAYA_BUNDLE for actual Laya weights; fresh process"]
fn mixed_real_laya_chat_first() {
    mixed(true, true);
}

#[test]
#[ignore = "same native variables plus GEN2_LAYA_BUNDLE for actual Laya weights; fresh process"]
fn mixed_real_laya_diarization_first() {
    mixed(false, true);
}

fn mixed(chat_first: bool, real_laya: bool) {
    let assets = std::env::var("GEN2_NEMOTRON_ASSETS").unwrap();
    let library = std::env::var("GEN2_NEMO_LIBRARY").unwrap();
    let bark = std::env::var("GEN2_BARK_BUNDLE").unwrap();
    let ort = PathBuf::from(std::env::var("GEN2_ORT_LIBRARY").unwrap());
    let chat_path = std::env::var("GEN2_CHAT_GGUF").unwrap();
    let laya_bundle = if real_laya {
        PathBuf::from(std::env::var("GEN2_LAYA_BUNDLE").expect("actual Laya bundle required"))
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke")
    };
    let laya_mb = gen2::decision::LayaBundle::open(&laya_bundle)
        .unwrap()
        .manifest()
        .envelope
        .reservation_mb;
    // Fill the exact aggregate budget so chat restoration must be refused.
    let budget = 4096 + 1024 + laya_mb;
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(budget)
        .build()
        .unwrap();
    let (chat, diarizer) = if chat_first {
        let chat = runtime.load(&chat_path).unwrap();
        (
            chat,
            runtime
                .load_diarizer(&assets, load_options(&library))
                .unwrap(),
        )
    } else {
        let diarizer = runtime
            .load_diarizer(&assets, load_options(&library))
            .unwrap();
        (runtime.load(&chat_path).unwrap(), diarizer)
    };
    let generate = || {
        chat.generate("Reply with the single word hello.")
            .max_tokens(8)
            .temperature(0.)
            .run()
    };
    assert!(!generate().unwrap().text().is_empty());
    verify_chat_placement(&chat, "initial");
    let laya = runtime
        .load_decider(
            &laya_bundle,
            gen2::decision::LoadOptions {
                native_library: Some(ort.clone()),
                ..Default::default()
            },
        )
        .unwrap();
    let speech = runtime
        .load_transcriber(
            &bark,
            gen2::speech::LoadOptions {
                native_library: Some(ort),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(runtime.stats().estimated_resident_mb, budget);
    assert!(!runtime.residency().is_resident(chat.id()));
    assert!(matches!(
        generate(),
        Err(gen2::Error::Diarization(DiarizationError::ResourceLimit(_)))
    ));
    assert!(matches!(
        runtime.load_diarizer(&assets, load_options(&library)),
        Err(gen2::Error::Diarization(DiarizationError::ResourceLimit(_)))
    ));
    assert_eq!(runtime.diarization_reserved_mb(), 1024);
    let result = diarizer
        .diarize(
            Audio::mono_16khz(vec![0.; 16000]).unwrap(),
            Default::default(),
        )
        .unwrap();
    assert!(result.turns.is_empty());
    match load_options(&library).device {
        gen2::diarization::ExecutionDevice::Cpu => {
            assert_eq!(result.backend, "cpu");
            assert!(result.device.is_none());
        }
        gen2::diarization::ExecutionDevice::Metal => {
            assert_eq!(result.backend, "metal");
            assert!(result.device.is_some_and(|device| !device.is_empty()));
        }
    }
    laya.decide(
        gen2::decision::DecisionRequest::text("a b")
            .question("q", gen2::decision::Question::yes_no("a?")),
        Default::default(),
    )
    .unwrap();
    speech
        .transcribe(
            gen2::speech::Audio::mono_16khz(vec![0.; 320]).unwrap(),
            Default::default(),
        )
        .unwrap();
    speech.shutdown();
    assert_eq!(runtime.speech_reserved_mb(), 0);
    assert!(!generate().unwrap().text().is_empty());
    verify_chat_placement(&chat, "restored");
    diarizer.shutdown();
    laya.shutdown();
    assert_eq!(runtime.diarization_reserved_mb(), 0);
    assert_eq!(runtime.decision_reserved_mb(), 0);
    let reloaded = runtime
        .load_diarizer(&assets, load_options(&library))
        .unwrap();
    reloaded
        .diarize(
            Audio::mono_16khz(vec![0.; 320]).unwrap(),
            Default::default(),
        )
        .unwrap();
    reloaded.shutdown();
    assert_eq!(runtime.diarization_reserved_mb(), 0);
}
