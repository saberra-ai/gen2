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

#[test]
#[ignore = "requires GEN2_NEMOTRON_ASSETS, GEN2_NEMO_LIBRARY, GEN2_BARK_BUNDLE, GEN2_ORT_LIBRARY, GEN2_CHAT_GGUF; fresh process"]
fn mixed_chat_first() {
    mixed(true);
}

#[test]
#[ignore = "requires GEN2_NEMOTRON_ASSETS, GEN2_NEMO_LIBRARY, GEN2_BARK_BUNDLE, GEN2_ORT_LIBRARY, GEN2_CHAT_GGUF; fresh process"]
fn mixed_diarization_first() {
    mixed(false);
}

fn mixed(chat_first: bool) {
    let assets = std::env::var("GEN2_NEMOTRON_ASSETS").unwrap();
    let library = std::env::var("GEN2_NEMO_LIBRARY").unwrap();
    let bark = std::env::var("GEN2_BARK_BUNDLE").unwrap();
    let ort = PathBuf::from(std::env::var("GEN2_ORT_LIBRARY").unwrap());
    let chat_path = std::env::var("GEN2_CHAT_GGUF").unwrap();
    // Exact auxiliary budget: 4096 MiB Bark + 1024 diarization + 1 smoke Laya.
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(5121)
        .build()
        .unwrap();
    let (chat, diarizer) = if chat_first {
        let chat = runtime.load(&chat_path).unwrap();
        (
            chat,
            runtime
                .load_diarizer(&assets, LoadOptions::cpu(&library))
                .unwrap(),
        )
    } else {
        let diarizer = runtime
            .load_diarizer(&assets, LoadOptions::cpu(&library))
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
    let laya = runtime
        .load_decider(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke"),
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
    assert_eq!(runtime.stats().estimated_resident_mb, 5121);
    assert!(!runtime.residency().is_resident(chat.id()));
    assert!(matches!(
        generate(),
        Err(gen2::Error::Diarization(DiarizationError::ResourceLimit(_)))
    ));
    assert!(matches!(
        runtime.load_diarizer(&assets, LoadOptions::cpu(&library)),
        Err(gen2::Error::Diarization(DiarizationError::ResourceLimit(_)))
    ));
    assert_eq!(runtime.diarization_reserved_mb(), 1024);
    assert!(
        diarizer
            .diarize(
                Audio::mono_16khz(vec![0.; 16000]).unwrap(),
                Default::default()
            )
            .unwrap()
            .turns
            .is_empty()
    );
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
    diarizer.shutdown();
    laya.shutdown();
    assert_eq!(runtime.diarization_reserved_mb(), 0);
    assert_eq!(runtime.decision_reserved_mb(), 0);
    let reloaded = runtime
        .load_diarizer(&assets, LoadOptions::cpu(&library))
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
