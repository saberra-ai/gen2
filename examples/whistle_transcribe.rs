//! cargo run --release --no-default-features --features asr-dynamic,hf --example whistle_transcribe -- provision CACHE
//! cargo run --release --no-default-features --features asr-dynamic --example whistle_transcribe -- BUNDLE AUDIO.wav [LANGUAGE] [MAX_TOKENS]
use gen2::{Runtime, speech::*};
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("provision") {
        let cache = args
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("supply a cache directory"))?;
        println!("{}", download_whistle(cache)?.display());
        return Ok(());
    }
    let bundle = args
        .first()
        .ok_or_else(|| anyhow::anyhow!("supply BUNDLE AUDIO.wav [LANGUAGE] [MAX_TOKENS]"))?;
    let wav = args
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("supply an audio WAV"))?;
    let language = args
        .get(2)
        .map(|lang| serde_json::from_value(serde_json::Value::String(lang.clone())))
        .transpose()?;
    let max_tokens = args.get(3).map(|n| n.parse()).transpose()?;
    let library = std::env::var("GEN2_ORT_LIBRARY")
        .map_err(|_| anyhow::anyhow!("set GEN2_ORT_LIBRARY to ONNX Runtime 1.24"))?;
    let runtime = Runtime::builder().resident_memory_budget_mb(8192).build()?;
    let speech = runtime.load_speech(
        SpeechSource::whistle(bundle),
        LoadOptions {
            native_library: Some(library.into()),
            reservation_mb: 2048,
            ..Default::default()
        },
    )?;
    let result = speech.transcribe_request(TranscriptionRequest {
        audio: Audio::from_wav(wav)?,
        options: TranscribeOptions::default(),
        settings: TranscriptionSettings {
            language,
            max_tokens,
            allow_partial: true,
        },
    })?;
    speech.shutdown();
    println!("{}", serde_json::to_string_pretty(&result)?);
    if result.completion == Completion::TokenLimit {
        anyhow::bail!(
            "partial transcript: increase the bounded token limit or split the recording"
        );
    }
    Ok(())
}
