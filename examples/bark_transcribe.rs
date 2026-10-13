//! Transcribe local WAV files with a single resident Bark model.
use gen2::{
    Runtime,
    speech::{Audio, LoadOptions, TranscribeOptions},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .ok_or("usage: bark_transcribe BUNDLE WAV... (or --download CACHE WAV... with hf)")?;
    let bundle = if first == "--download" {
        gen2::speech::download_bark(args.next().ok_or("--download requires CACHE")?)?
    } else {
        std::path::PathBuf::from(first)
    };
    let wavs: Vec<_> = args.collect();
    if wavs.is_empty() {
        return Err("supply at least one WAV file".into());
    }
    let library = std::env::var("GEN2_ORT_LIBRARY")
        .map_err(|_| "set GEN2_ORT_LIBRARY to the ONNX Runtime 1.24 library")?;
    let runtime = Runtime::builder().build()?;
    let speech = runtime.load_transcriber(
        bundle,
        LoadOptions {
            native_library: Some(library.into()),
            ..Default::default()
        },
    )?;
    for wav in wavs {
        let result = speech.transcribe(Audio::from_wav(&wav)?, TranscribeOptions::default())?;
        println!("{}", serde_json::to_string(&result)?);
    }
    speech.shutdown();
    Ok(())
}
