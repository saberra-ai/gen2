//! Completed-recording speaker timeline using the admitted Gen2 Rust API.
use gen2::{
    Runtime,
    diarization::{Audio, DiarizeOptions, LoadOptions},
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 3 && args[1] == "--download" {
        println!(
            "{}",
            gen2::diarization::download_nemotron(&args[2])?.display()
        );
        return Ok(());
    }
    if args.len() != 4 {
        return Err(
            "usage: nemotron_diarize MODEL_DIRECTORY NATIVE_C_LIBRARY AUDIO.wav | --download CACHE"
                .into(),
        );
    }
    let runtime = Runtime::builder().resident_memory_budget_mb(2048).build()?;
    let diarizer = runtime.load_diarizer(&args[1], LoadOptions::cpu(PathBuf::from(&args[2])))?;
    let result = diarizer.diarize(Audio::from_wav(&args[3])?, DiarizeOptions::default())?;
    println!("{}", serde_json::to_string(&result)?);
    diarizer.shutdown();
    Ok(())
}
