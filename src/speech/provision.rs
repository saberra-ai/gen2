use super::*;
use std::path::{Path, PathBuf};

/// Provision the complete pinned release into an atomically published cache entry.
/// A validated warm entry returns before constructing a network client. Corrupt
/// entries are preserved with a `.corrupt-*` name when a replacement is ready.
pub fn download_bark(cache: impl AsRef<Path>) -> Result<PathBuf> {
    static DOWNLOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = DOWNLOAD
        .lock()
        .map_err(|_| SpeechError::Execution("speech download lock poisoned".into()))?;
    let cache = cache.as_ref();
    std::fs::create_dir_all(cache).map_err(io)?;
    let destination = cache.join(format!("bark-{MODEL_REVISION}"));
    if SpeechBundle::open(&destination).is_ok() {
        return Ok(destination);
    }
    #[cfg(not(feature = "hf"))]
    {
        Err(SpeechError::BackendUnavailable(
            "enable hf to provision uncached Bark weights".into(),
        ))
    }
    #[cfg(feature = "hf")]
    {
        // The same synchronous entrypoint is safe when called from a Tokio host.
        let cache = cache.to_path_buf();
        std::thread::spawn(move || {
            let staging = tempfile::Builder::new()
                .prefix("bark-staging-")
                .tempdir_in(&cache)
                .map_err(io)?;
            let client = hf_hub::HFClient::builder()
                .cache_dir(cache.join("hub"))
                .build_sync()
                .map_err(|e| SpeechError::Execution(e.to_string()))?;
            let repo = client.model("ConwayResearch", "bark-0.6B-4bit-v2.0");
            for force in [false, true] {
                for name in [
                    "manifest.json",
                    "LICENSE",
                    "config.json",
                    "greyhound.json",
                    "vocab.txt",
                    "encoder-model.onnx",
                    "encoder-model.onnx.data",
                    "decoder_joint-model.onnx",
                    "decoder_joint-model.onnx.data",
                ] {
                    let source = repo
                        .download_file()
                        .filename(format!("onnx/{name}"))
                        .revision(MODEL_REVISION)
                        .force_download(force)
                        .send()
                        .map_err(|e| SpeechError::Execution(e.to_string()))?;
                    std::fs::copy(source, staging.path().join(name)).map_err(io)?;
                }
                // A corrupt Hub blob is not repaired by copying it repeatedly.
                // Retry once with a forced network fetch before publishing.
                match SpeechBundle::open(staging.path()) {
                    Ok(_) => break,
                    Err(e) if force => return Err(e),
                    Err(_) => continue,
                }
            }
            if destination.exists() {
                if SpeechBundle::open(&destination).is_ok() {
                    return Ok(destination);
                }
                let quarantine = cache.join(format!(
                    "bark-{MODEL_REVISION}.corrupt-{}",
                    uuid::Uuid::new_v4()
                ));
                std::fs::rename(&destination, quarantine).map_err(io)?;
            }
            std::fs::rename(staging.path(), &destination).map_err(io)?;
            Ok(destination)
        })
        .join()
        .map_err(|_| SpeechError::Execution("speech provisioner panicked".into()))?
    }
}
fn io(e: std::io::Error) -> SpeechError {
    SpeechError::Execution(e.to_string())
}
