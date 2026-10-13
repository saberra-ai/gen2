use super::*;
use std::path::{Path, PathBuf};

/// Provision the complete pinned release into an atomically published cache entry.
/// A validated warm entry returns before constructing a network client. Corrupt
/// entries are preserved with a `.corrupt-*` name when a replacement is ready.
pub fn download_bark(cache: impl AsRef<Path>) -> Result<PathBuf> {
    download(cache.as_ref(), SpeechModel::Bark)
}

/// Provision the pinned FP32 Whistle graphs, external weights and host pack.
pub fn download_whistle(cache: impl AsRef<Path>) -> Result<PathBuf> {
    download(cache.as_ref(), SpeechModel::Whistle)
}

fn open(model: SpeechModel, path: &Path) -> Result<SpeechBundle> {
    match model {
        SpeechModel::Bark => SpeechBundle::open(path),
        SpeechModel::Whistle => WhistleBundle::open(path).map(|b| b.0),
    }
}

fn download(cache: &Path, model: SpeechModel) -> Result<PathBuf> {
    static DOWNLOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = DOWNLOAD
        .lock()
        .map_err(|_| SpeechError::Execution("speech download lock poisoned".into()))?;
    let (name, revision) = match model {
        SpeechModel::Bark => ("bark", MODEL_REVISION),
        SpeechModel::Whistle => ("whistle", WHISTLE_REVISION),
    };
    std::fs::create_dir_all(cache).map_err(io)?;
    let destination = cache.join(format!("{name}-{revision}"));
    if open(model, &destination).is_ok() {
        return Ok(destination);
    }
    #[cfg(not(feature = "hf"))]
    {
        Err(SpeechError::BackendUnavailable(format!(
            "enable hf to provision uncached {name} weights"
        )))
    }
    #[cfg(feature = "hf")]
    {
        // The same synchronous entrypoint is safe when called from a Tokio host.
        let cache = cache.to_path_buf();
        std::thread::spawn(move || {
            let staging = tempfile::Builder::new()
                .prefix(&format!("{name}-staging-"))
                .tempdir_in(&cache)
                .map_err(io)?;
            let client = hf_hub::HFClient::builder()
                .cache_dir(cache.join("hub"))
                .build_sync()
                .map_err(|e| SpeechError::Execution(e.to_string()))?;
            let (owner, repository) = match model {
                SpeechModel::Bark => ("ConwayResearch", "bark-0.6B-4bit-v2.0"),
                SpeechModel::Whistle => ("mrfakename", "whistle-ONNX"),
            };
            let repo = client.model(owner, repository);
            let files: Vec<(String, String)> = match model {
                SpeechModel::Bark => [
                    "manifest.json",
                    "LICENSE",
                    "config.json",
                    "greyhound.json",
                    "vocab.txt",
                    "encoder-model.onnx",
                    "encoder-model.onnx.data",
                    "decoder_joint-model.onnx",
                    "decoder_joint-model.onnx.data",
                ]
                .iter()
                .map(|file| (format!("onnx/{file}"), file.to_string()))
                .collect(),
                SpeechModel::Whistle => {
                    serde_json::from_str::<super::whistle_bundle::Lock>(super::whistle_bundle::LOCK)
                        .map_err(|e| SpeechError::InvalidBundle(e.to_string()))?
                        .files
                        .keys()
                        .map(|file| (file.clone(), file.clone()))
                        .collect()
                }
            };
            for force in [false, true] {
                for (remote, local) in &files {
                    let source = repo
                        .download_file()
                        .filename(remote)
                        .revision(revision)
                        .force_download(force)
                        .send()
                        .map_err(|e| SpeechError::Execution(e.to_string()))?;
                    let output = staging.path().join(local);
                    if let Some(parent) = output.parent() {
                        std::fs::create_dir_all(parent).map_err(io)?;
                    }
                    std::fs::copy(source, output).map_err(io)?;
                }
                // A corrupt Hub blob is not repaired by copying it repeatedly.
                // Retry once with a forced network fetch before publishing.
                match open(model, staging.path()) {
                    Ok(_) => break,
                    Err(e) if force => return Err(e),
                    Err(_) => continue,
                }
            }
            if destination.exists() {
                if open(model, &destination).is_ok() {
                    return Ok(destination);
                }
                let quarantine = cache.join(format!(
                    "{name}-{revision}.corrupt-{}",
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
