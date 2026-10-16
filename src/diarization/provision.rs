use super::{DiarizationBundle, DiarizationError, MODEL_FILENAME, MODEL_REVISION, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
};

fn io(error: std::io::Error) -> DiarizationError {
    DiarizationError::Execution(error.to_string())
}
fn complete(path: &Path) -> bool {
    let notice = path.join("README.md");
    DiarizationBundle::open(path).is_ok()
        && std::fs::symlink_metadata(&notice).is_ok_and(|m| m.is_file() && m.len() == 43_239)
        && std::fs::read(notice).is_ok_and(|bytes| {
            hex::encode(Sha256::digest(bytes))
                == "5fe88ca4020d2ee134f806a0a02c52c19855b73047bc2e6f25b9171e1abeebfd"
        })
}

/// Explicitly acquire the pinned Q8 model and its model card into an immutable cache.
/// Warm verified entries work without `hf` or network. A kernel file lock serializes
/// publication across processes; interrupted downloads never replace an entry.
pub fn download_nemotron(cache: impl AsRef<Path>) -> Result<PathBuf> {
    let cache = cache.as_ref();
    std::fs::create_dir_all(cache).map_err(io)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(cache.join(".nemotron.lock"))
        .map_err(io)?;
    lock.lock().map_err(io)?;
    let destination = cache.join(format!("nemotron-{MODEL_REVISION}"));
    if complete(&destination) {
        return Ok(destination);
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = (lock, MODEL_FILENAME);
        Err(DiarizationError::BackendUnavailable(
            "enable hf to provision an uncached model".into(),
        ))
    }
    #[cfg(feature = "hf")]
    {
        let cache = cache.to_path_buf();
        std::thread::spawn(move || {
            // Moving the File into this worker keeps the publication lock held.
            let _lock = lock;
            let staging = tempfile::Builder::new()
                .prefix("nemotron-staging-")
                .tempdir_in(&cache)
                .map_err(io)?;
            let client = hf_hub::HFClient::builder()
                .cache_dir(cache.join("hub"))
                .build_sync()
                .map_err(|e| DiarizationError::Execution(e.to_string()))?;
            let repo = client.model("nvidia", "Nemotron-3-Diarization");
            for force in [false, true] {
                for name in [MODEL_FILENAME, "README.md"] {
                    let source = repo
                        .download_file()
                        .filename(name)
                        .revision(MODEL_REVISION)
                        .force_download(force)
                        .send()
                        .map_err(|e| DiarizationError::Execution(e.to_string()))?;
                    let target = staging.path().join(name);
                    std::fs::copy(source, &target).map_err(io)?;
                    std::fs::File::open(target)
                        .map_err(io)?
                        .sync_all()
                        .map_err(io)?;
                }
                if complete(staging.path()) {
                    break;
                }
                if force {
                    return Err(DiarizationError::InvalidBundle(
                        "downloaded model/card checksum mismatch".into(),
                    ));
                }
            }
            if destination.exists() {
                std::fs::rename(
                    &destination,
                    cache.join(format!(
                        "nemotron-{MODEL_REVISION}.corrupt-{}",
                        uuid::Uuid::new_v4()
                    )),
                )
                .map_err(io)?;
            }
            std::fs::rename(staging.path(), &destination).map_err(io)?;
            Ok(destination)
        })
        .join()
        .map_err(|_| DiarizationError::Execution("diarization provisioner panicked".into()))?
    }
}
