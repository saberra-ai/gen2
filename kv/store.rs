//! On-disk store for keepwarm KV blobs (unsloth-adoption ticket 13).
//!
//! Files are keyed by chat id (hashed — chat ids can be arbitrary
//! strings) under one directory. Blobs are large (measured 221 MB at
//! 12.5K tokens), so a hard budget is enforced on every save: keep the
//! newest `PIO_KV_KEEP_FILES` (default 2) and stay under
//! `PIO_KV_BUDGET_MB` (default 1024) — oldest evicted first.
//!
//! Enablement is env-gated for v1 (`PIO_KV_KEEPWARM=1`): defaults off on
//! desktop (eviction/switching UX unchanged), the Nest daemon opts in.
//! Model identity is NOT part of the key — the blob header carries the
//! full identity and a mismatched restore is a lenient miss, which also
//! self-cleans via the budget.

use std::path::{Path, PathBuf};

/// True when keepwarm persistence is enabled for this process.
pub fn keepwarm_enabled() -> bool {
    std::env::var("PIO_KV_KEEPWARM").ok().as_deref() == Some("1")
}

/// Directory for KV blobs: `PIO_KV_DIR`, else `<local-data>/pio/kv`.
pub fn kv_dir() -> PathBuf {
    if let Ok(d) = std::env::var("PIO_KV_DIR") {
        return PathBuf::from(d);
    }
    dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("pio")
        .join("kv")
}

/// Blob path for a chat id (SHA-256-hashed to a safe filename).
pub fn path_for_chat(dir: &Path, chat_id: &str) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(chat_id.as_bytes());
    let digest = h.finalize();
    let mut name = String::with_capacity(36);
    for b in &digest[..16] {
        name.push_str(&format!("{b:02x}"));
    }
    name.push_str(".piokv");
    dir.join(name)
}

/// Existing blob for a chat, if any.
pub fn candidate_for_chat(dir: &Path, chat_id: &str) -> Option<PathBuf> {
    let p = path_for_chat(dir, chat_id);
    p.is_file().then_some(p)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Enforce the store budget: newest `keep_files` blobs, total size under
/// `budget_mb`. Call after every save. Returns the number of evicted files.
pub fn enforce_budget(dir: &Path) -> usize {
    let keep_files = env_u64("PIO_KV_KEEP_FILES", 2) as usize;
    let budget_bytes = env_u64("PIO_KV_BUDGET_MB", 1024) * 1024 * 1024;

    let mut blobs: Vec<(PathBuf, std::time::SystemTime, u64)> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "piokv"))
            .filter_map(|e| {
                let md = e.metadata().ok()?;
                Some((e.path(), md.modified().ok()?, md.len()))
            })
            .collect(),
        Err(_) => return 0,
    };
    // Newest first.
    blobs.sort_by_key(|b| std::cmp::Reverse(b.1));

    let mut evicted = 0;
    let mut total: u64 = 0;
    for (i, (path, _, size)) in blobs.iter().enumerate() {
        total += size;
        if i >= keep_files || total > budget_bytes {
            if std::fs::remove_file(path).is_ok() {
                evicted += 1;
                tracing::info!(
                    target: "pio::gen2::kv::keepwarm",
                    path = %path.display(),
                    "evicted KV blob (store budget)"
                );
            }
            total -= size;
        }
    }
    evicted
}

/// Remove a blob that failed identity/transcript checks — it can never
/// restore again for this chat and just burns budget.
pub fn remove_stale(path: &Path) {
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_paths_are_stable_and_distinct() {
        let dir = Path::new("/tmp/kv");
        let a1 = path_for_chat(dir, "chat-a");
        let a2 = path_for_chat(dir, "chat-a");
        let b = path_for_chat(dir, "chat-b");
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        assert!(a1.to_string_lossy().ends_with(".piokv"));
    }

    #[test]
    fn budget_keeps_newest_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        for (name, age_secs) in [("old", 300), ("mid", 200), ("new", 100)] {
            let p = path_for_chat(dir, name);
            std::fs::write(&p, vec![0u8; 128]).unwrap();
            let mtime = std::time::SystemTime::now() - std::time::Duration::from_secs(age_secs);
            let f = std::fs::File::options().write(true).open(&p).unwrap();
            f.set_modified(mtime).unwrap();
        }
        // keep_files default is 2 — the oldest must go.
        let evicted = enforce_budget(dir);
        assert_eq!(evicted, 1);
        assert!(!path_for_chat(dir, "old").exists());
        assert!(path_for_chat(dir, "mid").exists());
        assert!(path_for_chat(dir, "new").exists());
    }
}
