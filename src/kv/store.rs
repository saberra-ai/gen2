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
    // Saturating: `PIO_KV_BUDGET_MB` is a string from the environment, and a
    // value near u64::MAX would otherwise overflow this multiply. Saturating
    // to "effectively unlimited" is what an absurd budget asked for.
    let budget_bytes = env_u64("PIO_KV_BUDGET_MB", 1024).saturating_mul(1024 * 1024);

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
            // Discount the file only if it is actually gone. A removal that
            // failed (read-only directory, file held open) leaves the bytes
            // on disk, and discounting them anyway makes the running total
            // under-count — the store then reports itself inside a budget
            // it is not inside, and stays over it indefinitely.
            if std::fs::remove_file(path).is_ok() {
                evicted += 1;
                total -= size;
                tracing::info!(
                    target: "pio::gen2::kv::keepwarm",
                    path = %path.display(),
                    "evicted KV blob (store budget)"
                );
            }
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
        // keep_files default is 2 — the oldest must go. Pinned explicitly
        // rather than left to the ambient environment: `enforce_budget`
        // reads process-global vars, so a concurrent test that sets them
        // would otherwise decide this one's outcome.
        let evicted = with_env(
            &[("PIO_KV_KEEP_FILES", "2"), ("PIO_KV_BUDGET_MB", "1024")],
            || enforce_budget(dir),
        );
        assert_eq!(evicted, 1);
        assert!(!path_for_chat(dir, "old").exists());
        assert!(path_for_chat(dir, "mid").exists());
        assert!(path_for_chat(dir, "new").exists());
    }

    /// Write `count` blobs of `size` bytes each, oldest (`chat-0`) first,
    /// ten seconds apart so the eviction order is unambiguous.
    fn seed_blobs(dir: &Path, count: usize, size: usize) -> Vec<PathBuf> {
        let now = std::time::SystemTime::now();
        (0..count)
            .map(|i| {
                let p = path_for_chat(dir, &format!("chat-{i}"));
                std::fs::write(&p, vec![0u8; size]).unwrap();
                let age = std::time::Duration::from_secs((count - i) as u64 * 10);
                std::fs::File::options()
                    .write(true)
                    .open(&p)
                    .unwrap()
                    .set_modified(now - age)
                    .unwrap();
                p
            })
            .collect()
    }

    fn live_bytes(dir: &Path) -> u64 {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "piokv"))
            .map(|e| e.metadata().unwrap().len())
            .sum()
    }

    /// `enforce_budget` reads its limits from the environment, which is
    /// process-global — these tests must not run concurrently.
    fn with_env<T>(vars: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<_> = vars
            .iter()
            .map(|(k, _)| (*k, std::env::var(k).ok()))
            .collect();
        for (k, v) in vars {
            // SAFETY: serialised by LOCK; no other thread in this test binary
            // reads the environment while the guard is held.
            unsafe { std::env::set_var(k, v) };
        }
        let out = f();
        for (k, v) in saved {
            match v {
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
        out
    }

    #[test]
    fn a_hashed_chat_path_stays_inside_the_store_directory() {
        // Chat ids are arbitrary strings and reach this function unfiltered;
        // hashing is what keeps `../../etc/passwd` or an absolute path from
        // becoming a filename component.
        let dir = Path::new("/tmp/kv");
        for hostile in [
            "../../../../etc/passwd",
            "/etc/passwd",
            "..",
            "a/b/c",
            "chat\u{0}id",
            "C:\\Windows\\System32",
            "",
        ] {
            let p = path_for_chat(dir, hostile);
            assert_eq!(
                p.parent(),
                Some(dir),
                "{hostile:?} escaped the store directory: {p:?}"
            );
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            assert_eq!(name.len(), 32 + ".piokv".len(), "{hostile:?} -> {name}");
            assert!(
                name.trim_end_matches(".piokv")
                    .chars()
                    .all(|c| c.is_ascii_hexdigit()),
                "{hostile:?} -> {name}"
            );
        }
    }

    #[test]
    fn the_byte_budget_evicts_beyond_the_file_count() {
        // Two files are inside `keep_files` but past the byte budget, so the
        // budget arm has to fire on its own.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        seed_blobs(dir, 3, 600 * 1024);
        let evicted = with_env(
            &[("PIO_KV_KEEP_FILES", "3"), ("PIO_KV_BUDGET_MB", "1")],
            || enforce_budget(dir),
        );
        assert!(evicted > 0, "byte budget never fired");
        assert!(
            live_bytes(dir) <= 1024 * 1024,
            "store is over its byte budget after enforcement: {} bytes",
            live_bytes(dir)
        );
    }

    #[test]
    fn an_absurd_byte_budget_saturates_rather_than_overflowing() {
        // `PIO_KV_BUDGET_MB` is a string from the environment; the MB→bytes
        // multiply overflowed u64 (a debug-build panic) before it saturated.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        seed_blobs(dir, 1, 64);
        let evicted = with_env(
            &[
                ("PIO_KV_KEEP_FILES", "8"),
                ("PIO_KV_BUDGET_MB", &u64::MAX.to_string()),
            ],
            || enforce_budget(dir),
        );
        assert_eq!(evicted, 0, "an unlimited budget must evict nothing");
    }

    #[test]
    fn a_removal_that_fails_is_not_counted_as_an_eviction() {
        // Fault injection: a read-only store directory still lists, but
        // every unlink fails. The steady state is honest accounting — the
        // caller must not be told bytes were reclaimed that were not.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        seed_blobs(dir, 4, 4096);
        let before = live_bytes(dir);

        let mut perms = std::fs::metadata(dir).unwrap().permissions();
        let original = perms.clone();
        {
            use std::os::unix::fs::PermissionsExt;
            perms.set_mode(0o500); // r-x: listable, not writable
        }
        std::fs::set_permissions(dir, perms).unwrap();

        let evicted = with_env(&[("PIO_KV_KEEP_FILES", "1")], || enforce_budget(dir));

        std::fs::set_permissions(dir, original).unwrap();

        assert_eq!(evicted, 0, "reported evictions that did not happen");
        assert_eq!(before, live_bytes(dir), "files vanished unexpectedly");
    }

    /// macOS/BSD only: `chflags uchg` makes one file's `unlink` fail while
    /// the rest of the directory stays writable. Linux's equivalent
    /// (`chattr +i`) needs root, so this experiment is Apple-gated.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn set_immutable(path: &Path, on: bool) {
        let flag = if on { "uchg" } else { "nouchg" };
        let status = std::process::Command::new("chflags")
            .arg(flag)
            .arg(path)
            .status()
            .expect("chflags");
        assert!(status.success(), "chflags {flag} failed");
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn a_blob_that_cannot_be_removed_does_not_stop_the_sweep_evicting_the_rest() {
        // Fault injection with a *partial* failure, which is where the
        // accounting actually bites. Steady state: the sweep evicts
        // everything it can, so the store ends as close to its budget as
        // the stuck file allows. Discounting an unremovable file from the
        // running total makes the very next file look affordable, and the
        // sweep stops early — leaving the store further over budget than
        // it had to be, with nothing reporting that it did.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let now = std::time::SystemTime::now();
        // Newest first: 300 KiB, then an unremovable 800 KiB, then 300 KiB,
        // against a 1 MiB budget. Only the byte budget decides (keep_files
        // covers all three), and the middle file is the one that sticks.
        let mut paths = Vec::new();
        for (name, size, age) in [
            ("newest", 300 * 1024, 10u64),
            ("stuck", 800 * 1024, 20),
            ("oldest", 300 * 1024, 30),
        ] {
            let p = path_for_chat(dir, name);
            std::fs::write(&p, vec![0u8; size]).unwrap();
            std::fs::File::options()
                .write(true)
                .open(&p)
                .unwrap()
                .set_modified(now - std::time::Duration::from_secs(age))
                .unwrap();
            paths.push(p);
        }
        set_immutable(&paths[1], true);

        let evicted = with_env(
            &[("PIO_KV_KEEP_FILES", "3"), ("PIO_KV_BUDGET_MB", "1")],
            || enforce_budget(dir),
        );

        let newest_exists = paths[0].exists();
        let stuck_exists = paths[1].exists();
        let oldest_exists = paths[2].exists();
        set_immutable(&paths[1], false); // before any assert, so tempdir can drop

        assert!(stuck_exists, "the immutable blob should have survived");
        assert!(newest_exists, "the newest blob was inside budget");
        assert!(
            !oldest_exists,
            "the sweep stopped early: it discounted a file it failed to remove"
        );
        assert_eq!(evicted, 1);
    }

    #[test]
    fn enforcement_is_idempotent_and_leaves_the_newest_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        seed_blobs(dir, 5, 1024);
        let first = with_env(&[("PIO_KV_KEEP_FILES", "2")], || enforce_budget(dir));
        assert_eq!(first, 3);
        let second = with_env(&[("PIO_KV_KEEP_FILES", "2")], || enforce_budget(dir));
        assert_eq!(second, 0, "a second sweep must find nothing left to evict");
        assert!(path_for_chat(dir, "chat-3").exists());
        assert!(path_for_chat(dir, "chat-4").exists());
    }

    #[test]
    fn a_missing_store_directory_is_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(enforce_budget(&tmp.path().join("never-created")), 0);
    }

    #[test]
    fn files_that_are_not_blobs_are_left_alone() {
        // The store shares its directory with whatever else lands there;
        // enforcement must key on the extension, not sweep the folder.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        seed_blobs(dir, 3, 1024);
        std::fs::write(dir.join("notes.txt"), b"keep me").unwrap();
        std::fs::create_dir(dir.join("subdir")).unwrap();
        with_env(&[("PIO_KV_KEEP_FILES", "1")], || enforce_budget(dir));
        assert!(dir.join("notes.txt").exists());
        assert!(dir.join("subdir").exists());
    }

    #[test]
    fn removing_a_blob_that_is_already_gone_is_harmless() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nothing.piokv");
        remove_stale(&missing);
        assert!(!missing.exists());
    }
}
