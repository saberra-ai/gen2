//! GGUF header parsing over arbitrary bytes.
//!
//! Invariant: arbitrary bytes may parse (`Ok`) or be refused (`Err`), but
//! must never panic, abort, hang, or allocate against a length the file
//! merely *declares*. The estimators downstream of the parse are in scope
//! too — they consume header-supplied dimensions, so an overflow there is
//! reachable from the same bytes.
//!
//! `ModelInfo::read` takes a path, so each input is written to one reused
//! scratch file rather than a fresh temp file per iteration.

#![no_main]

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;

fn scratch_path() -> &'static PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let mut p = std::env::temp_dir();
        p.push(format!("gen2-fuzz-gguf-{}.gguf", std::process::id()));
        p
    })
}

fuzz_target!(|data: &[u8]| {
    let path = scratch_path();
    let Ok(mut f) = std::fs::File::create(path) else {
        return;
    };
    if f.write_all(data).is_err() {
        return;
    }
    drop(f);

    if let Ok(info) = gen2::ModelInfo::read(path) {
        // Whatever the header claimed, every derived number must survive
        // being computed. These are the same calls a host app makes right
        // after reading a model file.
        let _ = info.memory_needed(0);
        let _ = info.memory_needed(u32::MAX);
        let _ = info.parameters;
        let _ = info.train_context;
    }
});
