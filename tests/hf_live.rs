//! The five-minute first run, end to end: an `hf:` reference downloads from
//! the Hugging Face Hub into an empty cache, loads, and generates; the same
//! reference then loads again with no network at all.
//!
//! Needs the network and ~400 MB, so it runs only when asked:
//!
//! ```sh
//! GEN2_TEST_HF=1 cargo test --test hf_live -- --nocapture
//! ```
//!
//! Without `GEN2_TEST_HF=1` it skips. With it, every failure is a failure.
//! One test function on purpose: it sets process environment variables
//! (`GEN2_MODELS_DIR`, `HF_ENDPOINT`), which two tests in one binary would
//! race on.

#![cfg(all(feature = "backend-llamacpp", feature = "hf"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use gen2::hf::{HfError, HfModel};

const REPO: &str = "unsloth/Qwen3-0.6B-GGUF";
const REFERENCE: &str = "hf:unsloth/Qwen3-0.6B-GGUF";
const EXPECTED_FILE: &str = "Qwen3-0.6B-Q4_K_M.gguf";
/// The size the Hub reported for the file when this test was written; the
/// assertion is loose so a re-upload with a different tokenizer blob does
/// not fail it, but a wrong quant (Q8_0 is 639 MB) does.
const EXPECTED_BYTES: u64 = 396_705_472;

#[test]
fn a_reference_downloads_loads_generates_and_then_loads_offline() {
    if std::env::var("GEN2_TEST_HF").as_deref() != Ok("1") {
        eprintln!("SKIP: set GEN2_TEST_HF=1 to download from the Hugging Face Hub");
        return;
    }
    let cache = tempfile::tempdir().expect("temp cache dir");
    // SAFETY: single-threaded at this point — one test in the binary, and
    // nothing has spawned yet.
    unsafe { std::env::set_var(gen2::hf::MODELS_DIR_VAR, cache.path()) };
    assert_eq!(gen2::hf::cache_dir(), cache.path());

    // ── The other grammar forms resolve against the live listing ────────
    let q8 = HfModel::parse("hf:unsloth/Qwen3-0.6B-GGUF:Q8_0")
        .unwrap()
        .resolve()
        .expect(":Q8_0 resolves");
    assert_eq!(q8.file, "Qwen3-0.6B-Q8_0.gguf");
    assert!(!q8.from_cache);
    let exact = HfModel::parse("hf:unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q5_K_M.gguf")
        .unwrap()
        .resolve()
        .expect(":file.gguf resolves");
    assert_eq!(exact.file, "Qwen3-0.6B-Q5_K_M.gguf");
    match HfModel::parse("hf:unsloth/Qwen3-0.6B-GGUF:Q9_9")
        .unwrap()
        .resolve()
    {
        Err(HfError::QuantNotFound { available, .. }) => {
            assert!(available.iter().any(|t| t == "Q4_K_M"), "{available:?}");
            eprintln!("--- Q9_9 refused; repo has: {}", available.join(", "));
        }
        other => panic!("a missing quant should list what exists, got {other:?}"),
    }
    match HfModel::parse("hf:unsloth/this-repo-does-not-exist-gen2")
        .unwrap()
        .resolve()
    {
        // Observed: 404 with `x-error-code: RepoNotFound` anonymously. A
        // private repo answers 401 instead, which is the gated text.
        Err(HfError::RepoNotFound { .. } | HfError::Gated { .. }) => {}
        other => panic!("a missing repo should be a distinct error, got {other:?}"),
    }

    // ── Cold: download (with a progress hook), load, generate ───────────
    let events = Arc::new(AtomicU64::new(0));
    let seen_total = Arc::new(AtomicU64::new(0));
    let (e, t) = (Arc::clone(&events), Arc::clone(&seen_total));
    let with_hook = HfModel::parse(REFERENCE).unwrap().on_progress(move |p| {
        e.fetch_add(1, Ordering::SeqCst);
        t.fetch_max(p.total, Ordering::SeqCst);
    });
    let runtime = gen2::Runtime::new().unwrap();
    let started = Instant::now();
    let model = runtime.load_hf(with_hook).expect("cold load from the Hub");
    let cold = started.elapsed();
    let info = model.info();
    eprintln!(
        "--- cold load (download + load): {:.1}s; {} progress events, total {} bytes; source {:?}",
        cold.as_secs_f64(),
        events.load(Ordering::SeqCst),
        seen_total.load(Ordering::SeqCst),
        info.source
    );
    assert!(
        events.load(Ordering::SeqCst) > 0,
        "the progress hook never fired on a real download"
    );
    assert!(
        seen_total.load(Ordering::SeqCst).abs_diff(EXPECTED_BYTES) < 20_000_000,
        "progress reported a total of {} bytes",
        seen_total.load(Ordering::SeqCst)
    );
    assert_eq!(
        info.source,
        gen2::model::ModelSourceKind::HuggingFace {
            repo: REPO.into(),
            file: EXPECTED_FILE.into(),
        }
    );
    assert!(info.local);
    let cached = gen2::hf::cached_gguf_files(cache.path(), REPO);
    assert_eq!(cached.len(), 1, "exactly the chosen file: {cached:?}");
    assert_eq!(cached[0].name, EXPECTED_FILE);
    let bytes = cached[0].size.unwrap_or(0);
    assert!(
        bytes.abs_diff(EXPECTED_BYTES) < 20_000_000,
        "downloaded {bytes} bytes, expected about {EXPECTED_BYTES}"
    );

    let text = model
        .generate("Reply with exactly one word: hello")
        .max_tokens(8)
        .text()
        .expect("generation");
    eprintln!("--- generated: {text:?}");
    assert!(
        !text.trim().is_empty(),
        "the model loaded but produced nothing"
    );

    // ── Warm: the same reference, with the Hub unreachable ──────────────
    // A closed local port: any request at all now fails, so a passing load
    // is proof of zero network, not of a lucky cache check.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    // SAFETY: as above — the engine's threads never read the environment.
    unsafe { std::env::set_var("HF_ENDPOINT", format!("http://127.0.0.1:{port}")) };

    let fired = Arc::new(AtomicU64::new(0));
    let hook = Arc::clone(&fired);
    let typed = HfModel::parse(REFERENCE).unwrap().on_progress(move |_| {
        hook.fetch_add(1, Ordering::SeqCst);
    });
    let download = typed
        .download()
        .expect("warm download resolves from the cache");
    assert!(download.cache_hit, "{download:?}");
    assert!(download.resolved.from_cache);
    assert_eq!(
        fired.load(Ordering::SeqCst),
        0,
        "the progress hook fired on a cache hit"
    );

    let started = Instant::now();
    let again = gen2::load(REFERENCE).expect("warm load with the Hub unreachable");
    eprintln!("--- warm load: {:.1}s", started.elapsed().as_secs_f64());
    let again_text = again
        .generate("Reply with exactly one word: hello")
        .max_tokens(8)
        .text()
        .expect("generation after a warm load");
    assert!(!again_text.trim().is_empty());

    // The typed form loads from the cache too, and reports the same source.
    let typed_model = runtime
        .load_hf(typed)
        .expect("Runtime::load_hf from the cache");
    assert_eq!(typed_model.info().source, info.source);
    assert_eq!(fired.load(Ordering::SeqCst), 0);

    // And a reference the cache cannot answer fails as a network error, not
    // a hang or a panic.
    match HfModel::parse("hf:unsloth/Qwen3-0.6B-GGUF:Q6_K")
        .unwrap()
        .resolve()
    {
        Err(HfError::Network { .. }) => {}
        other => panic!("offline miss should be a network error, got {other:?}"),
    }
}
