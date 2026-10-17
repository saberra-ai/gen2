//! Public Runtime preflight contracts over a sparse synthetic GGUF, never weights.
use gen2::{
    Runtime,
    advanced::{fit::ModelInfo, generation::Settings},
};
#[path = "../src/test_support/local_gguf.rs"]
mod fixtures;
use fixtures::hybrid_header;

#[test]
fn denied_inspection_preserves_the_full_explicit_plan_without_loading() {
    let directory = tempfile::tempdir().unwrap();
    let path = hybrid_header(directory.path());
    let info = ModelInfo::read(&path).unwrap();
    assert_eq!(
        info.memory_needed(1024) - info.memory_needed(0),
        12 * 1024 * 1024
    );
    let mut settings = Settings::default();
    settings.system.ctx_size = Some(1024);
    settings.system.gpu_layers = Some(0);
    settings.system.batch_size = Some(32);
    settings.system.threads = Some(2);
    settings.system.threads_batch = Some(2);
    let runtime = Runtime::builder()
        .settings(settings)
        .max_active_sessions(1)
        // Synthetic refusal fixture only. No application/phone budget override.
        .resident_memory_budget_mb(0)
        .build()
        .unwrap();
    let before = runtime.stats();
    for _ in 0..2 {
        let report = runtime.inspect_local_admission(&path).unwrap();
        assert!(!report.fits_without_eviction);
        assert_eq!(report.plan.weights_mb, 698);
        assert_eq!(report.plan.working_mb, 500);
        assert_eq!(report.plan.context_mb, 12);
        assert_eq!(report.plan.total_mb, 1210);
        assert_eq!(report.plan.context_tokens, 1024);
        assert_eq!(report.plan.concurrent_sessions, 1);
        assert!(!report.plan.context_reduced);
        assert_eq!(report.snapshot.budgets.inference_resident_mb, 0);
        assert_eq!(runtime.stats(), before);
        assert!(runtime.models().is_empty());
    }
}

/// Opt-in acquired-artifact check; never downloads or initializes a backend.
/// The host policy observation is not an installed Android admission result.
#[test]
#[ignore = "requires PNOLA_PHONE_GGUF pointing to the exact pinned acquired artifact"]
fn acquired_phone_header_and_plan_without_native_loading() {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let path = std::path::PathBuf::from(std::env::var_os("PNOLA_PHONE_GGUF").unwrap());
    let verify = || {
        let mut file = std::fs::File::open(&path).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 730895168);
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        assert_eq!(
            hex::encode(hash.finalize()),
            "b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5"
        );
    };
    verify();
    let info = ModelInfo::read(&path).unwrap();
    assert_eq!(info.architecture.as_deref(), Some("lfm2"));
    assert_eq!(info.train_context, Some(128000));
    assert_eq!(info.metadata.block_count, Some(16));
    assert_eq!(info.metadata.embedding_length, Some(2048));
    assert_eq!(info.metadata.head_count, Some(32));
    assert_eq!(
        info.metadata.head_count_kv_per_layer.as_deref(),
        Some([0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0].as_slice())
    );
    assert_eq!(info.memory_needed(1024) - info.memory_needed(0), 12582912);
    let mut settings = Settings::default();
    settings.system.ctx_size = Some(1024);
    settings.system.gpu_layers = Some(0);
    settings.system.batch_size = Some(32);
    settings.system.threads = Some(2);
    settings.system.threads_batch = Some(2);
    let runtime = Runtime::builder()
        .settings(settings)
        .max_active_sessions(1)
        .build()
        .unwrap();
    let before = runtime.stats();
    let report = runtime.inspect_local_admission(&path).unwrap();
    assert_eq!(report.plan.weights_mb, 698);
    assert_eq!(report.plan.working_mb, 500);
    assert_eq!(report.plan.context_mb, 12);
    assert_eq!(report.plan.total_mb, 1210);
    assert_eq!(report.plan.context_tokens, 1024);
    assert_eq!(report.plan.concurrent_sessions, 1);
    assert!(!report.plan.context_reduced);
    assert_eq!(runtime.stats(), before);
    assert!(runtime.models().is_empty());
    println!("acquired_header_plan={report:#?}");
    verify();
}
