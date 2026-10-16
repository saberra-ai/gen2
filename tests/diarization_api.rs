use gen2::{
    Runtime,
    diarization::{Audio, DiarizationError, LoadOptions},
};

#[test]
fn input_requires_bounded_finite_mono_pcm() {
    assert!(Audio::mono_16khz(vec![0.; 320]).is_ok());
    assert!(matches!(
        Audio::mono_16khz(Vec::new()),
        Err(DiarizationError::InvalidAudio(_))
    ));
    assert!(matches!(
        Audio::mono_16khz(vec![f32::NAN; 320]),
        Err(DiarizationError::InvalidAudio(_))
    ));
    assert!(matches!(
        Audio::mono_16khz(vec![0.; 120 * 16_000 + 1]),
        Err(DiarizationError::InvalidAudio(_))
    ));
}

#[test]
fn unavailable_runtime_does_not_reserve_memory() {
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(2048)
        .build()
        .unwrap();
    let missing = tempfile::tempdir().unwrap();
    assert!(matches!(
        runtime.load_diarizer(
            missing.path(),
            LoadOptions::cpu(missing.path().join("missing"))
        ),
        Err(gen2::Error::Diarization(
            DiarizationError::BackendUnavailable(_)
        ))
    ));
    assert_eq!(runtime.diarization_reserved_mb(), 0);
}

#[cfg(not(feature = "hf"))]
#[test]
fn uncached_provisioning_requires_download_feature() {
    let cache = tempfile::tempdir().unwrap();
    assert!(matches!(
        gen2::diarization::download_nemotron(cache.path()),
        Err(DiarizationError::BackendUnavailable(_))
    ));
}

#[test]
fn placement_requires_an_explicit_backend_without_implicit_gpu_fallback() {
    use gen2::diarization::ExecutionDevice;
    assert_eq!(LoadOptions::cpu("native").device, ExecutionDevice::Cpu);
    assert_eq!(LoadOptions::metal("native").device, ExecutionDevice::Metal);
}
