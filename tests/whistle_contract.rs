use gen2::{
    Runtime,
    speech::{LoadOptions, SpeechError, SpeechSource, WhistleBundle},
};

#[test]
fn whistle_loading_requires_the_complete_pinned_package() {
    let directory = tempfile::tempdir().unwrap();
    assert!(matches!(
        WhistleBundle::open(directory.path()),
        Err(SpeechError::InvalidBundle(_))
    ));
    let runtime = Runtime::builder().build().unwrap();
    assert!(
        runtime
            .load_speech(
                SpeechSource::whistle(directory.path()),
                LoadOptions::default()
            )
            .is_err()
    );
    assert_eq!(runtime.speech_reserved_mb(), 0);
}
