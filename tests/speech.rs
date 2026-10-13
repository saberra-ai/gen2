//! Contract tests for the roadmap's agreed speech interface.
use gen2::speech::{Audio, SpeechError};

#[test]
fn pcm_input_rejects_invalid_samples_and_duration() {
    assert!(matches!(
        Audio::mono_16khz(vec![]),
        Err(SpeechError::InvalidAudio(_))
    ));
    assert!(matches!(
        Audio::mono_16khz(vec![f32::NAN; 320]),
        Err(SpeechError::InvalidAudio(_))
    ));
    assert!(matches!(
        Audio::mono_16khz(vec![0.; 960_001]),
        Err(SpeechError::InvalidAudio(_))
    ));
    let audio = Audio::mono_16khz(vec![0.; 16_000]).unwrap();
    assert_eq!(audio.duration_seconds(), 1.0);
}
