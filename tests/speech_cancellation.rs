use gen2::speech::Cancellation;

#[test]
fn a_failed_or_dropped_child_operation_cannot_cancel_its_sibling() {
    let recording = Cancellation::default();
    let diarization = recording.child_token();
    let transcription = recording.child_token();
    diarization.cancel();
    assert!(diarization.is_cancelled());
    assert!(!recording.is_cancelled());
    assert!(!transcription.is_cancelled());
    recording.cancel();
    assert!(transcription.is_cancelled());
    assert!(transcription.child_token().is_cancelled());
}
