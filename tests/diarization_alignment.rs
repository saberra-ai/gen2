use gen2::{
    diarization::*,
    speech::{Timestamp, Transcription},
};

fn transcript(parts: &[(u64, u64, &str)], duration: f64) -> Transcription {
    Transcription {
        fulltext: parts.iter().map(|p| p.2).collect(),
        timestamps: parts
            .iter()
            .map(|p| Timestamp {
                startms: p.0,
                endms: p.1,
                text: p.2.into(),
            })
            .collect(),
        duration_seconds: duration,
        model_revision: "fixture",
        manifest_sha256: "fixture",
        provider: "fixture",
        queue_micros: 0,
        frontend_micros: 0,
        encoder_micros: 0,
        decoder_micros: 0,
        execution_micros: 0,
        token_ids: vec![0; parts.len()],
    }
}
fn timeline(turns: Vec<SpeakerTurn>) -> Diarization {
    Diarization {
        turns,
        audio_duration_seconds: 10.,
        native_frame_count: 1001,
        frame_probabilities: None,
        model_revision: MODEL_REVISION,
        runtime_version: "fixture".into(),
        geometry: Preset::default().geometry(),
        execution_micros: 0,
        native_compute_micros: 0,
        backend: "cpu",
    }
}
fn turn(start: f64, end: f64, speaker: u8) -> SpeakerTurn {
    SpeakerTurn {
        start_seconds: start,
        end_seconds: end,
        speaker,
    }
}

#[test]
fn words_use_whole_token_evidence_and_recording_offsets() {
    let speech = transcript(
        &[(80, 160, "Hel"), (160, 400, "lo"), (400, 560, " world!")],
        1.,
    );
    let result = attribute_bark(&speech, &timeline(vec![turn(5., 6., 2)]), 5.).unwrap();
    assert_eq!(result.text, "Hello world!");
    assert_eq!(result.words.len(), 2);
    assert_eq!(result.words[0].text, "Hello");
    assert_eq!(result.words[0].start_seconds, 5.08);
    assert_eq!(result.words[0].end_seconds, 5.4);
    assert_eq!(result.words[1].start_seconds, 5.4);
    assert_eq!(
        result.words[0].attribution,
        Attribution::Speaker { speaker: 2 }
    );
    assert_eq!(result.timing_method, "bark_token_intervals_v1");
}

#[test]
fn overlap_transitions_gaps_and_zero_duration_stay_explicit() {
    let speech = transcript(
        &[
            (0, 400, "Both"),
            (400, 800, " transition"),
            (800, 2000, " gap"),
            (2000, 2000, " zero"),
        ],
        2.,
    );
    let result = attribute_bark(
        &speech,
        &timeline(vec![turn(0., 0.6, 1), turn(0., 0.4, 2), turn(0.6, 0.8, 2)]),
        0.,
    )
    .unwrap();
    assert_eq!(
        result.words[0].attribution,
        Attribution::Overlap {
            speakers: vec![1, 2]
        }
    );
    assert_eq!(
        result.words[1].attribution,
        Attribution::Unknown {
            reason: AttributionUncertainty::SpeakerTransition
        }
    );
    assert_eq!(result.words[1].candidates.len(), 2);
    assert_eq!(
        result.words[2].attribution,
        Attribution::Unknown {
            reason: AttributionUncertainty::TimingGap
        }
    );
    assert_eq!(
        result.words[3].attribution,
        Attribution::Unknown {
            reason: AttributionUncertainty::ZeroDuration
        }
    );
}

#[test]
fn overlap_counts_cumulative_common_activity() {
    let speech = transcript(&[(0, 40, "Both")], 1.);
    let result = attribute_bark(
        &speech,
        &timeline(vec![
            turn(0., 0.04, 1),
            turn(0., 0.01, 2),
            turn(0.02, 0.03, 2),
        ]),
        0.,
    )
    .unwrap();
    assert_eq!(
        result.words[0].attribution,
        Attribution::Overlap {
            speakers: vec![1, 2]
        }
    );
}

#[test]
fn pairwise_overlap_does_not_include_a_sequential_third_speaker() {
    let speech = transcript(&[(0, 400, "Ambiguous")], 1.);
    let result = attribute_bark(
        &speech,
        &timeline(vec![turn(0., 0.2, 1), turn(0., 0.2, 2), turn(0.2, 0.4, 3)]),
        0.,
    )
    .unwrap();
    assert_eq!(
        result.words[0].attribution,
        Attribution::Unknown {
            reason: AttributionUncertainty::SpeakerTransition
        }
    );
    assert_eq!(
        result.words[0]
            .candidates
            .iter()
            .map(|candidate| candidate.speaker)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn edited_or_invalid_timing_is_rejected_without_retiming_text() {
    let mut speech = transcript(&[(0, 400, "Original")], 1.);
    speech.fulltext = "Edited text".into();
    assert!(matches!(
        attribute_bark(&speech, &timeline(vec![]), 0.),
        Err(DiarizationError::InvalidOutput(_))
    ));
    let speech = transcript(&[(400, 0, "Reversed")], 1.);
    assert!(attribute_bark(&speech, &timeline(vec![]), 0.).is_err());
    let speech = transcript(&[(0, 400, "Beyond end")], 1.);
    assert!(attribute_bark(&speech, &timeline(vec![]), 9.5).is_err());
}
