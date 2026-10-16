use super::{Diarization, DiarizationError, Result};
use crate::speech::Transcription;

/// Conservative attribution from approximate token timing, not calibrated confidence.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Attribution {
    Speaker {
        speaker: u8,
    },
    /// Multiple native speakers are simultaneously active in this word interval.
    /// This does not establish which voice uttered the cleaned word.
    Overlap {
        speakers: Vec<u8>,
    },
    Unknown {
        reason: AttributionUncertainty,
    },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributionUncertainty {
    ZeroDuration,
    TimingGap,
    InsufficientSpeech,
    SpeakerTransition,
    NonLexicalText,
}

/// Fraction of the measured word interval covered by this speaker's turn union.
/// Coverage is temporal evidence; it is not a probability of correct attribution.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeakerCoverage {
    pub speaker: u8,
    pub fraction: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AttributedWord {
    pub text: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub attribution: Attribution,
    pub candidates: Vec<SpeakerCoverage>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AttributedTranscript {
    /// Exact cleaned Bark text. Editing it requires discarding the old word mapping.
    pub text: String,
    pub words: Vec<AttributedWord>,
    pub offset_seconds: f64,
    pub timing_method: &'static str,
}

type Interval = (f64, f64);

/// Group whole Bark token intervals and attribute by measured temporal overlap.
///
/// `offset_seconds` locates this completed Bark window in the same continuous
/// recording as `timeline`. No character offsets are converted into timestamps.
/// Token intervals longer than 800 ms, zero duration, uncovered speech and
/// speaker transitions remain explicitly uncertain. The 800 ms/50%/20% rules
/// are conservative policy, not a calibrated word aligner or confidence score.
pub fn attribute_bark(
    transcription: &Transcription,
    timeline: &Diarization,
    offset_seconds: f64,
) -> Result<AttributedTranscript> {
    let invalid = |message: &str| DiarizationError::InvalidOutput(message.into());
    let duration = transcription.duration_seconds;
    if !duration.is_finite()
        || !(0.02..=60.).contains(&duration)
        || !offset_seconds.is_finite()
        || offset_seconds < 0.
        || !timeline.audio_duration_seconds.is_finite()
        || !(0.02..=3600.).contains(&timeline.audio_duration_seconds)
        || offset_seconds + duration > timeline.audio_duration_seconds + 1e-6
        || transcription.timestamps.len() > 16_384
        || transcription.fulltext.len() > 1_048_576
        || timeline.turns.len() > 100_000
    {
        return Err(invalid("invalid alignment timebase or output bounds"));
    }
    let mut reconstructed = String::new();
    let mut previous_start = 0;
    let mut previous_end = 0;
    for token in &transcription.timestamps {
        if token.startms > token.endms
            || token.startms < previous_start
            || token.endms < previous_end
            || token.endms as f64 > duration * 1000. + 1e-6
        {
            return Err(invalid("invalid/nonmonotonic Bark token interval"));
        }
        reconstructed.push_str(&token.text);
        if reconstructed.len() > 1_048_576 {
            return Err(invalid("excessive token text"));
        }
        previous_start = token.startms;
        previous_end = token.endms;
    }
    if reconstructed != transcription.fulltext {
        return Err(invalid("Bark text differs from token timing evidence"));
    }
    // Merge each speaker's intervals so duplicated/overlapping turns never
    // double-count coverage. The merged ends are monotonic for binary lookup.
    let mut speakers: [Vec<Interval>; 8] = std::array::from_fn(|_| Vec::new());
    let mut previous = 0.;
    for turn in &timeline.turns {
        if !turn.start_seconds.is_finite()
            || !turn.end_seconds.is_finite()
            || turn.start_seconds < previous
            || turn.start_seconds < 0.
            || turn.end_seconds < turn.start_seconds
            || turn.end_seconds > timeline.audio_duration_seconds + 1e-6
            || !(1..=8).contains(&turn.speaker)
        {
            return Err(invalid("invalid speaker timeline for alignment"));
        }
        previous = turn.start_seconds;
        let intervals = &mut speakers[usize::from(turn.speaker - 1)];
        if let Some(last) = intervals
            .last_mut()
            .filter(|last| turn.start_seconds <= last.1)
        {
            last.1 = last.1.max(turn.end_seconds);
        } else {
            intervals.push((turn.start_seconds, turn.end_seconds));
        }
    }
    let mut words = Vec::new();
    let mut text = String::new();
    let mut interval: Option<Interval> = None;
    for token in &transcription.timestamps {
        for character in token.text.chars() {
            if character.is_whitespace() {
                if let Some((start, end)) = interval.take() {
                    words.push(attribute_word(
                        std::mem::take(&mut text),
                        start,
                        end,
                        &speakers,
                    ));
                    if words.len() > 16_384 {
                        return Err(DiarizationError::ResourceLimit(
                            "too many words in one Bark window".into(),
                        ));
                    }
                }
            } else {
                text.push(character);
                let start = offset_seconds + token.startms as f64 / 1000.;
                let end = offset_seconds + token.endms as f64 / 1000.;
                interval = Some(match interval {
                    None => (start, end),
                    Some((first, last)) => (first.min(start), last.max(end)),
                });
            }
        }
    }
    if let Some((start, end)) = interval {
        words.push(attribute_word(text, start, end, &speakers));
        if words.len() > 16_384 {
            return Err(DiarizationError::ResourceLimit(
                "too many words in one Bark window".into(),
            ));
        }
    }
    Ok(AttributedTranscript {
        text: reconstructed,
        words,
        offset_seconds,
        timing_method: "bark_token_intervals_v1",
    })
}

fn attribute_word(
    text: String,
    start: f64,
    end: f64,
    speakers: &[Vec<Interval>; 8],
) -> AttributedWord {
    let duration = end - start;
    let mut clipped: [Vec<Interval>; 8] = std::array::from_fn(|_| Vec::new());
    let mut candidates = Vec::new();
    if duration > 0. {
        for (index, intervals) in speakers.iter().enumerate() {
            let first = intervals.partition_point(|(_, end)| *end <= start);
            let mut coverage = 0.;
            for (left, right) in intervals[first..]
                .iter()
                .take_while(|(left, _)| *left < end)
            {
                let left = left.max(start);
                let right = right.min(end);
                coverage += (right - left).max(0.);
                clipped[index].push((left, right));
            }
            if coverage > 0. {
                candidates.push(SpeakerCoverage {
                    speaker: index as u8 + 1,
                    fraction: (coverage / duration).min(1.),
                });
            }
        }
    }
    let unknown = |reason| Attribution::Unknown { reason };
    let relevant: Vec<_> = candidates
        .iter()
        .filter(|c| c.fraction >= 0.2)
        .map(|c| c.speaker)
        .collect();
    let attribution = if !text.chars().any(char::is_alphanumeric) {
        unknown(AttributionUncertainty::NonLexicalText)
    } else if duration <= 0. {
        unknown(AttributionUncertainty::ZeroDuration)
    } else if duration > 0.8 + 1e-9 {
        unknown(AttributionUncertainty::TimingGap)
    } else if relevant.len() > 1 {
        // Every reported ID must share the same activity. A qualifying pair
        // cannot establish overlap with another, sequential speaker. Sum the
        // intersection union so brief, separated coactivity still counts.
        let mut common = clipped[usize::from(relevant[0] - 1)].clone();
        for speaker in &relevant[1..] {
            let intervals = &clipped[usize::from(*speaker - 1)];
            let mut intersection = Vec::new();
            let (mut left, mut right) = (0, 0);
            while left < common.len() && right < intervals.len() {
                let start = common[left].0.max(intervals[right].0);
                let end = common[left].1.min(intervals[right].1);
                if end > start {
                    intersection.push((start, end));
                }
                if common[left].1 <= intervals[right].1 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
            common = intersection;
        }
        let simultaneous =
            common.iter().map(|(start, end)| end - start).sum::<f64>() >= 0.02 - 1e-9;
        if simultaneous {
            Attribution::Overlap { speakers: relevant }
        } else {
            unknown(AttributionUncertainty::SpeakerTransition)
        }
    } else if let Some(candidate) = candidates.iter().find(|c| c.fraction >= 0.5) {
        Attribution::Speaker {
            speaker: candidate.speaker,
        }
    } else {
        unknown(AttributionUncertainty::InsufficientSpeech)
    };
    AttributedWord {
        text,
        start_seconds: start,
        end_seconds: end,
        attribution,
        candidates,
    }
}
