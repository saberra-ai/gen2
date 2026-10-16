use crate::{audio, engine::Job};
use gen2::speech::{Audio, TranscribeOptions};
use serde::Serialize;
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[derive(Serialize)]
pub struct Transcript {
    pub text: String,
    pub duration_seconds: f64,
    pub execution_micros: u64,
    pub parts: usize,
    pub parts_completed: usize,
    pub asr_error: Option<String>,
    pub attribution: crate::attribution::SpeakerAttribution,
    pub audio_path: String,
    pub audio_sha256: String,
}

pub fn run(
    job: Job,
    input: &Path,
    send: impl Fn(serde_json::Value) -> Result<(), String>,
) -> Result<Transcript, String> {
    let source = audio::info(input)?;
    let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
    let normalized = directory.path().join("normalized.wav");
    send(serde_json::json!({"type":"progress","stage":"converting"}))?;
    audio::convert(input, &normalized, &job.cancellation)?;
    let (frames, parts) = audio::plan(&normalized, &job.cancellation)?;
    send(serde_json::json!({"type":"progress","stage":"diarizing"}))?;
    let mut attribution = crate::attribution::run(&job, &normalized, frames, |accepted, total| {
        send(
            serde_json::json!({"type":"progress","stage":"diarizing","accepted_audio_seconds":accepted,"total_audio_seconds":total}),
        )
    });
    if job.cancellation.is_cancelled() {
        return Err("Transcription cancelled.".into());
    }
    send(serde_json::json!({"type":"timeline","attribution":attribution}))?;
    send(
        serde_json::json!({"type":"progress","stage":"transcribing","index":0,"total":parts.len()}),
    )?;
    let mut texts = Vec::new();
    let mut execution = 0;
    let mut asr_error = None;
    for (index, (start, count)) in parts.iter().enumerate() {
        if job.cancellation.is_cancelled() {
            return Err("Transcription cancelled.".into());
        }
        let samples = match audio::read_part(&normalized, *start, *count) {
            Ok(samples) => samples,
            Err(error) => {
                asr_error = Some(error);
                break;
            }
        };
        let result = job.require_speech().and_then(|speech| {
            speech
                .transcribe(
                    Audio::mono_16khz(samples).map_err(|e| e.to_string())?,
                    TranscribeOptions {
                        cancellation: job.cancellation.clone(),
                        deadline: Some(Instant::now() + Duration::from_secs(120)),
                    },
                )
                .map_err(|e| e.to_string())
        });
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                asr_error = Some(error);
                break;
            }
        };
        attribution.add_words(&result, *start as f64 / 16000.);
        let mut payload = serde_json::to_value(&result).map_err(|e| e.to_string())?;
        let text = payload
            .get("fulltext")
            .or_else(|| payload.get("text"))
            .and_then(|v| v.as_str())
            .ok_or("Bark returned no transcript text.")?
            .to_owned();
        payload["text"] = serde_json::json!(text);
        texts.push(text);
        execution += result.execution_micros;
        send(
            serde_json::json!({"type":"part","index":index+1,"total":parts.len(),"result":payload}),
        )?;
    }
    if job.cancellation.is_cancelled() {
        return Err("Transcription cancelled.".into());
    }
    attribution.finish_words(asr_error.is_none());
    Ok(Transcript {
        text: texts.join("\n\n"),
        duration_seconds: frames as f64 / 16000.,
        execution_micros: execution,
        parts: parts.len(),
        parts_completed: texts.len(),
        asr_error,
        attribution,
        audio_path: source.path,
        audio_sha256: source.sha256,
    })
}

pub fn verify_bundle() -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let resources = executable
        .parent()
        .and_then(Path::parent)
        .ok_or("Could not locate app resources.")?
        .join("Resources");
    let input = std::env::args()
        .nth(2)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| resources.join("sample.wav"));
    #[cfg(target_os = "macos")]
    {
        let library = resources
            .parent()
            .unwrap()
            .join("Frameworks/libonnxruntime.dylib");
        let loaded = unsafe { libloading::Library::new(library) }
            .map_err(|error| format!("Bundled ONNX library failed: {error:?}"))?;
        drop(loaded);
    }
    let engine = crate::engine::Engine::new();
    let _shutdown = engine.shutdown_guard();
    engine.load(&resources);
    let job = engine
        .claim("bundle-verification".into())
        .map_err(|_| engine.status(false).detail)?;
    let result = run(job, &input, |_| Ok(()))?;
    engine.shutdown();
    if let Some(output) = std::env::args().nth(3) {
        std::fs::write(
            output,
            serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    println!(
        "{}",
        serde_json::json!({"duration_seconds":result.duration_seconds,"parts":result.parts,"words":result.text.split_whitespace().count(),"execution_micros":result.execution_micros,"bundled_resources":true})
    );
    Ok(())
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    fn execute() -> serde_json::Value {
        let engine = crate::engine::Engine::new();
        engine.load(Path::new("/missing-packaged-resources"));
        let input = std::env::var("PNOLA_VERIFY_AUDIO").expect("PNOLA_VERIFY_AUDIO is required");
        let job = engine.claim("qualification".into()).unwrap();
        let result = run(job, Path::new(&input), |_| Ok(())).unwrap();
        engine.shutdown();
        let value = serde_json::to_value(result).unwrap();
        if let Ok(path) = std::env::var("PNOLA_VERIFY_OUTPUT") {
            std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        }
        value
    }
    #[test]
    #[ignore = "requires real Bark/Nemotron native variables, PNOLA_VERIFY_AUDIO and PNOLA_BARK_REFERENCE"]
    fn real_transcript_and_speaker_timeline_preserve_measured_text() {
        let result = execute();
        let expected: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::env::var("PNOLA_BARK_REFERENCE").unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(result["text"], expected["fulltext"]);
        assert!(result["asr_error"].is_null());
        assert_eq!(result["attribution"]["status"], "complete");
        assert_eq!(
            result["attribution"]["timeline"]["native_frame_count"],
            6000
        );
        assert_eq!(
            result["attribution"]["timeline"]["turns"]
                .as_array()
                .unwrap()
                .len(),
            27
        );
        assert_eq!(
            result["attribution"]["words"].as_array().unwrap().len(),
            117
        );
        assert_eq!(result["attribution"]["words_complete"], true);
        assert_eq!(
            result["audio_sha256"],
            "f00f92e53115a4a6724aec0ddaf9c675df6d5fabd3a489b43fffe56ba52a3ae9"
        );
        assert!(
            result["attribution"]["words"]
                .as_array()
                .unwrap()
                .iter()
                .any(|w| w["attribution"]["kind"] == "unknown")
        );
    }
    #[test]
    #[ignore = "use a missing GEN2_BARK_BUNDLE with real Nemotron native variables and PNOLA_VERIFY_AUDIO; fresh process"]
    fn unavailable_asr_preserves_a_standalone_timeline() {
        let result = execute();
        assert_eq!(result["text"], "");
        assert!(result["asr_error"].is_string());
        assert_eq!(result["attribution"]["status"], "complete");
        assert_eq!(
            result["attribution"]["timeline"]["native_frame_count"],
            6000
        );
        assert_eq!(result["attribution"]["words_complete"], false);
    }
    #[test]
    #[ignore = "use a missing GEN2_NEMO_LIBRARY with real Bark native variables and PNOLA_VERIFY_AUDIO; fresh process"]
    fn unavailable_diarization_preserves_a_successful_transcript() {
        let result = execute();
        assert!(!result["text"].as_str().unwrap().is_empty());
        assert!(result["asr_error"].is_null());
        assert_eq!(result["attribution"]["status"], "unavailable");
        assert!(result["attribution"]["reason"].is_string());
    }
}
