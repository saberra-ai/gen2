#![cfg(feature = "backend-llamacpp")]

use std::path::PathBuf;

use gen2::output::{FinishReason, OutputMode};
use gen2::{Error, Event, Message, Runtime, Session, VoiceInput};

#[test]
#[ignore = "requires LFM3 Omni GGUF fixtures, pineapple.f32le, and red.png"]
fn lfm3_encodes_microphone_before_commit_and_streams_a_reply() -> gen2::Result<()> {
    let fixtures =
        PathBuf::from(std::env::var_os("GEN2_LFM3_FIXTURES").expect("fixture directory"));
    let mut settings = gen2::advanced::generation::Settings::default();
    settings.system.ctx_size = Some(4096);
    settings.prompt.include_meta = Some(false);
    let runtime = Runtime::builder().settings(settings).build()?;
    let model = runtime.load_with_projector(
        fixtures.join("LFM3-Omni-2B-Q4_K_M.gguf"),
        fixtures.join("mmproj-LFM3-Omni-2B-Q4_K_M.gguf"),
    )?;
    assert!(model.capabilities().microphone);

    let pcm_bytes = std::fs::read(fixtures.join("pineapple.f32le")).expect("audio fixture");
    let (frames, remainder) = pcm_bytes.as_chunks::<4>();
    assert!(
        remainder.is_empty(),
        "fixture must contain complete f32 samples"
    );
    let pcm: Vec<_> = frames.iter().copied().map(f32::from_le_bytes).collect();

    let mut transcription = Session::new().with_system("Perform ASR in English.");
    let mut mic = model
        .turn(&mut transcription)
        .greedy()
        .max_tokens(48)
        .start_microphone()?;
    assert!(mic.push(&[f32::NAN]).is_err());
    let mut prefilled_rows = 0;
    for chunk in pcm.chunks(1600) {
        prefilled_rows += mic.push(chunk)?;
    }
    assert!(
        prefilled_rows > 0,
        "encoder must advance the decoder during capture"
    );
    let response = mic.commit()?.finish()?;
    eprintln!(
        "microphone encoded {prefilled_rows} rows before commit, heard {:?}",
        response.text()
    );
    assert!(response.text().to_lowercase().contains("pineapple"));
    assert_eq!(transcription.messages().len(), 2);

    let mut conversation = Session::new().with_system(
        "Repeat the secret word in the user's audio. Respond with interleaved text and audio.",
    );
    let mut mic = model
        .turn(&mut conversation)
        .output_mode(OutputMode::TextAndSpeech)
        .greedy()
        .max_tokens(128)
        .start_microphone()?;
    for chunk in pcm.chunks(1000) {
        mic.push(chunk)?;
    }

    let mut events = mic.commit()?;
    let mut speech_samples = 0;
    let mut reply = String::new();
    for event in &mut events {
        match event? {
            Event::SpeechDelta(chunk) => {
                assert_eq!(chunk.sample_rate, 24_000);
                assert!(chunk.samples.iter().all(|sample| sample.is_finite()));
                speech_samples += chunk.samples.len();
            }
            Event::TextDelta(delta) => reply.push_str(&delta),
            _ => {}
        }
    }
    let response = events.finish()?;
    assert_eq!(response.text(), reply);
    assert!(reply.to_lowercase().contains("pineapple"), "{reply:?}");
    assert!(speech_samples > 2_400);
    eprintln!("microphone reply: {reply:?}, {speech_samples} streamed samples");

    let initial_messages = conversation.messages().len();
    {
        let mut mic = model
            .turn(&mut conversation)
            .user("This turn should be discarded.")
            .start_microphone()?;
        assert!(mic.push(&pcm[..12_800])? > 0);
    }
    assert_eq!(conversation.messages().len(), initial_messages);

    let reply = model
        .turn(&mut conversation)
        .user("What was the secret word? Answer in one word.")
        .greedy()
        .max_tokens(32)
        .run()?;
    assert_eq!(reply.finish_reason(), &FinishReason::Stop);
    assert!(reply.text().to_lowercase().contains("pineapple"));

    let mut mic = model
        .turn(&mut conversation)
        .greedy()
        .max_tokens(48)
        .start_microphone()?;
    let mut rows = 0;
    for chunk in pcm.chunks(5120) {
        rows += mic.push(chunk)?;
    }
    assert!(rows > 0);
    let reply = mic.commit()?.finish()?.text();
    assert!(reply.to_lowercase().contains("pineapple"), "{reply:?}");

    let journal = serde_json::to_vec(&conversation).unwrap();
    let mut short = model
        .turn(&mut conversation)
        .user("Discard the short capture and this message.")
        .start_microphone()?;
    assert_eq!(short.push(&pcm[..319])?, 0);
    assert!(short.commit().is_err());
    assert_eq!(serde_json::to_vec(&conversation).unwrap(), journal);

    let mut cancelled = model
        .turn(&mut conversation)
        .user("Discard the cancelled capture and this message.")
        .start_microphone()?;
    assert!(cancelled.push(&pcm[..12_800])? > 0);
    let canceller = cancelled.canceller();
    std::thread::spawn(move || canceller.cancel())
        .join()
        .unwrap();
    assert_eq!(
        cancelled.push(&pcm[..320]).unwrap_err().code(),
        Some("cancelled")
    );
    drop(cancelled);
    assert_eq!(serde_json::to_vec(&conversation).unwrap(), journal);
    let reply = model
        .turn(&mut conversation)
        .user("What was the secret word? Answer in one word.")
        .greedy()
        .max_tokens(32)
        .run()?;
    assert!(reply.text().to_lowercase().contains("pineapple"));

    let frame = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
    std::fs::copy(fixtures.join("red.png"), frame.path()).unwrap();
    let mut warm = Session::new();
    let reply = model
        .turn(&mut warm)
        .image(frame.path())
        .user("Name the color in the image.")
        .greedy()
        .max_tokens(32)
        .run()?;
    assert!(reply.text().to_lowercase().contains("red"));
    frame.close().unwrap();
    let before = serde_json::to_vec(&warm).unwrap();
    // Appending a caller-authored assistant message requires a replay. The
    // previous image is gone, so a continuation that silently skips the new
    // assistant message would wrongly return a successful capture handle.
    let err = model
        .turn(&mut warm)
        .message(Message::assistant_structured("I also saw a circle.", None))
        .start_microphone()
        .unwrap_err();
    assert!(err.to_string().contains("open image"), "{err}");
    assert_eq!(serde_json::to_vec(&warm).unwrap(), before);

    let stereo_48khz: Vec<_> = pcm.iter().flat_map(|&sample| [sample; 6]).collect();
    let mut voice = Session::new().with_system(
        "Repeat the secret word in the user's audio. Respond with interleaved text and audio.",
    );
    let mut played = 0;
    let response = model.turn(&mut voice).greedy().max_tokens(128).voice(
        VoiceInput::new(stereo_48khz.chunks(997).map(Ok), 48_000, 2),
        |speech| {
            assert_eq!(speech.sample_rate, 24_000);
            assert!(speech.samples.iter().all(|sample| sample.is_finite()));
            played += speech.samples.len();
            Ok(())
        },
    )?;
    assert!(response.text().to_lowercase().contains("pineapple"));
    assert!(played > 2400, "the host must receive speech chunks");

    let before = voice.messages().len();
    let sink_error = model
        .turn(&mut voice)
        .greedy()
        .max_tokens(128)
        .voice(VoiceInput::new(pcm.chunks(1600).map(Ok), 16_000, 1), |_| {
            Err(Error::Generation {
                code: "speaker_failed".into(),
                message: "playback stopped".into(),
            })
        })
        .unwrap_err();
    assert_eq!(sink_error.code(), Some("speaker_failed"));
    assert_eq!(
        voice.messages().len(),
        before + 1,
        "capture committed before playback"
    );
    Ok(())
}
