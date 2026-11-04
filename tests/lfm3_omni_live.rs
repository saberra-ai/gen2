#![cfg(feature = "backend-llamacpp")]

use std::path::PathBuf;

use gen2::input::{Image, Video};
use gen2::output::{OutputMode, OutputPart};
use gen2::{Event, Input, Runtime, Session, ToolDefinition};

#[test]
#[ignore = "requires the LFM3 Omni GGUF, projector, red.png, and pineapple.f32le fixtures"]
fn lfm3_omni_embedded_turns() -> gen2::Result<()> {
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
    let caps = model.capabilities();
    assert!(caps.text && caps.images && caps.audio && caps.video && caps.speech);

    let text = model
        .generate("What is the capital of France? Answer in one word.")
        .greedy()
        .max_tokens(24)
        .text()?;
    eprintln!("text: {text:?}");
    assert!(text.contains("Paris"));

    let image = fixtures.join("red.png");
    let vision = model
        .generate(Input::new().image(&image).text("What color is this image?"))
        .greedy()
        .max_tokens(24)
        .text()?;
    eprintln!("image: {vision:?}");
    assert!(vision.to_lowercase().contains("red"));

    let pcm_bytes = std::fs::read(fixtures.join("pineapple.f32le")).expect("audio fixture");
    let pcm: Vec<_> = pcm_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let heard = model
        .generate(Input::new().audio_pcm(pcm.clone()))
        .system("Perform ASR in English.")
        .greedy()
        .max_tokens(48)
        .text()?;
    eprintln!("audio: {heard:?}");
    assert!(heard.to_lowercase().contains("pineapple"));

    let video = Video::new(vec![Image::from_path(&image); 4], pcm.clone());
    let saw_and_heard = model
        .generate(
            Input::new()
                .video(video)
                .text("What color is on screen and what is the secret word in the audio?"),
        )
        .system("Answer questions about the video using what you see and hear.")
        .greedy()
        .max_tokens(64)
        .text()?;
    eprintln!("video with sound: {saw_and_heard:?}");
    let answer = saw_and_heard.to_lowercase();
    assert!(answer.contains("red") && answer.contains("pineapple"));

    let spoken = model
        .generate("The quick brown fox jumps over the lazy dog.")
        .system("Perform TTS in American English using a female voice.")
        .output_mode(OutputMode::Speech)
        .greedy()
        .max_tokens(100)
        .run()?;
    assert!(
        spoken
            .message
            .content
            .iter()
            .any(|part| matches!(part, OutputPart::Speech(chunk)
            if chunk.sample_rate == 24_000 && chunk.samples.len() > 2_400
                && chunk.samples.iter().all(|x| x.is_finite())
                && chunk.samples.iter().any(|x| x.abs() > 0.005)))
    );

    assert!(spoken.text().trim().is_empty(), "{:?}", spoken.text());

    let mut session = Session::new().with_system("Respond with interleaved text and audio.");
    let mut stream = model
        .turn(&mut session)
        .input(
            Input::new()
                .audio_pcm(pcm.clone())
                .text("Repeat the secret word."),
        )
        .output_mode(OutputMode::TextAndSpeech)
        .greedy()
        .max_tokens(128)
        .stream()?;
    let mut streamed_text = String::new();
    let mut streamed_samples = 0;
    for event in &mut stream {
        match event? {
            Event::TextDelta(text) => streamed_text.push_str(&text),
            Event::SpeechDelta(chunk) => streamed_samples += chunk.samples.len(),
            _ => {}
        }
    }
    let response = stream.finish()?;
    eprintln!("interleaved: {streamed_text:?}, {streamed_samples} samples");
    assert!(streamed_text.to_lowercase().contains("pineapple"));
    assert!(streamed_samples > 2_400);
    assert_eq!(response.text(), streamed_text);
    assert_eq!(
        response
            .message
            .content
            .iter()
            .filter_map(|part| match part {
                OutputPart::Speech(chunk) => Some(chunk.samples.len()),
                _ => None,
            })
            .sum::<usize>(),
        streamed_samples,
    );

    let followup = model
        .turn(&mut session)
        .user("What word did I ask you to repeat? Answer in one word.")
        .greedy()
        .max_tokens(24)
        .run()?;
    eprintln!("followup: {:?}", followup.text());
    assert_eq!(followup.text().trim().to_lowercase(), "pineapple");

    let frame = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
    std::fs::copy(&image, frame.path()).unwrap();
    let mut media_session = Session::new().with_system("Respond in text.");
    let first = model
        .turn(&mut media_session)
        .input(
            Input::new()
                .image(frame.path())
                .text("Name the color on screen."),
        )
        .greedy()
        .max_tokens(32)
        .run()?;
    assert!(first.text().to_lowercase().contains("red"));
    // Replaying the transcript would try to reopen the deleted image.
    frame.close().unwrap();
    let second =
        model
            .turn(&mut media_session)
            .input(Input::new().audio_pcm(pcm).text(
                "What was the previous image's color and what secret word did I say just now?",
            ))
            .greedy()
            .max_tokens(48)
            .run()?;
    let second_text = second.text().to_lowercase();
    eprintln!("warm image then audio: {second_text:?}");
    assert!(second_text.contains("red") && second_text.contains("pineapple"));

    #[derive(gen2::schemars::JsonSchema)]
    #[allow(dead_code)]
    struct WeatherArgs {
        city: String,
    }
    let calls = model
        .generate("Call get_weather for Paris. Do not guess the weather.")
        .tools([ToolDefinition::new("get_weather")
            .description("Get the current weather for a city.")
            .input_schema::<WeatherArgs>()])
        .greedy()
        .max_tokens(120)
        .run()?;
    eprintln!("tools: {:?} {:?}", calls.text(), calls.tool_calls());
    assert!(calls.tool_calls().iter().any(|call| {
        call.name() == "get_weather"
            && call.arguments()["city"]
                .as_str()
                .is_some_and(|city| city.contains("Paris"))
    }));
    Ok(())
}
