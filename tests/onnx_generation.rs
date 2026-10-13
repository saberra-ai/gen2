//! ONNX text generation through the same public API used by GGUF callers.
#![cfg(feature = "backend-onnx")]

use gen2::model::ThinkingMode;

#[test]
fn genai_bundle_routes_to_onnx_before_other_local_formats() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("genai_config.json"), b"{}").unwrap();
    assert_eq!(
        gen2::compat::backend::Engine::detect_backend_for_path(directory.path()),
        "onnx"
    );
}

#[test]
#[ignore = "requires PIO_TEST_ONNX_MODEL, GEN2_ONNX_LIBRARY and ORT_LIB_PATH"]
fn native_generation_preserves_turns_and_honours_greedy_and_schema() {
    let path = std::env::var("PIO_TEST_ONNX_MODEL").expect("set PIO_TEST_ONNX_MODEL");
    let model = gen2::load(&path).expect("ONNX bundle loads through gen2::load");
    let ask = || {
        model
            .generate("Name one colour.")
            .reasoning(ThinkingMode::Off)
            .greedy()
            .max_tokens(24)
            .run()
            .expect("ONNX generates")
    };
    let first = ask();
    assert!(!first.text().trim().is_empty(), "must decode real text");
    assert_eq!(first.text(), ask().text(), "greedy repeats");
    assert!(first.usage().completion_tokens > 0);
    assert!(first.usage().completion_tokens <= 24);
    eprintln!(
        "ONNX decoded: {:?}; usage={:?}",
        first.text(),
        first.usage()
    );

    let mut session = gen2::Session::new();
    model
        .turn(&mut session)
        .user("My name is Alice. Say hello.")
        .reasoning(ThinkingMode::Off)
        .greedy()
        .max_tokens(32)
        .run()
        .unwrap();
    let next = model
        .turn(&mut session)
        .user("What is my name?")
        .reasoning(ThinkingMode::Off)
        .greedy()
        .max_tokens(32)
        .run()
        .unwrap();
    assert!(
        next.text().contains("Alice"),
        "history lost: {:?}",
        next.text()
    );
    assert_eq!(session.len(), 4);

    #[derive(Debug, serde::Deserialize, gen2::schemars::JsonSchema)]
    struct FieldValue {
        text: String,
    }
    assert!(model.capabilities().structured_output);
    let field: FieldValue = model
        .generate("Return a JSON object with text set to Alice.")
        .reasoning(ThinkingMode::Off)
        .greedy()
        .max_tokens(48)
        .structured()
        .expect("schema enforced by native ONNX decoding");
    assert_eq!(field.text, "Alice");
}

#[test]
#[ignore = "requires PIO_TEST_ONNX_MODEL, GEN2_ONNX_LIBRARY and ORT_LIB_PATH"]
fn native_constraints_and_session_lifetime_survive_engine_unload() {
    use gen2::compat::backend::common::grammar::GrammarSpec;
    use gen2::compat::backend::{Backend, onnx::OnnxEngine};
    use gen2::compat::engine::LoadRequest;
    use gen2::compat::generation::{GenSpec, TokenEvent};
    use gen2::compat::session_rt::SessionSpec;
    use gen2::compat::types::message::Message;
    let path = std::env::var("PIO_TEST_ONNX_MODEL").expect("set PIO_TEST_ONNX_MODEL");
    let engine = OnnxEngine::default();
    engine
        .load_model(LoadRequest {
            model_path: path.into(),
            ..Default::default()
        })
        .unwrap();
    engine.reload_model().expect("reload must not deadlock");
    let session = engine
        .start_session(SessionSpec {
            messages: vec![Message::user("Say Alice.")],
            thinking: ThinkingMode::Off,
            ..Default::default()
        })
        .unwrap();
    engine.unload_model();
    drop(engine);
    // The session owns its weights and native runtime, not just a borrowed
    // handle. Its tokenizer, generator and constraints still work after unload.
    for grammar in [
        GrammarSpec::Regex("Alice".into()),
        GrammarSpec::Lark("start: \"Alice\"".into()),
    ] {
        let mut puller = session
            .pull(GenSpec {
                max_tokens: Some(16),
                temperature: Some(0.0),
                grammar: Some(grammar),
                ..Default::default()
            })
            .unwrap();
        let mut text = String::new();
        while let Some(event) = puller.next_event() {
            if let TokenEvent::Token(token) = event.unwrap() {
                text.push_str(&token.text);
            }
        }
        assert_eq!(text, "Alice");
    }
}

#[test]
#[ignore = "requires PIO_TEST_ONNX_MODEL"]
fn missing_runtime_is_a_normal_load_error() {
    if std::env::var_os("GEN2_ONNX_MISSING_RUNTIME_CHILD").is_some() {
        let path = std::env::var("PIO_TEST_ONNX_MODEL").unwrap();
        let error = gen2::load(path).expect_err("missing ORT must fail before loading GenAI");
        assert!(error.to_string().contains("ORT_LIB_PATH"), "{error}");
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "missing_runtime_is_a_normal_load_error",
            "--ignored",
            "--nocapture",
        ])
        .env("GEN2_ONNX_MISSING_RUNTIME_CHILD", "1")
        .env_remove("ORT_LIB_PATH")
        .status()
        .unwrap();
    assert!(
        status.success(),
        "missing runtime must not abort the process: {status}"
    );
}

#[test]
#[ignore = "requires the qualified Woof export and native ONNX libraries"]
fn native_woof_fills_email_and_unicode_city_fields() {
    #[derive(Debug, serde::Deserialize, gen2::schemars::JsonSchema)]
    struct FieldValue {
        text: String,
    }
    let model = gen2::load(std::env::var("PIO_TEST_ONNX_MODEL").unwrap()).unwrap();
    let system = "You fill the selected browser field from the user's goal and page context. Return only a JSON object with a text property containing the value to type. Use only the requested component: a first-name field needs the first name, not the full name. Never invent personal details.";
    // Stable JSON field ordering: changing the request string is a prompting
    // experiment, not an interchangeable serialization for this small model.
    for (prompt, expected) in [
        (
            r#"{"goal":"Sign up using alice@example.com.","page_context":{"fields":["First name","Email address","City"],"title":"Contact details"},"selected_field":{"current_value":"","label":"Email address","tag":"input"}}"#,
            "alice@example.com",
        ),
        (
            r#"{"goal":"Complete my shipping address in Montréal, Canada.","page_context":{"fields":["First name","Email address","City"],"title":"Contact details"},"selected_field":{"current_value":"","label":"City","tag":"input"}}"#,
            "Montréal",
        ),
    ] {
        let value: FieldValue = model
            .generate(prompt)
            .system(system)
            .reasoning(ThinkingMode::Off)
            .greedy()
            .max_tokens(64)
            .structured()
            .unwrap();
        assert_eq!(value.text, expected);
    }
}
