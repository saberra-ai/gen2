//! The mistral.rs backend.
//!
//! One backend covering GGUF, safetensors, UQFF and Hugging Face repositories,
//! with its own paged attention, quantization, device mapping and prefix
//! caching. None of that is configured from gen2: this backend consumes the
//! `LoadRequest`, `Settings`, `SessionSpec` and `GenSpec` that already exist
//! and produces the `TokenEvent`s that already exist.
//!
//! That is the rule the whole module is built to keep. Nothing named
//! `MistralRs*` appears in `src/api`, `Settings`, `GenSpec`, `Session` or the
//! agent, and a caller cannot tell from the public API which backend is
//! running. `src/backend/mistralrs/` is the membrane.
//!
//! # What stays gen2's
//!
//! Tools are declared to the model and executed here by gen2 — mistral.rs is
//! never handed a callback, so approvals, deferred tools and sub-agents keep
//! working unchanged. The transcript is gen2's, so `append_messages` remains
//! authoritative and mistral.rs's own session handling is unused. Grammar is
//! gen2's `GrammarSpec`, enforced with mistral.rs's native constrained
//! decoding rather than gen2's matcher.
//!
//! # Threading
//!
//! `BlockingModel` owns a tokio runtime and panics if constructed inside one.
//! The controller loop it runs on is a plain thread, which is the reason the
//! backend boundary is synchronous in the first place.

mod convert;
mod engine;
mod loader;
mod puller;
mod session;

pub(crate) use engine::MistralRsEngine;

#[cfg(test)]
mod live {
    /// The whole stack over mistral.rs, driven the way a consumer drives it.
    ///
    /// The point is that nothing in this test mentions mistral.rs. If a
    /// backend needs the public API to know about it, the boundary is in the
    /// wrong place.
    #[test]
    #[ignore = "needs PIO_TEST_MISTRALRS_MODEL"]
    fn a_consumer_can_chat_without_knowing_which_backend_ran() {
        let Ok(path) = std::env::var("PIO_TEST_MISTRALRS_MODEL") else {
            eprintln!("SKIP: set PIO_TEST_MISTRALRS_MODEL");
            return;
        };
        let engine = crate::api::Engine::load(&path).expect("the model should load");
        let mut session = crate::api::Session::new();

        let done = engine
            .chat(&mut session)
            .user("Say hello.")
            .max_tokens(24)
            .send()
            .expect("the turn should complete");

        assert!(
            !done.text.trim().is_empty(),
            "the backend produced no text at all"
        );
        assert_eq!(
            session.messages().len(),
            2,
            "the caller's transcript should hold the question and the reply"
        );

        // A second turn continues rather than restarting.
        engine
            .chat(&mut session)
            .user("And again.")
            .max_tokens(16)
            .send()
            .expect("the second turn should complete");
        assert_eq!(session.messages().len(), 4);
    }

    /// The round trip the backend is not proven without.
    ///
    /// A puller test shows a tool call reaching gen2. What it cannot show is
    /// the other half: that the call and its result, once in the transcript,
    /// replay to the model as a call and a result rather than as an empty
    /// assistant turn. That half broke silently until `tool_calls_of` existed,
    /// and nothing but an end-to-end run would have said so.
    #[test]
    #[ignore = "needs PIO_TEST_MISTRALRS_MODEL"]
    fn a_tool_call_and_its_result_replay_to_the_model() {
        use crate::api::{Engine, FunctionTool, Session, ToolOutput};
        use schemars::JsonSchema;
        use serde::Deserialize;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Deserialize, JsonSchema)]
        struct City {
            /// City to look up.
            city: String,
        }

        let Ok(path) = std::env::var("PIO_TEST_MISTRALRS_MODEL") else {
            eprintln!("SKIP: set PIO_TEST_MISTRALRS_MODEL");
            return;
        };
        let engine = Engine::load(&path).expect("the model should load");

        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let weather = FunctionTool::new(
            "get_weather",
            "Current weather for a city",
            move |_ctx, a: City| {
                let counter = Arc::clone(&counter);
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(ToolOutput::from(format!("18C and sunny in {}", a.city)))
                }
            },
        );

        let mut session = Session::new();
        let done = engine
            .agent(&mut session)
            .add_tool(weather)
            .max_steps(4)
            .goal("What is the weather in Paris? Use the tool.")
            .expect("the run should complete");

        // The model deciding not to call is a model choice, not a backend
        // failure, so the assertion is about what happens when it does.
        if calls.load(Ordering::SeqCst) == 0 {
            eprintln!("model made no tool call; nothing to assert about replay");
            return;
        }

        let transcript: String = session
            .messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            transcript.contains("get_weather"),
            "the call the model made is missing from the transcript:\n{transcript}"
        );
        assert!(
            transcript.contains("18C and sunny"),
            "the tool result the answer rests on is missing:\n{transcript}"
        );
        assert!(done.tool_rounds >= 1);
        eprintln!(
            "replay verified: {} tool call(s), {} round(s), transcript has the call and its result",
            calls.load(Ordering::SeqCst),
            done.tool_rounds
        );
    }
}
