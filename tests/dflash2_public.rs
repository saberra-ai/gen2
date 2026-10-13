use gen2::advanced::generation::Settings;

#[test]
fn invalid_dflash2_width_is_rejected_by_public_settings() {
    let mut value = serde_json::to_value(Settings::default()).unwrap();
    value["system"]["dflash2"] = serde_json::json!({
        "model_path": "/missing/draft.gguf", "max_tokens": 0
    });
    let settings: Settings = serde_json::from_value(value).unwrap();
    assert!(
        settings.validate().is_err(),
        "DFlash2 width zero must not silently become ordinary decoding"
    );
}

#[test]
fn invalid_draft_paths_widths_and_kv_types_are_rejected() {
    use gen2::advanced::generation::DFlash2Config;
    for (path, width) in [("", 5), ("draft.gguf", 65)] {
        let mut settings = Settings::default();
        settings.system.dflash2 = Some(DFlash2Config {
            model_path: path.into(),
            max_tokens: width,
        });
        assert!(settings.validate().is_err());
    }
    let mut settings = Settings::default();
    settings.system.kv_cache_type = Some("not-a-kv-type".into());
    assert!(settings.validate().is_err());
}

#[cfg(not(feature = "llama-dflash2"))]
#[test]
fn draft_configuration_requires_the_native_feature() {
    use gen2::advanced::generation::DFlash2Config;
    let mut settings = Settings::default();
    settings.system.dflash2 = Some(DFlash2Config {
        model_path: "draft.gguf".into(),
        max_tokens: 5,
    });
    assert!(settings.validate().is_err());
}

#[cfg(feature = "llama-dflash2")]
#[test]
fn a_batch_smaller_than_the_draft_block_is_rejected_before_native_use() {
    use gen2::advanced::generation::DFlash2Config;
    let mut settings = Settings::default();
    settings.system.batch_size = Some(1);
    settings.system.dflash2 = Some(DFlash2Config {
        model_path: "draft.gguf".into(),
        max_tokens: 5,
    });
    assert!(
        settings.validate().is_err(),
        "native block requires capacity for anchor plus proposals"
    );
}

#[cfg(feature = "llama-dflash2")]
#[test]
fn missing_draft_fails_through_public_model_loading() {
    use gen2::{Runtime, advanced::generation::DFlash2Config};
    let mut settings = Settings::default();
    settings.system.dflash2 = Some(DFlash2Config {
        model_path: "/missing/gen2-dflash2-draft.gguf".into(),
        max_tokens: 5,
    });
    let runtime = Runtime::builder().settings(settings).build().unwrap();
    let target = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gguf/corpus/full_llama.gguf");
    let error = runtime.load(target).unwrap_err().to_string();
    assert!(
        error.contains("draft"),
        "must fail for missing draft before native target loading: {error}"
    );
}

#[cfg(feature = "llama-dflash2")]
mod live {
    use super::*;
    use gen2::{
        Runtime,
        advanced::generation::{DFlash2Config, ThinkingMode},
    };

    const PROMPT: &str = "Write a complete Python implementation of a thread-safe LRU cache with get, put, eviction, capacity validation and unit tests. Explain the design briefly, then provide the full code.";

    fn runtime(drafting: bool) -> Runtime {
        runtime_with_context(drafting, 8192)
    }

    fn runtime_with_context(drafting: bool, context: u32) -> Runtime {
        runtime_with_configuration(drafting, context, 5)
    }

    fn runtime_with_configuration(drafting: bool, context: u32, width: usize) -> Runtime {
        let mut settings = Settings::default();
        settings.system.ctx_size = Some(context);
        settings.system.batch_size = Some(512);
        settings.system.threads = Some(4);
        settings.system.threads_batch = Some(4);
        settings.system.gpu_layers = Some(99);
        settings.system.flash_attn = Some(true);
        settings.system.kv_cache_type = Some("q8_0".into());
        settings.prompt.include_meta = Some(false);
        if drafting {
            settings.system.dflash2 = Some(DFlash2Config {
                model_path: std::env::var("GEN2_DFLASH2_DRAFT")
                    .expect("set GEN2_DFLASH2_DRAFT")
                    .into(),
                max_tokens: width,
            });
        }
        Runtime::builder().settings(settings).build().unwrap()
    }

    #[test]
    #[ignore = "requires local Saluki and DFlash2 GGUFs; run serially"]
    fn embedded_dflash2_generates_with_thinking_off_and_reports_acceptance() {
        let runtime = runtime(true);
        let model = runtime
            .load(std::env::var("GEN2_DFLASH2_TARGET").expect("set GEN2_DFLASH2_TARGET"))
            .unwrap();
        let reply = model
            .generate(PROMPT)
            .greedy()
            .reasoning(ThinkingMode::Off)
            .max_tokens(128)
            .run()
            .unwrap();
        assert!(!reply.text().trim().is_empty());
        assert!(reply.reasoning().is_none_or(|s| s.is_empty()));
        assert_eq!(reply.usage.completion_tokens, 128);
        assert!(
            reply.stats.execution().spec_drafted > 0,
            "drafting was requested but did not run"
        );
        assert!(reply.stats.execution().spec_accepted > 0);
    }

    #[test]
    #[ignore = "requires local Saluki and DFlash2 GGUFs; run serially"]
    fn greedy_output_matches_drafting_disabled_at_block_boundaries() {
        let path = std::env::var("GEN2_DFLASH2_TARGET").unwrap();
        let expected = {
            let runtime = runtime(false);
            let model = runtime.load(&path).unwrap();
            [1, 2, 7, 128].map(|limit| {
                let reply = model
                    .generate(PROMPT)
                    .greedy()
                    .reasoning(ThinkingMode::Off)
                    .max_tokens(limit)
                    .run()
                    .unwrap();
                assert_eq!(reply.usage.completion_tokens, limit as u32);
                reply.text()
            })
        };
        let runtime = runtime(true);
        let model = runtime.load(&path).unwrap();
        for (i, limit) in [1, 2, 7, 128].into_iter().enumerate() {
            let reply = model
                .generate(PROMPT)
                .greedy()
                .reasoning(ThinkingMode::Off)
                .max_tokens(limit)
                .run()
                .unwrap();
            assert_eq!(reply.usage.completion_tokens, limit as u32);
            assert_eq!(reply.text(), expected[i], "output diverged at cap {limit}");
        }
    }

    #[test]
    #[ignore = "requires local Saluki and DFlash2 GGUFs; run serially"]
    fn cancellation_and_multiturn_reuse_remain_usable() {
        use gen2::{Session, event::Event, output::FinishReason};
        let runtime = runtime(true);
        let model = runtime
            .load(std::env::var("GEN2_DFLASH2_TARGET").unwrap())
            .unwrap();
        let mut session = Session::new();
        let mut stream = model
            .turn(&mut session)
            .user(PROMPT)
            .greedy()
            .reasoning(ThinkingMode::Off)
            .max_tokens(128)
            .stream()
            .unwrap();
        let cancel = stream.canceller();
        let mut deltas = 0;
        while let Some(event) = stream.next() {
            if let Event::TextDelta(_) = event.unwrap() {
                deltas += 1;
                if deltas == 4 {
                    cancel.cancel();
                }
            }
        }
        let reply = stream.finish().unwrap();
        assert_eq!(reply.finish_reason, FinishReason::Cancelled);
        assert!(reply.usage.completion_tokens < 128);
        for prompt in [
            "Continue with the implementation.",
            "Now summarize it in one sentence.",
        ] {
            let reply = model
                .turn(&mut session)
                .user(prompt)
                .greedy()
                .reasoning(ThinkingMode::Off)
                .max_tokens(16)
                .run()
                .unwrap();
            assert!(!reply.text().is_empty());
            assert!(reply.reasoning().is_none_or(|s| s.is_empty()));
            assert!(reply.stats.execution().spec_drafted > 0);
        }
    }

    #[test]
    #[ignore = "requires local Saluki and DFlash2 GGUFs; run serially"]
    fn context_compaction_resets_both_caches_and_keeps_thinking_off() {
        let runtime = runtime_with_context(true, 512);
        let model = runtime
            .load(std::env::var("GEN2_DFLASH2_TARGET").unwrap())
            .unwrap();
        let mut session = gen2::Session::new();
        let prompt = format!(
            "{} Summarize these notes.",
            "The river carries sediment downstream. ".repeat(20)
        );
        for _ in 0..4 {
            let reply = model
                .turn(&mut session)
                .user(&prompt)
                .greedy()
                .reasoning(ThinkingMode::Off)
                .max_tokens(8)
                .run()
                .unwrap();
            assert!(!reply.text().is_empty());
            assert!(reply.reasoning().is_none_or(|s| s.is_empty()));
            assert_eq!(reply.usage.completion_tokens, 8);
        }
    }

    #[test]
    #[ignore = "requires local Saluki and DFlash2 GGUFs; run serially"]
    fn proposal_larger_than_trained_block_is_rejected_on_load() {
        let runtime = runtime_with_configuration(true, 512, 64);
        let error = runtime
            .load(std::env::var("GEN2_DFLASH2_TARGET").unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("DFlash2") || error.contains("trained block"),
            "{error}"
        );
    }
}
