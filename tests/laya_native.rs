#![cfg(feature = "laya-dynamic")]
use gen2::{Runtime, decision::*};
use std::path::PathBuf;

#[test]
#[ignore = "set GEN2_ORT_LIBRARY, GEN2_LAYA_BUNDLE and GEN2_LAYA_LONG_REFERENCE"]
fn native_long_scan_matches_upstream_windows_and_answers() {
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("GEN2_LAYA_LONG_REFERENCE").expect("long reference")).unwrap(),
    )
    .unwrap();
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(4096)
        .build()
        .unwrap();
    let model = runtime
        .load_decider(
            std::env::var("GEN2_LAYA_BUNDLE").unwrap(),
            LoadOptions {
                native_library: Some(std::env::var("GEN2_ORT_LIBRARY").unwrap().into()),
                ..Default::default()
            },
        )
        .unwrap();
    let request = serde_json::from_value(reference["request"].clone()).unwrap();
    let result = model
        .decide_long(
            request,
            LongStateOptions {
                window_tokens: Some(32),
                stride_tokens: Some(16),
                max_windows: 100,
            },
            DecisionOptions::default(),
        )
        .unwrap();
    let expected = &reference["result"];
    assert_eq!(
        result.windows.len(),
        expected["usage"]["windows"].as_u64().unwrap() as usize
    );
    assert_eq!(
        result.windows[0].manifest_sha256,
        reference["manifest_sha256"].as_str().unwrap()
    );
    for answer in result.answers {
        let e = &expected["answers"][&answer.id];
        let w = &e["window"];
        assert_eq!(
            answer.window_index,
            w["index"].as_u64().unwrap() as usize,
            "{} selected window",
            answer.id
        );
        assert_eq!(
            answer.token_start,
            w["token_start"].as_u64().unwrap() as usize
        );
        assert_eq!(answer.token_end, w["token_end"].as_u64().unwrap() as usize);
        match answer.answer.value {
            AnswerValue::Choice { label, .. } => {
                assert_eq!(serde_json::to_value(label).unwrap(), e["choice"])
            }
            AnswerValue::Score { expectation, .. } => {
                assert!((expectation - e["score"].as_f64().unwrap()).abs() < 1e-4)
            }
            AnswerValue::YesNo { probability_true } => {
                assert!((probability_true - e["noul"].as_f64().unwrap()).abs() < 1e-4)
            }
        }
        assert!(
            (answer.answer.answer_confidence - e["answer_confidence"].as_f64().unwrap()).abs()
                < 1e-4
        );
    }
    model.shutdown();
}

#[test]
#[ignore = "set GEN2_ORT_LIBRARY; uses a synthetic graph, no Laya weights"]
fn native_worker_infers_and_releases_its_reservation() {
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(16)
        .build()
        .unwrap();
    let model = runtime
        .load_decider(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke"),
            LoadOptions {
                native_library: Some(
                    std::env::var("GEN2_ORT_LIBRARY")
                        .expect("runtime library")
                        .into(),
                ),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(runtime.decision_reserved_mb(), 1);
    assert!(model.capabilities().multi_state_batch);
    assert_eq!(
        model.capabilities().execution.provider,
        ExecutionProvider::Cpu
    );
    let result = model
        .decide(
            DecisionRequest::text("a b")
                .question("q", Question::choice("pick", [("a", "low"), ("b", "high")])),
            DecisionOptions::default(),
        )
        .unwrap();
    assert_eq!(result.answers.len(), 1);
    assert_eq!(result.execution.provider, ExecutionProvider::Cpu);
    assert!(result.execution_micros > 0);
    assert_eq!(runtime.decision_models().len(), 1);
    let long = model
        .decide_long(
            DecisionRequest::text("a b ".repeat(20)).question("q", Question::yes_no("a?")),
            LongStateOptions {
                window_tokens: Some(10),
                stride_tokens: Some(5),
                max_windows: 10,
            },
            DecisionOptions::default(),
        )
        .unwrap();
    assert_eq!(long.windows.len(), 7);
    assert_eq!(long.state_tokens, 40);
    assert_eq!(long.answers[0].window_index, 0);
    assert!(matches!(
        model.decide_long(
            DecisionRequest::text("a b ".repeat(20)).question("q", Question::yes_no("a?")),
            LongStateOptions {
                window_tokens: Some(10),
                stride_tokens: Some(11),
                max_windows: 10
            },
            DecisionOptions::default()
        ),
        Err(DecisionError::InvalidRequest(_))
    ));
    assert!((result.answers[0].1.probabilities.iter().sum::<f64>() - 1.).abs() < 1e-12);
    model.shutdown();
    assert_eq!(runtime.decision_reserved_mb(), 0);
    assert_eq!(model.status(), DecisionStatus::Unloaded);
    let resumed = runtime
        .reload_decider(
            &model,
            LoadOptions {
                native_library: Some(std::env::var("GEN2_ORT_LIBRARY").unwrap().into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(runtime.decision_reserved_mb(), 1);
    resumed.shutdown();
    assert_eq!(runtime.decision_reserved_mb(), 0);
    #[cfg(not(target_vendor = "apple"))]
    {
        let unavailable = runtime.load_decider(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke"),
            LoadOptions {
                native_library: Some(std::env::var("GEN2_ORT_LIBRARY").unwrap().into()),
                execution: ExecutionOptions {
                    provider: ExecutionProvider::CoreMl,
                    allow_cpu_fallback: true,
                },
                ..Default::default()
            },
        );
        assert!(matches!(
            unavailable,
            Err(gen2::Error::Decision(DecisionError::BackendUnavailable(_)))
        ));
        assert_eq!(
            runtime.decision_reserved_mb(),
            0,
            "provider load failure rolls back"
        );
    }
}

#[test]
#[ignore = "set GEN2_ORT_LIBRARY, GEN2_LAYA_BUNDLE, GEN2_LAYA_REFERENCE; real weights and independent Python fixture required"]
fn native_matches_independent_python_reference() {
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("GEN2_LAYA_REFERENCE").expect("reference")).unwrap(),
    )
    .unwrap();
    let tolerance = if fixture["mode"] == "eager" {
        1e-3
    } else {
        1e-4
    };
    let bundle = std::env::var("GEN2_LAYA_BUNDLE").expect("bundle");
    let manifest: BundleManifest = serde_json::from_slice(
        &std::fs::read(PathBuf::from(&bundle).join("manifest.json")).unwrap(),
    )
    .unwrap();
    let budget = manifest.envelope.reservation_mb;
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(budget)
        .build()
        .unwrap();
    let model = runtime
        .load_decider(
            bundle,
            LoadOptions {
                native_library: Some(
                    std::env::var("GEN2_ORT_LIBRARY")
                        .expect("runtime library")
                        .into(),
                ),
                ..Default::default()
            },
        )
        .unwrap();
    let mut maximum = 0.0f64;
    // Mix states, question counts, lengths and an empty state in one native job.
    // Use cases sharing an encode policy so expected independent Python tensors
    // remain exact even when Rust pads rows together across state boundaries.
    let policy = &fixture["cases"][0]["options"];
    let cases: Vec<_> = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| &case["options"] == policy)
        .collect();
    let batched = model
        .decide_batch(
            cases
                .iter()
                .map(|case| serde_json::from_value(case["request"].clone()).unwrap())
                .collect(),
            DecisionOptions {
                encode: Some(serde_json::from_value(policy.clone()).unwrap()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(batched.len(), cases.len());
    for (result, case) in batched.iter().zip(&cases) {
        let inputs: Vec<EncodedQuestion> = serde_json::from_value(case["inputs"].clone()).unwrap();
        assert_eq!(result.inputs, inputs, "batched state preprocessing");
        let expected = case["probabilities"].as_array().unwrap();
        assert_eq!(result.answers.len(), expected.len());
        for ((id, answer), (row, question)) in result.answers.iter().zip(
            expected
                .iter()
                .zip(case["request"]["questions"].as_array().unwrap()),
        ) {
            assert_eq!(id, question[0].as_str().unwrap());
            let expected: Vec<f64> = serde_json::from_value(row.clone()).unwrap();
            assert_eq!(answer.probabilities.len(), expected.len());
            for (actual, expected) in answer.probabilities.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "cross-state batch {}",
                    case["name"]
                );
            }
        }
    }
    for case in fixture["cases"].as_array().unwrap() {
        let request: DecisionRequest = serde_json::from_value(case["request"].clone()).unwrap();
        let options = DecisionOptions {
            encode: Some(serde_json::from_value(case["options"].clone()).unwrap()),
            ..Default::default()
        };
        let result = model.decide(request, options).unwrap();
        if let Some(inputs) = case.get("inputs") {
            let expected: Vec<EncodedQuestion> = serde_json::from_value(inputs.clone()).unwrap();
            assert_eq!(
                result.inputs, expected,
                "exact preprocessing for {}",
                case["name"]
            );
        }
        if let Some(expected) = fixture.get("manifest_sha256").and_then(|v| v.as_str()) {
            assert_eq!(result.manifest_sha256, expected);
        }
        let expected = case["probabilities"].as_array().unwrap();
        assert_eq!(result.answers.len(), expected.len());
        for (i, ((_, answer), row)) in result.answers.iter().zip(expected).enumerate() {
            let expected: Vec<f64> = serde_json::from_value(row.clone()).unwrap();
            assert_eq!(answer.probabilities.len(), expected.len());
            let mut ranked: Vec<_> = expected.iter().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(a.1));
            if let AnswerValue::Choice { index, .. } = answer.value
                && (ranked.len() == 1 || ranked[0].1 - ranked[1].1 > 1e-5)
            {
                assert_eq!(index, ranked[0].0, "choice outside a near-tie");
            }
            for (actual, expected) in answer.probabilities.iter().zip(expected) {
                let delta = (actual - expected).abs();
                maximum = maximum.max(delta);
                assert!(
                    delta <= tolerance,
                    "{} probability difference {delta}",
                    case["name"]
                );
            }
            let action = case["act_probabilities"][i].as_f64().unwrap();
            assert!(
                (answer.act_probability - action).abs() <= tolerance,
                "{} action mismatch",
                case["name"]
            );
        }
    }
    eprintln!("maximum probability difference: {maximum}");
    model.shutdown();
    assert_eq!(runtime.decision_reserved_mb(), 0);
}
