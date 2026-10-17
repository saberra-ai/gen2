//! Real public Runtime/Model/Turn regression. Run explicitly with the acquired
//! Gemma270 GGUF in GEN2_GRAMMAR_TEST_MODEL; no private cache flush or load bypass.
use gen2::{
    Runtime, Session,
    advanced::generation::{Settings, ThinkingMode},
};
use serde_json::json;

#[derive(serde::Deserialize, gen2::schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct Decision {
    approved: bool,
}

fn turn(model: &gen2::Model, session: &mut Session) {
    let response = model
        .turn(session)
        .user("Return JSON with approved true: {\"approved\":true}.")
        .max_tokens(96)
        .greedy()
        .seed(42)
        .reasoning(ThinkingMode::Off)
        .json_schema::<Decision>()
        .unwrap()
        .full_context()
        .run()
        .unwrap();
    let value: Decision = serde_json::from_str(&response.text()).unwrap();
    // Either boolean is valid; semantic correctness is not this lifetime test.
    println!(
        "{}",
        json!({"phase":"turn","approved":value.approved,"finish":format!("{:?}",response.finish_reason),"usage":{"promptTokens":response.usage.prompt_tokens,"completionTokens":response.usage.completion_tokens},"execution":response.stats.execution()})
    );
    assert_eq!(response.finish_reason, gen2::output::FinishReason::Stop);
    let context = response.stats.request_context().expect("actual context");
    assert_eq!(context.capacity_tokens, 512);
    assert_eq!(context.output_tokens, 96);
    assert!(context.prompt_tokens + 96 <= 512);
}

#[test]
#[ignore = "requires explicitly supplied real Gemma270 artifact and ordinary CPU admission"]
fn grammar_vocabulary_releases_at_public_eviction() {
    let path = std::env::var("GEN2_GRAMMAR_TEST_MODEL").expect("GEN2_GRAMMAR_TEST_MODEL");
    let bytes = std::fs::read(&path).unwrap();
    use sha2::{Digest, Sha256};
    assert_eq!(
        hex::encode(Sha256::digest(&bytes)),
        "3626e245220ca4a1c5911eb4010b3ecb7bdbf5bc53c79403c21355354d1e2dc6"
    );
    drop(bytes);
    let mut settings = Settings::default();
    settings.system.ctx_size = Some(512);
    settings.system.gpu_layers = Some(0);
    settings.system.batch_size = Some(16);
    settings.system.threads = Some(2);
    settings.system.threads_batch = Some(2);
    settings.prompt.include_meta = Some(false);
    let runtime = Runtime::builder()
        .settings(settings.clone())
        .max_active_sessions(1)
        .build()
        .unwrap();
    let admission = runtime.inspect_local_admission(&path).unwrap();
    println!(
        "{}",
        json!({"phase":"admission","fitsWithoutEviction":admission.fits_without_eviction,"plan":format!("{:?}",admission.plan),"snapshot":admission.snapshot})
    );
    assert!(
        admission.fits_without_eviction,
        "ordinary admission refused"
    );
    let model = runtime.load(&path).unwrap();
    assert_eq!(model.info().offload.unwrap().layers, 0);
    assert_eq!(
        model.grammar_ownership().unwrap().unwrap(),
        gen2::model::GrammarOwnership::default()
    );
    let mut first = Session::new();
    turn(&model, &mut first);
    let after_first = model.grammar_ownership().unwrap().unwrap();
    println!("{}", json!({"phase":"first","ownership":after_first}));
    assert_eq!(
        (
            after_first.vocabulary_builds,
            after_first.vocabulary_releases,
            after_first.live_vocabularies
        ),
        (1, 0, 1)
    );
    let mut second = Session::new();
    turn(&model, &mut second);
    let after_second = model.grammar_ownership().unwrap().unwrap();
    println!("{}", json!({"phase":"second","ownership":after_second}));
    assert_eq!(
        after_second.vocabulary_builds, 1,
        "reuse the loaded model vocabulary"
    );
    runtime.evict(&model).unwrap();
    let evicted = model.grammar_ownership().unwrap().unwrap();
    println!(
        "{}",
        json!({"phase":"evicted-handles-still-alive","ownership":evicted,"residentModels":runtime.stats().resident_models})
    );
    assert_eq!(runtime.stats().resident_models, 0);
    assert_eq!(evicted.active_matchers, 0);
    assert_eq!(
        evicted.live_vocabularies, 0,
        "eviction must release the prepared vocabulary even while logical Model and Session handles live"
    );
    assert_eq!(evicted.vocabulary_releases, 1);
    // Restore the same logical Model and retained conversation, then evict
    // it while an independent model with the *same vocab* remains usable.
    turn(&model, &mut first);
    let restored = model.grammar_ownership().unwrap().unwrap();
    println!("{}", json!({"phase":"restored","ownership":restored}));
    assert_eq!(
        (
            restored.vocabulary_builds,
            restored.vocabulary_releases,
            restored.live_vocabularies
        ),
        (2, 1, 1)
    );
    let other_runtime = Runtime::builder()
        .settings(settings)
        .max_active_sessions(1)
        .build()
        .unwrap();
    let other_admission = other_runtime.inspect_local_admission(&path).unwrap();
    assert!(other_admission.fits_without_eviction);
    let other = other_runtime.load(&path).unwrap();
    let mut other_session = Session::new();
    turn(&other, &mut other_session);
    let independent = other.grammar_ownership().unwrap().unwrap();
    assert_eq!(
        (
            independent.vocabulary_builds,
            independent.vocabulary_releases,
            independent.live_vocabularies
        ),
        (1, 0, 1)
    );
    runtime.evict(&model).unwrap();
    let first_released = model.grammar_ownership().unwrap().unwrap();
    assert_eq!(
        (
            first_released.vocabulary_builds,
            first_released.vocabulary_releases,
            first_released.live_vocabularies,
            first_released.active_matchers
        ),
        (2, 2, 0, 0)
    );
    assert_eq!(other.grammar_ownership().unwrap().unwrap(), independent);
    turn(&other, &mut other_session);
    assert_eq!(
        other
            .grammar_ownership()
            .unwrap()
            .unwrap()
            .vocabulary_builds,
        1
    );
    println!(
        "{}",
        json!({"phase":"independent-owner","first":first_released,"second":other.grammar_ownership().unwrap().unwrap()})
    );
    let overflow = other
        .turn(&mut Session::new())
        .user("word ".repeat(800))
        .max_tokens(96)
        .full_context()
        .run()
        .unwrap_err();
    println!(
        "{}",
        json!({"phase":"overflow","code":overflow.code(),"ownership":other.grammar_ownership().unwrap().unwrap()})
    );
    assert_eq!(overflow.code(), Some("context_overflow"));

    let mut active_session = Session::new();
    let mut stream = other
        .turn(&mut active_session)
        .user("Return eighty booleans in values.")
        .max_tokens(96)
        .greedy()
        .seed(42)
        .reasoning(ThinkingMode::Off)
        .json_schema::<Sequence>()
        .unwrap()
        .full_context()
        .stream()
        .unwrap();
    loop {
        let event = stream.next().expect("stream must start").unwrap();
        if matches!(event,gen2::event::Event::TextDelta(ref text) if !text.is_empty()) {
            break;
        }
    }
    let active = other.grammar_ownership().unwrap().unwrap();
    assert_eq!(
        active.active_matchers, 1,
        "observe actual constrained decoding before eviction"
    );
    other_runtime.evict(&other).unwrap();
    let drained = other.grammar_ownership().unwrap().unwrap();
    let stopped = stream.finish().unwrap();
    println!(
        "{}",
        json!({"phase":"active-eviction","before":active,"after":drained,"finish":format!("{:?}",stopped.finish_reason)})
    );
    assert_eq!(stopped.finish_reason, gen2::output::FinishReason::Cancelled);
    assert_eq!(
        (
            drained.vocabulary_builds,
            drained.vocabulary_releases,
            drained.live_vocabularies,
            drained.active_matchers
        ),
        (1, 1, 0, 0)
    );
    assert_eq!(other_runtime.stats().resident_models, 0);

    let mut cancelled_session = Session::new();
    let mut cancelled = other
        .turn(&mut cancelled_session)
        .user("Return eighty booleans in values.")
        .max_tokens(96)
        .greedy()
        .seed(42)
        .reasoning(ThinkingMode::Off)
        .json_schema::<Sequence>()
        .unwrap()
        .full_context()
        .stream()
        .unwrap();
    loop {
        let event = cancelled.next().expect("cancel turn must start").unwrap();
        if matches!(event,gen2::event::Event::TextDelta(ref text) if !text.is_empty()) {
            break;
        }
    }
    let cancel_active = other.grammar_ownership().unwrap().unwrap();
    assert_eq!(cancel_active.active_matchers, 1);
    cancelled.canceller().cancel();
    let response = cancelled.finish().unwrap();
    assert_eq!(
        response.finish_reason,
        gen2::output::FinishReason::Cancelled
    );
    let after_cancel = other.grammar_ownership().unwrap().unwrap();
    assert_eq!(
        (
            after_cancel.vocabulary_builds,
            after_cancel.vocabulary_releases,
            after_cancel.live_vocabularies,
            after_cancel.active_matchers
        ),
        (2, 1, 1, 0)
    );
    other_runtime.evict(&other).unwrap();
    let final_counts = other.grammar_ownership().unwrap().unwrap();
    assert_eq!(
        (
            final_counts.vocabulary_builds,
            final_counts.vocabulary_releases,
            final_counts.live_vocabularies,
            final_counts.active_matchers
        ),
        (2, 2, 0, 0)
    );
    println!(
        "{}",
        json!({"phase":"cancel-and-final-release","before":cancel_active,"afterCancel":after_cancel,"final":final_counts,"finish":format!("{:?}",response.finish_reason)})
    );
    drop((
        first,
        second,
        other_session,
        active_session,
        cancelled_session,
        model,
        runtime,
        other,
        other_runtime,
    ));
}

// A long constrained sequence leaves an observable active matcher after the
// first output fragment. It is deliberately cancelled, never accepted as JSON.
#[derive(serde::Deserialize, gen2::schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Sequence {
    #[schemars(length(min = 80, max = 80))]
    values: Vec<bool>,
}
