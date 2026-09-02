//! The LiteRT-LM backend.
//!
//! Google's on-device runtime, reached through its C ABI. It runs `.litertlm`
//! bundles — a packaged format carrying the weights, tokenizer and prompt
//! template together — with CPU, GPU and NPU execution paths of its own.
//! None of that is configured from gen2: this backend consumes the
//! `LoadRequest`, `Settings`, `SessionSpec` and `GenSpec` that already exist
//! and produces the `TokenEvent`s that already exist.
//!
//! That is the rule the whole module is built to keep. Nothing named
//! `LiteRtLm*` appears in `src/api`, `Settings`, `GenSpec`, `Session` or the
//! agent, and a caller cannot tell from the public API which backend is
//! running. `src/backend/litertlm/` is the membrane.
//!
//! # Getting the runtime
//!
//! gen2 does not vendor, link or download LiteRT-LM. The feature compiles on a
//! machine that has never heard of it; the library is looked for when a model
//! is loaded, by the name Google's own prebuilts use
//! (`liblitert-lm.so` / `.dylib` / `litert-lm.dll`), or at
//! `GEN2_LITERTLM_LIBRARY` if that is set. Prebuilt libraries are published
//! with each LiteRT-LM release as `litert_lm_c_api-<version>.zip`.
//!
//! # What stays gen2's
//!
//! Tools are declared to the model and executed here by gen2 — LiteRT-LM is
//! never handed a callback, so approvals, deferred tools and sub-agents keep
//! working unchanged. Calls are recovered from the text stream by gen2's own
//! cross-backend parser, the same one the llama and MLX backends use.
//!
//! # What is shared
//!
//! The transcript is the one genuine two-sided arrangement. LiteRT-LM's
//! `Conversation` is stateful and holds its own prefilled KV, so gen2 owns
//! what the conversation *is* and LiteRT-LM owns what it has already computed.
//! Only newly appended messages are sent; when the caller edits or clears
//! history the controller rebuilds the session, which discards the
//! conversation with it. LiteRT-LM can never be left holding a turn gen2 has
//! deleted.
//!
//! # Context
//!
//! `n_ctx()` is never a guess. The capability API that reports a bundle's
//! context window is not exported by the v0.16.0 prebuilt, so a load either
//! introspects it, takes an explicit `ctx_size` from the caller, or fails with
//! an error saying which. See [`capabilities`].

mod capabilities;
mod convert;
mod engine;
mod ffi;
mod puller;
mod session;

pub(crate) use engine::LiteRtLmEngine;

#[cfg(test)]
mod live {
    //! Tests that need a real runtime and a real `.litertlm` bundle.
    //!
    //! Both are set through the environment because neither ships with gen2:
    //! `GEN2_LITERTLM_LIBRARY` for Google's shared library and
    //! `PIO_TEST_LITERTLM_MODEL` for the model.

    fn model() -> Option<String> {
        std::env::var("PIO_TEST_LITERTLM_MODEL")
            .ok()
            .filter(|p| !p.is_empty())
    }

    /// The whole stack over LiteRT-LM, driven the way a consumer drives it.
    ///
    /// The point is that nothing in this test mentions LiteRT-LM. If a backend
    /// needs the public API to know about it, the boundary is in the wrong
    /// place.
    #[test]
    #[ignore = "needs PIO_TEST_LITERTLM_MODEL and a LiteRT-LM runtime"]
    fn a_consumer_can_chat_without_knowing_which_backend_ran() {
        let Some(path) = model() else {
            eprintln!("SKIP: set PIO_TEST_LITERTLM_MODEL");
            return;
        };
        let engine = crate::api::Engine::builder()
            .model(&path)
            // LiteRT-LM's shipped runtime cannot report a bundle's context
            // window, so the caller states it — which is exactly the path the
            // backend documents rather than a test-only workaround.
            .context(4096)
            .build()
            .expect("the model should load");
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
        eprintln!("litertlm decoded: {:?}", done.text);
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

    /// Per-turn sampling has to reach the model, not just the struct.
    ///
    /// LiteRT-LM fixes the sampler when a conversation is created, so a turn
    /// asking for different sampling is only honoured if the backend notices
    /// and reopens.
    ///
    /// The observable is the seed, not determinism. Determinism alone proves
    /// nothing here: the sampler this runtime defaults to is itself
    /// deterministic, so `.greedy()` twice matched even with the fix reverted.
    /// Two different seeds diverging cannot happen unless the seed reached the
    /// sampler, and the same seed reproducing cannot happen unless it reached
    /// it consistently. Together they pin it.
    #[test]
    #[ignore = "needs PIO_TEST_LITERTLM_MODEL and a LiteRT-LM runtime"]
    fn a_per_turn_sampling_change_reaches_the_model() {
        let Some(path) = model() else {
            eprintln!("SKIP: set PIO_TEST_LITERTLM_MODEL");
            return;
        };
        let engine = crate::api::Engine::builder()
            .model(&path)
            .context(4096)
            .build()
            .expect("the model should load");

        let ask = |seed: u64| {
            let mut session = crate::api::Session::new();
            engine
                .chat(&mut session)
                .user("Write one sentence about an unusual animal.")
                .max_tokens(40)
                .temperature(1.5)
                .seed(seed)
                .send()
                .expect("the turn should complete")
                .text
        };

        let a = ask(1);
        let b = ask(1);
        let c = ask(999_999);

        assert!(!a.trim().is_empty(), "sampled decoding produced nothing");
        assert_eq!(
            a, b,
            "the same seed gave two different answers, so the seed is not \
             reaching the sampler consistently"
        );
        assert_ne!(
            a, c,
            "two different seeds gave identical output, so the sampler this \
             turn asked for never reached the model — the conversation is \
             still running whatever it was opened with"
        );
        eprintln!("seed reaches the sampler: seed 1 and seed 999999 diverge");
    }

    /// `.greedy()` has to map to something the runtime implements.
    ///
    /// The sampler type literally named "greedy" answers `UNIMPLEMENTED` on
    /// the shipped runtime, so a backend that picked it by name fails here
    /// rather than at a code review.
    #[test]
    #[ignore = "needs PIO_TEST_LITERTLM_MODEL and a LiteRT-LM runtime"]
    fn greedy_decoding_runs_and_repeats_itself() {
        let Some(path) = model() else {
            eprintln!("SKIP: set PIO_TEST_LITERTLM_MODEL");
            return;
        };
        let engine = crate::api::Engine::builder()
            .model(&path)
            .context(4096)
            .build()
            .expect("the model should load");

        let ask = || {
            let mut session = crate::api::Session::new();
            engine
                .chat(&mut session)
                .user("Name exactly one colour.")
                .max_tokens(12)
                .greedy()
                .send()
                .expect("greedy decoding should complete")
                .text
        };

        let first = ask();
        assert!(!first.trim().is_empty(), "greedy produced nothing at all");
        assert_eq!(first, ask(), "greedy decoding is not deterministic");
    }

    /// Sampling this backend cannot honour is refused, not ignored.
    #[test]
    #[ignore = "needs PIO_TEST_LITERTLM_MODEL and a LiteRT-LM runtime"]
    fn sampling_the_runtime_lacks_is_refused_end_to_end() {
        let Some(path) = model() else {
            eprintln!("SKIP: set PIO_TEST_LITERTLM_MODEL");
            return;
        };
        let engine = crate::api::Engine::builder()
            .model(&path)
            .context(4096)
            .build()
            .expect("the model should load");
        let mut session = crate::api::Session::new();

        let outcome = engine
            .chat(&mut session)
            .user("hello")
            .max_tokens(8)
            .min_p(0.1)
            .send();
        assert!(
            outcome.is_err(),
            "min_p has no LiteRT-LM equivalent, so accepting it would give the \
             caller different sampling than they asked for"
        );
    }

    /// A conversation LiteRT-LM keeps prefilled must still match gen2's.
    ///
    /// This is the backend's one shared-state risk: only new messages are
    /// sent, so a session whose history the caller edits could leave the model
    /// working from a transcript that no longer exists.
    #[test]
    #[ignore = "needs PIO_TEST_LITERTLM_MODEL and a LiteRT-LM runtime"]
    fn an_edited_transcript_is_not_still_in_the_model() {
        let Some(path) = model() else {
            eprintln!("SKIP: set PIO_TEST_LITERTLM_MODEL");
            return;
        };
        let engine = crate::api::Engine::builder()
            .model(&path)
            .context(4096)
            .build()
            .expect("the model should load");
        let mut session = crate::api::Session::new();

        engine
            .chat(&mut session)
            .user("My favourite colour is chartreuse. Remember it.")
            .max_tokens(24)
            .send()
            .expect("the first turn should complete");

        // Take it back out, the way `edit` is documented to.
        session.edit(|m| m.retain(|msg| !msg.text().contains("chartreuse")));

        let done = engine
            .chat(&mut session)
            .user("What is my favourite colour? If you were not told, say you were not told.")
            .max_tokens(48)
            .send()
            .expect("the second turn should complete");

        assert!(
            !done.text.to_lowercase().contains("chartreuse"),
            "the model still had a message the caller deleted: {}",
            done.text
        );
    }
}
