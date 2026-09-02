//! What the loaded model can actually do.
//!
//! LiteRT-LM's upstream tree has a `loaded_file` API that reports a bundle's
//! context window and which modalities it supports. The v0.16.0 prebuilt does
//! not export it — the release ships `engine.h` and `conversation.h` and
//! neither the header nor the symbols for capabilities.
//!
//! So this asks, and copes with the answer being "I cannot tell you". That is
//! the whole design: a newer runtime gets introspection, an older one gets an
//! explicit `ctx_size` from the caller, and a runtime with neither fails the
//! load with an error saying so. What it never does is pick a number.

use std::sync::Arc;

use crate::engine::ExecError;

use super::ffi::{Runtime, c_string};

/// Facts about a bundle, established at load time.
#[derive(Debug)]
pub(super) struct ModelFacts {
    /// The context window the engine is configured with. Always either
    /// reported by the runtime or supplied by the caller.
    pub max_context_tokens: u32,
    /// Whether the bundle declares function calling. `None` when the runtime
    /// cannot say.
    #[allow(dead_code)]
    pub supports_function_calling: Option<bool>,
}

impl ModelFacts {
    /// Ask the runtime what it knows, and fall back to what the caller said.
    ///
    /// The order matters. Introspection first, because a model's real context
    /// window beats anything guessed from outside it. An explicit `ctx_size`
    /// second, because a caller who states one has said what they want and the
    /// engine is configured to exactly that. And then failure — because the
    /// remaining option is to invent a number, and a backend that lies to the
    /// controller about context capacity turns every truncation decision into
    /// a decision made against a fiction.
    pub(super) fn probe(
        runtime: &Arc<Runtime>,
        model_path: &str,
        ctx_size: Option<u32>,
    ) -> Result<Self, ExecError> {
        decide(introspect(runtime, model_path)?, ctx_size, model_path)
    }
}

/// The decision itself, with no FFI around it.
///
/// Split from [`ModelFacts::probe`] rather than duplicated, so the rule that
/// matters — never invent a context size — is testable on a machine with no
/// LiteRT-LM runtime installed, which is every machine in CI, and cannot drift
/// from what the real path does.
///
/// Three separate ideas, deliberately not collapsed: what the model can hold,
/// what the caller asked for, and what the engine gets configured with. An
/// earlier version returned the introspected maximum whenever it had one,
/// which meant a future runtime reporting a 128K window would quietly ignore
/// `.context(4096)` and allocate 128K of KV — worst on exactly the small
/// devices this backend exists for.
fn decide(
    introspected: Option<(u32, bool)>,
    requested: Option<u32>,
    model_path: &str,
) -> Result<ModelFacts, ExecError> {
    // A runtime that answers zero has not answered. Treating it as a context
    // window would configure an engine that can hold nothing.
    let model_max = introspected.map(|(t, _)| t).filter(|t| *t > 0);
    let supports_function_calling = introspected.map(|(_, tools)| tools);
    let requested = requested.filter(|n| *n > 0);

    let configured = match (requested, model_max) {
        // The caller's number, clamped to what the model can actually hold.
        // Asking for more than the model has is a mistake worth correcting
        // quietly; asking for less is a choice worth honouring exactly.
        (Some(want), Some(max)) => want.min(max),
        (Some(want), None) => want,
        (None, Some(max)) => max,
        (None, None) => {
            return Err(ExecError::InvalidModelFile(format!(
                "this LiteRT-LM runtime cannot report the context window of \
                 `{model_path}`, and gen2 will not guess one. Set an explicit \
                 context size (`Engine::builder().context(n)`) matching the \
                 bundle — the `ekv` number in its filename is usually it — or \
                 use a LiteRT-LM build that exports the loaded-file capability \
                 API."
            )));
        }
    };

    Ok(ModelFacts {
        max_context_tokens: configured,
        supports_function_calling,
    })
}

/// What the runtime can tell us, if it has the capability API at all.
fn introspect(runtime: &Arc<Runtime>, model_path: &str) -> Result<Option<(u32, bool)>, ExecError> {
    let Some(api) = runtime.capabilities.as_ref() else {
        return Ok(None);
    };
    let path = c_string(model_path)?;

    // SAFETY: `path` outlives the call; the handle is null-checked and freed
    // on every path out.
    let file = unsafe { (api.loaded_file_create)(path.as_ptr()) };
    if file.is_null() {
        // The runtime has the API but could not read the file. That is a real
        // answer about this file, not a missing capability, so it is left to
        // engine creation to produce the specific error.
        return Ok(None);
    }
    // SAFETY: `file` is a live handle, used only here and freed below.
    Ok(Some(unsafe {
        let tokens = (api.loaded_file_max_context_tokens)(file);
        let tools = (api.loaded_file_supports_function_calling)(file);
        (api.loaded_file_delete)(file);
        (tokens, tools)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule the whole module is built around: no invented context sizes.
    ///
    /// Without introspection and without an explicit size, the only honest
    /// outcome is a failed load. A backend that picks 4096 here reports a
    /// capacity nothing configured, and the controller truncates — or fails to
    /// truncate — against it.
    #[test]
    fn a_load_that_cannot_know_the_context_window_fails_instead_of_guessing() {
        let err = decide(None, None, "model.litertlm")
            .expect_err("with no context information, the load must fail");
        let text = err.to_string();
        assert!(
            text.contains("will not guess"),
            "the error should say why it refused: {text}"
        );
        assert!(
            text.contains("context size"),
            "the error should tell the caller what to set: {text}"
        );
        assert!(
            text.contains("model.litertlm"),
            "the error should name the file it is about: {text}"
        );
    }

    #[test]
    fn an_explicit_context_size_is_used_exactly_as_given() {
        let facts = decide(None, Some(4096), "m").expect("an explicit size is enough");
        assert_eq!(facts.max_context_tokens, 4096);
    }

    /// A request larger than the model can hold is clamped to the model.
    #[test]
    fn a_request_beyond_the_model_is_clamped_to_what_the_model_holds() {
        let facts = decide(Some((1280, true)), Some(4096), "m").expect("introspection answered");
        assert_eq!(facts.max_context_tokens, 1280);
        assert_eq!(facts.supports_function_calling, Some(true));
    }

    /// A request smaller than the model can hold is honoured exactly.
    ///
    /// The case that matters on a phone. Returning the model's maximum here
    /// would allocate a KV cache many times the size the caller budgeted for,
    /// and the caller would have no way to tell.
    #[test]
    fn a_request_smaller_than_the_model_is_honoured_rather_than_widened() {
        let facts = decide(Some((131_072, true)), Some(4096), "m").expect("introspection answered");
        assert_eq!(
            facts.max_context_tokens, 4096,
            "asking for 4K on a 128K model must give 4K, not 128K"
        );
    }

    /// With no request, the model's own window stands.
    #[test]
    fn the_model_window_is_used_when_the_caller_asks_for_nothing() {
        let facts = decide(Some((8192, false)), None, "m").expect("introspection answered");
        assert_eq!(facts.max_context_tokens, 8192);
    }

    #[test]
    fn a_reported_zero_is_not_an_answer() {
        // Some runtimes return 0 for "I don't know". Configuring an engine
        // with a zero-token context would fail a long way from the cause.
        let facts = decide(Some((0, false)), Some(2048), "m")
            .expect("the caller's size should be used instead");
        assert_eq!(facts.max_context_tokens, 2048);
    }

    /// Zero with nothing to fall back on is still a refusal, not a zero-token
    /// engine.
    #[test]
    fn a_reported_zero_with_no_fallback_still_refuses() {
        assert!(decide(Some((0, false)), None, "m").is_err());
    }
}
