use super::*;
use std::collections::BTreeMap;
/// Explicit routing among loaded checkpoints. No model is downloaded or
/// substituted automatically; missing routes are actionable errors.
#[derive(Debug, Clone, Default)]
pub struct DecisionRouter {
    models: BTreeMap<Checkpoint, DecisionModel>,
}
impl DecisionRouter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, model: DecisionModel) -> Option<DecisionModel> {
        self.models.insert(model.info().checkpoint, model)
    }
    pub fn model(&self, checkpoint: Checkpoint) -> Result<&DecisionModel> {
        self.models.get(&checkpoint).ok_or_else(|| {
            DecisionError::BackendUnavailable(format!("{checkpoint:?} checkpoint is not loaded"))
        })
    }
    pub fn decide(
        &self,
        checkpoint: Checkpoint,
        request: DecisionRequest,
        options: DecisionOptions,
    ) -> Result<DecisionResult> {
        self.model(checkpoint)?.decide(request, options)
    }
    /// Select English for an explicit `en` language tag, multilingual otherwise.
    /// This does not claim to detect a language from text. Typed decisions use
    /// the explicit checkpoint method regardless of language.
    pub fn decide_for_language(
        &self,
        language: &str,
        request: DecisionRequest,
        mut options: DecisionOptions,
    ) -> Result<DecisionResult> {
        if language.trim().is_empty() {
            return Err(DecisionError::InvalidRequest(
                "language tag is empty".into(),
            ));
        }
        let checkpoint = if language
            .split('-')
            .next()
            .unwrap_or("")
            .eq_ignore_ascii_case("en")
        {
            Checkpoint::English
        } else {
            Checkpoint::Multilingual
        };
        options.language = Some(language.into());
        self.decide(checkpoint, request, options)
    }
}
