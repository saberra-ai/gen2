//! Cross-backend grammar-constrained decoding.
//!
//! Wraps the `llguidance` crate (JSON-schema / Lark / regex / GBNF →
//! per-step token mask) behind a small stable API that every gen2
//! backend can share. The backend just calls `apply_mask` on its
//! pre-softmax logits and `observe` on the sampled token; llguidance
//! handles the grammar parsing, token-trie traversal, and mask
//! computation.
//!
//! This is the same engine `llama_cpp_2::llguidance_sampler` uses, so a
//! grammar that works against the llama backend also works in MLX /
//! ONNX mode — output constraint semantics are backend-agnostic.

use std::sync::Arc;

use anyhow::{Context, Result};
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie};

use super::tokenizer::HfTokenizer;

/// High-level grammar specification. Backend-agnostic — pass one to any
/// gen2 session's `GenSpec` and every backend that wires up
/// [`GrammarMatcher`] honours it.
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarSpec {
    /// Constrain output to a JSON object (shorthand for schema
    /// `{"type": "object"}`). Fastest way to force "reply with valid
    /// JSON" without writing a schema.
    JsonObject,
    /// Constrain to a specific JSON schema.
    JsonSchema(serde_json::Value),
    /// Constrain to a regex (RE2-compatible via llguidance).
    Regex(String),
    /// Lark grammar text (a superset of GBNF for practical purposes).
    Lark(String),
}

impl GrammarSpec {
    fn into_top_level(self) -> Result<TopLevelGrammar> {
        Ok(match self {
            GrammarSpec::JsonObject => {
                TopLevelGrammar::from_json_schema(serde_json::json!({"type": "object"}))
            }
            GrammarSpec::JsonSchema(v) => TopLevelGrammar::from_json_schema(v),
            GrammarSpec::Regex(rx) => TopLevelGrammar::from_regex(&rx),
            GrammarSpec::Lark(lark) => TopLevelGrammar::from_lark(lark),
        })
    }
}

/// Per-session grammar matcher. Holds an `llguidance::Matcher` tied to
/// the session's tokenizer + the chosen [`GrammarSpec`]. Cheap to `new`
/// (grammar parsing is eager but fast); expensive to `apply_mask` on
/// wide vocabs — mask computation touches the full token trie.
pub struct GrammarMatcher {
    matcher: Matcher,
    vocab_size: usize,
}

impl GrammarMatcher {
    /// Build a matcher that constrains output to `spec` over the tokens
    /// of `tokenizer`. Builds a toktrie from every token's byte form so
    /// llguidance can compute exact per-token masks.
    pub fn new(tokenizer: &HfTokenizer, spec: GrammarSpec) -> Result<Self> {
        let tok_env = build_tok_env(tokenizer)?;
        let vocab_size = tok_env.tok_trie().vocab_size();
        let mut factory = ParserFactory::new_simple(&tok_env)
            .context("build llguidance ParserFactory")?;
        factory.quiet();
        let top = spec.into_top_level()?;
        let parser = factory
            .create_parser(top)
            .context("create grammar parser")?;
        let matcher = Matcher::new(Ok(parser));
        Ok(Self {
            matcher,
            vocab_size,
        })
    }

    /// Zero out logits for tokens that would leave the grammar in an
    /// unreachable state. Mutates `logits` in-place. The mask is
    /// computed by llguidance against the current matcher state.
    ///
    /// Note: this sets disallowed logits to `f32::NEG_INFINITY`. Soft
    /// masks (strong penalty instead of hard exclusion) are possible by
    /// calling into the matcher directly, but for the common case of
    /// "force schema compliance" hard is what callers want.
    pub fn apply_mask(&mut self, logits: &mut [f32]) -> Result<()> {
        let mask = self
            .matcher
            .compute_mask()
            .context("compute grammar mask")?;
        let n = logits.len().min(self.vocab_size);
        for i in 0..n {
            if !mask.is_allowed(i as u32) {
                logits[i] = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    /// Advance the matcher state with a sampled token. Must be called
    /// exactly once per accepted token (after sampling + any post-
    /// sampling filters).
    pub fn observe(&mut self, token_id: u32) -> Result<()> {
        self.matcher
            .consume_token(token_id)
            .context("consume grammar token")
    }

    /// True when the grammar has reached an accepting state — the
    /// caller can legitimately stop generation here. Orthogonal to
    /// model-emitted EOS.
    pub fn is_accepting(&mut self) -> bool {
        self.matcher.is_accepting().unwrap_or(false)
    }

    /// True when the matcher has permanently stopped (error or grammar
    /// fully consumed) — no further `observe` is useful.
    pub fn is_stopped(&self) -> bool {
        self.matcher.is_stopped()
    }
}

/// Build an `llguidance` TokEnv from our HfTokenizer. For each token
/// id in the vocab, decode it to bytes (preserving special tokens) and
/// feed the byte sequence into a TokTrie. Mirrors the approach
/// `llama_cpp_2::llguidance_sampler::build_tok_env` uses for llama.cpp
/// so cross-backend grammar semantics stay aligned.
fn build_tok_env(tokenizer: &HfTokenizer) -> Result<TokEnv> {
    let vocab_size = tokenizer.vocab_size();
    let mut words: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
    for id in 0..vocab_size {
        // `decode_keep_specials` renders special tokens as their text
        // form (e.g. `<turn|>`). We prefix those with 0xFF so the
        // tokenizer / trie treats them as atomic special-token bytes —
        // same convention llguidance / toktrie use elsewhere.
        let normal = tokenizer
            .decode(&[id as u32])
            .unwrap_or_default();
        if !normal.is_empty() {
            words.push(normal.into_bytes());
            continue;
        }
        let special = tokenizer
            .decode_keep_specials(&[id as u32])
            .unwrap_or_default();
        if !special.is_empty() {
            let mut buf = Vec::with_capacity(special.len() + 1);
            buf.push(0xFF);
            buf.extend_from_slice(special.as_bytes());
            words.push(buf);
        } else {
            words.push(Vec::new());
        }
    }
    let info = TokRxInfo {
        vocab_size: vocab_size as u32,
        tok_eos: tokenizer.eos_id().unwrap_or(0),
        tok_bos: tokenizer.bos_id(),
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    let trie = TokTrie::from(&info, &words);
    let approx = ApproximateTokEnv::new(trie);
    let tok_env: TokEnv = Arc::new(approx);
    let _ = InferenceCapabilities::default(); // touch to ensure import stays
    Ok(tok_env)
}

#[cfg(test)]
mod tests {
    // Full integration tests require a loaded tokenizer; those live in
    // the backend-specific integration test suites. Unit tests here
    // cover the spec-to-grammar translation.
    use super::*;

    #[test]
    fn json_object_spec_builds() {
        let _top = GrammarSpec::JsonObject.into_top_level().unwrap();
    }

    #[test]
    fn regex_spec_builds() {
        let _top = GrammarSpec::Regex(r"[0-9]+".into())
            .into_top_level()
            .unwrap();
    }

    #[test]
    fn lark_spec_builds() {
        let _top = GrammarSpec::Lark(
            r#"start: "yes" | "no""#.to_string(),
        )
        .into_top_level()
        .unwrap();
    }
}
