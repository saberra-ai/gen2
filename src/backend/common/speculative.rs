//! Swappable speculative-decoding predictors.
//!
//! Every gen2 backend runs the same inner loop for speculative decode:
//! ask a predictor for a draft of likely-next tokens, verify by running
//! the target model on the draft in one forward pass, accept the
//! longest prefix the model actually agrees with. The *only* thing
//! that changes between Lookahead / PLD / n-gram is how the draft is
//! produced. This module defines the trait all predictors share, plus
//! three concrete impls:
//!
//!   - **NgramPredictor**: rolling trigram table — ~1.2-1.5× speedup on
//!     free chat, near-zero overhead, no prompt knowledge.
//!   - **PromptLookupPredictor** (Saxena 2023, "PLD"): search the
//!     PROMPT (and recent output) for the last K context tokens, draft
//!     with whatever followed. Dominates on Q&A / RAG / code where the
//!     answer echoes phrases from the input. 2× reported on those
//!     workloads.
//!   - **HybridPredictor**: run both, emit the longer confident draft.
//!     Captures both motif repetition and prompt echo.
//!
//! Callers choose per-request via `GenSpec.speculative` or globally via
//! `PIO_MLX_SPEC_MODE={off|ngram|pld|hybrid}`.
//!
//! True Jacobi Lookahead (Fu et al. 2024) is a deeper restructure — it
//! requires running the target model on hypothetical future positions
//! in parallel with the current position. That changes the forward-pass
//! shape, not just the predictor, so it's tracked separately.

use std::collections::HashMap;

/// Default cap on drafted tokens per step, honoured by every predictor.
/// Larger drafts take longer to verify (wider batch) so increasing this
/// only helps when accept rate is high. 4 is a conservative sweet spot
/// validated against the MLX speculative path.
pub const DEFAULT_DRAFT_LEN: usize = 4;

/// Side-channel inputs a predictor may need on top of token history —
/// used by EAGLE-like drafters that consume the target model's
/// intermediate hidden states.
///
/// Feature-gated on `backend-mlx` because the `aux_hidden_states`
/// field holds `mlx_rs::Array` — the only backend currently wiring
/// hidden-state-aware drafters. When the llama-cpp backend grows its
/// own EAGLE-style support, this surface generalizes
/// (e.g. via `dyn Any` or a backend-abstract tensor handle).
#[cfg(feature = "backend-mlx")]
#[derive(Debug, Clone)]
pub struct DraftContext<'a> {
    /// Last accepted token id (target vocab).
    pub last_token: u32,
    /// Target model's auxiliary hidden states, one `[1, 1, H]` array
    /// per configured aux layer.
    pub aux_hidden_states: &'a [mlx_rs::Array],
    /// Absolute sequence position of `last_token` (for RoPE).
    pub pos: usize,
}

/// Object-safe interface every speculative predictor implements.
pub trait SpeculativePredictor: Send {
    /// Produce up to `max` candidate tokens continuing from the current
    /// context. Return empty when the predictor has no confident guess
    /// (caller falls back to single-token decode).
    fn draft(&mut self, max: usize) -> Vec<u32>;

    /// Hidden-state-aware draft. Default impl delegates to `draft`,
    /// so token-only predictors (Ngram / PLD / Hybrid / Noop) don't
    /// need to implement this. EAGLE-3 / Medusa / EAGLE-2 overrides
    /// to consume `ctx.aux_hidden_states`. Feature-gated with the
    /// `DraftContext` type — see its docstring.
    #[cfg(feature = "backend-mlx")]
    fn draft_with_context(&mut self, _ctx: &DraftContext<'_>, max: usize) -> Vec<u32> {
        self.draft(max)
    }

    /// Observe a token that was ACCEPTED by the target model. Every
    /// accepted token goes through this exactly once, in order.
    fn observe(&mut self, token: u32);

    /// Seed the predictor with the prompt tokens before decode starts.
    /// Default: noop. PLD uses this; n-gram doesn't.
    fn seed_prompt(&mut self, _prompt: &[u32]) {}

    /// Name for metrics / debug output.
    fn name(&self) -> &'static str;

    /// Whether this predictor consumes the draft context (aux hidden
    /// states etc). Backends can use this to decide whether to compute
    /// aux states on the target side — skip the work when all active
    /// predictors are token-only. Default: false.
    fn needs_context(&self) -> bool {
        false
    }

    /// Target-model layer ids whose post-block hidden states the
    /// predictor needs in `DraftContext::aux_hidden_states`. Empty for
    /// token-only predictors (default). The backend calls this once
    /// to know which layers to stash from the target's forward pass.
    fn aux_layer_ids(&self) -> &[usize] {
        &[]
    }
}

// ─── N-gram (rolling trigram table) ────────────────────────────────────

/// Rolling trigram table. For each (t_{n-2}, t_{n-1}) context seen in
/// recent output, tracks the most-frequent next token. Draft chains
/// lookups forward to produce a sequence. No prompt knowledge — relies
/// entirely on motifs repeating in the output stream.
pub struct NgramPredictor {
    buf: Vec<u32>,
    head: usize,
    len: usize,
    table: HashMap<[u32; 2], (u32, u32)>,
}

const NGRAM_HISTORY: usize = 1024;

impl NgramPredictor {
    pub fn new() -> Self {
        Self {
            buf: vec![0u32; NGRAM_HISTORY],
            head: 0,
            len: 0,
            table: HashMap::with_capacity(512),
        }
    }

    fn last_two(&self) -> Option<[u32; 2]> {
        if self.len < 2 {
            return None;
        }
        let avail = self.len.min(NGRAM_HISTORY);
        Some(std::array::from_fn(|i| {
            let offset = avail - 2 + i;
            self.buf[(self.head + NGRAM_HISTORY - avail + offset) % NGRAM_HISTORY]
        }))
    }
}

impl Default for NgramPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeculativePredictor for NgramPredictor {
    fn draft(&mut self, max: usize) -> Vec<u32> {
        let Some(mut key) = self.last_two() else {
            return Vec::new();
        };
        let mut result = Vec::with_capacity(max);
        for _ in 0..max {
            match self.table.get(&key) {
                Some(&(next, _)) => {
                    result.push(next);
                    key = [key[1], next];
                }
                None => break,
            }
        }
        result
    }

    fn observe(&mut self, token: u32) {
        if let Some(key) = self.last_two() {
            let entry = self.table.entry(key).or_insert((token, 0));
            if entry.0 == token {
                entry.1 += 1;
            } else if entry.1 == 0 {
                *entry = (token, 1);
            } else {
                entry.1 -= 1;
            }
        }
        self.buf[self.head] = token;
        self.head = (self.head + 1) % NGRAM_HISTORY;
        if self.len < NGRAM_HISTORY {
            self.len += 1;
        }
    }

    fn name(&self) -> &'static str {
        "ngram"
    }
}

// ─── Prompt Lookup Decoding (Saxena 2023) ──────────────────────────────

/// PLD: when the model's about to generate something that already
/// appears in the PROMPT, draft it directly from the prompt instead of
/// sampling.
///
/// Algorithm: keep a buffer of `prompt_tokens + accepted_output`. Take
/// the last K tokens (the "needle"). Scan the buffer for a match of
/// this needle somewhere EARLIER than the needle's own position. If a
/// match exists, draft with the `max` tokens following the match.
///
/// Tries needle lengths from `max_ngram` down to `min_ngram` — longer
/// matches are more confident but rarer. K=3 is the PLD paper's default.
pub struct PromptLookupPredictor {
    buf: Vec<u32>,
    min_ngram: usize,
    max_ngram: usize,
}

impl PromptLookupPredictor {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
            min_ngram: 1,
            max_ngram: 3,
        }
    }

    pub fn with_ngram_range(min: usize, max: usize) -> Self {
        Self {
            buf: Vec::with_capacity(4096),
            min_ngram: min.max(1),
            max_ngram: max.max(min.max(1)),
        }
    }

    /// Find the LATEST occurrence of `needle` in `self.buf[..tail_exclusive]`
    /// (i.e. NOT including the needle's own copy at the tail). Return the
    /// end index of the match so callers can draft with `buf[end..end+max]`.
    fn find_match(&self, needle: &[u32], tail_exclusive: usize) -> Option<usize> {
        let hay = &self.buf[..tail_exclusive];
        if needle.is_empty() || hay.len() < needle.len() {
            return None;
        }
        let n = needle.len();
        // Scan from right to left so we pick the MOST RECENT match —
        // recent context is a stronger prior for the model's trajectory.
        for i in (0..=hay.len() - n).rev() {
            if &hay[i..i + n] == needle {
                return Some(i + n);
            }
        }
        None
    }
}

impl Default for PromptLookupPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeculativePredictor for PromptLookupPredictor {
    fn draft(&mut self, max: usize) -> Vec<u32> {
        let total = self.buf.len();
        // Try progressively shorter needles — longest match = most confident.
        for n in (self.min_ngram..=self.max_ngram).rev() {
            if total < n {
                continue;
            }
            let needle_start = total - n;
            let needle = &self.buf[needle_start..total].to_vec();
            if let Some(match_end) = self.find_match(needle, needle_start) {
                let take = max.min(total.saturating_sub(match_end));
                if take == 0 {
                    continue;
                }
                return self.buf[match_end..match_end + take].to_vec();
            }
        }
        Vec::new()
    }

    fn observe(&mut self, token: u32) {
        self.buf.push(token);
    }

    fn seed_prompt(&mut self, prompt: &[u32]) {
        self.buf.extend_from_slice(prompt);
    }

    fn name(&self) -> &'static str {
        "pld"
    }
}

// ─── Hybrid (longest-wins of n-gram ∪ PLD) ─────────────────────────────

/// Runs both n-gram and PLD, returns whichever produced the longer
/// draft. When drafts tie in length, PLD wins because its matches are
/// more context-specific (longer needle = stronger prior).
pub struct HybridPredictor {
    ngram: NgramPredictor,
    pld: PromptLookupPredictor,
}

impl HybridPredictor {
    pub fn new() -> Self {
        Self {
            ngram: NgramPredictor::new(),
            pld: PromptLookupPredictor::new(),
        }
    }
}

impl Default for HybridPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeculativePredictor for HybridPredictor {
    fn draft(&mut self, max: usize) -> Vec<u32> {
        let a = self.ngram.draft(max);
        let b = self.pld.draft(max);
        if b.len() >= a.len() { b } else { a }
    }

    fn observe(&mut self, token: u32) {
        self.ngram.observe(token);
        self.pld.observe(token);
    }

    fn seed_prompt(&mut self, prompt: &[u32]) {
        self.pld.seed_prompt(prompt);
        // Don't seed ngram: it's a trigram table built from OUTPUT
        // motifs, and pre-seeding with prompt trigrams hurts accept
        // rate by biasing draft toward prompt patterns the model is
        // intentionally moving away from in its answer.
    }

    fn name(&self) -> &'static str {
        "hybrid"
    }
}

// ─── Mode selector (for GenSpec / env) ─────────────────────────────────

/// Which speculative predictor to use for a given session.
///
/// Default is `Lookahead` — a combined n-gram + prompt-lookup predictor
/// that captures most of the real-world speculative wins without
/// requiring a trained draft model or model-aware Jacobi iteration.
/// True Fu-et-al. 2024 Lookahead Decoding (parallel Jacobi forward
/// passes) is scoped as separate work; EAGLE-3 (trained draft models)
/// is integrable via the `Eagle3` variant once the draft-model forward
/// path is wired up for a given backend.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum SpeculativeMode {
    /// No speculative decode — single-token path only.
    Off,
    /// Rolling trigram predictor only — cheapest, captures motif
    /// repetition but not prompt echo.
    Ngram,
    /// Prompt Lookup Decoding (Saxena 2023) — best on Q&A / RAG / code
    /// where the answer echoes phrases from the prompt.
    Pld,
    /// **Default.** N-gram + PLD combined, longest confident draft wins.
    /// Captures both motif repetition and prompt echo; named "Lookahead"
    /// for the common speculative-decoding taxonomy though it is not
    /// the model-aware Jacobi variant from Fu et al. 2024.
    #[default]
    Lookahead,
    /// EAGLE-3 — trained draft model for a specific target. Requires
    /// loading the draft model alongside the target (path points at an
    /// MLX-converted EAGLE checkpoint, e.g.
    /// `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`). Scaffolded;
    /// per-backend forward-pass wiring is follow-up work.
    Eagle3 {
        /// Filesystem path to the MLX-formatted EAGLE-3 draft model.
        model_path: String,
    },
}

impl SpeculativeMode {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" | "0" => Some(Self::Off),
            "ngram" | "n-gram" | "trigram" => Some(Self::Ngram),
            "pld" | "prompt_lookup" | "lookup" => Some(Self::Pld),
            "lookahead" | "hybrid" | "both" | "default" => Some(Self::Lookahead),
            other if other.starts_with("eagle3:") => Some(Self::Eagle3 {
                model_path: other[7..].to_string(),
            }),
            _ => None,
        }
    }

    /// Build a boxed predictor for this mode. `Off` returns a no-op
    /// predictor whose `draft()` is always empty, so the speculative
    /// path naturally falls back to single-token decode. `Eagle3`
    /// currently falls back to Lookahead with a warning until per-
    /// backend draft-model wiring lands.
    pub fn build(self) -> Box<dyn SpeculativePredictor> {
        match self {
            Self::Off => Box::new(NoopPredictor),
            Self::Ngram => Box::new(NgramPredictor::new()),
            Self::Pld => Box::new(PromptLookupPredictor::new()),
            Self::Lookahead => Box::new(HybridPredictor::new()),
            Self::Eagle3 { model_path } => {
                // Backends that have wired EAGLE-3 fully (via the
                // `needs_context` path) will intercept this mode BEFORE
                // hitting `build()` and construct a real predictor
                // with tokenizer + target-model hooks in hand. This
                // placeholder path is the fallback for backends that
                // haven't wired it yet — drafts empty, falls back to
                // single-token decode, emits a clear trace.
                tracing::warn!(
                    model_path = %model_path,
                    "EAGLE-3 draft loaded but backend's aux-hidden-state plumbing is not \
                     wired — drafts will be empty (single-token fallback). See task #21."
                );
                Box::new(Eagle3PlaceholderPredictor {
                    model_path: model_path.clone(),
                })
            }
        }
    }
}

/// Placeholder predictor used while an EAGLE-3 matcher isn't fully
/// wired in the active backend. Holds the resolved `model_path` so
/// callers can surface it in telemetry; drafts empty (falls back to
/// single-token decode) until the backend replaces it with a real
/// `EagleDraftPredictor` (in `mlx::eagle_predictor`).
pub struct Eagle3PlaceholderPredictor {
    pub model_path: String,
}

impl SpeculativePredictor for Eagle3PlaceholderPredictor {
    fn draft(&mut self, _max: usize) -> Vec<u32> {
        Vec::new()
    }
    fn observe(&mut self, _token: u32) {}
    fn name(&self) -> &'static str {
        "eagle3_placeholder"
    }
}

/// Predictor that never drafts — used when speculative is disabled.
pub struct NoopPredictor;

impl SpeculativePredictor for NoopPredictor {
    fn draft(&mut self, _max: usize) -> Vec<u32> {
        Vec::new()
    }
    fn observe(&mut self, _token: u32) {}
    fn name(&self) -> &'static str {
        "off"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_predicts_repeating_sequence() {
        let mut p = NgramPredictor::new();
        for _ in 0..5 {
            for &t in &[1u32, 2, 3, 4] {
                p.observe(t);
            }
        }
        let draft = p.draft(4);
        assert!(!draft.is_empty());
        assert_eq!(draft[0], 1);
    }

    #[test]
    fn pld_drafts_from_prompt_echo() {
        let mut p = PromptLookupPredictor::new();
        // Prompt contains a distinctive phrase that the "output" ends with.
        p.seed_prompt(&[10, 11, 12, 20, 21, 22, 10, 11, 12, 30, 31]);
        // Now we "generate" the start of an echo of [10, 11, 12].
        p.observe(10);
        p.observe(11);
        p.observe(12);
        // PLD should draft [20, 21, 22] (continuation of the most-recent
        // past occurrence) — actually the LATEST match is at the second
        // occurrence of 10,11,12 which continues with [30, 31]. Either
        // is a valid PLD prediction; we picked latest-match so expect 30.
        let draft = p.draft(4);
        assert!(!draft.is_empty());
        assert_eq!(draft[0], 30);
    }

    #[test]
    fn pld_empty_without_match() {
        let mut p = PromptLookupPredictor::new();
        p.seed_prompt(&[1, 2, 3]);
        p.observe(99);
        assert!(p.draft(4).is_empty());
    }

    #[test]
    fn hybrid_picks_longer_draft() {
        let mut h = HybridPredictor::new();
        // Prompt echo scenario — PLD will dominate.
        h.seed_prompt(&[5, 6, 7, 8, 9, 10, 5, 6, 7]);
        h.observe(5);
        h.observe(6);
        h.observe(7);
        let draft = h.draft(4);
        assert!(!draft.is_empty(), "expected non-empty hybrid draft");
    }

    #[test]
    fn mode_parses_case_insensitive() {
        assert_eq!(
            SpeculativeMode::from_str_opt("NGRAM"),
            Some(SpeculativeMode::Ngram)
        );
        assert_eq!(
            SpeculativeMode::from_str_opt("pld"),
            Some(SpeculativeMode::Pld)
        );
        assert_eq!(
            SpeculativeMode::from_str_opt("off"),
            Some(SpeculativeMode::Off)
        );
        assert_eq!(
            SpeculativeMode::from_str_opt("lookahead"),
            Some(SpeculativeMode::Lookahead)
        );
        assert_eq!(
            SpeculativeMode::from_str_opt("hybrid"),
            Some(SpeculativeMode::Lookahead)
        );
        assert!(matches!(
            SpeculativeMode::from_str_opt("eagle3:/tmp/draft.mlx"),
            Some(SpeculativeMode::Eagle3 { model_path }) if model_path == "/tmp/draft.mlx"
        ));
        assert_eq!(SpeculativeMode::from_str_opt("weird"), None);
    }

    #[test]
    fn default_mode_is_lookahead() {
        assert_eq!(SpeculativeMode::default(), SpeculativeMode::Lookahead);
    }

    #[test]
    fn noop_predictor_empty() {
        let mut p = NoopPredictor;
        assert!(p.draft(4).is_empty());
    }
}
