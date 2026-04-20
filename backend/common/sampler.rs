//! Token sampling from logit slices, shared across backends.

use std::collections::VecDeque;

use rand::Rng;

/// Window over which repetition penalty + loop detectors act. 128 lets the
/// cycle-period detector see up through a ~60-token repeat unit (needs
/// `2 * period` tokens buffered) — catches typical "system prompt loop"
/// failures on Gemma 4 26B. Llama.cpp default is 64, fine for rep penalty
/// but too tight for cycle detection.
const REPETITION_WINDOW: usize = 128;

// consumed by workspace dependents (src-tauri, pio-daemon)
#[allow(dead_code)]
pub struct Sampler {
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<i32>,
    /// Repetition penalty factor. `None` or `Some(1.0)` disables. Values
    /// > 1.0 reduce the probability of recently-emitted tokens; llama.cpp
    /// default is `1.1`, HuggingFace default is `1.0` (off).
    repetition_penalty: Option<f32>,
    /// Ring buffer of recently emitted token ids (bounded by
    /// [`REPETITION_WINDOW`]). Populated by [`Sampler::observe`] from the
    /// caller's decode loop.
    recent: VecDeque<u32>,
    rng: rand::rngs::ThreadRng,
}

// consumed by workspace dependents (src-tauri, pio-daemon)
#[allow(dead_code)]
impl Sampler {
    pub fn new(
        temperature: f32,
        top_p: Option<f32>,
        top_k: Option<i32>,
        repetition_penalty: Option<f32>,
    ) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            recent: VecDeque::with_capacity(REPETITION_WINDOW),
            rng: rand::rng(),
        }
    }

    /// Record a token as "recently emitted" so subsequent `sample_from_logits`
    /// calls apply the repetition penalty to it. Call once per accepted token
    /// in the decode loop.
    pub fn observe(&mut self, token: u32) {
        if self.recent.len() == REPETITION_WINDOW {
            self.recent.pop_front();
        }
        self.recent.push_back(token);
    }

    /// Detect a tight post-answer token loop — the last `window` emitted
    /// tokens contain at most `max_unique` distinct values. Gemma 4 26B and
    /// 31B exhibit this after a complete answer: "...umbrellas! l l l l l l …"
    /// or "…serviced their door. la la la la …" running to max_tokens. The
    /// model's *content* is done; it's just not emitting EOT. Callers should
    /// force an Eos when this returns true.
    ///
    /// Defaults tuned conservatively: `window=16` tokens, `max_unique=2`.
    /// Legitimate prose won't hit this — natural text mixes 5-10+ distinct
    /// tokens in any 16-token window — but a low-entropy filler loop will.
    pub fn is_in_token_loop(&self, window: usize, max_unique: usize) -> bool {
        if self.recent.len() < window {
            return false;
        }
        let mut unique: Vec<u32> = Vec::with_capacity(max_unique + 1);
        for &t in self.recent.iter().rev().take(window) {
            if !unique.contains(&t) {
                if unique.len() >= max_unique {
                    return false;
                }
                unique.push(t);
            }
        }
        true
    }

    /// Detect n-gram-level sentence repetition — the last `n` tokens exactly
    /// equal the `n` tokens before them. Catches verbatim phrase loops like
    /// "It's a great place to experience. It's a great place to experience."
    /// that the raw-token loop detector misses (the phrase has plenty of
    /// unique tokens, but the n-gram repeats). Both mlx-lm and our code
    /// produce these on Gemma 4 26B/31B at long context when the model's
    /// stop tokens aren't being chosen by the sampler.
    ///
    /// `n=8` catches short phrase cycles; longer cycles (e.g. a repeated
    /// full sentence) need a larger `n`.
    pub fn is_in_ngram_loop(&self, n: usize) -> bool {
        let len = self.recent.len();
        if len < 2 * n {
            return false;
        }
        let mut iter = self.recent.iter().rev();
        let last: Vec<u32> = iter.by_ref().take(n).copied().collect();
        let prev: Vec<u32> = iter.take(n).copied().collect();
        last == prev
    }

    /// Combined phrase-loop detector: fires if any of a set of common cycle
    /// sizes exhibits an immediate repeat. Covers 4-, 8-, 16-, and 32-token
    /// cycles — enough range to catch short stutters ("la la la") up through
    /// "system prompt regurgitation" loops on Gemma 4 26B (~27-token unit).
    /// Pure natural text very rarely produces a 32-token exact match.
    pub fn is_in_any_ngram_loop(&self) -> bool {
        for n in [4usize, 8, 16, 32] {
            if self.is_in_ngram_loop(n) {
                return true;
            }
        }
        false
    }

    /// Cycle-period scanner. For each period `p` in `1..=max_period`, checks
    /// whether the last `p` tokens equal the `p` tokens before them —
    /// i.e. the emission is in a cycle of length `p`. `is_in_ngram_loop` only
    /// fires when the cycle length is exactly the n-gram size; this runs over
    /// every period and catches the in-between cases (e.g. the 27-token
    /// "you are Pio Chat… Device: macos-aarch64… Current date: April 20,
    /// 2026" loop we see on Gemma 4 26B that a period-16 or period-32 check
    /// misses entirely).
    ///
    /// O(`max_period²`) per call, but max_period=48 is fine: runs once per
    /// emitted token and the inner loop is a handful of integer compares.
    pub fn is_in_cycle(&self, max_period: usize) -> bool {
        let len = self.recent.len();
        if len < 2 {
            return false;
        }
        let end = len; // recent[len-1] is most-recent
        let cap = max_period.min(len / 2);
        // Access via VecDeque indexing (O(1) per element — VecDeque is a
        // ring buffer). Checking from shortest period first so tight loops
        // (single-token, alternating) return before the expensive cases.
        for p in 1..=cap {
            let mut all_match = true;
            for i in 0..p {
                let a = self.recent[end - 1 - i];
                let b = self.recent[end - 1 - p - i];
                if a != b {
                    all_match = false;
                    break;
                }
            }
            if all_match {
                return true;
            }
        }
        false
    }

    /// Apply repetition penalty in-place, matching the llama.cpp convention:
    /// `logit = logit / penalty` when `logit > 0`, `logit * penalty` when
    /// `logit < 0`. Both cases push the logit toward `-inf` when penalty > 1,
    /// reducing the probability of re-emitting recent tokens. No-op when
    /// penalty is `None` / `1.0` / the recent-buffer is empty.
    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
        let penalty = match self.repetition_penalty {
            Some(p) if p > 1.0 + f32::EPSILON => p,
            _ => return,
        };
        if self.recent.is_empty() {
            return;
        }
        for &tok in &self.recent {
            let i = tok as usize;
            if i >= logits.len() {
                continue;
            }
            let v = logits[i];
            logits[i] = if v > 0.0 { v / penalty } else { v * penalty };
        }
    }

    /// Sample a token ID from a logits slice of shape (vocab_size,).
    pub fn sample_from_logits(&mut self, logits: &[f32]) -> u32 {
        // Apply repetition penalty BEFORE temperature/top-p/top-k. This
        // mirrors llama.cpp order-of-operations and matches what the HF
        // `RepetitionPenaltyLogitsProcessor` does (it runs in the `PRE`
        // slot before temperature).
        let mut penalized: Vec<f32> = logits.to_vec();
        self.apply_repetition_penalty(&mut penalized);

        if self.temperature == 0.0 {
            return Self::argmax(&penalized);
        }

        // Apply temperature
        let scaled: Vec<f32> = if self.temperature != 1.0 {
            penalized.iter().map(|&v| v / self.temperature).collect()
        } else {
            penalized
        };

        // Softmax
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();

        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Top-k
        if let Some(k) = self.top_k {
            indexed.truncate(k.max(1) as usize);
        }

        // Top-p (nucleus)
        if let Some(p) = self.top_p {
            let mut cumulative = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, (_, prob)) in indexed.iter().enumerate() {
                cumulative += prob;
                if cumulative >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }

        // Normalize and sample
        let total: f32 = indexed.iter().map(|(_, p)| p).sum();
        if total <= 0.0 {
            return indexed.first().map(|(idx, _)| *idx as u32).unwrap_or(0);
        }

        let r: f32 = self.rng.random::<f32>() * total;
        let mut cumulative = 0.0f32;
        for (idx, prob) in &indexed {
            cumulative += prob;
            if cumulative >= r {
                return *idx as u32;
            }
        }

        indexed.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
    }

    fn argmax(values: &[f32]) -> u32 {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }
}
