//! Token sampling from logit slices, shared across backends.

use std::collections::VecDeque;

use rand::Rng;

/// Window over which repetition penalty acts. Matches llama.cpp default
/// (`--repeat-last-n 64`). Tokens emitted more than 64 steps ago no longer
/// suppress their own re-emission.
const REPETITION_WINDOW: usize = 64;

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
