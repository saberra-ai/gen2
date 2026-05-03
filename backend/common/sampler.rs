//! Token sampling from logit slices, shared across backends.

use std::collections::VecDeque;

use rand::Rng;

/// Window over which repetition penalty + loop detectors act. 128 lets the
/// cycle-period detector see up through a ~60-token repeat unit (needs
/// `2 * period` tokens buffered) — catches typical "system prompt loop"
/// failures on Gemma 4 26B. Llama.cpp default is 64, fine for rep penalty
/// but too tight for cycle detection.
const REPETITION_WINDOW: usize = 128;

/// DRY (Don't Repeat Yourself) penalty parameters.
///
/// Penalises tokens that would extend an n-gram that already appears in
/// recent output. For each candidate token `t`, we find the longest match
/// `L` between the suffix of the recent history ending at the current
/// position (i.e. if we emitted `t`, how many preceding tokens would now
/// form the tail of a previously-seen sequence?). When `L >= allowed_length`,
/// subtract `multiplier * base^(L - allowed_length)` from the candidate's
/// logit. Standard values: multiplier=0.8, base=1.75, allowed_length=2.
#[derive(Debug, Clone, Copy)]
pub struct DryParams {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
}

impl Default for DryParams {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 2,
        }
    }
}

/// XTC (Exclude Top Choices) parameters.
///
/// With probability `probability`, remove all candidate tokens whose prob
/// is >= `threshold` EXCEPT the lowest one — forcing the sampler to pick
/// a "second-tier" choice. Only activates when temperature > 0 because
/// greedy makes the operation meaningless. Standard values: probability
/// in [0.0, 1.0], threshold ~0.05–0.2. Applied after nucleus filtering
/// so it operates on the already-truncated candidate set.
#[derive(Debug, Clone, Copy)]
pub struct XtcParams {
    pub probability: f32,
    pub threshold: f32,
}

// consumed by workspace dependents (src-tauri, pio-daemon)
#[allow(dead_code)]
pub struct Sampler {
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<i32>,
    /// Repetition penalty factor. `None` or `Some(1.0)` disables. Values
    /// above `1.0` reduce the probability of recently-emitted tokens;
    /// llama.cpp default is `1.1`, HuggingFace default is `1.0` (off).
    repetition_penalty: Option<f32>,
    /// Presence penalty: a fixed additive damp subtracted from the logit
    /// of every token that has appeared at least once in the recent
    /// window. `None` or `Some(0.0)` disables. Different from
    /// repetition_penalty (multiplicative) and frequency_penalty
    /// (count-scaled). HF / OpenAI convention: typical values 0.0–2.0;
    /// the Qwen3.5/3.6 family upstream README recommends `1.5` for
    /// non-thinking general-tasks mode.
    presence_penalty: Option<f32>,
    /// Min-p threshold (Apfelmus 2023). Remove tokens whose prob <
    /// `min_p * max_prob`. None or 0.0 disables.
    min_p: Option<f32>,
    /// DRY n-gram repetition penalty parameters (pi6am, 2024).
    dry: Option<DryParams>,
    /// XTC (Exclude Top Choices) sampler parameters (p-e-w, 2024).
    xtc: Option<XtcParams>,
    /// Additive logit bias for end-of-turn tokens. Quantized Gemma 4 26B
    /// trained `\n` just above `<turn|>` at answer boundaries — the top two
    /// candidates sit within ~0.6 logits of each other. Mid-sentence the
    /// gap to any EOT is several logits wide, so a +1–2.0 bias doesn't
    /// cause premature termination there.
    eot_bias: Option<(Vec<u32>, f32)>,
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
            presence_penalty: None,
            min_p: None,
            dry: None,
            xtc: None,
            eot_bias: None,
            recent: VecDeque::with_capacity(REPETITION_WINDOW),
            rng: rand::rng(),
        }
    }

    /// Enable presence penalty — additive damp on tokens that appeared
    /// at least once in the recent window. `None` or `0.0` disables.
    /// See field docs for relationship to other penalty types.
    pub fn with_presence_penalty(mut self, penalty: Option<f32>) -> Self {
        self.presence_penalty = match penalty {
            Some(p) if p > 0.0 => Some(p),
            _ => None,
        };
        self
    }

    /// Enable an additive logit bias on end-of-turn tokens. See field docs
    /// on `eot_bias` for why this is needed on Gemma 4 26B.
    pub fn with_eot_bias(mut self, ids: Vec<u32>, bias: f32) -> Self {
        if !ids.is_empty() && bias != 0.0 {
            self.eot_bias = Some((ids, bias));
        }
        self
    }

    /// Enable min-p nucleus filtering. `None` or `0.0` disables.
    pub fn with_min_p(mut self, min_p: Option<f32>) -> Self {
        self.min_p = match min_p {
            Some(p) if p > 0.0 && p < 1.0 => Some(p),
            _ => None,
        };
        self
    }

    /// Enable DRY n-gram repetition penalty. `None` / `multiplier <= 0` disables.
    pub fn with_dry(mut self, params: Option<DryParams>) -> Self {
        self.dry = match params {
            Some(p) if p.multiplier > 0.0 && p.base > 1.0 => Some(p),
            _ => None,
        };
        self
    }

    /// Enable XTC top-choice exclusion. Only effective when temperature > 0;
    /// greedy decoding always picks argmax and XTC is a no-op there.
    pub fn with_xtc(mut self, params: Option<XtcParams>) -> Self {
        self.xtc = match params {
            Some(p) if p.probability > 0.0 && p.threshold > 0.0 && p.threshold < 1.0 => Some(p),
            _ => None,
        };
        self
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

    /// Apply presence penalty in-place. For each *unique* token id in
    /// the recent window, subtract `presence_penalty` once from its
    /// logit. `None` / `0.0` / empty-recent disables. The "once per
    /// unique token" semantic is what distinguishes presence_penalty
    /// from frequency_penalty (which would scale by occurrence count).
    fn apply_presence_penalty(&self, logits: &mut [f32]) {
        let penalty = match self.presence_penalty {
            Some(p) if p > 0.0 => p,
            _ => return,
        };
        if self.recent.is_empty() {
            return;
        }
        // Dedup by collecting unique ids first; the recent buffer is
        // bounded by REPETITION_WINDOW (128), so a small HashSet is fine.
        use std::collections::HashSet;
        let mut seen: HashSet<u32> = HashSet::with_capacity(self.recent.len());
        for &tok in &self.recent {
            if !seen.insert(tok) {
                continue;
            }
            let i = tok as usize;
            if i < logits.len() {
                logits[i] -= penalty;
            }
        }
    }

    /// Apply DRY n-gram repetition penalty in-place. For each candidate
    /// token `t`, find the longest overlap `L` where the sequence
    /// `[...recent, t]` ends with a substring that appears earlier in
    /// `recent`. When `L > allowed_length`, subtract
    /// `multiplier * base^(L - allowed_length)` from `logit[t]`.
    ///
    /// Implementation: for each occurrence of `self.recent.last()` in
    /// `recent[..recent.len()-1]`, count how far the sequence backwards
    /// agrees; the token that would extend the match gets penalized.
    fn apply_dry_penalty(&self, logits: &mut [f32]) {
        let Some(d) = self.dry else {
            return;
        };
        let n = self.recent.len();
        if n < 2 {
            return;
        }
        let anchor = self.recent[n - 1];

        // For each earlier occurrence of `anchor`, measure how many tokens
        // of the tail match backwards, then record the token that would
        // extend the match (recent[match_end+1]).
        use std::collections::HashMap;
        let mut max_extend_len: HashMap<u32, usize> = HashMap::new();
        for i in 0..n - 1 {
            if self.recent[i] != anchor {
                continue;
            }
            let mut l = 1usize;
            while i >= l && n > l && self.recent[i - l] == self.recent[n - 1 - l] {
                l += 1;
            }
            if i + 1 >= n {
                continue;
            }
            let next = self.recent[i + 1];
            if l >= d.allowed_length {
                let entry = max_extend_len.entry(next).or_insert(0);
                if l > *entry {
                    *entry = l;
                }
            }
        }

        for (tok, l) in max_extend_len {
            let idx = tok as usize;
            if idx >= logits.len() {
                continue;
            }
            let excess = (l - d.allowed_length) as i32;
            let penalty = d.multiplier * d.base.powi(excess);
            logits[idx] -= penalty;
        }
    }

    /// Sample a token ID from a logits slice of shape (vocab_size,).
    ///
    /// Pipeline (llama.cpp order of operations):
    ///   1. repetition penalty (pre-temperature)
    ///   2. presence penalty (pre-temperature)
    ///   3. DRY n-gram penalty (pre-temperature)
    ///   4. EOT logit bias (pre-temperature so the bias survives T>1 scaling)
    ///   5. temperature (division)
    ///   6. softmax
    ///   7. top-k
    ///   8. min-p
    ///   9. top-p
    ///   10. XTC
    ///   11. sample (categorical over remaining distribution)
    pub fn sample_from_logits(&mut self, logits: &[f32]) -> u32 {
        let mut penalized: Vec<f32> = logits.to_vec();
        self.apply_repetition_penalty(&mut penalized);
        self.apply_presence_penalty(&mut penalized);
        self.apply_dry_penalty(&mut penalized);
        if let Some((ids, bias)) = &self.eot_bias {
            for &id in ids {
                let i = id as usize;
                if i < penalized.len() {
                    penalized[i] += *bias;
                }
            }
        }

        if self.temperature == 0.0 {
            return Self::argmax(&penalized);
        }

        let scaled: Vec<f32> = if self.temperature != 1.0 {
            penalized.iter().map(|&v| v / self.temperature).collect()
        } else {
            penalized
        };

        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();

        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(k) = self.top_k {
            indexed.truncate(k.max(1) as usize);
        }

        // min-p: keep only tokens whose prob >= min_p * top_prob. Applied
        // BEFORE top-p per llama.cpp convention. On greedy/deterministic
        // inputs with a dominant mode, min-p leaves just the top candidate.
        if let Some(mp) = self.min_p
            && !indexed.is_empty()
        {
            let top = indexed[0].1;
            let cutoff = mp * top;
            let keep = indexed.iter().position(|(_, p)| *p < cutoff);
            if let Some(k) = keep {
                indexed.truncate(k.max(1));
            }
        }

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

        // XTC: with probability `probability`, drop all candidates above
        // `threshold` EXCEPT the LAST one above threshold (the weakest
        // "top choice"). If fewer than 2 tokens exceed the threshold, XTC
        // is a no-op — we'd just be re-selecting the only eligible token.
        if let Some(xtc) = self.xtc
            && self.rng.random::<f32>() < xtc.probability
        {
            // `indexed` is already sorted by prob desc.
            let above: usize = indexed
                .iter()
                .take_while(|(_, p)| *p >= xtc.threshold)
                .count();
            if above >= 2 {
                // Keep tokens BELOW threshold (everything from `above`
                // onwards) plus the single weakest above-threshold token
                // (indexed[above - 1]).
                let keeper = indexed[above - 1];
                let mut kept: Vec<(usize, f32)> = Vec::with_capacity(indexed.len() - above + 1);
                kept.push(keeper);
                kept.extend(indexed.iter().skip(above).copied());
                indexed = kept;
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn probs_from_logits(s: &mut Sampler, logits: &[f32]) -> Vec<(usize, f32)> {
        // Quick helper: run everything up to (not including) sampling and
        // return the kept (id, prob) pairs. Mirrors `sample_from_logits`
        // but stops before the RNG draw so tests can assert on the set.
        let mut pen = logits.to_vec();
        s.apply_repetition_penalty(&mut pen);
        s.apply_presence_penalty(&mut pen);
        s.apply_dry_penalty(&mut pen);
        if let Some((ids, bias)) = &s.eot_bias {
            for &id in ids {
                let i = id as usize;
                if i < pen.len() {
                    pen[i] += *bias;
                }
            }
        }
        let scaled: Vec<f32> = if s.temperature != 0.0 && s.temperature != 1.0 {
            pen.iter().map(|&v| v / s.temperature).collect()
        } else {
            pen
        };
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(k) = s.top_k {
            indexed.truncate(k.max(1) as usize);
        }
        if let Some(mp) = s.min_p
            && !indexed.is_empty()
        {
            let top = indexed[0].1;
            let cutoff = mp * top;
            if let Some(k) = indexed.iter().position(|(_, p)| *p < cutoff) {
                indexed.truncate(k.max(1));
            }
        }
        if let Some(p) = s.top_p {
            let mut cum = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, (_, q)) in indexed.iter().enumerate() {
                cum += q;
                if cum >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }
        indexed
    }

    /// Tracer for presence_penalty: each unique token in the recent
    /// window has a fixed `penalty` subtracted from its logit, *once*
    /// regardless of how many times it appeared. Distinct from
    /// `repetition_penalty` (multiplicative) and `frequency_penalty`
    /// (additive scaled by count). The Qwen3.5/3.6 family explicitly
    /// recommends `1.5` here for non-thinking general tasks; before
    /// this Sampler enhancement that recommendation was silently
    /// dropped on the MLX backend.
    #[test]
    fn presence_penalty_subtracts_fixed_value_once_per_unique_token() {
        let mut s = Sampler::new(1.0, None, None, None).with_presence_penalty(Some(0.5));
        // Observe token 7 three times — penalty should still apply once.
        s.observe(7);
        s.observe(7);
        s.observe(7);
        // Observe token 3 once.
        s.observe(3);
        // Token 5 unobserved.
        let logits = vec![1.0; 10];
        let mut pen = logits.clone();
        s.apply_presence_penalty(&mut pen);
        assert!(
            (pen[7] - 0.5).abs() < f32::EPSILON,
            "logit[7] should be 1.0 - 0.5 = 0.5, got {}",
            pen[7],
        );
        assert!(
            (pen[3] - 0.5).abs() < f32::EPSILON,
            "logit[3] should be 1.0 - 0.5 = 0.5, got {}",
            pen[3],
        );
        assert!(
            (pen[5] - 1.0).abs() < f32::EPSILON,
            "logit[5] (unobserved) must be unchanged, got {}",
            pen[5],
        );
    }

    #[test]
    fn presence_penalty_disabled_when_none_or_zero() {
        for penalty in [None, Some(0.0)] {
            let mut s = Sampler::new(1.0, None, None, None).with_presence_penalty(penalty);
            s.observe(2);
            let logits = vec![1.0, 1.0, 1.0];
            let mut pen = logits.clone();
            s.apply_presence_penalty(&mut pen);
            assert_eq!(pen, logits, "penalty={penalty:?} should be a no-op",);
        }
    }

    #[test]
    fn min_p_removes_low_probability_tokens() {
        // Top=100, others are 1 — after softmax top dominates. With
        // min_p=0.5, any token with prob < 0.5*top_prob should be removed.
        // Since top_prob≈1.0 (after temp=1), cutoff is 0.5 → only top kept.
        let mut s = Sampler::new(1.0, None, None, None).with_min_p(Some(0.5));
        let mut logits = vec![0.0; 10];
        logits[3] = 100.0;
        let kept = probs_from_logits(&mut s, &logits);
        assert_eq!(kept.len(), 1, "only the top token survives min_p=0.5");
        assert_eq!(kept[0].0, 3);
    }

    #[test]
    fn min_p_keeps_close_tokens() {
        // Two tokens with very close logits — both should survive min_p=0.5.
        let mut s = Sampler::new(1.0, None, None, None).with_min_p(Some(0.5));
        let mut logits = vec![0.0; 5];
        logits[0] = 10.0;
        logits[1] = 9.9;
        let kept = probs_from_logits(&mut s, &logits);
        assert!(kept.len() >= 2, "near-equal tokens should both survive");
        assert!(kept.iter().any(|(id, _)| *id == 0));
        assert!(kept.iter().any(|(id, _)| *id == 1));
    }

    #[test]
    fn dry_penalizes_extending_repeat() {
        // Build recent = [A, B, C, A, B], allowed_length=2.
        // If we emit C next, sequence ends `...A, B, C` which matches
        // earlier `A, B, C`, extending a 3-gram match. Penalty fires.
        let mut s = Sampler::new(0.0, None, None, None).with_dry(Some(DryParams {
            multiplier: 1.0,
            base: 2.0,
            allowed_length: 2,
        }));
        for t in [10u32, 20, 30, 10, 20] {
            s.observe(t);
        }
        let mut logits = vec![0.0f32; 50];
        s.apply_dry_penalty(&mut logits);
        // Token 30 (C) should have its logit reduced.
        assert!(
            logits[30] < 0.0,
            "DRY should penalize token 30 that would extend the [A,B,C] match; got {}",
            logits[30]
        );
        // Unrelated token should be untouched.
        assert_eq!(logits[7], 0.0);
    }

    #[test]
    fn dry_no_penalty_without_repeat() {
        let mut s = Sampler::new(0.0, None, None, None).with_dry(Some(DryParams {
            multiplier: 1.0,
            base: 2.0,
            allowed_length: 2,
        }));
        for t in [1u32, 2, 3, 4] {
            s.observe(t);
        }
        let mut logits = vec![0.0f32; 10];
        s.apply_dry_penalty(&mut logits);
        for v in &logits {
            assert_eq!(*v, 0.0, "no repetition, no penalty");
        }
    }

    #[test]
    fn xtc_drops_top_with_prob_1() {
        // Force XTC to fire every sample. Temperature > 0 required.
        // Top two tokens have ~50% each; threshold=0.1 → both above.
        // XTC drops the strongest, keeps the weakest of the two + any
        // below-threshold tokens.
        let mut s = Sampler::new(1.0, None, None, None).with_xtc(Some(XtcParams {
            probability: 1.0,
            threshold: 0.1,
        }));
        let mut logits = vec![-100.0f32; 5];
        logits[0] = 10.0;
        logits[1] = 9.99;
        // 1000 trials — token 0 should NEVER be sampled (always excluded).
        let mut token0_count = 0usize;
        let mut token1_count = 0usize;
        for _ in 0..1000 {
            let t = s.sample_from_logits(&logits);
            if t == 0 {
                token0_count += 1;
            }
            if t == 1 {
                token1_count += 1;
            }
        }
        assert_eq!(token0_count, 0, "token 0 should always be excluded by XTC");
        assert!(
            token1_count > 500,
            "token 1 should be sampled most of the time (got {})",
            token1_count
        );
    }
}
