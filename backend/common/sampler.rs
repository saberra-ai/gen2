//! Token sampling from logit slices, shared across backends.

use rand::Rng;

#[allow(dead_code)]
pub struct Sampler {
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<i32>,
    rng: rand::rngs::ThreadRng,
}

#[allow(dead_code)]
impl Sampler {
    pub fn new(temperature: f32, top_p: Option<f32>, top_k: Option<i32>) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            rng: rand::rng(),
        }
    }

    /// Sample a token ID from a logits slice of shape (vocab_size,).
    pub fn sample_from_logits(&mut self, logits: &[f32]) -> u32 {
        if self.temperature == 0.0 {
            return Self::argmax(logits);
        }

        // Apply temperature
        let scaled: Vec<f32> = if self.temperature != 1.0 {
            logits.iter().map(|&v| v / self.temperature).collect()
        } else {
            logits.to_vec()
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
