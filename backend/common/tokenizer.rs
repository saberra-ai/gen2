//! HuggingFace tokenizer wrapper shared across backends.

use std::path::Path;
use tokenizers::Tokenizer;

// consumed by workspace dependents (src-tauri, pio-daemon)
#[allow(dead_code)]
pub struct HfTokenizer {
    inner: Tokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    /// All stop-token ids for this model: EOS plus any end-of-turn markers
    /// discovered in the added vocab (e.g. `<turn|>` for Gemma 4, `<|eot_id|>`
    /// for Llama 3). Callers should treat hitting any of these as end-of-stream.
    stop_ids: Vec<u32>,
}

// consumed by workspace dependents (src-tauri, pio-daemon)
#[allow(dead_code)]
impl HfTokenizer {
    /// Load tokenizer from a directory containing `tokenizer.json`.
    pub fn from_dir(model_dir: &Path) -> anyhow::Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;

        // Try to resolve BOS/EOS from added tokens
        let bos_id = inner
            .get_added_vocabulary()
            .get_vocab()
            .get("<|begin_of_text|>")
            .or_else(|| inner.get_added_vocabulary().get_vocab().get("<s>"))
            .copied();

        let added = inner.get_added_vocabulary().get_vocab();
        let lookup = |name: &str| added.get(name).copied();

        // Primary EOS: match Llama, Mistral, Qwen, Gemma. `<eos>` covers Gemma
        // (1/2/3/4); `<|eot_id|>` is Llama-3's chat EOT; the others are older.
        let eos_id = lookup("<|end_of_text|>")
            .or_else(|| lookup("</s>"))
            .or_else(|| lookup("<|eot_id|>"))
            .or_else(|| lookup("<eos>"));

        // Stop set: every token whose presence in the stream should end decoding.
        // Gemma 4 needs `<turn|>` (end-of-turn) in addition to `<eos>` because
        // chat mode emits EOT, not EOS. Llama 3 has the same dual-marker pattern.
        let mut stop_ids: Vec<u32> = Vec::new();
        if let Some(id) = eos_id {
            stop_ids.push(id);
        }
        for name in ["<turn|>", "<|eot_id|>", "<|im_end|>", "<end_of_turn>"] {
            if let Some(id) = lookup(name)
                && !stop_ids.contains(&id)
            {
                stop_ids.push(id);
            }
        }

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            stop_ids,
        })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {}", e))
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// All stop-token ids: EOS plus any end-of-turn markers (e.g. `<turn|>`).
    pub fn stop_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
