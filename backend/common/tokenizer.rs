//! HuggingFace tokenizer wrapper shared across backends.

use std::path::Path;
use tokenizers::Tokenizer;

// consumed by workspace dependents (src-tauri, pio-daemon)
#[allow(dead_code)]
pub struct HfTokenizer {
    inner: Tokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
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

        let eos_id = inner
            .get_added_vocabulary()
            .get_vocab()
            .get("<|end_of_text|>")
            .or_else(|| inner.get_added_vocabulary().get_vocab().get("</s>"))
            .or_else(|| inner.get_added_vocabulary().get_vocab().get("<|eot_id|>"))
            .copied();

        Ok(Self {
            inner,
            bos_id,
            eos_id,
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

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
