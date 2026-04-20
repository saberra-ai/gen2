//! HuggingFace tokenizer wrapper shared across backends.

use std::path::Path;
use tokenizers::Tokenizer;

/// Parse `generation_config.json` in `model_dir` and return its `eos_token_id`
/// values as a `Vec<u32>`. Accepts either an integer (single EOS) or an array
/// (multiple EOS). Returns empty when the file is missing or malformed —
/// callers always fall back to name-based EOS resolution from `tokenizer.json`.
fn read_generation_config_eos_ids(model_dir: &Path) -> Vec<u32> {
    let path = model_dir.join("generation_config.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return Vec::new();
    };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    let field = match json.get("eos_token_id") {
        Some(v) => v,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    match field {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                out.push(u as u32);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                if let Some(u) = v.as_u64() {
                    out.push(u as u32);
                }
            }
        }
        _ => {}
    }
    out
}

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
    ///
    /// Also reads `generation_config.json` in the same directory when present
    /// and merges its `eos_token_id` (int or array) into the stop-id set.
    /// This is how model authors declare all the tokens that should terminate
    /// decoding — e.g. Gemma 4 E2B / 31B list `[1, 106, 50]` (`<eos>`,
    /// `<turn|>`, `<|tool_response>`); Llama 3 lists `[128001, 128009]`
    /// (`<|end_of_text|>`, `<|eot_id|>`). Relying on our hardcoded
    /// name-based list alone misses model-specific tokens like 50.
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

        // Stop set: authoritative EOS ids from `generation_config.json` first
        // (per-model, supplied by the trainer), then our hardcoded fallback
        // by name for models that don't ship a generation_config.
        let mut stop_ids: Vec<u32> = Vec::new();
        for id in read_generation_config_eos_ids(model_dir) {
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
        }
        if let Some(id) = eos_id {
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
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
