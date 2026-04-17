//! GGUF binary metadata parsing + RAM estimation.
//!
//! Pure file-format logic: parses the GGUF header/kv table, classifies the
//! quantization tier, and estimates inference memory. Kept in `gen2/bundle/`
//! because these types and functions are coupled to the model file format,
//! not to chat or app-level orchestration.

use std::fs;
use std::io::{self, Read};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::error::PioError;
use crate::types::{Model, ModelMetadata};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Trim whitespace from an `Option<String>`; return `None` if the result is empty.
///
/// Used by callers that parse raw GGUF string fields (often zero-padded or
/// whitespace-wrapped) into meaningful model metadata fields.
pub fn trim_optional(input: Option<String>) -> Option<String> {
    input.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

// ── GGUF binary constants ──────────────────────────────────────────────────

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// ── GGUF metadata struct ───────────────────────────────────────────────────

/// Raw metadata fields extracted from a GGUF file header.
#[derive(Debug, Default, Clone)]
pub struct GgufMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub organization: Option<String>,
    pub contact: Option<String>,
    pub license: Option<String>,
    pub context_length: Option<u64>,
    pub architecture: Option<String>,
    pub file_type: Option<u64>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub vocab_size: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub chat_template: Option<String>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
}

// ── Low-level GGUF binary readers ──────────────────────────────────────────

fn read_len_prefixed_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = reader.read_u64::<LittleEndian>()?;
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_len_prefixed_string<R: Read>(reader: &mut R) -> io::Result<()> {
    let len = reader.read_u64::<LittleEndian>()?;
    io::copy(&mut reader.take(len), &mut io::sink())?;
    Ok(())
}

fn primitive_byte_size(value_type: u32) -> Option<u64> {
    match value_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => Some(1),
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => Some(2),
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => Some(4),
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

fn skip_array<R: Read>(reader: &mut R, element_type: u32, len: u64) -> io::Result<()> {
    match element_type {
        GGUF_TYPE_STRING => {
            for _ in 0..len {
                skip_len_prefixed_string(reader)?;
            }
        }
        GGUF_TYPE_ARRAY => {
            for _ in 0..len {
                let nested_type = reader.read_u32::<LittleEndian>()?;
                let nested_len = reader.read_u64::<LittleEndian>()?;
                skip_array(reader, nested_type, nested_len)?;
            }
        }
        ty => {
            if let Some(size) = primitive_byte_size(ty) {
                let bytes = size
                    .checked_mul(len)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "array too large"))?;
                io::copy(&mut reader.take(bytes), &mut io::sink())?;
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported GGUF array element type: {ty}"),
                ));
            }
        }
    }
    Ok(())
}

fn skip_value<R: Read>(reader: &mut R, value_type: u32) -> io::Result<()> {
    match value_type {
        GGUF_TYPE_STRING => skip_len_prefixed_string(reader),
        GGUF_TYPE_ARRAY => {
            let element_type = reader.read_u32::<LittleEndian>()?;
            let len = reader.read_u64::<LittleEndian>()?;
            skip_array(reader, element_type, len)
        }
        ty => {
            if let Some(size) = primitive_byte_size(ty) {
                io::copy(&mut reader.take(size), &mut io::sink())?;
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported GGUF value type: {ty}"),
                ))
            }
        }
    }
}

fn read_value_as_string<R: Read>(reader: &mut R, value_type: u32) -> io::Result<Option<String>> {
    if value_type == GGUF_TYPE_STRING {
        read_len_prefixed_string(reader).map(Some)
    } else {
        skip_value(reader, value_type)?;
        Ok(None)
    }
}

fn read_value_as_u64<R: Read>(reader: &mut R, value_type: u32) -> io::Result<Option<u64>> {
    let value = match value_type {
        GGUF_TYPE_UINT8 => reader.read_u8()? as u64,
        GGUF_TYPE_INT8 => {
            let v = reader.read_i8()? as i64;
            if v < 0 {
                return Ok(None);
            }
            v as u64
        }
        GGUF_TYPE_UINT16 => reader.read_u16::<LittleEndian>()? as u64,
        GGUF_TYPE_INT16 => {
            let v = reader.read_i16::<LittleEndian>()? as i64;
            if v < 0 {
                return Ok(None);
            }
            v as u64
        }
        GGUF_TYPE_UINT32 => reader.read_u32::<LittleEndian>()? as u64,
        GGUF_TYPE_INT32 => {
            let v = reader.read_i32::<LittleEndian>()? as i64;
            if v < 0 {
                return Ok(None);
            }
            v as u64
        }
        GGUF_TYPE_UINT64 => reader.read_u64::<LittleEndian>()?,
        GGUF_TYPE_INT64 => {
            let v = reader.read_i64::<LittleEndian>()?;
            if v < 0 {
                return Ok(None);
            }
            v as u64
        }
        _ => {
            skip_value(reader, value_type)?;
            return Ok(None);
        }
    };
    Ok(Some(value))
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Parse raw metadata fields from a GGUF file header.
///
/// Reads the magic bytes, version, and iterates KV pairs to extract model
/// metadata (name, architecture, context length, quantization, etc.).
pub fn parse_gguf_metadata(path: &Path) -> Result<GgufMetadata, PioError> {
    let file = fs::File::open(path).map_err(PioError::io)?;
    let mut reader = io::BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(PioError::io)?;
    if &magic != b"GGUF" {
        return Err(PioError::io("file is not in GGUF format"));
    }

    let version = reader.read_u32::<LittleEndian>().map_err(PioError::io)?;
    if version == 0 || version > 3 {
        return Err(PioError::io(format!("unsupported GGUF version: {version}")));
    }

    let _tensor_count = reader.read_u64::<LittleEndian>().map_err(PioError::io)?;
    let kv_count = reader.read_u64::<LittleEndian>().map_err(PioError::io)?;

    let mut metadata = GgufMetadata::default();

    for _ in 0..kv_count {
        let key = read_len_prefixed_string(&mut reader).map_err(PioError::io)?;
        let value_type = reader.read_u32::<LittleEndian>().map_err(PioError::io)?;

        match key.as_str() {
            "general.name" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.name = Some(value.trim().to_string());
                }
            }
            "general.description" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.description = Some(value.trim().to_string());
                }
            }
            "general.author" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.author = Some(value.trim().to_string());
                }
            }
            "general.organization" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.organization = Some(value.trim().to_string());
                }
            }
            "general.contact" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.contact = Some(value.trim().to_string());
                }
            }
            "general.license" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.license = Some(value.trim().to_string());
                }
            }
            "general.architecture" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.architecture = Some(value.trim().to_string());
                }
            }
            "general.file_type" => {
                if let Some(value) =
                    read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.file_type = Some(value);
                }
            }
            "tokenizer.chat_template" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(PioError::io)?
                {
                    metadata.chat_template = Some(value);
                }
            }
            // Arch-prefixed keys: use suffix matching to handle all architectures
            _ => {
                let key_str = key.as_str();
                if key_str.ends_with(".context_length") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.context_length = Some(value);
                    }
                } else if key_str.ends_with(".embedding_length") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.embedding_length = Some(value);
                    }
                } else if key_str.ends_with(".block_count") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.block_count = Some(value);
                    }
                } else if key_str.ends_with(".attention.head_count_kv") {
                    // Must check before .head_count to avoid partial suffix match
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.head_count_kv = Some(value);
                    }
                } else if key_str.ends_with(".attention.head_count") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.head_count = Some(value);
                    }
                } else if key_str.ends_with(".vocab_size") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.vocab_size = Some(value);
                    }
                } else if key_str.ends_with(".feed_forward_length") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.feed_forward_length = Some(value);
                    }
                } else if key_str.ends_with(".expert_count") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.expert_count = Some(value);
                    }
                } else if key_str.ends_with(".expert_used_count") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(PioError::io)?
                    {
                        metadata.expert_used_count = Some(value);
                    }
                } else {
                    skip_value(&mut reader, value_type).map_err(PioError::io)?;
                }
            }
        }
    }

    Ok(metadata)
}

/// Map GGUF `general.file_type` enum value to a human-readable quantization label.
pub fn file_type_to_quantization_label(ft: u64) -> Option<&'static str> {
    match ft {
        0 => Some("F32"),
        1 => Some("F16"),
        2 => Some("Q4_0"),
        3 => Some("Q4_1"),
        7 => Some("Q8_0"),
        8 => Some("Q5_0"),
        9 => Some("Q5_1"),
        10 => Some("Q2_K"),
        11 => Some("Q3_K_S"),
        12 => Some("Q3_K_M"),
        13 => Some("Q3_K_L"),
        14 => Some("Q4_K_S"),
        15 => Some("Q4_K_M"),
        16 => Some("Q5_K_S"),
        17 => Some("Q5_K_M"),
        18 => Some("Q6_K"),
        19 => Some("IQ2_XXS"),
        20 => Some("IQ2_XS"),
        21 => Some("IQ3_XXS"),
        22 => Some("IQ1_S"),
        23 => Some("IQ4_NL"),
        24 => Some("IQ3_S"),
        25 => Some("IQ3_M"),
        26 => Some("IQ2_S"),
        27 => Some("IQ2_M"),
        28 => Some("IQ4_XS"),
        29 => Some("IQ1_M"),
        30 => Some("BF16"),
        31 => Some("Q4_0_4_4"),
        32 => Some("Q4_0_4_8"),
        33 => Some("Q4_0_8_8"),
        34 => Some("TQ1_0"),
        35 => Some("TQ2_0"),
        _ => None,
    }
}

/// Approximate bits-per-weight for a given GGUF file_type, used for param estimation.
pub fn file_type_bits_per_weight(ft: u64) -> Option<f64> {
    match ft {
        0 => Some(32.0),  // F32
        1 => Some(16.0),  // F16
        2 => Some(4.5),   // Q4_0
        3 => Some(5.0),   // Q4_1
        7 => Some(8.5),   // Q8_0
        8 => Some(5.5),   // Q5_0
        9 => Some(5.5),   // Q5_1
        10 => Some(3.35), // Q2_K
        11 => Some(3.50), // Q3_K_S
        12 => Some(3.70), // Q3_K_M
        13 => Some(3.90), // Q3_K_L
        14 => Some(4.58), // Q4_K_S
        15 => Some(4.83), // Q4_K_M
        16 => Some(5.54), // Q5_K_S
        17 => Some(5.69), // Q5_K_M
        18 => Some(6.59), // Q6_K
        19 => Some(2.06), // IQ2_XXS
        20 => Some(2.31), // IQ2_XS
        21 => Some(3.06), // IQ3_XXS
        22 => Some(1.56), // IQ1_S
        23 => Some(4.5),  // IQ4_NL
        24 => Some(3.44), // IQ3_S
        25 => Some(3.44), // IQ3_M
        26 => Some(2.5),  // IQ2_S
        27 => Some(2.7),  // IQ2_M
        28 => Some(4.25), // IQ4_XS
        29 => Some(1.75), // IQ1_M
        30 => Some(16.0), // BF16
        34 => Some(1.69), // TQ1_0
        35 => Some(2.06), // TQ2_0
        _ => None,
    }
}

/// Estimate total parameter count from architecture dimensions or file size.
///
/// For MoE models, returns the *total* param count (all experts).
pub fn estimate_parameter_count(meta: &GgufMetadata, file_size: Option<u64>) -> Option<u64> {
    // Primary: compute from architecture params
    if let (Some(d), Some(n_layer)) = (meta.embedding_length, meta.block_count) {
        let d_ff = meta.feed_forward_length.unwrap_or((d as f64 * 2.67) as u64);
        let vocab = meta.vocab_size.unwrap_or(32000);

        // Attention: Q,K,V,O projections per layer
        let attention_params = n_layer * 4 * d * d;
        // FFN: SwiGLU (gate + up + down) per layer
        let ffn_params_per_expert = n_layer * 3 * d * d_ff;
        // MoE: multiply FFN by expert count (each expert has its own FFN)
        let n_experts = meta.expert_count.unwrap_or(1);
        let ffn_params = ffn_params_per_expert * n_experts;
        let embed_params = vocab * d;

        return Some(attention_params + ffn_params + embed_params);
    }

    // Fallback: estimate from file size and quantization
    if let (Some(size), Some(ft)) = (file_size, meta.file_type)
        && let Some(bpw) = file_type_bits_per_weight(ft)
    {
        // Subtract ~10% for metadata/KV overhead
        let effective_size = (size as f64 * 0.9) as u64;
        return Some((effective_size as f64 * 8.0 / bpw) as u64);
    }

    None
}

/// Estimate RAM usage in bytes to run a model at the given context size.
///
/// Uses architecture details (layer count, KV heads, embedding dim) to
/// compute KV cache size when available, otherwise falls back to a
/// 1.2x file-size heuristic plus 500 MB runtime overhead.
pub fn estimate_ram_bytes(metadata: &ModelMetadata, file_size: u64, context_size: u32) -> u64 {
    const OVERHEAD: u64 = 500 * 1024 * 1024; // 500 MB runtime overhead

    // If we have architecture details, compute KV cache estimate
    if let (Some(n_layer), Some(n_kv), Some(d), Some(n_head)) = (
        metadata.block_count,
        metadata.head_count_kv,
        metadata.embedding_length,
        metadata.head_count,
    ) && n_head > 0
    {
        let head_dim = d / n_head;
        // KV cache: 2 (K+V) * layers * kv_heads * head_dim * context * 2 bytes (fp16)
        let kv_cache = 2 * n_layer * n_kv * head_dim * (context_size as u64) * 2;
        return file_size + kv_cache + OVERHEAD;
    }

    // Fallback: 1.2x file size + overhead
    ((file_size as f64 * 1.2) as u64) + OVERHEAD
}

/// Build a [`ModelMetadata`] from raw GGUF header fields.
///
/// Returns `None` if the GGUF header contains no useful architecture info
/// (no architecture, no file_type, no block_count).
pub fn build_model_metadata(gguf: &GgufMetadata, file_size: Option<u64>) -> Option<ModelMetadata> {
    // Only build metadata if we extracted at least architecture or file_type
    if gguf.architecture.is_none() && gguf.file_type.is_none() && gguf.block_count.is_none() {
        return None;
    }

    let quantization = gguf
        .file_type
        .and_then(file_type_to_quantization_label)
        .map(|s| s.to_string());

    let parameter_count = estimate_parameter_count(gguf, file_size);

    let supports_tools = gguf.chat_template.as_ref().map(|t| t.contains("tools"));

    Some(ModelMetadata {
        architecture: gguf.architecture.clone(),
        quantization,
        file_type: gguf.file_type.map(|v| v as u32),
        parameter_count,
        context_length: gguf.context_length,
        embedding_length: gguf.embedding_length,
        block_count: gguf.block_count,
        head_count: gguf.head_count,
        head_count_kv: gguf.head_count_kv,
        vocab_size: gguf.vocab_size,
        feed_forward_length: gguf.feed_forward_length,
        supports_tools,
        expert_count: gguf.expert_count,
        expert_used_count: gguf.expert_used_count,
    })
}

/// Detect model format from a file or directory path.
///
/// Inspects file extensions for single files, or directory contents for
/// multi-file model bundles. Defaults to `"gguf"` when the format cannot
/// be determined.
pub fn detect_format_from_path(path: &Path) -> String {
    if path.is_file()
        && let Some(ext) = path.extension().and_then(|e| e.to_str())
    {
        return match ext {
            "gguf" => "gguf",
            "onnx" => "onnx",
            "safetensors" => "mlx",
            _ => "gguf", // default assumption for unknown single files
        }
        .to_string();
    }
    if path.is_dir() {
        if path.join("model.onnx").exists() {
            return "onnx".to_string();
        }
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                if entry.path().extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    return "mlx".to_string();
                }
            }
        }
    }
    "gguf".to_string()
}

/// Try to backfill metadata for a GGUF model that has none stored.
///
/// Returns `true` if metadata was populated (caller should persist the model).
pub fn backfill_metadata(model: &mut Model) -> bool {
    if model.metadata.is_some() {
        return false;
    }
    let path_str = match model.model_path.as_deref() {
        Some(p) if p.ends_with(".gguf") => p,
        _ => return false,
    };
    let p = Path::new(path_str);
    if !p.is_file() {
        return false;
    }
    let file_size = fs::metadata(p).ok().map(|m| m.len());
    match parse_gguf_metadata(p) {
        Ok(gguf) => {
            if let Some(meta) = build_model_metadata(&gguf, file_size) {
                // Also fix context_size if it was the 8192 default and GGUF says otherwise
                if let Some(ctx) = gguf.context_length
                    && model.config.context_size == 8192
                    && ctx > 0
                    && ctx != 8192
                {
                    model.config.context_size = ctx.min(u32::MAX as u64) as u32;
                }
                model.metadata = Some(meta);
                true
            } else {
                false
            }
        }
        Err(_) => false,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    // ── file_type_to_quantization_label ────────────────────────────────

    #[test]
    fn quantization_label_known_types() {
        assert_eq!(file_type_to_quantization_label(0), Some("F32"));
        assert_eq!(file_type_to_quantization_label(1), Some("F16"));
        assert_eq!(file_type_to_quantization_label(15), Some("Q4_K_M"));
        assert_eq!(file_type_to_quantization_label(18), Some("Q6_K"));
        assert_eq!(file_type_to_quantization_label(30), Some("BF16"));
        assert_eq!(file_type_to_quantization_label(34), Some("TQ1_0"));
    }

    #[test]
    fn quantization_label_unknown_type() {
        assert_eq!(file_type_to_quantization_label(99), None);
        assert_eq!(file_type_to_quantization_label(1000), None);
    }

    // ── file_type_bits_per_weight ──────────────────────────────────────

    #[test]
    fn bits_per_weight_known_types() {
        assert_eq!(file_type_bits_per_weight(0), Some(32.0));
        assert_eq!(file_type_bits_per_weight(1), Some(16.0));
        assert_eq!(file_type_bits_per_weight(15), Some(4.83));
        assert_eq!(file_type_bits_per_weight(18), Some(6.59));
    }

    #[test]
    fn bits_per_weight_unknown_type() {
        assert_eq!(file_type_bits_per_weight(99), None);
    }

    // ── estimate_parameter_count ───────────────────────────────────────

    #[test]
    fn param_count_from_architecture() {
        // Simulate a small model: 32 layers, d=4096, 32 heads, 32k vocab
        let meta = GgufMetadata {
            embedding_length: Some(4096),
            block_count: Some(32),
            feed_forward_length: Some(11008),
            vocab_size: Some(32000),
            head_count: Some(32),
            ..Default::default()
        };

        let count = estimate_parameter_count(&meta, None);
        assert!(count.is_some());
        let c = count.unwrap();
        // Rough check: should be in the ~7B range
        assert!(c > 5_000_000_000, "expected >5B params, got {c}");
        assert!(c < 10_000_000_000, "expected <10B params, got {c}");
    }

    #[test]
    fn param_count_from_file_size_fallback() {
        // 4 GB file with Q4_K_M (ft=15, bpw=4.83)
        let meta = GgufMetadata {
            file_type: Some(15),
            ..Default::default()
        };
        let file_size = 4 * 1024 * 1024 * 1024_u64; // 4 GB

        let count = estimate_parameter_count(&meta, Some(file_size));
        assert!(count.is_some());
        let c = count.unwrap();
        // 4GB * 0.9 * 8 / 4.83 ~ 5.97B
        assert!(c > 4_000_000_000, "expected >4B params, got {c}");
        assert!(c < 8_000_000_000, "expected <8B params, got {c}");
    }

    #[test]
    fn param_count_no_info() {
        let meta = GgufMetadata::default();
        assert_eq!(estimate_parameter_count(&meta, None), None);
    }

    #[test]
    fn param_count_moe_model() {
        // Simulate Mixtral-like: 32 layers, d=4096, 8 experts
        let meta = GgufMetadata {
            embedding_length: Some(4096),
            block_count: Some(32),
            feed_forward_length: Some(14336),
            vocab_size: Some(32000),
            head_count: Some(32),
            expert_count: Some(8),
            expert_used_count: Some(2),
            ..Default::default()
        };

        let count = estimate_parameter_count(&meta, None);
        assert!(count.is_some());
        let c = count.unwrap();
        // Mixtral-8x7B has ~46.7B total params
        assert!(c > 30_000_000_000, "expected >30B params, got {c}");
        assert!(c < 60_000_000_000, "expected <60B params, got {c}");
    }

    // ── estimate_ram_bytes ─────────────────────────────────────────────

    #[test]
    fn ram_estimate_with_architecture() {
        let metadata = ModelMetadata {
            block_count: Some(32),
            head_count: Some(32),
            head_count_kv: Some(8),
            embedding_length: Some(4096),
            ..Default::default()
        };
        let file_size = 4 * 1024 * 1024 * 1024_u64; // 4 GB
        let context_size = 4096;

        let ram = estimate_ram_bytes(&metadata, file_size, context_size);

        // File size + KV cache + overhead
        // KV cache: 2 * 32 * 8 * 128 * 4096 * 2 = 536,870,912 bytes (~512 MB)
        // Total: ~4GB + ~512MB + 500MB = ~5GB
        assert!(ram > file_size, "RAM should exceed file size");
        let overhead_mb = (ram - file_size) / (1024 * 1024);
        assert!(
            overhead_mb > 900,
            "expected >900MB overhead, got {overhead_mb}MB"
        );
    }

    #[test]
    fn ram_estimate_fallback() {
        let metadata = ModelMetadata::default();
        let file_size = 4 * 1024 * 1024 * 1024_u64; // 4 GB

        let ram = estimate_ram_bytes(&metadata, file_size, 4096);

        // Fallback: 1.2 * file_size + 500MB
        let expected = ((file_size as f64 * 1.2) as u64) + 500 * 1024 * 1024;
        assert_eq!(ram, expected);
    }

    // ── build_model_metadata ───────────────────────────────────────────

    #[test]
    fn build_metadata_with_architecture() {
        let gguf = GgufMetadata {
            architecture: Some("llama".to_string()),
            file_type: Some(15),
            embedding_length: Some(4096),
            block_count: Some(32),
            head_count: Some(32),
            head_count_kv: Some(8),
            vocab_size: Some(32000),
            feed_forward_length: Some(11008),
            chat_template: Some("{% if tools %}...{% endif %}".to_string()),
            ..Default::default()
        };

        let meta = build_model_metadata(&gguf, Some(4_000_000_000));
        assert!(meta.is_some());
        let m = meta.unwrap();
        assert_eq!(m.architecture.as_deref(), Some("llama"));
        assert_eq!(m.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(m.file_type, Some(15));
        assert_eq!(m.supports_tools, Some(true));
        assert!(m.parameter_count.is_some());
    }

    #[test]
    fn build_metadata_empty_gguf() {
        let gguf = GgufMetadata::default();
        assert!(build_model_metadata(&gguf, None).is_none());
    }

    #[test]
    fn build_metadata_tools_detection() {
        // Template WITHOUT tools
        let gguf = GgufMetadata {
            architecture: Some("qwen2".to_string()),
            chat_template: Some("{{ messages }}".to_string()),
            ..Default::default()
        };
        let meta = build_model_metadata(&gguf, None).unwrap();
        assert_eq!(meta.supports_tools, Some(false));
    }

    // ── detect_format_from_path ────────────────────────────────────────

    #[test]
    fn format_detection_gguf_extension() {
        let p = Path::new("/tmp/model.gguf");
        // File doesn't exist so is_file() returns false, falls through to default
        assert_eq!(detect_format_from_path(p), "gguf");
    }

    #[test]
    fn format_detection_default() {
        let p = Path::new("/tmp/nonexistent_file.xyz");
        assert_eq!(detect_format_from_path(p), "gguf");
    }

    // ── trim_optional ──────────────────────────────────────────────────

    #[test]
    fn trim_optional_whitespace() {
        assert_eq!(
            trim_optional(Some("  hello  ".into())),
            Some("hello".into())
        );
        assert_eq!(trim_optional(Some("   ".into())), None);
        assert_eq!(trim_optional(Some("".into())), None);
        assert_eq!(trim_optional(None), None);
    }

    // ── parse_gguf_metadata ────────────────────────────────────────────

    #[test]
    fn parse_gguf_nonexistent_file() {
        let result = parse_gguf_metadata(Path::new("/tmp/nonexistent_model.gguf"));
        assert!(result.is_err());
    }

    #[test]
    fn parse_gguf_non_gguf_file() {
        // Create a temp file with non-GGUF content
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.gguf");
        fs::write(&path, b"NOT_GGUF_DATA_HERE").unwrap();

        let result = parse_gguf_metadata(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("not in GGUF format"),
            "expected GGUF format error, got: {}",
            err.message
        );
    }

    #[test]
    fn parse_gguf_minimal_valid_header() {
        // Build a minimal valid GGUF v3 header with 0 tensors, 0 KV pairs
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count = 0

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("minimal.gguf");
        fs::write(&path, &buf).unwrap();

        let result = parse_gguf_metadata(&path);
        assert!(result.is_ok());
        let meta = result.unwrap();
        assert!(meta.name.is_none());
        assert!(meta.architecture.is_none());
    }

    #[test]
    fn parse_gguf_with_string_kv() {
        // Build a GGUF v3 header with one KV pair: general.name = "TestModel"
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count = 1

        // Key: "general.name"
        let key = b"general.name";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);

        // Value type: STRING (8)
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());

        // Value: "TestModel"
        let value = b"TestModel";
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("named.gguf");
        fs::write(&path, &buf).unwrap();

        let result = parse_gguf_metadata(&path);
        assert!(result.is_ok());
        let meta = result.unwrap();
        assert_eq!(meta.name.as_deref(), Some("TestModel"));
    }

    #[test]
    fn parse_gguf_with_uint32_kv() {
        // Build a GGUF v3 header with one KV pair: general.file_type = 15 (Q4_K_M)
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count = 1

        // Key: "general.file_type"
        let key = b"general.file_type";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);

        // Value type: UINT32 (4)
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());

        // Value: 15
        buf.extend_from_slice(&15u32.to_le_bytes());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("typed.gguf");
        fs::write(&path, &buf).unwrap();

        let result = parse_gguf_metadata(&path);
        assert!(result.is_ok());
        let meta = result.unwrap();
        assert_eq!(meta.file_type, Some(15));
    }
}
