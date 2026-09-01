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

use crate::engine::ExecError;
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

/// Ceiling on any single length-prefixed GGUF string. Every string in the
/// file declares its length *before* its bytes exist, so the declaration is
/// attacker-controlled and must be bounded before it reaches an allocator —
/// an unbounded `vec![0u8; len]` aborts the process, which no `Result` can
/// catch. The largest real field is `tokenizer.chat_template` (~100 KB).
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;

/// Ceiling on GGUF array nesting. The format lets arrays hold arrays with no
/// self-imposed limit and `skip_array` recurses, so ~12 bytes of input buy
/// one stack frame; without this bound a small file overflows the stack,
/// which aborts the process rather than returning an error.
const MAX_ARRAY_DEPTH: u32 = 64;

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
    if len > MAX_STRING_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF string declares {len} bytes, over the {MAX_STRING_BYTES}-byte limit"),
        ));
    }
    // Grow from what is actually read, never from the declared length: the
    // file may be far shorter than it claims, and `Take::read_to_end`
    // reserves against bytes delivered rather than bytes promised.
    let mut buf = Vec::new();
    let read = reader.take(len).read_to_end(&mut buf)?;
    if read as u64 != len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "GGUF string is truncated",
        ));
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Discard exactly `len` bytes. `io::copy` reports a short copy as success,
/// so a truncated tail would otherwise skip "successfully" and let a
/// truncated file parse as a valid one.
fn skip_exact<R: Read>(reader: &mut R, len: u64) -> io::Result<()> {
    let skipped = io::copy(&mut reader.take(len), &mut io::sink())?;
    if skipped != len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "GGUF value is truncated",
        ));
    }
    Ok(())
}

fn skip_len_prefixed_string<R: Read>(reader: &mut R) -> io::Result<()> {
    let len = reader.read_u64::<LittleEndian>()?;
    skip_exact(reader, len)
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

fn skip_array<R: Read>(reader: &mut R, element_type: u32, len: u64, depth: u32) -> io::Result<()> {
    if depth > MAX_ARRAY_DEPTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF array nesting deeper than {MAX_ARRAY_DEPTH}"),
        ));
    }
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
                skip_array(reader, nested_type, nested_len, depth + 1)?;
            }
        }
        ty => {
            if let Some(size) = primitive_byte_size(ty) {
                let bytes = size
                    .checked_mul(len)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "array too large"))?;
                skip_exact(reader, bytes)?;
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
    skip_value_at_depth(reader, value_type, 0)
}

fn skip_value_at_depth<R: Read>(reader: &mut R, value_type: u32, depth: u32) -> io::Result<()> {
    match value_type {
        GGUF_TYPE_STRING => skip_len_prefixed_string(reader),
        GGUF_TYPE_ARRAY => {
            let element_type = reader.read_u32::<LittleEndian>()?;
            let len = reader.read_u64::<LittleEndian>()?;
            skip_array(reader, element_type, len, depth)
        }
        ty => {
            if let Some(size) = primitive_byte_size(ty) {
                skip_exact(reader, size)
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
pub fn parse_gguf_metadata(path: &Path) -> Result<GgufMetadata, ExecError> {
    let file = fs::File::open(path).map_err(ExecError::io)?;
    let mut reader = io::BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(ExecError::io)?;
    if &magic != b"GGUF" {
        return Err(ExecError::io("file is not in GGUF format"));
    }

    let version = reader.read_u32::<LittleEndian>().map_err(ExecError::io)?;
    if version == 0 || version > 3 {
        return Err(ExecError::io(format!(
            "unsupported GGUF version: {version}"
        )));
    }

    let _tensor_count = reader.read_u64::<LittleEndian>().map_err(ExecError::io)?;
    let kv_count = reader.read_u64::<LittleEndian>().map_err(ExecError::io)?;

    let mut metadata = GgufMetadata::default();

    for _ in 0..kv_count {
        let key = read_len_prefixed_string(&mut reader).map_err(ExecError::io)?;
        let value_type = reader.read_u32::<LittleEndian>().map_err(ExecError::io)?;

        match key.as_str() {
            "general.name" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.name = Some(value.trim().to_string());
                }
            }
            "general.description" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.description = Some(value.trim().to_string());
                }
            }
            "general.author" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.author = Some(value.trim().to_string());
                }
            }
            "general.organization" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.organization = Some(value.trim().to_string());
                }
            }
            "general.contact" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.contact = Some(value.trim().to_string());
                }
            }
            "general.license" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.license = Some(value.trim().to_string());
                }
            }
            "general.architecture" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.architecture = Some(value.trim().to_string());
                }
            }
            "general.file_type" => {
                if let Some(value) =
                    read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.file_type = Some(value);
                }
            }
            "tokenizer.chat_template" => {
                if let Some(value) =
                    read_value_as_string(&mut reader, value_type).map_err(ExecError::io)?
                {
                    metadata.chat_template = Some(value);
                }
            }
            // Arch-prefixed keys: use suffix matching to handle all architectures
            _ => {
                let key_str = key.as_str();
                if key_str.ends_with(".context_length") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.context_length = Some(value);
                    }
                } else if key_str.ends_with(".embedding_length") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.embedding_length = Some(value);
                    }
                } else if key_str.ends_with(".block_count") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.block_count = Some(value);
                    }
                } else if key_str.ends_with(".attention.head_count_kv") {
                    // Must check before .head_count to avoid partial suffix match
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.head_count_kv = Some(value);
                    }
                } else if key_str.ends_with(".attention.head_count") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.head_count = Some(value);
                    }
                } else if key_str.ends_with(".vocab_size") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.vocab_size = Some(value);
                    }
                } else if key_str.ends_with(".feed_forward_length") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.feed_forward_length = Some(value);
                    }
                } else if key_str.ends_with(".expert_count") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.expert_count = Some(value);
                    }
                } else if key_str.ends_with(".expert_used_count") {
                    if let Some(value) =
                        read_value_as_u64(&mut reader, value_type).map_err(ExecError::io)?
                    {
                        metadata.expert_used_count = Some(value);
                    }
                } else {
                    skip_value(&mut reader, value_type).map_err(ExecError::io)?;
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
        // MoE: each expert has its own FFN.
        let n_experts = meta.expert_count.unwrap_or(1);

        // These dimensions come straight out of an untrusted header, and
        // their product overflows u64 for absurd values — which panics in a
        // debug build and wraps to a nonsense count in a release one.
        // Overflow means "not a real model's dimensions", so fall through
        // to the file-size estimate rather than reporting a wrapped number.
        let arch_params = (|| {
            // Attention: Q,K,V,O projections per layer.
            let attention = n_layer.checked_mul(4)?.checked_mul(d)?.checked_mul(d)?;
            // FFN: SwiGLU (gate + up + down) per layer, per expert.
            let ffn = n_layer
                .checked_mul(3)?
                .checked_mul(d)?
                .checked_mul(d_ff)?
                .checked_mul(n_experts)?;
            let embed = vocab.checked_mul(d)?;
            attention.checked_add(ffn)?.checked_add(embed)
        })();
        if let Some(total) = arch_params {
            return Some(total);
        }
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

/// KV-cache bytes per token of context: K+V, fp16, per layer per KV head.
///
/// Single definition site for the formula `estimate_ram_bytes`,
/// `auto_tune_ctx`, and the load-time context clamp all share.
///
/// Saturates rather than overflows: the three dimensions come from an
/// untrusted header, and a saturated cost makes every downstream fit
/// decision refuse, which is the right answer for absurd dimensions.
pub fn kv_bytes_per_token(n_layer: u64, n_head_kv: u64, head_dim: u64) -> u64 {
    2u64.saturating_mul(n_layer)
        .saturating_mul(n_head_kv)
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .max(1)
}

/// Largest context that fits the memory budget once model weights and
/// runtime overhead are paid, capped by the model's training context and
/// an optional tier cap. Closed form — KV cost is linear in context.
///
/// Floors at 2048: below that a model is effectively unusable, and
/// whether it should load at all is residency admission's decision, not
/// context sizing's.
pub fn fit_context(
    budget_bytes: u64,
    model_resident_bytes: u64,
    kv_per_token: u64,
    train_ctx: u32,
    tier_cap: Option<u32>,
) -> u32 {
    const OVERHEAD: u64 = 500 * 1024 * 1024; // matches estimate_ram_bytes
    const FLOOR: u32 = 2048;
    let remaining = budget_bytes
        .saturating_sub(model_resident_bytes)
        .saturating_sub(OVERHEAD);
    let max_tokens = (remaining / kv_per_token.max(1)).min(u64::from(u32::MAX)) as u32;
    // Round down to a 1K boundary so llama.cpp gets a tidy context.
    let fitted = ((max_tokens / 1024) * 1024).max(FLOOR);
    let capped = tier_cap.map_or(fitted, |cap| fitted.min(cap.max(FLOOR)));
    capped.min(train_ctx.max(FLOOR))
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
        // Saturating throughout: every input here is header-derived.
        let kv_cache =
            kv_bytes_per_token(n_layer, n_kv, head_dim).saturating_mul(context_size as u64);
        return file_size.saturating_add(kv_cache).saturating_add(OVERHEAD);
    }

    // Fallback: 1.2x file size + overhead
    ((file_size as f64 * 1.2) as u64).saturating_add(OVERHEAD)
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
        let msg = err.to_string();
        assert!(
            msg.contains("not in GGUF format"),
            "expected GGUF format error, got: {msg}"
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

    // ── fit_context: load-time context clamp ───────────────────────

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Llama-3.1-8B-ish dims: 32 layers, 8 KV heads, head_dim 128.
    fn llama8b_kv() -> u64 {
        kv_bytes_per_token(32, 8, 128)
    }

    #[test]
    fn fit_context_clamps_huge_train_ctx() {
        // 16 GiB budget, 5 GiB weights: 131072-token KV (~16 GiB at these
        // dims) cannot fit — the clamp must land well below train ctx.
        let fitted = fit_context(16 * GIB, 5 * GIB, llama8b_kv(), 131_072, None);
        assert!(fitted < 131_072, "must clamp, got {fitted}");
        assert!(fitted >= 2048, "floor holds, got {fitted}");
        // And the fitted KV actually fits the remaining budget.
        let kv_total = llama8b_kv() * fitted as u64;
        assert!(kv_total <= 16 * GIB - 5 * GIB, "fitted KV overflows budget");
    }

    #[test]
    fn fit_context_respects_train_ctx_when_room() {
        // Tiny model, huge budget: train ctx is the binding constraint.
        let fitted = fit_context(64 * GIB, GIB, kv_bytes_per_token(16, 2, 64), 8_192, None);
        assert_eq!(fitted, 8_192);
    }

    #[test]
    fn fit_context_tier_cap_binds() {
        let fitted = fit_context(
            64 * GIB,
            GIB,
            kv_bytes_per_token(16, 2, 64),
            131_072,
            Some(16_384),
        );
        assert_eq!(fitted, 16_384);
    }

    #[test]
    fn fit_context_floors_at_2048_when_budget_exhausted() {
        // Weights alone exceed budget: context sizing still returns the
        // floor — refusing the load is residency admission's decision.
        let fitted = fit_context(8 * GIB, 12 * GIB, llama8b_kv(), 131_072, None);
        assert_eq!(fitted, 2048);
    }

    #[test]
    fn fit_context_rounds_to_1k() {
        let fitted = fit_context(10 * GIB, 5 * GIB, llama8b_kv(), 131_072, None);
        assert_eq!(fitted % 1024, 0, "context should land on a 1K boundary");
    }

    #[test]
    fn estimate_and_fit_share_the_formula() {
        // estimate_ram_bytes at the fitted context must sit within budget
        // (they share kv_bytes_per_token, so this pins the coupling).
        let budget = 24 * GIB;
        let file = 6 * GIB;
        let fitted = fit_context(budget, file, llama8b_kv(), 131_072, None);
        let meta = crate::types::ModelMetadata {
            block_count: Some(32),
            head_count_kv: Some(8),
            head_count: Some(32),
            embedding_length: Some(4096),
            ..Default::default()
        };
        let est = estimate_ram_bytes(&meta, file, fitted);
        assert!(
            est <= budget,
            "estimate {est} exceeds budget {budget} at fitted ctx {fitted}"
        );
    }

    // ── Adversarial GGUF header fixtures ───────────────────────────────
    //
    // Every fixture is built byte-by-byte here rather than checked in as a
    // binary, so the exact bytes under test are readable and diffable. The
    // contract these pin: `parse_gguf_metadata` on ARBITRARY bytes returns
    // `Ok` or `Err`, and never panics, aborts, hangs, or allocates against
    // an attacker-declared length.

    /// Incremental builder for GGUF byte fixtures.
    #[derive(Default)]
    struct GgufBuilder {
        bytes: Vec<u8>,
    }

    impl GgufBuilder {
        /// `magic` + `version` + `tensor_count` + `kv_count`, in the
        /// 64-bit-count layout this parser defines for every version.
        fn header(version: u32, tensor_count: u64, kv_count: u64) -> Self {
            let mut b = Self::default();
            b.bytes.extend_from_slice(b"GGUF");
            b.bytes.extend_from_slice(&version.to_le_bytes());
            b.bytes.extend_from_slice(&tensor_count.to_le_bytes());
            b.bytes.extend_from_slice(&kv_count.to_le_bytes());
            b
        }

        fn u32(mut self, v: u32) -> Self {
            self.bytes.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn u64(mut self, v: u64) -> Self {
            self.bytes.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn raw(mut self, v: &[u8]) -> Self {
            self.bytes.extend_from_slice(v);
            self
        }

        /// A length-prefixed string whose declared length matches its bytes.
        fn string(self, s: &[u8]) -> Self {
            self.u64(s.len() as u64).raw(s)
        }

        /// A length-prefixed string that LIES about its length.
        fn string_declaring(self, declared: u64, actual: &[u8]) -> Self {
            self.u64(declared).raw(actual)
        }

        fn key(self, k: &str) -> Self {
            self.string(k.as_bytes())
        }

        fn parse(self) -> Result<GgufMetadata, ExecError> {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("fixture.gguf");
            fs::write(&path, &self.bytes).unwrap();
            parse_gguf_metadata(&path)
        }
    }

    /// One `general.architecture = "llama"` KV pair, appended to a builder.
    fn with_arch(b: GgufBuilder) -> GgufBuilder {
        b.key("general.architecture")
            .u32(GGUF_TYPE_STRING)
            .string(b"llama")
    }

    // ── version gate ───────────────────────────────────────────────────

    #[test]
    fn versions_one_through_three_are_accepted() {
        for version in 1..=3u32 {
            let meta = with_arch(GgufBuilder::header(version, 0, 1))
                .parse()
                .unwrap_or_else(|e| panic!("v{version} header rejected: {e}"));
            assert_eq!(
                meta.architecture.as_deref(),
                Some("llama"),
                "v{version} lost its architecture"
            );
        }
    }

    #[test]
    fn version_zero_and_versions_past_three_are_refused() {
        for version in [0u32, 4, 100, u32::MAX] {
            let err = GgufBuilder::header(version, 0, 0)
                .parse()
                .expect_err("unsupported version must be refused");
            assert!(
                err.to_string().contains("unsupported GGUF version"),
                "v{version} gave the wrong error: {err}"
            );
        }
    }

    #[test]
    fn the_wrong_magic_is_refused_before_anything_else_is_read() {
        // Same bytes as a valid v3 header apart from the first four.
        let err = GgufBuilder::default()
            .raw(b"GGUE")
            .u32(3)
            .u64(0)
            .u64(0)
            .parse()
            .expect_err("non-GGUF magic must be refused");
        assert!(err.to_string().contains("not in GGUF format"), "{err}");
    }

    #[test]
    fn an_empty_file_is_refused_not_read_past_its_end() {
        assert!(GgufBuilder::default().parse().is_err());
    }

    #[test]
    fn a_header_truncated_at_every_prefix_length_is_refused() {
        // The full 24-byte header of a valid, empty v3 file. Every strict
        // prefix of it must be an error, never a partial-read success.
        let full = with_arch(GgufBuilder::header(3, 0, 1)).bytes;
        for cut in 0..full.len() {
            let result = GgufBuilder::default().raw(&full[..cut]).parse();
            assert!(
                result.is_err(),
                "prefix of length {cut} parsed as valid: {result:?}"
            );
        }
        // ...and the untruncated fixture is the one that parses.
        assert!(GgufBuilder::default().raw(&full).parse().is_ok());
    }

    // ── allocation bombs ───────────────────────────────────────────────

    #[test]
    fn an_enormous_declared_string_length_is_refused_not_allocated() {
        // A 28-byte file that claims a 140-terabyte key. Before the length
        // bound this reached `vec![0u8; len]` and aborted the process with
        // SIGABRT ("memory allocation of 140737488355327 bytes failed") —
        // an abort no caller can catch.
        let err = GgufBuilder::header(3, 0, 1)
            .u64(0x0000_7FFF_FFFF_FFFF)
            .parse()
            .expect_err("an absurd string length must be refused");
        assert!(err.to_string().contains("over the"), "{err}");
    }

    #[test]
    fn a_string_length_of_u64_max_is_refused_not_allocated() {
        let err = GgufBuilder::header(3, 0, 1)
            .u64(u64::MAX)
            .parse()
            .expect_err("u64::MAX string length must be refused");
        assert!(err.to_string().contains("over the"), "{err}");
    }

    #[test]
    fn a_value_string_just_over_the_limit_is_refused_and_just_under_is_read() {
        // Pins the `<` vs `<=` edge of the string bound without writing
        // 64 MiB to disk: only the over-limit side is exercised as a
        // declaration, the under-limit side as a real short string.
        let over = GgufBuilder::header(3, 0, 1)
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .u64(MAX_STRING_BYTES + 1)
            .parse();
        assert!(over.is_err(), "MAX+1 must be refused");

        let under = GgufBuilder::header(3, 0, 1)
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(b"ok")
            .parse()
            .unwrap();
        assert_eq!(under.name.as_deref(), Some("ok"));
    }

    #[test]
    fn an_enormous_declared_array_length_is_refused_not_allocated() {
        // `[u32; u64::MAX]` — the byte count overflows u64 and must be
        // caught, and even a non-overflowing count must not be trusted
        // past the end of the file.
        let overflowing = GgufBuilder::header(3, 0, 1)
            .key("unknown.array")
            .u32(GGUF_TYPE_ARRAY)
            .u32(GGUF_TYPE_UINT32)
            .u64(u64::MAX)
            .parse();
        assert!(
            overflowing.is_err(),
            "overflowing array size must be refused"
        );

        let past_eof = GgufBuilder::header(3, 0, 1)
            .key("unknown.array")
            .u32(GGUF_TYPE_ARRAY)
            .u32(GGUF_TYPE_UINT8)
            .u64(1_000_000_000)
            .raw(b"three")
            .parse();
        assert!(past_eof.is_err(), "array running past EOF must be refused");
    }

    #[test]
    fn an_enormous_kv_count_terminates_at_end_of_file() {
        // kv_count is a u64 the file chooses; the loop must be bounded by
        // the bytes actually present, not by the declared count.
        let err = GgufBuilder::header(3, 0, u64::MAX)
            .parse()
            .expect_err("a kv_count with no KV pairs behind it must error");
        assert!(!err.to_string().is_empty());
    }

    // ── nesting ────────────────────────────────────────────────────────

    #[test]
    fn a_modestly_nested_array_is_skipped_and_the_next_key_still_parses() {
        // `[[[u8; 0]]]` followed by a real key: nesting inside the bound
        // is skipped exactly, leaving the reader on the next KV pair.
        let meta = GgufBuilder::header(3, 0, 2)
            .key("unknown.nested")
            .u32(GGUF_TYPE_ARRAY)
            .u32(GGUF_TYPE_ARRAY)
            .u64(1)
            .u32(GGUF_TYPE_ARRAY)
            .u64(1)
            .u32(GGUF_TYPE_UINT8)
            .u64(0)
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(b"after-nesting")
            .parse()
            .unwrap();
        assert_eq!(meta.name.as_deref(), Some("after-nesting"));
    }

    #[test]
    fn array_nesting_past_the_depth_limit_is_refused_not_recursed() {
        // 12 bytes of input buy one stack frame, so an unbounded
        // `skip_array` overflowed the stack (SIGABRT: "has overflowed its
        // stack") on a ~2 MB file. Build a fixture past the bound and
        // assert it is refused with an error instead.
        let mut b = GgufBuilder::header(3, 0, 1)
            .key("unknown.deep")
            .u32(GGUF_TYPE_ARRAY);
        for _ in 0..(MAX_ARRAY_DEPTH as usize + 8) {
            b = b.u32(GGUF_TYPE_ARRAY).u64(1);
        }
        b = b.u32(GGUF_TYPE_UINT8).u64(0);
        let err = b.parse().expect_err("over-deep nesting must be refused");
        assert!(err.to_string().contains("nesting deeper than"), "{err}");
    }

    #[test]
    fn a_deeply_nested_array_fixture_from_disk_is_refused() {
        // The stack-overflow repro at the scale that actually crashed:
        // 200_000 nesting levels, ~2.4 MB. Kept as a checked-in fixture so
        // the crash is reproducible without regenerating it.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gguf/deeply_nested_arrays.gguf");
        let err = parse_gguf_metadata(&path).expect_err("deep nesting must be refused");
        assert!(err.to_string().contains("nesting deeper than"), "{err}");
    }

    // ── value types ────────────────────────────────────────────────────

    #[test]
    fn an_unknown_primitive_value_type_is_refused() {
        // 13 and up are unassigned; the parser cannot know a value's width
        // so it must stop rather than resynchronise on garbage.
        for ty in [13u32, 255, u32::MAX] {
            let err = GgufBuilder::header(3, 0, 1)
                .key("unknown.key")
                .u32(ty)
                .u64(0)
                .parse()
                .expect_err("unknown value type must be refused");
            assert!(
                err.to_string().contains("unsupported GGUF value type"),
                "type {ty} gave: {err}"
            );
        }
    }

    #[test]
    fn an_unknown_array_element_type_is_refused() {
        let err = GgufBuilder::header(3, 0, 1)
            .key("unknown.key")
            .u32(GGUF_TYPE_ARRAY)
            .u32(99)
            .u64(4)
            .parse()
            .expect_err("unknown array element type must be refused");
        assert!(
            err.to_string()
                .contains("unsupported GGUF array element type"),
            "{err}"
        );
    }

    #[test]
    fn every_declared_primitive_type_is_skippable_as_an_unknown_key() {
        // Any type the format defines must be skippable by width alone,
        // so an unknown key never desynchronises the KV walk.
        for ty in [
            GGUF_TYPE_UINT8,
            GGUF_TYPE_INT8,
            GGUF_TYPE_UINT16,
            GGUF_TYPE_INT16,
            GGUF_TYPE_UINT32,
            GGUF_TYPE_INT32,
            GGUF_TYPE_FLOAT32,
            GGUF_TYPE_BOOL,
            GGUF_TYPE_UINT64,
            GGUF_TYPE_INT64,
            GGUF_TYPE_FLOAT64,
        ] {
            let width = primitive_byte_size(ty).unwrap() as usize;
            let meta = GgufBuilder::header(3, 0, 2)
                .key("some.unknown.key")
                .u32(ty)
                .raw(&vec![0u8; width])
                .key("general.name")
                .u32(GGUF_TYPE_STRING)
                .string(b"survivor")
                .parse()
                .unwrap_or_else(|e| panic!("type {ty} broke the KV walk: {e}"));
            assert_eq!(meta.name.as_deref(), Some("survivor"), "type {ty}");
        }
    }

    #[test]
    fn negative_signed_metadata_is_dropped_rather_than_wrapped_to_a_huge_u64() {
        // `-1` as a u64 is 18446744073709551615, which would flow into RAM
        // and parameter estimates. Every signed width must decline instead.
        for (ty, bytes) in [
            (GGUF_TYPE_INT8, (-1i8).to_le_bytes().to_vec()),
            (GGUF_TYPE_INT16, (-1i16).to_le_bytes().to_vec()),
            (GGUF_TYPE_INT32, (-1i32).to_le_bytes().to_vec()),
            (GGUF_TYPE_INT64, (-1i64).to_le_bytes().to_vec()),
        ] {
            let meta = GgufBuilder::header(3, 0, 1)
                .key("llama.block_count")
                .u32(ty)
                .raw(&bytes)
                .parse()
                .unwrap();
            assert_eq!(meta.block_count, None, "negative type {ty} leaked through");
        }
    }

    #[test]
    fn a_positive_signed_value_is_still_read() {
        // The guard above must reject only the negative half.
        let meta = GgufBuilder::header(3, 0, 1)
            .key("llama.block_count")
            .u32(GGUF_TYPE_INT32)
            .raw(&32i32.to_le_bytes())
            .parse()
            .unwrap();
        assert_eq!(meta.block_count, Some(32));
    }

    #[test]
    fn a_typed_field_carrying_the_wrong_type_is_skipped_not_misread() {
        // `general.name` declared as a u32: the value is skipped by width
        // and `name` stays unset, rather than four bytes being read as text.
        let meta = GgufBuilder::header(3, 0, 2)
            .key("general.name")
            .u32(GGUF_TYPE_UINT32)
            .u32(7)
            .key("general.architecture")
            .u32(GGUF_TYPE_STRING)
            .string(b"llama")
            .parse()
            .unwrap();
        assert_eq!(meta.name, None);
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
    }

    // ── truncation ─────────────────────────────────────────────────────

    #[test]
    fn a_string_declaring_more_bytes_than_it_carries_is_refused() {
        let err = GgufBuilder::header(3, 0, 1)
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string_declaring(1000, b"only-a-few")
            .parse()
            .expect_err("a truncated string must be refused");
        assert!(err.to_string().contains("truncated"), "{err}");
    }

    #[test]
    fn a_truncated_trailing_value_does_not_parse_as_a_complete_file() {
        // The last KV pair is the dangerous one: a short skip that reports
        // success ends the loop and returns `Ok` on a truncated file.
        // Pinned for a skipped string, a skipped primitive, and an array.
        let short_string = GgufBuilder::header(3, 0, 1)
            .key("unknown.key")
            .u32(GGUF_TYPE_STRING)
            .string_declaring(1000, b"only-a-few")
            .parse();
        assert!(short_string.is_err(), "{short_string:?}");

        let short_primitive = GgufBuilder::header(3, 0, 1)
            .key("unknown.key")
            .u32(GGUF_TYPE_UINT64)
            .raw(b"abc")
            .parse();
        assert!(short_primitive.is_err(), "{short_primitive:?}");

        let short_array = GgufBuilder::header(3, 0, 1)
            .key("unknown.key")
            .u32(GGUF_TYPE_ARRAY)
            .u32(GGUF_TYPE_UINT32)
            .u64(100)
            .raw(b"abcd")
            .parse();
        assert!(short_array.is_err(), "{short_array:?}");
    }

    #[test]
    fn a_v1_file_with_the_historical_32_bit_counts_is_refused_not_misread() {
        // GGUF v1 wrote `tensor_count`/`kv_count` as u32; this parser reads
        // u64 for every version. Such a file is not silently reinterpreted
        // into unbounded work — the misaligned read lands on an absurd
        // string length and is refused by the length bound.
        let result = GgufBuilder::default()
            .raw(b"GGUF")
            .u32(1)
            .u32(0) // 32-bit tensor_count
            .u32(1) // 32-bit kv_count
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(b"legacy")
            .parse();
        assert!(result.is_err(), "expected refusal, got {result:?}");
    }

    // ── key handling ───────────────────────────────────────────────────

    #[test]
    fn a_duplicate_metadata_key_takes_the_last_value_seen() {
        let meta = GgufBuilder::header(3, 0, 3)
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(b"first")
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(b"second")
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(b"third")
            .parse()
            .unwrap();
        assert_eq!(meta.name.as_deref(), Some("third"));
    }

    #[test]
    fn unknown_keys_are_skipped_without_disturbing_known_ones() {
        let meta = GgufBuilder::header(3, 0, 4)
            .key("some.vendor.extension")
            .u32(GGUF_TYPE_FLOAT64)
            .raw(&1.5f64.to_le_bytes())
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(b"Real Model")
            .key("another.unknown")
            .u32(GGUF_TYPE_ARRAY)
            .u32(GGUF_TYPE_STRING)
            .u64(2)
            .string(b"a")
            .string(b"b")
            .key("llama.context_length")
            .u32(GGUF_TYPE_UINT32)
            .u32(8192)
            .parse()
            .unwrap();
        assert_eq!(meta.name.as_deref(), Some("Real Model"));
        assert_eq!(meta.context_length, Some(8192));
    }

    #[test]
    fn an_empty_key_is_treated_as_an_unknown_key_not_an_error() {
        let meta = GgufBuilder::header(3, 0, 2)
            .key("")
            .u32(GGUF_TYPE_UINT32)
            .u32(1)
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(b"named")
            .parse()
            .unwrap();
        assert_eq!(meta.name.as_deref(), Some("named"));
    }

    #[test]
    fn head_count_kv_is_matched_before_the_shorter_head_count_suffix() {
        // `.attention.head_count_kv` also ends with nothing that
        // `.attention.head_count` matches, but the ordering is load-bearing:
        // a reordered match arm would put the KV head count in `head_count`.
        let meta = GgufBuilder::header(3, 0, 2)
            .key("llama.attention.head_count_kv")
            .u32(GGUF_TYPE_UINT32)
            .u32(8)
            .key("llama.attention.head_count")
            .u32(GGUF_TYPE_UINT32)
            .u32(32)
            .parse()
            .unwrap();
        assert_eq!(meta.head_count_kv, Some(8));
        assert_eq!(meta.head_count, Some(32));
    }

    #[test]
    fn invalid_utf8_in_a_string_value_is_replaced_not_rejected() {
        // Model names come from the file; lossy decoding keeps a otherwise
        // usable header readable instead of failing the whole parse.
        let meta = GgufBuilder::header(3, 0, 1)
            .key("general.name")
            .u32(GGUF_TYPE_STRING)
            .string(&[0x41, 0xFF, 0xFE, 0x42])
            .parse()
            .unwrap();
        let name = meta.name.unwrap();
        assert!(name.starts_with('A') && name.ends_with('B'), "{name:?}");
    }

    // ── downstream estimators on hostile dimensions ────────────────────

    #[test]
    fn each_factor_of_the_parameter_estimate_is_individually_guarded() {
        // One case per multiplication in the estimate. Each is sized so
        // that its own factor overflows while every other one stays small
        // AND the wrapped product lands *benign* (zero), so the final
        // `checked_add` cannot mask it. Without that sizing a wrapped
        // intermediate is merely huge, the sum overflows, and the case
        // passes on the strength of a different guard than the one it
        // claims to test.
        let cases: [(&str, GgufMetadata); 5] = [
            (
                "attention: n_layer * 4 * d * d",
                GgufMetadata {
                    block_count: Some(2),
                    embedding_length: Some(1 << 32),
                    feed_forward_length: Some(1),
                    vocab_size: Some(1),
                    ..Default::default()
                },
            ),
            (
                "ffn: n_layer * 3 * d * d_ff",
                GgufMetadata {
                    block_count: Some(1),
                    embedding_length: Some(2),
                    feed_forward_length: Some(1 << 63),
                    vocab_size: Some(1),
                    ..Default::default()
                },
            ),
            (
                "moe: ffn * n_experts",
                GgufMetadata {
                    block_count: Some(1),
                    embedding_length: Some(2),
                    feed_forward_length: Some(1 << 61),
                    vocab_size: Some(1),
                    expert_count: Some(4),
                    ..Default::default()
                },
            ),
            (
                "embedding: vocab * d",
                GgufMetadata {
                    block_count: Some(1),
                    embedding_length: Some(1 << 30),
                    feed_forward_length: Some(1),
                    vocab_size: Some(1 << 34),
                    ..Default::default()
                },
            ),
            (
                "the sum of the three terms",
                GgufMetadata {
                    block_count: Some(2),
                    embedding_length: Some(1 << 30),
                    feed_forward_length: Some(1),
                    vocab_size: Some(1 << 33),
                    ..Default::default()
                },
            ),
        ];
        for (which, meta) in cases {
            assert_eq!(
                estimate_parameter_count(&meta, None),
                None,
                "{which} did not decline on overflow"
            );
        }
    }

    #[test]
    fn a_plausible_header_still_gets_an_architecture_estimate() {
        // The overflow guard must decline only on overflow. Without this,
        // a guard that always declines would pass every test above.
        let meta = GgufMetadata {
            block_count: Some(32),
            embedding_length: Some(4096),
            feed_forward_length: Some(11008),
            vocab_size: Some(32000),
            ..Default::default()
        };
        let count = estimate_parameter_count(&meta, None).expect("a real header must estimate");
        assert!((5..10).contains(&(count / 1_000_000_000)), "got {count}");
    }

    #[test]
    fn parameter_estimation_on_absurd_dimensions_does_not_overflow() {
        // Before the checked arithmetic this panicked with "attempt to
        // multiply with overflow" in a debug build and wrapped silently in
        // a release one, straight off `ModelInfo::read` of a hostile file.
        let meta = GgufMetadata {
            embedding_length: Some(u64::MAX / 2),
            block_count: Some(u64::MAX / 2),
            feed_forward_length: Some(u64::MAX),
            vocab_size: Some(u64::MAX),
            expert_count: Some(u64::MAX),
            ..Default::default()
        };
        // No architecture estimate is possible, and with no file size there
        // is no fallback either — so the honest answer is "unknown".
        assert_eq!(estimate_parameter_count(&meta, None), None);
        // With a file size, the fallback still answers.
        let with_size = GgufMetadata {
            file_type: Some(15),
            ..meta
        };
        assert!(estimate_parameter_count(&with_size, Some(4 << 30)).is_some());
    }

    #[test]
    fn kv_cost_saturates_instead_of_overflowing_on_absurd_dimensions() {
        assert_eq!(kv_bytes_per_token(u64::MAX, u64::MAX, u64::MAX), u64::MAX);
        // ...and a saturated cost makes context sizing fall to its floor
        // rather than producing a nonsense context.
        assert_eq!(fit_context(64 * GIB, 0, u64::MAX, 131_072, None), 2048);
    }

    #[test]
    fn ram_estimation_on_absurd_dimensions_saturates_instead_of_overflowing() {
        // Same path as above, reached through `ModelInfo::memory_needed`.
        let metadata = ModelMetadata {
            block_count: Some(u64::MAX),
            head_count: Some(1),
            head_count_kv: Some(u64::MAX),
            embedding_length: Some(u64::MAX),
            ..Default::default()
        };
        assert_eq!(
            estimate_ram_bytes(&metadata, u64::MAX, u32::MAX),
            u64::MAX,
            "a saturated estimate must exceed every budget, not wrap under one"
        );
        // The fallback arm too.
        assert_eq!(
            estimate_ram_bytes(&ModelMetadata::default(), u64::MAX, 4096),
            u64::MAX
        );
    }

    #[test]
    fn a_zero_head_count_does_not_divide_by_zero() {
        let metadata = ModelMetadata {
            block_count: Some(32),
            head_count: Some(0),
            head_count_kv: Some(8),
            embedding_length: Some(4096),
            ..Default::default()
        };
        let ram = estimate_ram_bytes(&metadata, 1 << 30, 4096);
        assert!(ram > 1 << 30);
    }

    #[test]
    fn a_hostile_header_survives_the_whole_read_estimate_pipeline() {
        // End-to-end over the path `gen2::ModelInfo::read` walks: parse the
        // header, build metadata, then estimate params and RAM from it.
        // Absurd-but-well-formed dimensions must not panic anywhere.
        let meta = GgufBuilder::header(3, 0, 5)
            .key("general.architecture")
            .u32(GGUF_TYPE_STRING)
            .string(b"llama")
            .key("llama.block_count")
            .u32(GGUF_TYPE_UINT64)
            .u64(u64::MAX)
            .key("llama.embedding_length")
            .u32(GGUF_TYPE_UINT64)
            .u64(u64::MAX)
            .key("llama.attention.head_count")
            .u32(GGUF_TYPE_UINT64)
            .u64(1)
            .key("llama.attention.head_count_kv")
            .u32(GGUF_TYPE_UINT64)
            .u64(u64::MAX)
            .parse()
            .unwrap();
        let model = build_model_metadata(&meta, Some(u64::MAX)).unwrap();
        let _ = estimate_parameter_count(&meta, Some(u64::MAX));
        let _ = estimate_ram_bytes(&model, u64::MAX, u32::MAX);
        let _ = fit_context(
            u64::MAX,
            u64::MAX,
            kv_bytes_per_token(u64::MAX, u64::MAX, u64::MAX),
            u32::MAX,
            Some(u32::MAX),
        );
    }

    // ── the fuzz invariant, as a deterministic test ────────────────────

    #[test]
    fn arbitrary_bytes_behind_a_valid_magic_never_panic() {
        // The same invariant `fuzz_targets/gguf.rs` asserts, run over a
        // deterministic corpus so it holds in ordinary CI too: any result
        // is acceptable, a panic or abort is not.
        let mut rng_state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            rng_state
        };
        for case in 0..2000 {
            let len = (next() % 96) as usize;
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(&((next() % 5) as u32).to_le_bytes());
            for _ in 0..len {
                bytes.push((next() & 0xFF) as u8);
            }
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("fuzz.gguf");
            fs::write(&path, &bytes).unwrap();
            let result = parse_gguf_metadata(&path);
            if let Ok(meta) = result {
                // Whatever comes back must survive the estimators too.
                let _ = estimate_parameter_count(&meta, Some(bytes.len() as u64));
                if let Some(model) = build_model_metadata(&meta, Some(bytes.len() as u64)) {
                    let _ = estimate_ram_bytes(&model, bytes.len() as u64, 4096);
                }
            }
            let _ = case;
        }
    }

    #[test]
    fn every_checked_in_fuzz_seed_parses_or_is_refused_without_crashing() {
        // Guards the seed corpus against rot: a seed that stops being a
        // GGUF-shaped input stops steering the fuzzer, silently.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf/corpus");
        let mut seen = 0;
        for entry in fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
                continue;
            }
            seen += 1;
            let bytes = fs::read(&path).unwrap();
            assert_eq!(&bytes[..4], b"GGUF", "{path:?} lost its magic");
            let _ = parse_gguf_metadata(&path);
        }
        assert!(seen >= 9, "expected the full seed corpus, found {seen}");
    }
}
