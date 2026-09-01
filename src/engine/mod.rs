//! Engine: long-lived orchestrator.

mod error;
pub(crate) mod telemetry;
mod types;

// Backend-specific Engine is re-exported from gen2::backend
pub use crate::backend::Engine;
pub use error::ExecError;
// ExecutionStats is a leaf domain type (persisted on `types::Message`), so it
// lives in `types/` — the shared type module must never import from gen2.
// Re-exported here so `gen2::engine::ExecutionStats` callers are unchanged.
pub use crate::types::ExecutionStats;
pub use telemetry::{HookBus, HookEvent, HookListener};
pub use types::{
    Capabilities, ChatTemplateSpec, CtxParamsInput, EmbedLoadRequest, LoadRequest, MmSettings,
    ModelParamsInput, PromptSettings, SamplingSettings, Settings, StoppingSettings, SystemSettings,
};

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// GGUF magic bytes: `GGUF` (0x47 0x47 0x55 0x46).
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Architectures verified to work on Metal (Apple Silicon) with the current llama.cpp.
const VERIFIED_ARCHITECTURES: &[&str] = &[
    "llama",
    "qwen2",
    "qwen2moe",
    "phi2",
    "phi3",
    "starcoder2",
    "falcon",
    "gptneox",
    "mpt",
    "baichuan",
    "stablelm",
    "internlm2",
    "command-r",
    "deepseek2",
    "mistral",
    "minicpm",
    "minicpm3",
    "gemma",
    "gemma2",
    "gpt2",
    "gptj",
    "refact",
    "bloom",
    "plamo",
    "codeshell",
    "orion",
    "mamba",
    "xverse",
    "dbrx",
    "olmo",
    "olmo2",
    "openelm",
    "arctic",
    "chatglm",
    "exaone",
    "granite",
    "granitemoe",
    "rwkv6",
    "rwkv6qwen2",
    "qwen2vl",
    "cohere2",
    "olmoe",
    "qwen3",
    "qwen3moe",
];

/// Architectures with known Metal kernel crashes. These will be blocked.
const KNOWN_METAL_CRASHES: &[&str] = &[
    // NB: "gemma3" was removed (2026-06) after the llama-cpp-rs bump to 8625c7c —
    // it now loads + generates on Metal (re-validated with gemma-3-1b-it). The
    // 2026-03 abort was a stale-fork issue, not an arch limit.
    "qwen35", // Gated Delta Net tensor op — Metal abort, llama.cpp #20358; not re-tested (no local model)
];

/// Validate that `path` points to a non-empty file with a valid GGUF header.
///
/// Call this **before** passing a path to `LlamaModel::load_from_file()` to
/// avoid hangs in the C FFI layer when the file is empty or corrupt.
pub fn validate_model_file(path: &Path) -> Result<(), ExecError> {
    let md = path.metadata().map_err(|e| {
        ExecError::InvalidModelFile(format!("cannot read model file '{}': {e}", path.display()))
    })?;

    // Directory bundles (MLX safetensors, ONNX model.onnx layouts) are
    // validated by their backend loaders — they're not GGUF files and
    // don't have the 4-byte magic we check below. Delegating validation
    // to the backend here prevents the hard-coded GGUF check from
    // rejecting a perfectly good MLX model dir.
    if md.is_dir() {
        return Ok(());
    }

    if md.len() == 0 {
        return Err(ExecError::InvalidModelFile(format!(
            "model file is empty (0 bytes): {}",
            path.display()
        )));
    }

    // Read first 4 bytes and check GGUF magic.
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map_err(|e| {
            ExecError::InvalidModelFile(format!(
                "cannot read model header '{}': {e}",
                path.display()
            ))
        })?;

    if magic != GGUF_MAGIC {
        return Err(ExecError::InvalidModelFile(format!(
            "not a valid GGUF file (bad magic bytes): {}",
            path.display()
        )));
    }

    Ok(())
}

/// Read `general.architecture` from a GGUF file header without loading the model.
///
/// Parses just enough of the GGUF binary format to find the architecture key.
/// Returns `None` if the key isn't found or the file can't be parsed.
pub fn read_gguf_architecture(path: &Path) -> Option<String> {
    let mut f = File::open(path).ok()?;

    // Header: magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24 bytes
    let mut header = [0u8; 24];
    f.read_exact(&mut header).ok()?;

    // Verify magic
    if header[0..4] != GGUF_MAGIC {
        return None;
    }

    let version = u32::from_le_bytes(header[4..8].try_into().ok()?);
    if !(2..=3).contains(&version) {
        return None;
    }

    let kv_count = u64::from_le_bytes(header[16..24].try_into().ok()?) as usize;

    // Walk KV pairs looking for "general.architecture"
    for _ in 0..kv_count {
        // Key: length(8) + string bytes
        let key = read_gguf_string(&mut f)?;
        // Value type: u32
        let mut vtype_buf = [0u8; 4];
        f.read_exact(&mut vtype_buf).ok()?;
        let vtype = u32::from_le_bytes(vtype_buf);

        if key == "general.architecture" && vtype == 8 {
            // Type 8 = GGUF_TYPE_STRING
            return read_gguf_string(&mut f);
        } else {
            // Skip this value
            skip_gguf_value(&mut f, vtype)?;
        }
    }

    None
}

/// Read `general.file_type` (the quant enum) from a GGUF header without
/// loading the model.
///
/// `general.file_type` is a `u32` in GGUF metadata whose value is the
/// `llama_ftype` enum (e.g. 15 = `MOSTLY_Q4_K_M`). Used by the iOS
/// memory-budget preflight to enforce the on-device quant ceiling. Returns
/// `None` if the key isn't present or the file can't be parsed.
pub fn read_gguf_file_type(path: &Path) -> Option<u32> {
    let mut f = File::open(path).ok()?;

    // Header: magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24 bytes
    let mut header = [0u8; 24];
    f.read_exact(&mut header).ok()?;

    if header[0..4] != GGUF_MAGIC {
        return None;
    }

    let version = u32::from_le_bytes(header[4..8].try_into().ok()?);
    if !(2..=3).contains(&version) {
        return None;
    }

    let kv_count = u64::from_le_bytes(header[16..24].try_into().ok()?) as usize;

    for _ in 0..kv_count {
        let key = read_gguf_string(&mut f)?;
        let mut vtype_buf = [0u8; 4];
        f.read_exact(&mut vtype_buf).ok()?;
        let vtype = u32::from_le_bytes(vtype_buf);

        // Type 4 = GGUF_TYPE_UINT32.
        if key == "general.file_type" && vtype == 4 {
            let mut val = [0u8; 4];
            f.read_exact(&mut val).ok()?;
            return Some(u32::from_le_bytes(val));
        } else {
            skip_gguf_value(&mut f, vtype)?;
        }
    }

    None
}

/// Read a GGUF string: u64 length + UTF-8 bytes.
fn read_gguf_string(f: &mut File) -> Option<String> {
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).ok()?;
    let len = u64::from_le_bytes(len_buf) as usize;
    if len > 1024 * 1024 {
        return None; // sanity limit
    }
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

/// Skip a GGUF value based on its type tag.
fn skip_gguf_value(f: &mut File, vtype: u32) -> Option<()> {
    match vtype {
        0 => {
            f.seek(SeekFrom::Current(1)).ok()?;
        } // u8
        1 => {
            f.seek(SeekFrom::Current(1)).ok()?;
        } // i8
        2 => {
            f.seek(SeekFrom::Current(2)).ok()?;
        } // u16
        3 => {
            f.seek(SeekFrom::Current(2)).ok()?;
        } // i16
        4 => {
            f.seek(SeekFrom::Current(4)).ok()?;
        } // u32
        5 => {
            f.seek(SeekFrom::Current(4)).ok()?;
        } // i32
        6 => {
            f.seek(SeekFrom::Current(4)).ok()?;
        } // f32
        7 => {
            f.seek(SeekFrom::Current(1)).ok()?;
        } // bool
        8 => {
            read_gguf_string(f)?;
        } // string
        9 => {
            // array
            let mut atype_buf = [0u8; 4];
            f.read_exact(&mut atype_buf).ok()?;
            let atype = u32::from_le_bytes(atype_buf);
            let mut alen_buf = [0u8; 8];
            f.read_exact(&mut alen_buf).ok()?;
            let alen = u64::from_le_bytes(alen_buf) as usize;
            for _ in 0..alen {
                skip_gguf_value(f, atype)?;
            }
        }
        10 => {
            f.seek(SeekFrom::Current(8)).ok()?;
        } // u64
        11 => {
            f.seek(SeekFrom::Current(8)).ok()?;
        } // i64
        12 => {
            f.seek(SeekFrom::Current(8)).ok()?;
        } // f64
        _ => return None, // unknown type
    }
    Some(())
}

/// Validate a model architecture against known compatibility data.
///
/// - Verified architectures pass silently.
/// - Known-crash architectures (e.g. gemma3 on Metal) return an error.
/// - Unknown architectures log a warning but pass.
pub fn validate_model_architecture(arch: &str) -> Result<(), ExecError> {
    let normalized = arch.trim().to_lowercase();

    if KNOWN_METAL_CRASHES.iter().any(|&a| a == normalized) {
        return Err(ExecError::UnsupportedArchitecture(format!(
            "Model architecture '{arch}' has known compatibility issues on Apple Silicon \
             that cause the app to crash. Try Qwen 2.5, Llama 3.1, or Phi 3.5 instead."
        )));
    }

    if !VERIFIED_ARCHITECTURES.iter().any(|&a| a == normalized) {
        tracing::warn!(
            "Model architecture '{arch}' has not been verified with this version of Pio. \
             It may work, but if it crashes, try a different model."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn rejects_nonexistent_file() {
        let r = validate_model_file(Path::new("/tmp/pio_test_nonexistent.gguf"));
        assert!(r.is_err());
        assert!(
            r.unwrap_err()
                .to_string()
                .contains("cannot read model file")
        );
    }

    #[test]
    fn rejects_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("empty.gguf");
        File::create(&p).unwrap();
        let r = validate_model_file(&p);
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("empty (0 bytes)"));
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.gguf");
        let mut f = File::create(&p).unwrap();
        f.write_all(b"NOT_GGUF_DATA_HERE").unwrap();
        let r = validate_model_file(&p);
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("bad magic bytes"));
    }

    #[test]
    fn accepts_valid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("good.gguf");
        let mut f = File::create(&p).unwrap();
        f.write_all(&[0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00])
            .unwrap();
        assert!(validate_model_file(&p).is_ok());
    }

    #[test]
    fn blocks_known_metal_crash_architecture() {
        // qwen35 stays blocked (Gated Delta Net Metal abort; not re-tested).
        let r = validate_model_architecture("qwen35");
        assert!(r.is_err(), "qwen35 must be blocked");
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("compatibility issues"));
        assert!(msg.contains("Qwen 2.5"));
        // gemma3 was un-blocked after the llama.cpp bump — it must now PASS.
        assert!(validate_model_architecture("gemma3").is_ok());
    }

    #[test]
    fn allows_verified_architecture() {
        assert!(validate_model_architecture("llama").is_ok());
        assert!(validate_model_architecture("qwen2").is_ok());
        assert!(validate_model_architecture("phi3").is_ok());
        assert!(validate_model_architecture("gemma2").is_ok());
    }

    #[test]
    fn allows_unknown_architecture_with_warning() {
        // Unknown architectures pass (with a log warning, not tested here)
        assert!(validate_model_architecture("some_future_arch").is_ok());
    }

    #[test]
    fn architecture_check_is_case_insensitive() {
        // qwen35 is still blocked (gemma3 was un-blocked after the llama.cpp bump).
        assert!(validate_model_architecture("Qwen35").is_err());
        assert!(validate_model_architecture("QWEN35").is_err());
        assert!(validate_model_architecture("Llama").is_ok());
        assert!(validate_model_architecture("QWEN2").is_ok());
    }

    /// Helper: build a minimal GGUF file with one KV pair: general.architecture = <arch>.
    fn build_gguf_with_arch(path: &std::path::Path, arch: &str) {
        let mut f = File::create(path).unwrap();
        // Magic
        f.write_all(&GGUF_MAGIC).unwrap();
        // Version 3
        f.write_all(&3u32.to_le_bytes()).unwrap();
        // Tensor count: 0
        f.write_all(&0u64.to_le_bytes()).unwrap();
        // KV count: 1
        f.write_all(&1u64.to_le_bytes()).unwrap();
        // Key: "general.architecture"
        let key = b"general.architecture";
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        f.write_all(key).unwrap();
        // Value type: 8 (string)
        f.write_all(&8u32.to_le_bytes()).unwrap();
        // Value: the architecture string
        f.write_all(&(arch.len() as u64).to_le_bytes()).unwrap();
        f.write_all(arch.as_bytes()).unwrap();
    }

    #[test]
    fn reads_architecture_from_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("test.gguf");
        build_gguf_with_arch(&p, "llama");
        assert_eq!(read_gguf_architecture(&p), Some("llama".to_string()));
    }

    #[test]
    fn reads_gemma3_architecture_from_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("gemma3.gguf");
        build_gguf_with_arch(&p, "gemma3");
        assert_eq!(read_gguf_architecture(&p), Some("gemma3".to_string()));
    }
}
