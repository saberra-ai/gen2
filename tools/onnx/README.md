# Woof ONNX text generation

Qualified on 2026-10-12 with Apple M4 Pro, macOS 26.3, native GenAI 0.15.2
and ONNX Runtime 1.31.0, CPU execution. Inference is native Rust/C++; Python
is needed only for checkpoint conversion and the independent reference check.

## Export the exact checkpoint

Use Python 3.12 on Apple silicon. Install `requirements.txt` in an isolated
environment. Download these files from the model's `mlx/` directory, preferably
at the qualified revision `4f21a47f1a4c0d9393a4c3bf6593d9987f27e763`:

- `config.json`, `model.safetensors`, `manifest.json`
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
- `generation_config.json`, `LICENSE`

[Pinned model repository](https://huggingface.co/ConwayResearch/woof-textfiller-0.8B-4bit-v2.1/tree/4f21a47f1a4c0d9393a4c3bf6593d9987f27e763/mlx).
The qualified source weights' SHA-256 is
`46034af853890ab38ee0db869f32d4ba1a9b6aa6e529a89c0fdb8eb5064eb409`.
The exporter checks all downloaded inputs against the supplied manifest,
every reconstructed text tensor's name/shape against Transformers, and source
and checkpoint digests when reusing a previous reconstruction. The manifest
checks integrity, not authenticity; pin/trust the download source separately.

```sh
python -m pip install -r tools/onnx/requirements.txt
python tools/onnx/export_woof.py \
  --mlx /models/woof-mlx \
  --checkpoint /models/woof-hf \
  --output /models/woof-onnx
```

Choose fresh checkpoint/output directories. A reconstruction can be reused
only with its original `gen2_reconstruction.json` receipt and matching weights.
Output refuses an existing directory. INT4 is the default: 64-element symmetric
blocks, FP32 activations, and shared quantized embedding/LM-head weights. It
dequantizes the already-quantized MLX affine checkpoint before re-quantization;
it cannot recover pre-quantization weights. `--precision fp32` avoids the
additional quantization and is useful as an export oracle, but is much larger.
The exported graph data is about 407 MiB for INT4, plus tokenizer/config files.

The convolution axis swap and offset-norm subtraction invert mlx-lm's Qwen3.5
sanitization. Text-only export includes embedding lookup and uses ordinary 2D
position IDs; it does not require a vision pipeline. Licenses and provenance
are included. There is no model download or converter subprocess in Rust.

## Native inference

Install the host's [GenAI 0.15.2 release](https://github.com/microsoft/onnxruntime-genai/releases/tag/v0.15.2)
and ONNX Runtime 1.31.0. Set both paths before loading a model. `ORT_LIB_PATH`
is the full shared-library **file**, not its directory. A Python runtime wheel
contains it under `site-packages/onnxruntime/capi/` if using the export environment.
Do not change either library path while an ONNX engine/session is alive.

```sh
export GEN2_ONNX_LIBRARY=/runtime/libonnxruntime-genai.dylib
export ORT_LIB_PATH=/runtime/libonnxruntime.1.31.0.dylib
cargo run --features backend-onnx --example text_filler -- /models/woof-onnx
```

Other platforms use their corresponding `.so`/`.dll`. The Rust feature compiles
without native binaries; only macOS arm64 CPU inference has been qualified.
Execution providers come from the GenAI bundle. Gen2 reports no GPU placement
unless a backend can establish it; these CPU results are not a Metal claim.
Missing/bad ORT paths fail before GenAI's native initializer can abort.
Handles keep both native libraries alive and shutdown waits for all engines
and sessions created through this backend.

## Qualification and limitations

The native tests exercise actual tokens, greedy repeatability, conversation
history, output limits, JSON schema, regex/Lark constraints, reload/unload
lifetimes, missing-runtime recovery, and email/Unicode field values:

```sh
export PIO_TEST_ONNX_MODEL=/models/woof-onnx
export PIO_DISABLE_META_PROMPT=1
cargo test --no-default-features --features backend-onnx \
  --test onnx_generation -- --ignored --nocapture
cargo test --no-default-features --features backend-onnx \
  --lib backend::conformance::onnx -- --nocapture
python tools/onnx/verify_woof.py --mlx /models/woof-mlx \
  --checkpoint /models/woof-hf --onnx /models/woof-fp32
```

The independent reference check found identical unconstrained greedy outputs
for MLX, reconstructed Transformers and FP32 ONNX on two prompts. The first
generated token's top-five logits agreed for MLX/Transformers, with mean
absolute differences about 0.060/0.062 (arithmetic/rounding is not identical).
No claim is made of bitwise-identical logits or full-corpus equivalence.

The compact export decoded about 49–67 tokens/s with 166–224 ms TTFT in two
`hello` smoke runs (different sampled completions and concurrent load).
The focused field example returns the correct email
and `Montréal` in schema-constrained JSON. It returns **Alice Chen**, not Alice,
for the first-name request; the same error occurs in the original MLX reference.
The original GGUF passed all three field checks on Gen2's current Metal backend
(25/25 layers offloaded), so quantized variants are not interchangeable quality
oracles. This is a backend qualification, not a field-filling accuracy eval.

Request ordering also matters for this small checkpoint. The example uses
typed JSON serialization with page context before the selected field; this
keeps the request stable across Cargo feature combinations. A differently
ordered request produced a wrong email value. Do not mistake valid JSON for
a guarantee that the chosen field value is correct.

Each turn rebuilds the native generator from the current transcript. KV
snapshot import and automatic context truncation are unsupported. Context is
bounded by the bundle (32768 here); prompt plus output budget must fit. Some
sampler controls are explicitly unsupported; see the root README. An arbitrary
ONNX graph, audio model, encoder or classifier is not a GenAI text bundle.
