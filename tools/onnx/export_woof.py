#!/usr/bin/env python3
"""Export Woof's MLX affine checkpoint to a text-only ONNX GenAI bundle.

Requires Apple silicon, mlx, torch, transformers, safetensors and
onnxruntime-genai==0.15.2 (including its builder dependencies). No remote code
is executed. Supply the downloaded mlx/ directory from the original repository.
The existing 4-bit weights are dequantized, NOT recovered to pre-quantization
weights; --precision int4 applies an additional ONNX-compatible quantization.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_source(source: Path) -> None:
    manifest = json.loads((source / "manifest.json").read_text())["files"]
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "generation_config.json", "LICENSE"):
        expected = manifest[name]
        path = source / name
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Downloaded source does not match its manifest: {name}")


def prepare(source: Path, destination: Path) -> None:
    import mlx.core as mx
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    original = json.loads((source / "config.json").read_text())
    quant = original["quantization"]
    if original["model_type"] != "qwen3_5" or quant != {
        "group_size": 64, "bits": 4, "mode": "affine"
    }:
        raise ValueError("Expected Woof's Qwen3.5 4-bit affine checkpoint")
    config = Qwen3_5TextConfig.from_dict(original["text_config"])
    config.architectures = ["Qwen3_5ForCausalLM"]
    config.max_position_embeddings = 32768  # Published Woof runtime limit.
    destination.mkdir(parents=True, exist_ok=False)
    config.save_pretrained(destination)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json", "LICENSE"):
        shutil.copy2(source / name, destination / name)

    packed = mx.load(str(source / "model.safetensors"))
    weights = {}
    offsets = (".input_layernorm.weight", ".post_attention_layernorm.weight",
               "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
    for key, value in packed.items():
        if key.endswith((".scales", ".biases")):
            continue
        if not key.startswith("language_model."):
            raise ValueError(f"Unexpected non-text tensor: {key}")
        name = key.removeprefix("language_model.")
        if key.endswith(".weight") and key[:-7] + ".scales" in packed:
            value = mx.dequantize(value, packed[key[:-7] + ".scales"],
                                 packed.get(key[:-7] + ".biases"), **quant)
        # Invert mlx-lm's Qwen3.5 sanitization. Linear-attention gated RMSNorm
        # is not offset; only the five explicitly listed norm families are.
        value = value.astype(mx.float32)
        if name.endswith(offsets):
            value = value - 1.0
        if name.endswith("conv1d.weight"):
            if value.shape[-1] != 1:
                raise ValueError(f"Checkpoint is not in MLX conv layout: {name}")
            value = value.moveaxis(2, 1)
        tensor = torch.from_numpy(np.array(value)).contiguous()
        weights[name] = tensor if name.endswith("A_log") else tensor.to(torch.bfloat16)
    # Check every text tensor name AND shape before saving; a silent partial
    # load would produce a graph with randomly initialized layers.
    with torch.device("meta"):
        expected = Qwen3_5ForCausalLM(config).state_dict()
    missing = set(expected) - set(weights) - {"lm_head.weight"}
    extra = set(weights) - set(expected)
    mismatches = [name for name in weights if name in expected
                  and weights[name].shape != expected[name].shape]
    if missing or extra or mismatches:
        raise ValueError(f"Weight mismatch: missing={missing}, extra={extra}, shapes={mismatches}")
    save_file(weights, destination / "model.safetensors")
    receipt = {"source_sha256": digest(source / "model.safetensors"),
               "checkpoint_sha256": digest(destination / "model.safetensors")}
    (destination / "gen2_reconstruction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Reconstructed all {len(weights)} text tensors from the MLX checkpoint", flush=True)


def export(checkpoint: Path, output: Path, precision: str) -> None:
    import torch
    from transformers import AutoConfig
    from onnxruntime_genai.models.builder import set_io_dtype, set_onnx_dtype
    from onnxruntime_genai.models.builders.qwen import Qwen35TextModel

    output.mkdir(parents=True, exist_ok=False)
    cache = output / ".export-cache"
    cache.mkdir()
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    options = {"exclude_embeds": False, "hf_token": False,
               "block_size": 64, "is_symmetric": True, "accuracy_level": 1,
               "shared_embeddings": True, "op_types_to_quantize": ("MatMul", "Gather")}
    with torch.no_grad():
        builder = Qwen35TextModel(config, set_io_dtype(precision, "cpu", options),
                                 set_onnx_dtype(precision, options), "cpu", str(cache), options)
        builder.make_model(str(checkpoint))
        builder.save_model(str(output))
        builder.make_genai_config(str(checkpoint), {}, str(output))
        builder.save_processing(str(checkpoint), {}, str(output))
    shutil.copy2(checkpoint / "chat_template.jinja", output / "chat_template.jinja")
    shutil.copy2(checkpoint / "LICENSE", output / "LICENSE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "int4"), default="int4")
    args = parser.parse_args()
    validate_source(args.mlx)
    if not args.checkpoint.exists():
        prepare(args.mlx.resolve(), args.checkpoint.resolve())
    else:
        receipt = json.loads((args.checkpoint / "gen2_reconstruction.json").read_text())
        if receipt["source_sha256"] != digest(args.mlx / "model.safetensors") or receipt["checkpoint_sha256"] != digest(args.checkpoint / "model.safetensors"):
            raise ValueError("Existing reconstruction does not match this source; choose a new checkpoint directory")
    export(args.checkpoint.resolve(), args.output.resolve(), args.precision)
    manifest = {
        "source": "ConwayResearch/woof-textfiller-0.8B-4bit-v2.1/mlx",
        "source_sha256": digest(args.mlx / "model.safetensors"),
        "runtime": "onnxruntime-genai 0.15.2", "execution_provider": "cpu",
        "precision": args.precision, "context_length": 32768,
        "note": "MLX affine dequantization; conv layout and offset norms reversed. INT4 is re-quantized.",
    }
    (args.output / "gen2_export.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
