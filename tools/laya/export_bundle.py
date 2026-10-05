"""Export an offline FP32 Laya bundle from immutable, reviewed sources.

Use the pinned environment in requirements.txt. Requires a git checkout of the
upstream source at CODE_REVISION. Downloads only the selected checkpoint.
This creates an experimental bundle; platform/device qualification is separate.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

CODE_REVISION = "8a6e1328cce2460a0e5aa348ad465bb1b5821cd2"
MODEL_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_digest(checkpoint):
    """Bind weights AND encoder/tokenizer/config to local source identity."""
    paths = [checkpoint / n for n in ("model.safetensors", "rl_agent_config.json")]
    for directory in ("encoder", "tokenizer"):
        paths.extend(p for p in (checkpoint / directory).rglob("*") if p.is_file())
    inventory = [(p.relative_to(checkpoint).as_posix(), sha(p)) for p in sorted(paths)]
    return hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()


def main(args):
    if min(args.threads, args.max_batch, args.max_options, args.reservation_mb) < 1:
        raise ValueError("execution and resource budgets must be positive")
    if args.checkpoint_dir and (not args.model_repository or not args.model_license):
        raise ValueError("local checkpoints require --model-repository and --model-license")
    if not args.checkpoint_dir and (args.model_repository or args.model_license):
        raise ValueError("custom provenance options require --checkpoint-dir")
    if args.model_repository and (not args.model_repository.strip() or any(ord(c) < 32 or ord(c) == 127 for c in args.model_repository)):
        raise ValueError("model repository must be a nonempty provenance name")
    if args.model_license and not args.model_license.is_file():
        raise ValueError("model license must be an existing file")
    actual = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if actual != CODE_REVISION or dirty:
        raise ValueError("upstream checkout must be clean and at the pinned revision")
    if args.output.exists():
        raise ValueError("output already exists; use a new directory to preserve previous bundles")
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import onnx
    from huggingface_hub import snapshot_download
    from laya.agent import Agent

    subfolder = "" if args.checkpoint == "english" else args.checkpoint
    prefix = subfolder + "/" if subfolder else ""
    if args.checkpoint_dir:
        checkpoint = args.checkpoint_dir.resolve()
        model_repository = args.model_repository
        model_revision = checkpoint_digest(checkpoint)
    else:
        snapshot = Path(snapshot_download("convaiinnovations/laya", revision=MODEL_REVISION,
            local_dir=str(args.cache / "checkpoints" / args.checkpoint), max_workers=1, allow_patterns=[prefix + s for s in
            ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")] ))
        checkpoint = snapshot / subfolder
        model_repository = "convaiinnovations/laya"
        model_revision = MODEL_REVISION
    # Agent normalizes tokenizer_config in place. Copy metadata so the immutable
    # Hub snapshot is never modified by that compatibility helper.
    args.output.mkdir(parents=True)
    staging = args.output / "source"
    staging.mkdir()
    for name in ("tokenizer", "encoder"):
        shutil.copytree(checkpoint / name, staging / name)
    shutil.copy2(checkpoint / "rl_agent_config.json", staging / "rl_agent_config.json")
    # Hard link avoids another large copy, and the reference only reads weights.
    try:
        (staging / "model.safetensors").hardlink_to(checkpoint / "model.safetensors")
    except OSError:
        shutil.copy2(checkpoint / "model.safetensors", staging / "model.safetensors")
    torch.set_num_threads(args.threads)
    print("Loading pinned checkpoint on CPU", flush=True)
    agent = Agent(str(staging), device="cpu", compile=False, backend="eager")
    agent.model.float().eval()
    batch, seq, markers = 2, 17, 3
    inputs = (torch.randint(0,100,(batch,seq)), torch.ones((batch,seq),dtype=torch.long),
        torch.tensor([[1,5,9]]*batch), torch.ones((batch,markers),dtype=torch.bool), torch.zeros(batch,dtype=torch.long))
    names = ["input_ids","attention_mask","marker_pos","marker_mask","qtype"]
    axes = {"input_ids":{0:"B",1:"S"},"attention_mask":{0:"B",1:"S"},
        "marker_pos":{0:"B",1:"K"},"marker_mask":{0:"B",1:"K"},"qtype":{0:"B"},
        "logits":{0:"B",1:"K"},"act_logits":{0:"B"}}
    print("Exporting encoder and decision heads", flush=True)
    graph = args.output / "model.onnx"
    torch.onnx.export(agent.model, inputs, str(graph), export_params=True, opset_version=18,
        do_constant_folding=True, input_names=names, output_names=["logits","act_logits"],
        dynamic_axes=axes, dynamo=False)
    # Keep a small graph and separate hashed weights, including models >2 GiB.
    model = onnx.load(str(graph))
    onnx.save_model(model, str(graph), save_as_external_data=True, all_tensors_to_one_file=True,
        location="model.data", size_threshold=1024, convert_attribute=False)
    del model
    onnx.checker.check_model(str(graph))
    shutil.copytree(staging / "tokenizer", args.output / "tokenizer")
    shutil.copy2(staging / "rl_agent_config.json", args.output / "rl_agent_config.json")
    shutil.copy2(args.source / "LICENSE", args.output / "LICENSE-LAYA.txt")
    licenses = ["LICENSE-LAYA.txt"]
    if args.model_license:
        shutil.copy2(args.model_license, args.output / "LICENSE-MODEL.txt")
        licenses.append("LICENSE-MODEL.txt")
    config = json.loads((args.output / "rl_agent_config.json").read_text())
    tok = agent.tok
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and staging not in path.parents:
            files[path.relative_to(args.output).as_posix()] = {"bytes":path.stat().st_size,"sha256":sha(path)}
    manifest = dict(schema="gen2-laya-bundle/v1",model_repository=model_repository,
        model_revision=model_revision,code_revision=CODE_REVISION,checkpoint=args.checkpoint,
        graph="model.onnx",tokenizer="tokenizer/tokenizer.json",config="rl_agent_config.json",
        calibration=None,licenses=licenses,dtype="fp32",opset=18,
        exporter_sha256=sha(Path(__file__)),versions={p:importlib.metadata.version(p) for p in
        ("torch","transformers","tokenizers","onnx","onnxruntime","numpy","safetensors")},
        envelope=dict(max_len=config.get("max_len",1024),head_max_len=config.get("head_max_len",256),
            max_batch=args.max_batch,max_options=args.max_options,reservation_mb=args.reservation_mb),
        special_tokens=dict(cls=tok.cls_token_id,sep=tok.sep_token_id,pad=tok.pad_token_id,
            mask=tok.mask_token_id,mask_text=tok.mask_token),files=files)
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Experimental bundle written to {args.output}; run parity and device qualification before release",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache",type=Path,default=Path("target/laya/hub"))
    parser.add_argument("--checkpoint",choices=["english","multilingual","typed-decisions"],default="english")
    parser.add_argument("--checkpoint-dir",type=Path,help="local compatible Laya checkpoint; no Hub download")
    parser.add_argument("--model-repository",help="provenance name for a local checkpoint")
    parser.add_argument("--model-license",type=Path,help="license supplied by the local checkpoint owner")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--max-batch",type=int,default=4)
    parser.add_argument("--max-options",type=int,default=12)
    parser.add_argument("--reservation-mb",type=int,required=True)
    main(parser.parse_args())
