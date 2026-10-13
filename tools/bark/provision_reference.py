"""Acquire only the revision-pinned Bark release and licensed test parquet."""
import argparse
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
bundle = args.destination / "model" / "onnx"
bundle.mkdir(parents=True, exist_ok=True)
revision = "0d5f814105042fe801bb019dbf34b4e6f7d3eae6"
for name in ("manifest.json", "LICENSE", "config.json", "greyhound.json", "vocab.txt",
             "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx", "decoder_joint-model.onnx.data"):
    source = hf_hub_download("ConwayResearch/bark-0.6B-4bit-v2.0", "onnx/"+name, revision=revision)
    shutil.copyfile(source, bundle / name)
for name in ("bark-release.json", "infer.py", "LICENSE"):
    source = hf_hub_download("ConwayResearch/bark-0.6B-4bit-v2.0", name, revision=revision)
    shutil.copyfile(source, args.destination / "model" / name)
source = hf_hub_download("amaai-lab/DisfluencySpeech", "data/test-00000-of-00001.parquet", repo_type="dataset",
                        revision="b7da294fe3a70dd96df6640893f2a5dfc2c87638")
shutil.copyfile(source, args.destination / "test.parquet")
