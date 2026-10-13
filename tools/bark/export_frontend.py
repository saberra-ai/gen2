"""Export the pinned ONNX ASR frontend resources and their license."""
import argparse
import importlib.metadata as metadata
from pathlib import Path

import numpy as np
import onnx_asr
from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy
from onnx_asr.preprocessors.preprocessor import OnnxPreprocessor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
assert metadata.version("onnx-asr") == "0.12.0"
root = Path(onnx_asr.__file__).parent
args.destination.mkdir(parents=True, exist_ok=True)
for name in ("nemo128.onnx",):
    (args.destination / name).write_bytes((root / "preprocessors/data" / name).read_bytes())
with np.load(root / "preprocessors/data/fbanks.npz") as data:
    (args.destination / "nemo128.f32").write_bytes(data["nemo128"].astype("<f4").tobytes())
distribution = metadata.distribution("onnx-asr")
license_file = next(f for f in distribution.files if str(f).endswith("licenses/LICENSE"))
(args.destination / "LICENSE").write_bytes(distribution.locate_file(license_file).read_bytes())
cpu = NemoPreprocessorNumpy("nemo128")
graph = OnnxPreprocessor("nemo128", {"providers": ["CPUExecutionProvider"]})
for length in (320, 16000, 480000, 960000):
    wave = np.random.default_rng(7).normal(0, .05, (1, length)).astype(np.float32)
    inputs = wave, np.array([length], dtype=np.int64)
    a, lengths_a = cpu(*inputs)
    b, lengths_b = graph(*inputs)
    print(length, a.shape, lengths_a, lengths_b, "max error", np.abs(a-b).max(), flush=True)
