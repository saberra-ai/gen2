"""Matched CPU duration sweep, from the same normalized PCM as bark_benchmark."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
from onnx_asr.loader import Manager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--wav", type=Path, required=True)
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
native = json.loads(args.native.read_text())
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
started = time.perf_counter()
asr = Manager(providers=["CPUExecutionProvider"], sess_options=options,
              preprocessor_config={"use_numpy_preprocessors": True}).create_asr(
                  "nemo-conformer-tdt", args.bundle, offline=True)
cold_seconds = time.perf_counter() - started
base, rate = sf.read(args.wav, dtype="float32")
assert rate == 16000 and base.ndim == 1
sweep = []
for entry in native["sweep"]:
    seconds = entry["seconds"]
    wave = np.resize(base, seconds * 16000)
    expected = asr.recognize(wave, sample_rate=16000).strip()
    assert expected == entry["text"], f"controlled sweep transcript mismatch at {seconds}s"
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        actual = asr.recognize(wave, sample_rate=16000).strip()
        samples.append(time.perf_counter() - started)
        assert actual == expected
    native_times = np.array([r["wall_micros"] / 1e6 for r in entry["samples"]])
    ratio = float(np.median(native_times) / np.median(samples))
    rtf = float(np.median(native_times) / seconds)
    sweep.append({"seconds": seconds, "python_seconds": samples,
                  "native_median_seconds": float(np.median(native_times)),
                  "native_p95_seconds": float(np.quantile(native_times, .95)),
                  "python_median_seconds": float(np.median(samples)),
                  "median_native_to_python_ratio": ratio, "native_rtf": rtf,
                  "passed": ratio <= 1.25 and rtf < 1.0})
    print(sweep[-1], flush=True)
receipt = {"native_git_sha": native["git_sha"], "provider": "CPUExecutionProvider",
           "intra_threads": 4, "inter_threads": 1, "python_cold_seconds": cold_seconds,
           "sweep": sweep, "passed": all(r["passed"] for r in sweep)}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
assert receipt["passed"], "controlled performance gate failed"
