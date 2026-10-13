"""Export independent NumPy features, encoder tensors and greedy token decisions."""
import argparse
import json
import contextlib
import io
import platform
import ctypes
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnx_asr.loader import Manager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
args = parser.parse_args()
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
asr = Manager(providers=["CPUExecutionProvider"], sess_options=options,
              preprocessor_config={"use_numpy_preprocessors": True}).create_asr(
                  "nemo-conformer-tdt", args.bundle, offline=True).asr
receipt = json.loads((args.reference / "reference.json").read_text())
probes = [0, 1, 2, 13, 34, 41, 43, 90, 99, 111, 249]
runtime = io.StringIO()
with contextlib.redirect_stdout(runtime):
    np.show_runtime()
blas_runtime = []
for library in (Path(np.__file__).parent.parent / "numpy.libs").glob("*openblas*"):
    handle = ctypes.CDLL(str(library))
    for symbol in ["scipy_openblas_get_corename64_", "scipy_openblas_get_config64_"]:
        function = getattr(handle, symbol, None)
        if function is not None:
            function.restype = ctypes.c_char_p
            blas_runtime.append({"library": library.name, "symbol": symbol,
                                 "value": function().decode()})
(args.reference / "numpy-runtime.json").write_text(json.dumps({
    "platform": platform.platform(), "numpy_runtime": runtime.getvalue(),
    "numpy_build": np.show_config(mode="dicts"), "blas_runtime": blas_runtime,
    "cpu_features": np._core._multiarray_umath.__cpu_features__,
}, indent=2) + "\n")
for record in receipt["records"]:
    if "token_ids" in record and record["id"] not in probes:
        continue
    wave = np.fromfile(args.reference / record["pcm"], dtype="<f4")
    # Observe the helper's original NumPy operations without changing their
    # inputs or results. Stage artifacts locate the first cross-platform drift.
    prefix = args.reference / f'{record["id"]:03}'
    original_matmul, original_log, original_rfft = np.matmul, np.log, np.fft.rfft
    def observe_matmul(*inputs, **kwargs):
        result = original_matmul(*inputs, **kwargs)
        result.astype("<f4").tofile(str(prefix) + ".mel.f32")
        return result
    def observe_log(*inputs, **kwargs):
        result = original_log(*inputs, **kwargs)
        result.astype("<f4").tofile(str(prefix) + ".log-mel.f32")
        return result
    def observe_rfft(*inputs, **kwargs):
        result = original_rfft(*inputs, **kwargs)
        (np.abs(result).astype(np.float32) ** 2).astype("<f4").tofile(str(prefix) + ".power.f32")
        return result
    try:
        np.matmul, np.log, np.fft.rfft = observe_matmul, observe_log, observe_rfft
        features, lengths = asr._preprocessor(wave[None], np.array([len(wave)], np.int64))
    finally:
        np.matmul, np.log, np.fft.rfft = original_matmul, original_log, original_rfft
    encoded, encoded_lengths = asr._encode(features, lengths)
    tokens, timestamps, _ = next(asr._decoding(encoded, encoded_lengths))
    record["token_ids"] = list(tokens)
    record["timestamps"] = list(timestamps)
    if record["id"] in probes:
        features.astype("<f4").tofile(str(prefix) + ".features.f32")
        encoded.transpose(0, 2, 1).astype("<f4").tofile(str(prefix) + ".encoder.f32")
        states = asr._create_state()
        logits, duration, states = asr._decode([], states, encoded[0, 0])
        logits.astype("<f4").tofile(str(prefix) + ".logits.f32")
        states[0].astype("<f4").tofile(str(prefix) + ".state1.f32")
        states[1].astype("<f4").tofile(str(prefix) + ".state2.f32")
        record["first_duration"] = int(duration)
        record["first_token"] = int(logits.argmax())
    print(record["id"], flush=True)
(args.reference / "reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
