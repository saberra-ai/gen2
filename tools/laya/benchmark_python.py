"""Measure Python ONNX with the exact rows and microbatch shapes from Rust.

This conservative baseline times padding and native forward passes from already
encoded rows. Rust timings additionally include preprocessing, decoding and its
worker boundary. It is not an end-to-end Python SDK benchmark.
"""
import argparse
import hashlib
import json
import time
import platform
from pathlib import Path
import numpy as np
import onnxruntime as ort
from qualify_bundle import peak_bytes


def emit(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def load_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(a):
    records = load_records(a.rust)
    header = records[0]
    if header.get("engine") != "rust" or records[-1].get("type") != "complete":
        raise ValueError("Rust benchmark is not complete")
    raw = (a.bundle / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != header["manifest_sha256"]:
        raise ValueError("manifest mismatch")
    manifest = json.loads(raw)
    capi = Path(ort.__file__).parent / "capi"
    native = list(capi.glob("onnxruntime.dll")) + list(capi.glob("libonnxruntime.so.*")) + list(capi.glob("libonnxruntime.*.dylib"))
    if not native:
        raise ValueError("cannot identify the packaged Python ONNX runtime library")
    native_sha = hashlib.sha256(native[0].read_bytes()).hexdigest()
    if native_sha != header["native_library_sha256"]:
        raise ValueError("Python and Rust must use the same native runtime library")
    options = ort.SessionOptions()
    options.intra_op_num_threads = header["threads"]
    start = time.perf_counter()
    session = ort.InferenceSession(str(a.bundle / manifest["graph"]), sess_options=options, providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - start
    with a.output.open("x", encoding="utf-8") as stream:
        emit(stream, dict(type="host", engine="python-onnx-preencoded", runtime=ort.__version__,
                          native_library_sha256=native_sha, platform=platform.platform(), architecture=platform.machine(),
                          envelope=manifest["envelope"],
                          manifest_sha256=header["manifest_sha256"], plan_sha256=header["plan_sha256"],
                          rust_report_sha256=hashlib.sha256(a.rust.read_bytes()).hexdigest(),
                          threads=header["threads"], warmups=header["warmups"], runs=header["runs"],
                          load_seconds=load_seconds, peak_process_bytes=peak_bytes()))
        for case in records:
            if case.get("type") != "case":
                continue
            rows = case["result"]["inputs"]
            def run():
                logits, actions = [], []
                for start in range(0, len(rows), manifest["envelope"]["max_batch"]):
                    chunk = rows[start:start + manifest["envelope"]["max_batch"]]
                    b, s, k = len(chunk), max(len(r["input_ids"]) for r in chunk), max(2, max(len(r["marker_pos"]) for r in chunk))
                    data = dict(input_ids=np.full((b, s), manifest["special_tokens"]["pad"], dtype=np.int64),
                                attention_mask=np.zeros((b, s), dtype=np.int64), marker_pos=np.zeros((b, k), dtype=np.int64),
                                marker_mask=np.zeros((b, k), dtype=bool), qtype=np.array([r["qtype"] for r in chunk], dtype=np.int64))
                    for i, row in enumerate(chunk):
                        n, width = len(row["input_ids"]), len(row["marker_pos"])
                        data["input_ids"][i, :n] = row["input_ids"]
                        data["attention_mask"][i, :n] = 1
                        data["marker_pos"][i, :width] = row["marker_pos"]
                        data["marker_mask"][i, :width] = True
                    values, act = session.run(["logits", "act_logits"], data)
                    if values.shape != (b, k) or act.shape != (b, 2) or not np.isfinite(values).all() or not np.isfinite(act).all():
                        raise ValueError("invalid native output during benchmark")
                    logits.extend(values[i, :len(row["marker_pos"])].copy() for i, row in enumerate(chunk))
                    actions.extend(act)
                return logits, actions
            durations = []
            for i in range(header["warmups"] + header["runs"]):
                start = time.perf_counter()
                logits, actions = run()
                elapsed = time.perf_counter() - start
                if i >= header["warmups"]:
                    durations.append(elapsed)
                if (i + 1) % 10 == 0:
                    print(f'{case["name"]}: {i+1}/{header["warmups"]+header["runs"]} calls', flush=True)
            delta = 0.0
            if len(logits) != len(case["result"]["answers"]):
                raise ValueError("missing benchmark answer rows")
            for values, (_, answer) in zip(logits, case["result"]["answers"]):
                z = values.astype(np.float64) / answer["temperature"]
                probabilities = np.exp(z-z.max())
                probabilities /= probabilities.sum()
                delta = max(delta, float(np.max(np.abs(probabilities-np.array(answer["probabilities"])))))
            if not np.isfinite(delta) or delta > 1e-4:
                raise ValueError(f'benchmark output drift: {delta}')
            emit(stream, dict(type="case", name=case["name"], durations_seconds=durations,
                              input_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                              maximum_probability_difference=delta, peak_process_bytes=peak_bytes()))
        emit(stream, dict(type="complete", peak_process_bytes=peak_bytes()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "rust", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
