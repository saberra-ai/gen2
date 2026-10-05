"""Summarize complete native/Python latency reports without hiding regressions."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("latencies must be finite and positive")
    position = (len(values)-1)*p
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def records(path):
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if result[-1].get("type") != "complete":
        raise ValueError(f"incomplete report: {path}")
    return result


def main(a):
    rust, python = records(a.rust), records(a.python)
    rh, ph = rust[0], python[0]
    for key in ["manifest_sha256", "plan_sha256", "native_library_sha256", "threads", "runs", "warmups"]:
        if rh[key] != ph[key]:
            raise ValueError(f"mismatched benchmark {key}")
    if ph["rust_report_sha256"] != hashlib.sha256(a.rust.read_bytes()).hexdigest():
        raise ValueError("Python benchmark belongs to a different Rust report")
    py_cases = {c["name"]: c for c in python if c.get("type") == "case"}
    summaries = []
    for case in rust:
        if case.get("type") != "case":
            continue
        other = py_cases.pop(case["name"])
        inputs = case["result"]["inputs"]
        if hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != other["input_sha256"]:
            raise ValueError("input row mismatch")
        if len(case["durations_seconds"]) != rh["runs"] or len(other["durations_seconds"]) != rh["runs"]:
            raise ValueError("missing measured samples")
        r95, p95 = percentile(case["durations_seconds"], .95), percentile(other["durations_seconds"], .95)
        summaries.append(dict(name=case["name"], questions=len(inputs),
                              sequence_min=min(len(r["input_ids"]) for r in inputs), sequence_max=max(len(r["input_ids"]) for r in inputs),
                              rust_p50_seconds=percentile(case["durations_seconds"], .5), rust_p95_seconds=r95,
                              python_p50_seconds=percentile(other["durations_seconds"], .5), python_p95_seconds=p95,
                              rust_questions_per_second=len(inputs)/(sum(case["durations_seconds"])/len(case["durations_seconds"])),
                              rust_p95_ratio=r95/p95, within_15_percent=r95 <= p95*1.15))
    expected = {f"q{q}-{length}" for q in [1, 5, 10, 32] for length in ["short", "medium", "boundary"]}
    complete_matrix = {c["name"] for c in summaries} == expected and not py_cases
    envelope = ph["envelope"]
    shape_coverage = complete_matrix and all(
        c["questions"] == int(c["name"].split("-")[0][1:]) and (
            c["sequence_max"] == envelope["max_len"] if c["name"].endswith("boundary") else
            envelope["max_len"]//4 <= c["sequence_max"] <= envelope["max_len"]*3//4 if c["name"].endswith("medium") else
            c["sequence_max"] <= envelope["max_len"]//4)
        for c in summaries)
    enough_samples = rh["warmups"] >= 5 and rh["runs"] >= 100
    result = dict(host=rh, python_header=ph, cases=summaries, complete_matrix=complete_matrix,
                  enough_samples=enough_samples, shape_coverage=shape_coverage, release_build=rh["release_build"],
                  latency_gate_passed=complete_matrix and shape_coverage and enough_samples and rh["release_build"] and all(c["within_15_percent"] for c in summaries),
                  rust_peak_process_bytes=max(c.get("peak_process_bytes") or 0 for c in rust),
                  python_peak_process_bytes=max(c.get("peak_process_bytes") or 0 for c in python),
                  limitation="Python baseline uses preencoded rows; Rust additionally times preprocessing/decoding/worker admission. Physical-device and chat-coexistence qualification remain separate.")
    result["measured_peaks_fit_reservation"] = all(
        0 < result[key] <= rh["reservation_mb"]*1024*1024 for key in ("rust_peak_process_bytes", "python_peak_process_bytes"))
    a.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["complete_matrix", "enough_samples", "release_build", "latency_gate_passed"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("rust", "python", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    main(p.parse_args())
