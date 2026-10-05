"""Compare a real exported checkpoint with independent Python on the current host.

Uses the committed preprocessing/permutation corpora and upstream long scan.
Reports correctness evidence only; this is not a latency or application-quality gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source, bundle, output):
    source, bundle, output = source.resolve(), bundle.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = manifest["checkpoint"]
    suffix = "" if checkpoint == "english" else "-" + checkpoint
    import onnxruntime
    capi = Path(onnxruntime.__file__).parent / "capi"
    matches = sorted({p.resolve() for p in [*capi.glob("onnxruntime.dll"), *capi.glob("libonnxruntime.so*"), *capi.glob("libonnxruntime*.dylib")]})
    if len(matches) != 1:
        raise RuntimeError(f"Expected one installed native runtime, got {matches}")
    environment = dict(os.environ, GEN2_ORT_LIBRARY=str(matches[0]), GEN2_LAYA_BUNDLE=str(bundle))

    def run(name, args, extra=None):
        path = output / (name + ".log")
        with path.open("wb") as log:
            subprocess.run(list(map(str, args)), env=dict(environment, **(extra or {})), stdout=log, stderr=subprocess.STDOUT, check=True)
        return path.read_text(encoding="utf-8", errors="replace")

    comparisons = []
    cargo = ["cargo", "test", "--locked", "--release", "--no-default-features", "--features", "laya-dynamic", "--test", "laya_native"]
    for corpus, fixture, modes in (
        ("preprocessing", Path(f"tests/fixtures/laya/preprocessing{suffix}.json"), ("onnx", "eager")),
        ("permutations", Path(f"tests/fixtures/laya/permutations-{checkpoint}.json"), ("onnx",)),
    ):
        cases = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
        for mode in modes:
            name = f"{corpus}-{mode}"
            reference = output / (name + ".json")
            print(f"{checkpoint}: recording and checking {name} ({len(cases)} cases)", flush=True)
            run(name + "-python", [sys.executable, "tools/laya/record_inference.py", "--source", source,
                "--bundle", bundle, "--fixture", fixture, "--mode", mode, "--output", reference])
            log = run(name + "-rust", cargo + ["native_matches_independent_python_reference", "--", "--ignored", "--exact", "--nocapture"],
                      {"GEN2_LAYA_REFERENCE": str(reference)})
            assert "test result: ok. 1 passed; 0 failed" in log, "Native comparison test did not execute"
            maximum = re.search(r"maximum probability difference: ([0-9.eE+-]+)", log)
            assert maximum, "Missing native probability comparison measurement"
            comparisons.append(dict(corpus=corpus, mode=mode, cases=len(cases), fixture_sha256=sha(fixture),
                                    max_probability_difference=float(maximum[1]), tolerance=1e-3 if mode == "eager" else 1e-4))
    reference = output / "long-reference.json"
    run("long-python", [sys.executable, "tools/laya/record_long.py", "--source", source, "--bundle", bundle, "--output", reference])
    log = run("long-rust", cargo + ["native_long_scan_matches_upstream_windows_and_answers", "--", "--ignored", "--exact", "--nocapture"],
              {"GEN2_LAYA_LONG_REFERENCE": str(reference)})
    assert "test result: ok. 1 passed; 0 failed" in log, "Native long scan test did not execute"
    shutil.copyfile(bundle / "manifest.json", output / "manifest.json")
    sources = [Path("Cargo.lock"), Path("Cargo.toml"), Path("src/api/runtime.rs"), Path("tests/laya_native.rs"),
               *Path("src/decision").glob("*.rs"), *Path("tools/laya").glob("*.py")]
    evidence = dict(scope="Real-checkpoint CPU preprocessing, Python ONNX/eager parity, cross-state batching, permutations and upstream long scans",
        checkpoint=checkpoint, passed=True, platform=platform.platform(), machine=platform.machine(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        rust=subprocess.check_output(["rustc", "--version"], text=True).strip(),
        python=sys.version, onnxruntime=onnxruntime.__version__, runtime_sha256=sha(matches[0]),
        manifest_sha256=sha(bundle / "manifest.json"), comparisons=comparisons, long_scan_passed=True,
        sources={p.as_posix(): sha(p) for p in sorted(sources)}, performance_qualified=False,
        accelerator_qualified=False, application_quality_claim=False)
    (output / "qualification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"{checkpoint}: real native CPU parity passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.bundle, args.output)
