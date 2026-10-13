"""Collect small, provenance-rich release evidence; never weights or audio."""
import argparse
import datetime
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True), "commit implementation before capturing receipts"
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

inputs = {
    "reference.json": args.root / "oracle124/reference.json",
    "native.json": args.root / "native.json",
    "qualification.json": args.root / "qualification.json",
    "audio-reference.json": args.root / "audio/audio-reference.json",
    "native-audio.json": args.root / "audio/native-audio.json",
    "normalization-errors.json": args.root / "audio/normalization-errors.json",
    "audio-quality.json": args.root / "audio-quality.json",
    "performance.json": args.root / "performance.json",
    "mixed-performance.json": args.root / "mixed-performance.json",
    "performance-comparison.json": args.root / "performance-comparison.json",
    "numeric.log": args.root / "numeric.log",
    "integration.log": args.root / "integration.log",
}
for path in inputs.values():
    assert path.is_file(), path
for name in ["performance.json", "mixed-performance.json", "qualification.json", "audio-quality.json"]:
    basis = json.loads(inputs[name].read_text())["git_sha"]
    subprocess.run(["git", "merge-base", "--is-ancestor", basis, sha], check=True)
    changed = subprocess.check_output(["git", "diff", "--name-only", basis, sha, "--", "src", "Cargo.toml", "Cargo.lock", "assets", "examples/bark_benchmark.rs", "tests"], text=True)
    assert not changed, f"implementation changed since receipt {name}: {changed}"
assert json.loads(inputs["audio-quality.json"].read_text())["passed"]
assert json.loads(inputs["performance-comparison.json"].read_text())["passed"]
for name in ["numeric.log", "integration.log"]:
    log = inputs[name].read_text()
    assert "test result: ok." in log and "FAILED" not in log, f"unpassed native gate: {name}"
args.output.mkdir(parents=True, exist_ok=True)
for name, path in inputs.items():
    shutil.copyfile(path, args.output / name)
source_files = [Path("Cargo.toml"), Path("Cargo.lock"), Path("tools/bark/reference.lock"),
                *sorted(Path("src/speech").glob("*.rs")), Path("src/onnx_runtime.rs"),
                Path("src/api/runtime.rs"),
                Path("assets/bark/nemo128.f32"), Path("tests/fixtures/bark/smoke.json"),
                Path("tools/bark/critical-content.json")]
cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() if platform.system() == "Darwin" else platform.processor()
provenance = {
    "implementation_git_sha": sha, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "machine": platform.platform(), "cpu": cpu, "architecture": platform.machine(),
    "native_runtime": {"version": "1.24.4", "library_name": args.library.name,
                       "sha256": digest(args.library)},
    "provider": "CPUExecutionProvider", "intra_threads": 4, "inter_threads": 1,
    "verification_features": "asr-dynamic,asr-resample-soxr,laya-dynamic; benchmark additionally backend-llamacpp",
    "numeric_gates": {"frontend_atol": 1e-4, "frontend_rtol": 1e-4,
                      "encoder_decoder_atol": 1e-4, "encoder_decoder_rtol": 1e-4,
                      "experimental_resampling_max_absolute_error": 1e-5},
    "frontend": "native NumPy-compatible DSP, float64 FFT, float32 mel/FMA/normalization",
    "source_sha256": {str(p): digest(p) for p in source_files},
    "evidence_sha256": {name: digest(path) for name, path in inputs.items()},
    "chat_model": {"repo": "unsloth/Qwen3-0.6B-GGUF", "revision": "50968a4468ef4233ed78cd7c3de230dd1d61a56b",
                   "sha256": digest(args.root / "chat/Qwen3-0.6B-Q4_K_M.gguf"), "gpu_layers": 0},
    "mixed_decision_model": "synthetic Laya native smoke; not real Laya checkpoint quality/performance",
    "experimental_resampling": True,
}
(args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(args.output)
