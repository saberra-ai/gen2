"""Archive completed, passing Whistle CPU qualification with provenance."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("target/whistle"))
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--native", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
read = lambda path: json.loads(path.read_text(encoding="utf-8"))
reference = read(args.reference)
quality = read(args.root / "quality.json")
performance = read(args.root / "performance.json")
comparison = read(args.root / "performance-comparison.json")
probes = read(args.root / "probe-errors.json")
assert len(reference["records"]) == quality["clips"] == 202, "incomplete corpus"
assert quality["passed"] and not quality["raw_parity_mismatches"], "failed parity"
assert comparison["passed"] and comparison["memory"]["passed"], "failed performance/memory"
assert performance["cpu_memory_arena"] and comparison["environment"]["session_options"]["enableCpuMemArena"], "unmatched allocator settings"
assert len(performance["soak_rss"]) == 100, "incomplete residency soak"
assert len(probes) == 19 and all(p["atol"] == p["rtol"] == 1e-4 for p in probes), "incomplete stage probes"
lock = pathlib.Path("tools/whistle/model-lock.json")
assert performance["manifest_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest(), "model lock mismatch"
assert quality["reference_sha256"] == hashlib.sha256(args.reference.read_bytes()).hexdigest(), "reference mismatch"
assert quality["forced_sha256"] == hashlib.sha256(args.native.with_suffix(".forced.json").read_bytes()).hexdigest(), "forced result mismatch"
assert quality["native_sha256"] == hashlib.sha256(args.native.read_bytes()).hexdigest(), "native result mismatch"
args.output.mkdir(parents=True, exist_ok=False)
files = [args.reference, args.native, args.native.with_suffix(".forced.json"), args.root / "quality.json",
         args.root / "performance.json", args.root / "performance-comparison.json", args.root / "probe-errors.json",
         pathlib.Path("tools/whistle/model-lock.json"), pathlib.Path("tools/whistle/reference/source-lock.json"),
         pathlib.Path("tools/whistle/reference/package-lock.json"), args.root / "corpus/corpus.json"]
hashes = {}
for path in files:
    shutil.copyfile(path, args.output / path.name)
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {"code_sha": performance["git_sha"], "capture_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "model_revision": performance["model_revision"], "runtime_library_sha256": performance["native_library_sha256"],
           "os": performance["os"], "cpu": performance["cpu"], "architecture": performance["architecture"],
           "provider": performance["provider"], "cpu_memory_arena": performance["cpu_memory_arena"], "intra_threads": performance["intra_threads"], "inter_threads": performance["inter_threads"],
           "corpus_cases": 202, "artifacts": hashes, "accepted": True}
(args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(args.output)
