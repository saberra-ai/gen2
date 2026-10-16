"""Admit N0 evidence only after real outputs, pins and loader checks agree."""
import argparse
import json
from pathlib import Path
import subprocess
import tomllib

from artifacts import digest, matches
from score import read_rttm


def load(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--chat-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tools = Path(__file__).resolve().parent
    lock = load(tools / "runtime.lock.json")
    for artifact in lock["artifacts"]:
        require(matches(a.assets / artifact["name"], artifact), f"corrupt asset: {artifact['name']}")
    build = load(a.evidence / "build/build.json")
    require(build["lock_sha256"] == digest(tools / "runtime.lock.json"), "stale build lock")
    require(build["runtime_revision"] == lock["runtime"]["revision"], "wrong native source")
    for name, item in build["libraries"].items():
        require(digest(a.sdk / "lib" / name) == item["sha256"], f"changed runtime library: {name}")
    require(build["shim"]["sha256"] == digest(a.evidence / "build/n0_probe.dylib"), "changed shim")
    require(build["shim"]["source_sha256"] == digest(tools / "native_probe.cpp"), "stale shim source")
    native = load(a.evidence / "native/native.json")
    require(native["shim_sha256"] == build["shim"]["sha256"], "native run used another shim")
    require(native["runner_sha256"] == digest(tools / "run_native.py"), "stale native runner")
    require(native["lock_sha256"] == digest(tools / "runtime.lock.json"), "stale native lock")
    records = {r["name"]: r for r in native["records"]}
    require(records["ami"]["segments"] == records["ami-repeat"]["segments"], "A–B–A drift")
    require(not records["silence"]["segments"] and records["tail"]["frames"] > 0, "silence/tail failed")
    require(records["ami"]["wav_sha256"] == digest(a.assets / "ami_en2002d_2132.wav"), "wrong native audio")
    reference = load(a.evidence / "reference-ami.json")
    require(reference["wav_sha256"] == records["ami"]["wav_sha256"], "oracle audio mismatch")
    require(reference["model_revision"] == lock["model"]["revision"], "wrong oracle model")
    require(reference["script_sha256"] == digest(tools / "reference.py"), "stale oracle script")
    require(reference["lock_sha256"] == digest(tools / "runtime.lock.json"), "stale oracle asset lock")
    require(reference["environment_lock_sha256"] == digest(tools / "reference/uv.lock"), "stale oracle environment lock")
    direct = reference["transformers_direct_url"]
    require(direct["vcs_info"]["commit_id"] == "683b721971417b3b2c64fba9001f525607547e5a",
            "wrong Transformers source commit")
    frozen = tomllib.loads((tools / "reference/uv.lock").read_text())
    packages = {p["name"].lower().replace('_', '-'): p["version"] for p in frozen["package"]}
    for name, version in reference["installed_packages"].items():
        require(packages.get(name.lower().replace('_', '-')) == version, f"unlocked oracle package: {name}")
    require(reference["device"] == "cpu" and reference["dtype"] == "float32", "wrong oracle precision/device")
    require(native["geometry"] == reference["geometry"] ==
            {"chunk": 9, "right": 4, "left": 0, "fifo": 264, "cache": 264, "update": 222},
            "mismatched effective native/oracle streaming geometry")
    comparison = load(a.evidence / "comparison.json")
    require(comparison["native_sha256"] == digest(a.evidence / "native/ami.json"), "stale native comparison")
    require(comparison["native_probabilities_sha256"] == digest(a.evidence / "native/ami.json.f32"),
            "stale native probability comparison")
    require(comparison["reference_sha256"] == digest(a.evidence / "reference-ami.npy"), "stale oracle comparison")
    require(comparison["reference_metadata_sha256"] == digest(a.evidence / "reference-ami.json"), "stale oracle metadata comparison")
    require(comparison["rttm_sha256"] == digest(a.assets / "ami_en2002d_2132.rttm"), "wrong reference labels")
    require(comparison["smoke_quality_passed"] and comparison["q8_minus_fp32_absolute_der"] <= .02,
            "quantization smoke quality failed")
    example = load(a.evidence / "native-example-settings/ami.json")["segments"]
    upstream = read_rttm(a.evidence / "upstream-ami.rttm")
    require(len(example) == len(upstream) and len(example) > 1, "upstream native reference missing")
    for actual, expected in zip(example, upstream):
        require(actual["speaker"] == int(expected["speaker"].split("_")[-1])
                and abs(actual["start"] - expected["start"]) <= .001
                and abs(actual["end"] - expected["end"]) <= .001, "upstream native mismatch")
    orders = {}
    expected_fingerprints = {"probe": digest(a.probe), "shim": build["shim"]["sha256"],
                             "diarization_model": digest(a.assets / "Nemotron-3-Diarization.q8_0.gguf"),
                             "wav": records["ami"]["wav_sha256"], "chat_model": digest(a.chat_model),
                             "probe_source": digest(tools / "coexistence/src/main.rs")}
    for order in ["chat-first", "diar-first"]:
        summary = load(a.evidence / f"{order}.json")
        require(summary["fingerprints"] == expected_fingerprints, f"unrelated/stale coexistence binary/model: {order}")
        require(summary["order"] == order and summary["chat_before"] == summary["chat_after"]
                and bool(summary["chat_before"])
                and summary["concurrent_chat_and_diarization_equal"], f"coexistence failed: {order}")
        loader = (a.evidence / f"{order}.log").read_text()
        require(str(a.sdk.resolve() / "lib/libggml-base.0.25.1.dylib") in loader,
                f"native ggml did not load from pinned SDK: {order}")
        for name in ["first", "second", "reloaded", "concurrent"]:
            require(load(a.evidence / order / f"{name}.json")["segments"] == records["ami"]["segments"],
                    f"coexistence changed native output: {order}/{name}")
        orders[order] = summary
    require(orders["chat-first"]["isolated_chat"] == orders["chat-first"]["chat_before"]
            == orders["diar-first"]["chat_before"], "chat changed from the isolated baseline")
    require(all(s["chat_after_unload"] == s["chat_before"] for s in orders.values()),
            "chat changed after destroying native models")
    macho = subprocess.check_output(["otool", "-hv", str(a.probe)], text=True)
    require("TWOLEVEL" in macho, "unqualified Mach-O symbol namespace")
    files = [p for p in tools.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and ".venv" not in p.parts and "target" not in p.parts]
    receipt = {"scope": "N0 macOS arm64 CPU, one real AMI clip plus silence and tail; not N1–N5",
               "packaging_decision": "in-process host-supplied SDK via dynamic C ABI; qualify other platforms separately",
               "passed": True, "model": lock["model"], "runtime_revision": lock["runtime"]["revision"],
               "lock_sha256": digest(tools / "runtime.lock.json"),
               "tools_sha256": {str(f.relative_to(tools)): digest(f) for f in files},
               "probe_sha256": digest(a.probe), "chat_model_sha256": digest(a.chat_model),
               "toolchain": build["toolchain"], "macho_header": macho,
               "native_build": build, "oracle_execution": reference,
               "coexistence": orders, "quality": comparison,
               "native_records": [{k: r[k] for k in ["name", "frames", "audio_seconds", "seconds", "wav_sha256"]}
                                  for r in native["records"]],
               "evidence_directory": str(a.evidence.resolve()),
               "limits": ["single fixture is not broad quality evidence",
                          "host Homebrew dependency closure is not a self-contained redistributable SDK",
                          "native output includes a padded terminal frame",
                          "public Gen2 diarization API and admission are not implemented"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
