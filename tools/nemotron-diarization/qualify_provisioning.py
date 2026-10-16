"""Exercise the real public provisioner across processes, corruption and offline reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    lock = json.loads(Path(__file__).with_name("runtime.lock.json").read_text())
    destination = args.cache / f"nemotron-{lock['model']['revision']}"
    if destination.exists():
        parser.error("use a fresh qualification cache")
    # Represents a process interrupted before publication. It must never become ready.
    stale = args.cache / "nemotron-staging-interrupted"
    stale.mkdir()
    (stale / "Nemotron-3-Diarization.q8_0.gguf").write_bytes(b"partial")
    command = [str(args.executable.resolve()), "--download", str(args.cache.resolve())]
    started = time.monotonic()
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    concurrent = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=300)
        if process.returncode:
            raise RuntimeError(f"provisioner failed: {stderr}")
        assert Path(stdout.strip()) == destination.resolve()
        concurrent.append({"exit_code": process.returncode, "path": stdout.strip()})
    cold_seconds = time.monotonic() - started
    for artifact in lock["artifacts"]:
        if artifact["role"] in ("model", "notice"):
            assert digest(destination / artifact["name"]) == artifact["sha256"]
    # A corrupted published entry is rejected and repaired using the pinned source.
    (destination / "README.md").write_text("corrupted")
    repaired = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    assert Path(repaired.stdout.strip()) == destination.resolve()
    quarantined = list(args.cache.glob("nemotron-*.corrupt-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "README.md").read_text() == "corrupted"
    offline = os.environ | {"HF_HUB_OFFLINE": "1", "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9"}
    warm = subprocess.run(command, env=offline, check=True, capture_output=True, text=True, timeout=30)
    assert Path(warm.stdout.strip()) == destination.resolve()
    assert (stale / "Nemotron-3-Diarization.q8_0.gguf").read_bytes() == b"partial"
    result = {"schema_version": 1, "executable_sha256": digest(args.executable), "model_revision": lock["model"]["revision"], "concurrent_cold_processes": concurrent, "cold_seconds": cold_seconds, "corrupt_entry_quarantined_and_repaired": True, "interrupted_staging_ignored": True, "offline_warm_reuse": True, "published": {artifact["name"]: digest(destination / artifact["name"]) for artifact in lock["artifacts"] if artifact["role"] in ("model", "notice")}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
