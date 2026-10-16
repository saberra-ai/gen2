"""Explicit acquisition/checking of the N0 revision-pinned verification assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def matches(path, artifact):
    return (path.is_file() and not path.is_symlink()
            and path.stat().st_size == artifact["size"]
            and digest(path) == artifact["sha256"])


def acquire(artifact, cache):
    target = cache / artifact["name"]
    if matches(target, artifact):
        return
    fd, temporary = tempfile.mkstemp(suffix=".part", dir=cache)
    stage = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output:
            with urllib.request.urlopen(artifact["url"], timeout=60) as source:
                total = 0
                while block := source.read(1024 * 1024):
                    total += len(block)
                    if total > artifact["size"]:
                        raise ValueError(f"oversized download: {artifact['name']}")
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if not matches(stage, artifact):
            raise ValueError(f"size/hash mismatch: {artifact['name']}")
        stage.replace(target)
    finally:
        stage.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("runtime.lock.json"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", action="store_true", help="also acquire original FP32 oracle weights")
    parser.add_argument("operation", choices=["provision", "check"])
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    entries = lock["artifacts"]
    names = [entry["name"] for entry in entries]
    if (len(names) != len(set(names)) or any(Path(name).name != name or name in (".", "..")
                                           for name in names)):
        parser.error("artifact names must be unique plain filenames")
    entries = [entry for entry in entries if args.reference or entry["role"] != "reference"]
    if args.operation == "provision":
        args.cache.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            acquire(entry, args.cache)
    bad = [entry["name"] for entry in entries if not matches(args.cache / entry["name"], entry)]
    if bad:
        parser.exit(1, f"missing/corrupt artifacts: {', '.join(bad)}\n")
    print(json.dumps({"verified": [entry["name"] for entry in entries], "lock_sha256": digest(args.lock)}))


if __name__ == "__main__":
    main()
