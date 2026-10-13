"""Provision the pinned model and upstream JavaScript oracle, validating hashes."""
import concurrent.futures
import hashlib
import json
import pathlib
import urllib.request

ROOT = pathlib.Path("target/whistle/model")
LOCK = json.loads(pathlib.Path("tools/whistle/model-lock.json").read_text())
ORACLE = json.loads(pathlib.Path("tools/whistle/reference/source-lock.json").read_text())


def fetch(item):
    name, expected = item
    dest = ROOT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    def valid():
        return dest.exists() and dest.stat().st_size == expected["bytes"] and hashlib.sha256(dest.read_bytes()).hexdigest() == expected["sha256"]
    if not valid():
        temporary = dest.with_suffix(dest.suffix + ".download")
        urllib.request.urlretrieve(f'https://huggingface.co/{LOCK["repository"]}/resolve/{LOCK["revision"]}/{name}', temporary)
        temporary.replace(dest)
    if not valid():
        raise ValueError(f"artifact checksum mismatch: {name}")
    return name


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for name in executor.map(fetch, {**LOCK["files"], **ORACLE["files"]}.items()):
            print(name, flush=True)
