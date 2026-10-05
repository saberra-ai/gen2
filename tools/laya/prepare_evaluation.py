"""Validate a labeled dataset and produce a one-call-per-sample native plan."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate_quality import validate_dataset


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "bundle", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    raw = a.dataset.read_bytes()
    dataset = json.loads(raw)
    records = validate_dataset(dataset)
    envelope = json.loads((a.bundle/"manifest.json").read_bytes())["envelope"]
    plan = dict(schema="gen2-laya-benchmark/v1", warmups=0, runs=1, threads=2,
                host=dict(purpose="quality evaluation; not latency qualification", dataset_sha256=hashlib.sha256(raw).hexdigest()),
                cases=[dict(name=r["id"], request=r["request"], encode=dict(max_len=envelope["max_len"],
                           head_max_len=envelope["head_max_len"], overflow="reject")) for r in records])
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
