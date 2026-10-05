"""Create the design's 1/5/10/32-question by short/medium/boundary matrix."""
import argparse
import json
import platform
import subprocess
from pathlib import Path


def main(a):
    manifest = json.loads((a.bundle / "manifest.json").read_text(encoding="utf-8"))
    envelope = manifest["envelope"]
    questions = [
        dict(type="choice", instructions="Who handles this?", options=[
            dict(label="billing", description="Payments"), dict(label="support", description="Product issues")]),
        dict(type="score", instructions="How urgent?", levels=["Routine", "Soon", "Blocking"]),
        dict(type="yes_no", instructions="Is a refund requested?", false_label="false", true_label="true",
             false_description=None, true_description=None),
    ]
    power = a.power_mode
    if power is None and platform.system() == "Windows":
        power = subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip()
    cases = []
    for count in [1, 5, 10, 32]:
        for length, repeats in [("short", 8), ("medium", envelope["max_len"] // 2), ("boundary", envelope["max_len"] * 2)]:
            cases.append(dict(name=f"q{count}-{length}", request=dict(
                state=dict(kind="text", value="evidence " * repeats),
                questions=[[f"q{i}", questions[i % 3]] for i in range(count)]),
                encode=dict(max_len=envelope["max_len"], head_max_len=envelope["head_max_len"], overflow="allow_with_diagnostics")))
    plan = dict(schema="gen2-laya-benchmark/v1", warmups=a.warmups, runs=a.runs, threads=a.threads,
                host=dict(platform=platform.platform(), architecture=platform.machine(), processor=platform.processor(),
                          power_mode=power or "unrecorded", note=a.host_note), cases=cases)
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--power-mode")
    p.add_argument("--host-note", default="Active power scheme recorded; AC and thermal state are not independently measured")
    args = p.parse_args()
    if min(args.runs, args.threads) < 1 or args.warmups < 0:
        p.error("runs/threads must be positive; warmups must be nonnegative")
    main(args)
