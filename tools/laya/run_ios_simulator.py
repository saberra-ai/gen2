"""Run the offline synthetic smoke app in a newly created, disposable simulator."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def command(*args, timeout=180):
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


def main(app, output):
    output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(command("xcrun", "simctl", "list", "--json"))
    runtimes = [r for r in inventory["runtimes"] if r.get("isAvailable") and ".iOS-" in r["identifier"]]
    if not runtimes:
        raise RuntimeError("No available iOS simulator runtime; install one with Xcode")
    runtime = max(runtimes, key=lambda r: tuple(map(int, r["version"].split("."))))
    devices = [d for d in inventory["devicetypes"] if d.get("productFamily") == "iPhone"]
    device = devices[-1]
    identifier = command("xcrun", "simctl", "create", "Gen2 Laya smoke", device["identifier"], runtime["identifier"])
    assert re.fullmatch(r"[0-9A-Fa-f-]{36}", identifier), "Unexpected created simulator ID"
    try:
        command("xcrun", "simctl", "boot", identifier)
        command("xcrun", "simctl", "bootstatus", identifier, "-b", timeout=300)
        command("xcrun", "simctl", "install", identifier, str(app.resolve()))
        bundle = "ai.saberra.gen2.laya.smoke"
        command("xcrun", "simctl", "launch", identifier, bundle, "--ci-smoke")
        container = Path(command("xcrun", "simctl", "get_app_container", identifier, bundle, "data"))
        report = container / "Documents/laya-smoke-report.json"
        deadline = time.monotonic() + 180
        while not report.is_file() and time.monotonic() < deadline:
            time.sleep(1)
        if not report.is_file():
            raise RuntimeError("Smoke app did not publish a report within 180 seconds")
        result = json.loads(report.read_text(encoding="utf-8"))
        result["simulator_runtime"] = runtime
        result["simulator_device_type"] = device["identifier"]
        (output / "simulator-evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert result.get("ok") is True, result
        assert result.get("synthetic") is True and result.get("physical_device") is False, result
        required = {"open", "capabilities", "Unicode single decision", "batch ordering and empty state",
                    "batch/single parity", "multi-window decision", "expired deadline rejected",
                    "suspended inference rejected", "resume parity", "closed handle rejected"}
        assert required <= set(result.get("checks", [])), result
        print("iOS simulator: native synthetic decisions, batching, windows and lifecycle passed")
    finally:
        # Only this invocation's freshly created simulator is touched.
        subprocess.run(["xcrun", "simctl", "io", identifier, "screenshot", str(output / "simulator.png")], timeout=30, check=False)
        subprocess.run(["xcrun", "simctl", "shutdown", identifier], timeout=60, check=False)
        subprocess.run(["xcrun", "simctl", "delete", identifier], timeout=60, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.app, args.output)
