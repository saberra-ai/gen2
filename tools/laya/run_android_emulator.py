"""Install and run the offline smoke APK on an explicitly selected emulator."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid


def main(args):
    if not args.serial.startswith("emulator-"):
        raise ValueError("This CI runner accepts emulator serials only")
    args.output.mkdir(parents=True, exist_ok=True)
    adb = [str(args.adb), "-s", args.serial]
    def command(*parts):
        return subprocess.check_output(adb + list(parts), text=True, timeout=120).strip()
    assert command("shell", "getprop", "ro.kernel.qemu") == "1", "Not an emulator"
    assert command("shell", "getprop", "ro.product.cpu.abi") == "x86_64", "Expected x86_64 emulator"
    package = "example.gen2.laya"
    token = uuid.uuid4().hex
    try:
        command("install", "-r", str(args.apk.resolve()))
        command("shell", "am", "force-stop", package)
        command("shell", "am", "start", "-W", "-n", package + "/example.gen2.MainActivity",
                "--ez", "ci_smoke", "true", "--es", "ci_run_id", token)
        deadline = time.monotonic() + 180
        result = None
        while time.monotonic() < deadline:
            read = subprocess.run(adb + ["exec-out", "run-as", package, "cat", "files/reports/smoke-smoke.json"],
                                  capture_output=True, text=True, timeout=15)
            try:
                candidate = json.loads(read.stdout)
                if candidate.get("run_id") == token:
                    result = candidate
                    break
            except json.JSONDecodeError:
                pass
            time.sleep(1)
        if result is None:
            raise RuntimeError("App did not publish this run's report within 180 seconds")
        result["emulator_execution"] = True
        result["physical_device_execution"] = False
        (args.output / "emulator-evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert result.get("passed") is True and result.get("synthetic") is True, result
        assert result["abi"] == "x86_64" and len(result["batch"]) == 2, result
        assert len(result["long_scan"]["windows"]) > 1, result
        assert result["deadline_error"] and result["suspended_error"] and result["closed_error"], result
        print("Android emulator: native synthetic single/batch/window inference, deadlines and lifecycle passed")
    finally:
        with (args.output / "emulator-logcat.txt").open("wb") as log:
            subprocess.run(adb + ["logcat", "-d", "-t", "1000"], stdout=log, stderr=subprocess.STDOUT, timeout=30, check=False)
        with (args.output / "emulator.png").open("wb") as screenshot:
            subprocess.run(adb + ["exec-out", "screencap", "-p"], stdout=screenshot, timeout=30, check=False)
        # The workflow owns the emulator lifecycle; never touch physical devices.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("adb", "apk", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--serial", default="emulator-5554")
    main(parser.parse_args())
