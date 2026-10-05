"""Inspect Apple SDK build products; never label linking as device execution."""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(output, runtime):
    framework = output / "Gen2Laya.xcframework"
    metadata = plistlib.loads((framework / "Info.plist").read_bytes())
    slices = metadata["AvailableLibraries"]
    assert len(slices) == 2, "Expected device and simulator slices"
    assert {s.get("SupportedPlatformVariant", "device") for s in slices} == {"device", "simulator"}
    artifacts = []
    symbols = {"gen2_laya_" + s for s in ("open", "decide", "invoke", "suspend", "resume", "close", "free")}
    for slice_info in slices:
        assert slice_info["SupportedPlatform"] == "ios"
        assert slice_info["SupportedArchitectures"] == ["arm64"]
        library = framework / slice_info["LibraryIdentifier"] / slice_info["LibraryPath"]
        assert command("xcrun", "lipo", "-archs", str(library)) == "arm64"
        exported = command("xcrun", "nm", "-gU", str(library))
        actual = {line.split()[-1].removeprefix("_") for line in exported.splitlines() if line.split()}
        assert symbols <= actual, f"Missing C exports: {symbols - actual}"
        artifacts.append({"path": str(library.relative_to(output)), "sha256": digest(library)})
    for target in ("aarch64-apple-ios", "aarch64-apple-ios-sim"):
        host = output / f"{target}-host.dylib"
        assert command("xcrun", "lipo", "-archs", str(host)) == "arm64"
        artifacts.append({"path": host.name, "sha256": digest(host),
                          "load_commands": command("xcrun", "otool", "-L", str(host))})
    evidence = {
        "scope": "Apple SDK device/simulator arm64 build, C exports, Swift final link and XCFramework packaging",
        "device_execution": False,
        "simulator_execution": False,
        "real_checkpoint_qualification": False,
        "deployment_target": "15.0",
        "commit": command("git", "rev-parse", "HEAD"),
        "rust": command("rustc", "--version"),
        "xcode": command("xcodebuild", "-version"),
        "runtime_info_sha256": digest(runtime / "Info.plist"),
        "runtime_binaries": [{"path": str(p.relative_to(runtime)), "sha256": digest(p)}
                             for p in sorted(runtime.rglob("onnxruntime")) if p.is_file()],
        "artifacts": artifacts,
    }
    (output / "build-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(evidence["scope"] + ": passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.output, args.runtime)
