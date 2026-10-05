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
    # Rust archives can contain bitcode newer than Xcode's LLVM reader. Use the
    # matching rustup component for archive symbols; Xcode still links the host.
    host = next(line.split(": ", 1)[1] for line in command("rustc", "-vV").splitlines()
                if line.startswith("host: "))
    nm = Path(command("rustc", "--print", "sysroot")) / "lib/rustlib" / host / "bin/llvm-nm"
    if not nm.is_file():
        raise RuntimeError("Install the active toolchain's llvm-tools-preview rustup component")
    runtime_slices = plistlib.loads((runtime / "Info.plist").read_bytes())["AvailableLibraries"]
    for runtime_slice in runtime_slices:
        if runtime_slice["SupportedPlatform"] == "ios":
            info = runtime / runtime_slice["LibraryIdentifier"] / runtime_slice["LibraryPath"] / "Info.plist"
            metadata = plistlib.loads(info.read_bytes())
            assert metadata["MinimumOSVersion"] == "15.1", "Requalify a changed runtime deployment floor"
            assert metadata["CFBundleShortVersionString"] == "1.24.2", "Requalify a changed runtime version"
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
        exported = command(str(nm), "--extern-only", "--defined-only", str(library))
        actual = {line.split()[-1].removeprefix("_") for line in exported.splitlines() if line.split()}
        assert symbols <= actual, f"Missing C exports: {symbols - actual}"
        artifacts.append({"path": str(library.relative_to(output)), "sha256": digest(library)})
    for target in ("aarch64-apple-ios", "aarch64-apple-ios-sim"):
        host = output / f"{target}-host.dylib"
        assert command("xcrun", "lipo", "-archs", str(host)) == "arm64"
        artifacts.append({"path": host.name, "sha256": digest(host),
                          "load_commands": command("xcrun", "otool", "-L", str(host))})
        app = output / target / "Gen2Laya.app"
        info = plistlib.loads((app / "Info.plist").read_bytes())
        assert info["CFBundleIdentifier"] == "ai.saberra.gen2.laya.smoke"
        assert info["MinimumOSVersion"] == "15.1"
        executable = app / info["CFBundleExecutable"]
        assert command("xcrun", "lipo", "-archs", str(executable)) == "arm64"
        smoke = app / "smoke"
        manifest = json.loads((smoke / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["versions"]["fixture"] == "SYNTHETIC_TEST_ONLY"
        for name, expected in manifest["files"].items():
            path = smoke / name
            assert digest(path) == expected["sha256"] and path.stat().st_size == expected["bytes"]
        artifacts.append({"path": str(executable.relative_to(output)), "sha256": digest(executable),
                          "build_version": command("xcrun", "vtool", "-show-build", str(executable))})
    evidence = {
        "scope": "Apple SDK device/simulator arm64 build, C exports, Swift final link, XCFramework and offline app packaging",
        "device_execution": False,
        "simulator_execution": False,
        "real_checkpoint_qualification": False,
        "deployment_target": "15.1",
        "commit": command("git", "rev-parse", "HEAD"),
        "rust": command("rustc", "--version"),
        "archive_symbol_tool": command(str(nm), "--version"),
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
