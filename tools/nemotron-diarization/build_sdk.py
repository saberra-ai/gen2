"""Build the pinned standalone CPU SDK and record its actual dependency closure."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess

from artifacts import digest


def capture(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    source, prefix = args.source.resolve(), args.prefix.resolve()
    lock_path = Path(__file__).with_name("runtime.lock.json")
    lock = json.loads(lock_path.read_text())["runtime"]
    if capture(["git", "rev-parse", "HEAD"], source) != lock["revision"]:
        parser.error("native source is not the locked revision")
    if capture(["git", "status", "--porcelain", "--untracked-files=no"], source):
        parser.error("native source has tracked changes")
    if capture(["git", "rev-parse", "HEAD"], source / "llama.cpp") != lock["llama_revision"]:
        parser.error("llama.cpp submodule does not match the lock")
    for name, expected in lock["source_sha256"].items():
        if digest(source / name) != expected:
            parser.error(f"locked native source changed: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    flags = [f"-D{k}={v}" for k, v in lock["cmake_definitions"].items()]
    flags.append(f"-DCMAKE_INSTALL_PREFIX={prefix}")
    commands = [
        ["scripts/configure.sh", lock["preset"], *flags],
        ["cmake", "--build", "--preset", lock["preset"], "--parallel", str(args.jobs)],
        ["cmake", "--install", f"build/{lock['preset']}"],
    ]
    for name, command in zip(["configure", "build", "install"], commands):
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    shim = args.output.resolve() / "n0_probe.dylib"
    if platform.system() != "Darwin":
        parser.error("N0 shim packaging is currently qualified only for macOS")
    shim_command = ["clang++", "-std=c++17", "-O2", "-dynamiclib",
                    str(Path(__file__).with_name("native_probe.cpp").resolve()),
                    f"-I{prefix / 'include'}", f"-I{source / 'examples'}",
                    f"-L{prefix / 'lib'}", "-lnemo_speech_asr_c",
                    f"-Wl,-rpath,{prefix / 'lib'}", "-o", str(shim)]
    subprocess.run(shim_command, check=True)
    files = {}
    for path in sorted((prefix / "lib").iterdir()):
        if path.is_file() and not path.is_symlink():
            item = {"sha256": digest(path), "size": path.stat().st_size}
            if platform.system() == "Darwin":
                item["dependencies"] = capture(["otool", "-L", str(path)])
                item["exports"] = capture(["nm", "-gU", str(path)])
            files[path.name] = item
    toolchain = {name: capture([name, "--version"]).splitlines()[0]
                 for name in ["cmake", "ninja", "clang++"]}
    if platform.system() == "Darwin":
        toolchain["sdk"] = capture(["xcrun", "--show-sdk-version"])
    if shutil.which("brew"):
        toolchain["packages"] = capture(["brew", "list", "--versions", "sentencepiece", "abseil", "protobuf"])
    receipt = {"runtime_revision": lock["revision"], "lock_sha256": digest(lock_path),
               "platform": platform.platform(), "machine": platform.machine(),
               "commands": commands, "toolchain": toolchain, "libraries": files,
               "shim": {"command": shim_command, "sha256": digest(shim),
                        "source_sha256": digest(Path(__file__).with_name("native_probe.cpp"))},
               "cmake_cache_sha256": digest(source / "build" / lock["preset"] / "CMakeCache.txt")}
    (args.output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output / "build.json")


if __name__ == "__main__":
    main()
