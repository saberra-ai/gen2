"""Relocate the qualified arm64 CPU SDK and its complete native dependency closure.

Packaging is explicit and offline. It never edits the source SDK or installed
Homebrew files. Ad-hoc signatures establish local integrity, not notarization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def system_dependency(name):
    return name.startswith(("/usr/lib/", "/System/Library/"))


def expand_loader_path(name, loader):
    if name == "@loader_path" or name.startswith("@loader_path/"):
        return loader.parent / name.removeprefix("@loader_path").lstrip("/")
    if name.startswith("@executable_path"):
        raise ValueError(f"host executable path is not relocatable: {name}")
    if not Path(name).is_absolute():
        raise ValueError(f"unsupported loader path: {name}")
    return Path(name)


def resolve_dependency(name, loader, rpaths):
    if name.startswith("@rpath/"):
        suffix = name.removeprefix("@rpath/")
        candidates = {p.resolve() for value in rpaths
                      if (p := expand_loader_path(value, loader) / suffix).is_file()}
        if len(candidates) > 1:
            raise ValueError(f"ambiguous dependency {name} from {loader}: {candidates}")
        if not candidates:
            raise FileNotFoundError(f"unresolved dependency {name} from {loader}")
        return next(iter(candidates))
    path = expand_loader_path(name, loader)
    if not path.is_file():
        raise FileNotFoundError(f"unresolved dependency {name} from {loader}")
    return path.resolve()


def inspect_library(path):
    identity = capture(["otool", "-D", str(path)]).splitlines()[1:]
    if len(identity) != 1:
        raise ValueError(f"expected one dylib identity: {path}")
    dependencies = re.findall(r"^\s+(.+) \(compatibility version .+\)$",
                              capture(["otool", "-L", str(path)]), re.MULTILINE)
    dependencies = [name for name in dependencies if name != identity[0]]
    commands = capture(["otool", "-l", str(path)])
    rpaths = re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.+) \(offset \d+\)", commands)
    architectures = capture(["lipo", "-archs", str(path)]).split()
    if architectures != ["arm64"]:
        raise ValueError(f"unqualified architecture in {path}: {architectures}")
    deployment = re.findall(r"cmd LC_BUILD_VERSION\n\s+cmdsize \d+\n\s+platform 1\n\s+minos ([\d.]+)", commands)
    if not deployment:
        deployment = re.findall(r"cmd LC_VERSION_MIN_MACOSX\n\s+cmdsize \d+\n\s+version ([\d.]+)", commands)
    if len(deployment) != 1:
        raise ValueError(f"missing or ambiguous macOS deployment target: {path}")
    return {"identity": identity[0], "dependencies": dependencies, "rpaths": rpaths,
            "minimum_macos": deployment[0]}


def discover(entry, sdk, build):
    pending = [entry.resolve()]
    libraries = {}
    while pending:
        path = pending.pop()
        if path in libraries:
            continue
        info = inspect_library(path)
        # The qualified SDK's runpaths are loader-relative; the explicit SDK
        # search path also resolves a dependency inherited from its parent.
        edges = {}
        for name in info["dependencies"]:
            if system_dependency(name):
                continue
            resolved = resolve_dependency(name, path, [*info["rpaths"], str(sdk / "lib")])
            edges[name] = resolved
            pending.append(resolved)
        info["edges"] = edges
        info["source_sha256"] = digest(path)
        if path.is_relative_to((sdk / "lib").resolve()):
            expected = build["libraries"].get(path.name)
            if not expected or info["source_sha256"] != expected["sha256"]:
                raise ValueError(f"SDK library differs from the native build receipt: {path}")
        libraries[path] = info
    return libraries


def copy_notices(sdk, build, libraries, output, lock):
    notices = output / "notices"
    original = sdk / "share/licenses/nemo-speech"
    for name in ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]:
        if digest(original / name) != lock["runtime"]["source_sha256"][name]:
            raise ValueError(f"native notice differs from the lock: {name}")
    if not (original / "third_party/ggml/LICENSE").is_file():
        raise ValueError("native SDK is missing the ggml license")
    shutil.copytree(original, notices / "nemo-speech")
    versions = dict(line.split(maxsplit=1) for line in build["toolchain"]["packages"].splitlines())
    components = {}
    for path in libraries:
        if path.is_relative_to((sdk / "lib").resolve()):
            continue
        roots = [parent for parent in path.parents
                 if (parent / "INSTALL_RECEIPT.json").is_file()]
        if len(roots) != 1:
            raise ValueError(f"dependency has no unambiguous package provenance: {path}")
        root = roots[0]
        name = root.parent.name
        if name not in {"sentencepiece", "abseil", "protobuf"} or versions.get(name) != root.name:
            raise ValueError(f"dependency was not qualified by this build receipt: {root}")
        if name in components:
            if components[name]["version"] != root.name:
                raise ValueError(f"multiple versions of {name}")
            continue
        target = notices / name
        target.mkdir()
        for filename in ["LICENSE", "INSTALL_RECEIPT.json", "sbom.spdx.json"]:
            shutil.copy2(root / filename, target / filename)
        for filename in ["AUTHORS", "NOTICE"]:
            if (root / filename).is_file():
                shutil.copy2(root / filename, target / filename)
        components[name] = {"version": root.name, "source": str(root),
                            "license_sha256": digest(root / "LICENSE")}
    if "sentencepiece" in components:
        if components["sentencepiece"]["version"] != "0.2.2":
            raise ValueError("requalify bundled SentencePiece notices for this version")
        bundled = Path(__file__).with_name("notices") / "sentencepiece-0.2.2"
        provenance = json.loads((bundled / "sources.json").read_text())
        for name, record in provenance["files"].items():
            if digest(bundled / name) != record["sha256"]:
                raise ValueError(f"bundled license checksum mismatch: {name}")
        shutil.copytree(bundled, notices / "sentencepiece/bundled")
    return components


def verify_files(package, files):
    reserved = {"package.json", "lib/libnemo_speech_asr_c.dylib"}
    actual = {str(path.relative_to(package)) for path in package.rglob("*")
              if path.is_file() or path.is_symlink()}
    if set(files) & reserved or actual != set(files) | reserved:
        raise ValueError("package file set differs from its manifest")
    for name, expected in files.items():
        path = package / name
        if not path.resolve().is_relative_to(package.resolve()) or path.is_symlink():
            raise ValueError(f"invalid package file path: {name}")
        if digest(path) != expected["sha256"] or path.stat().st_size != expected["size"]:
            raise ValueError(f"package checksum mismatch: {name}")


def version_tuple(value):
    numbers = [int(part) for part in value.split(".")]
    return tuple(numbers + [0] * (3 - len(numbers)))


def verify(package):
    manifest = json.loads((package / "package.json").read_text())
    if manifest.get("format") != "gen2-nemo-macos-cpu-v1":
        raise ValueError("unsupported native package format")
    verify_files(package, manifest["files"])
    deployments = []
    for name in manifest["libraries"]:
        if Path(name).name != name or not name.startswith("libgen2_nemo_") or f"lib/{name}" not in manifest["files"]:
            raise ValueError(f"invalid private library name: {name}")
        path = package / "lib" / name
        info = inspect_library(path)
        deployments.append(info["minimum_macos"])
        if info["identity"] != f"@rpath/{name}" or info["rpaths"]:
            raise ValueError(f"non-private identity or search path remains: {name}")
        for dependency in info["dependencies"]:
            if system_dependency(dependency):
                continue
            if not dependency.startswith("@loader_path/"):
                raise ValueError(f"external dependency remains in {name}: {dependency}")
            target = resolve_dependency(dependency, path, [])
            if target.parent != path.parent or target.name not in manifest["libraries"]:
                raise ValueError(f"dependency escapes the packaged closure: {dependency}")
        subprocess.run(["codesign", "--verify", "--strict", str(path)], check=True,
                       capture_output=True)
    if not deployments or max(deployments, key=version_tuple) != manifest.get("minimum_macos"):
        raise ValueError("package minimum macOS version differs from its libraries")
    alias = package / "lib/libnemo_speech_asr_c.dylib"
    if not alias.is_symlink() or alias.readlink().as_posix() != manifest["entry_library"]:
        raise ValueError("native package entry alias differs from its manifest")
    if manifest["entry_library"] not in manifest["libraries"]:
        raise ValueError("native package entry is outside its library closure")
    return manifest


def package_sdk(sdk, build_path, destination):
    if platform.system() != "Darwin":
        raise ValueError("this CPU package is qualified only on macOS arm64")
    if destination.exists():
        raise FileExistsError(f"refusing to replace an existing package: {destination}")
    sdk = sdk.resolve()
    lock_path = Path(__file__).with_name("runtime.lock.json")
    lock = json.loads(lock_path.read_text())
    build = json.loads(build_path.read_text())
    if build["runtime_revision"] != lock["runtime"]["revision"] or build["lock_sha256"] != digest(lock_path):
        raise ValueError("native build receipt does not match the immutable lock")
    if build["commands"][0][1] != "cpu-diar":
        raise ValueError("only the qualified CPU build preset can be packaged")
    entry = (sdk / "lib/libnemo_speech_asr_c.dylib").resolve()
    libraries = discover(entry, sdk, build)
    names = {path: "libgen2_nemo_" + path.name.removeprefix("lib") for path in libraries}
    if len(set(names.values())) != len(names):
        raise ValueError("native libraries have colliding output names")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".nemo-package-") as temporary:
        staged = Path(temporary) / "package"
        (staged / "lib").mkdir(parents=True)
        components = copy_notices(sdk, build, libraries, staged, lock)
        records = {}
        for source, info in libraries.items():
            name = names[source]
            output = staged / "lib" / name
            shutil.copy2(source, output)
            output.chmod(0o755)
            command = ["install_name_tool", "-id", f"@rpath/{name}"]
            for old, target in info["edges"].items():
                command.extend(["-change", old, f"@loader_path/{names[target]}"])
            for value in dict.fromkeys(info["rpaths"]):
                command.extend(["-delete_rpath", value])
            subprocess.run([*command, str(output)], check=True, capture_output=True)
            subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", str(output)],
                           check=True, capture_output=True)
            records[name] = {"source": str(source), "source_sha256": info["source_sha256"],
                             "original_identity": info["identity"],
                             "minimum_macos": info["minimum_macos"],
                             "dependencies": [f"@loader_path/{names[target]}" for target in info["edges"].values()]}
        (staged / "lib/libnemo_speech_asr_c.dylib").symlink_to(names[entry])
        manifest = {"format": "gen2-nemo-macos-cpu-v1", "architecture": "arm64", "backend": "cpu",
                    "minimum_macos": max((info["minimum_macos"] for info in libraries.values()), key=version_tuple),
                    "runtime_revision": lock["runtime"]["revision"], "lock_sha256": digest(lock_path),
                    "build_receipt_sha256": digest(build_path), "entry_library": names[entry],
                    "signing": "ad-hoc; not notarized", "components": components, "libraries": records,
                    "files": {str(path.relative_to(staged)): {"sha256": digest(path), "size": path.stat().st_size}
                              for path in sorted(staged.rglob("*")) if path.is_file() and not path.is_symlink()}}
        (staged / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(staged)
        staged.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-prefix", type=Path)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="verify a previously relocated package")
    args = parser.parse_args()
    if args.verify:
        manifest = verify(args.verify.resolve())
    else:
        if not all([args.sdk_prefix, args.build_receipt, args.output]):
            parser.error("supply --sdk-prefix, --build-receipt and --output, or --verify")
        manifest = package_sdk(args.sdk_prefix, args.build_receipt, args.output.resolve())
    print(json.dumps({"verified": True, "library_count": len(manifest["libraries"]),
                      "backend": manifest["backend"], "architecture": manifest["architecture"]}))


if __name__ == "__main__":
    main()
