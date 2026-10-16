"""Apply the pinned Pnola speaker integration, refusing unrelated local edits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plan(root, gen2, bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    changes = []
    root = root.resolve()
    for name, record in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid overlay path: {name}")
        target = root / name
        target.resolve().relative_to(root)
        if target.is_symlink():
            raise ValueError(f"refusing symlink target: {name}")
        content = (bundle / "overlay" / name).read_bytes()
        if digest(content) != record["overlay_sha256"]:
            raise ValueError(f"overlay checksum mismatch: {name}")
        if name == "src-tauri/Cargo.toml":
            relative_gen2 = Path(os.path.relpath(gen2.resolve(), target.parent)).as_posix()
            content = content.replace(b'"@GEN2_ROOT@"', json.dumps(relative_gen2).encode())
        previous = target.read_bytes() if target.exists() else None
        if previous == content:
            continue
        known = record.get("compatible_base_sha256", [record["base_sha256"]])
        previous_template=re.sub(rb'(?m)^gen2 = \{ path = "[^"]+"',b'gen2 = { path = "@GEN2_ROOT@"',previous) if name=="src-tauri/Cargo.toml" and previous is not None else None
        known_template=previous_template is not None and digest(previous_template)==record["overlay_sha256"]
        if not known_template and (digest(previous) if previous is not None else None) not in known:
            raise ValueError(f"unrelated local changes in {name}; integrate them before applying")
        changes.append((target, previous, content))
    return changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pnola-root", type=Path, required=True)
    p.add_argument("--gen2-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--check", action="store_true", help="validate without changing files")
    args = p.parse_args()
    if not (args.gen2_root / "src/diarization/alignment.rs").is_file():
        p.error("Gen2 root must contain the speaker-attribution implementation")
    changes = plan(args.pnola_root, args.gen2_root, Path(__file__).resolve().parent)
    if args.check:
        print(json.dumps({"valid": True, "files_to_update": len(changes)}))
        return
    backup = args.pnola_root / ".nemotron-backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if changes:
        backup.mkdir(parents=True)
        for target, previous, content in changes:
            if previous is not None:
                saved = backup / target.relative_to(args.pnola_root.resolve())
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(previous)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stage:
                stage.write(content)
                stage.flush()
                os.fsync(stage.fileno())
                staged = Path(stage.name)
            staged.replace(target)
    print(json.dumps({"updated": len(changes), "backup": str(backup) if changes else None}))


if __name__ == "__main__":
    main()
