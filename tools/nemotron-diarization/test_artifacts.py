"""Artifact admission through the public provision/check command."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("artifacts.py")


class ArtifactAdmission(unittest.TestCase):
    def test_corruption_never_replaces_a_verified_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "upstream.bin"
            source.write_bytes(b"abc")
            lock = root / "lock.json"
            lock.write_text(json.dumps({"artifacts": [{
                "name": "model.bin", "url": source.as_uri(), "size": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(), "role": "model",
            }]}))
            cache = root / "cache"
            cmd = [sys.executable, str(SCRIPT), "--lock", str(lock), "--cache", str(cache)]
            self.assertNotEqual(subprocess.run(cmd + ["check"], capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(cmd + ["provision"], capture_output=True).returncode, 0)
            self.assertEqual((cache / "model.bin").read_bytes(), b"abc")
            (cache / "model.bin").write_bytes(b"bad")
            self.assertNotEqual(subprocess.run(cmd + ["check"], capture_output=True).returncode, 0)
            source.write_bytes(b"truncated")
            self.assertNotEqual(subprocess.run(cmd + ["provision"], capture_output=True).returncode, 0)
            self.assertEqual((cache / "model.bin").read_bytes(), b"bad")
            self.assertEqual(list(cache.glob("*.part")), [])
            source.write_bytes(b"abc")
            self.assertEqual(subprocess.run(cmd + ["provision"], capture_output=True).returncode, 0)
            self.assertEqual((cache / "model.bin").read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
