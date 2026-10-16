import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from apply import plan


class OverlayTest(unittest.TestCase):
    def test_unrelated_file_prevents_any_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, bundle = root / "app", root / "bundle"
            app.mkdir()
            (bundle / "overlay").mkdir(parents=True)
            (app / "one").write_bytes(b"original")
            (app / "two").write_bytes(b"unrelated edit")
            for name in ["one", "two"]:
                (bundle / "overlay" / name).write_bytes(b"integrated")
            manifest = {"files": {name: {"base_sha256": hashlib.sha256(b"original").hexdigest(), "overlay_sha256": hashlib.sha256(b"integrated").hexdigest()} for name in ["one", "two"]}}
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unrelated"):
                plan(app, root, bundle)
            self.assertEqual((app / "one").read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
