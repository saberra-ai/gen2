import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from apply import plan


class OverlayTest(unittest.TestCase):
    def test_applied_overlay_is_idempotent_and_can_relocate_gen2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, bundle = root / "app", root / "bundle"
            target = app / "src-tauri/Cargo.toml"
            target.parent.mkdir(parents=True)
            overlay = bundle / "overlay/src-tauri/Cargo.toml"
            overlay.parent.mkdir(parents=True)
            content = b'gen2 = { path = "@GEN2_ROOT@", default-features = false }\n'
            overlay.write_bytes(content)
            (bundle / "manifest.json").write_text(json.dumps({"files": {"src-tauri/Cargo.toml": {"base_sha256": None, "overlay_sha256": hashlib.sha256(content).hexdigest()}}}))
            gen2 = root / "gen2"
            first = plan(app, gen2, bundle)
            target.write_bytes(first[0][2])
            self.assertEqual(plan(app, gen2, bundle), [])
            relocated = plan(app, root / "relocated/gen2", bundle)
            self.assertEqual(len(relocated), 1)
            self.assertIn(b'../../relocated/gen2', relocated[0][2])
            target.write_bytes(relocated[0][2])
            self.assertEqual(plan(app, root / "relocated/gen2", bundle), [])

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
