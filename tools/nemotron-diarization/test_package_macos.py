"""Loader resolution must not silently choose a different native dependency."""
from pathlib import Path
import tempfile
import unittest

import hashlib
from package_macos import resolve_dependency, verify_files, qualified_entry


class LoaderResolution(unittest.TestCase):
    def test_external_entry_symlink_cannot_bypass_the_native_build_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sdk = root / "sdk"
            (sdk / "lib").mkdir(parents=True)
            native = sdk / "lib/libnemo_speech_asr_c.1.dylib"
            native.write_bytes(b"qualified native library")
            alias = sdk / "lib/libnemo_speech_asr_c.dylib"
            alias.symlink_to(native.name)
            build = {"libraries": {native.name: {"sha256": hashlib.sha256(native.read_bytes()).hexdigest()}}}
            self.assertEqual(qualified_entry(sdk, build), native)
            alias.unlink()
            unrelated = root / native.name
            unrelated.write_bytes(native.read_bytes())
            alias.symlink_to(unrelated)
            with self.assertRaisesRegex(ValueError, "entry.*SDK"):
                qualified_entry(sdk, build)
            alias.unlink()
            alias.symlink_to(native.name)
            native.write_bytes(b"different library")
            with self.assertRaisesRegex(ValueError, "entry.*receipt"):
                qualified_entry(sdk, build)

    def test_unlisted_native_code_and_mutated_notices_fail_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "lib").mkdir()
            (root / "package.json").write_text('{}')
            (root / "lib/libnemo_speech_asr_c.dylib").write_bytes(b"entry alias placeholder")
            (root / "LICENSE").write_bytes(b"original")
            files = {"LICENSE": {"sha256": hashlib.sha256(b"original").hexdigest(), "size": 8}}
            verify_files(root, files)
            (root / "lib/unlisted.dylib").write_bytes(b"unlisted code")
            with self.assertRaisesRegex(ValueError, "file set"):
                verify_files(root, files)
            (root / "lib/unlisted.dylib").unlink()
            (root / "LICENSE").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_files(root, files)

    def test_relative_dependency_resolves_without_ambient_loader_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "libpart.dylib").write_bytes(b"library")
            self.assertEqual(resolve_dependency("@loader_path/libpart.dylib", root / "entry.dylib", []), root / "libpart.dylib")
            self.assertEqual(resolve_dependency("@rpath/libpart.dylib", root / "entry.dylib", ["@loader_path"]), root / "libpart.dylib")
            with self.assertRaises(FileNotFoundError):
                resolve_dependency("@rpath/libmissing.dylib", root / "entry.dylib", [])

    def test_conflicting_rpaths_and_host_executable_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ["one", "two"]:
                (root / name).mkdir()
                (root / name / "libpart.dylib").write_bytes(name.encode())
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_dependency("@rpath/libpart.dylib", root / "entry.dylib", [str(root / "one"), str(root / "two")])
            with self.assertRaisesRegex(ValueError, "executable"):
                resolve_dependency("@executable_path/libpart.dylib", root / "entry.dylib", [])


if __name__ == "__main__":
    unittest.main()
