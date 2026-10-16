"""Loader resolution must not silently choose a different native dependency."""
from pathlib import Path
import tempfile
import unittest

import hashlib
import copy
from runtime_profiles import load_profile, profile_provenance, validate_provenance
from package_macos import resolve_dependency, verify_files, qualified_entry, verify_entry, package_backend, private_prefix, verify_build_provenance, verify_package_provenance


class LoaderResolution(unittest.TestCase):
    def test_metal_requires_embedded_shaders_and_separate_loader_namespace(self):
        self.assertEqual(package_backend({"preset": "cpu-diar", "backend": "cpu"}), "cpu")
        metal = {"preset": "metal-diar", "backend": "metal", "metal_shaders_embedded": True}
        self.assertEqual(package_backend(metal), "metal")
        self.assertNotEqual(private_prefix("cpu"), private_prefix("metal"))
        for invalid in [{**metal, "metal_shaders_embedded": False}, {**metal, "backend": "cpu"}, {"preset": "cuda-diar", "backend": "cpu"}]:
            with self.assertRaises(ValueError):
                package_backend(invalid)

    def test_build_admission_rejects_legacy_and_mutated_semantic_provenance(self):
        lock_path, lock = load_profile()
        fields = profile_provenance(lock_path, lock)
        build = {**fields, "model_free_state_test_completed": True,
                 "libraries": {"libnemo_speech_asr_c.1.dylib": {"sha256": "1" * 64}}}
        self.assertEqual(verify_build_provenance(build), lock)
        for key in fields:
            broken = copy.deepcopy(build)
            del broken[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                verify_build_provenance(broken)
        for key, value in [("runtime_reported_version", "nemo-speech-asr 0.2.0"),
                           ("compatibility_patches", []), ("frame_contract", {}),
                           ("llama_revision", "0" * 40)]:
            with self.subTest(changed=key), self.assertRaises(ValueError):
                verify_build_provenance({**build, key: value})
        with self.assertRaises(ValueError):
            verify_build_provenance({**build, "model_free_state_test_completed": False})
        # A declaration alone does not replace the recorded real state test.
        self.assertEqual(validate_provenance(fields), lock)

    def test_package_binds_build_bytes_and_original_sdk_library_identity(self):
        import json
        lock_path, lock = load_profile()
        fields = profile_provenance(lock_path, lock)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'provenance').mkdir()
            receipt = root / 'provenance/build.json'
            build = {**fields, 'model_free_state_test_completed': True,
                     'backend': 'cpu', 'preset': 'cpu-diar', 'libraries': {'libnemo_speech_asr_c.1.dylib': {'sha256': '1' * 64}}}
            receipt.write_text(json.dumps(build))
            name = 'libgen2_nemo_nemo_speech_asr_c.1.dylib'
            manifest = {**fields, 'build_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
                        'entry_library': name, 'backend': 'cpu',
                        'libraries': {name: {'source_sha256': '1' * 64,
                                           'sdk_library': 'libnemo_speech_asr_c.1.dylib'}}}
            verify_package_provenance(root, manifest)
            for change in [{'source_sha256': '2' * 64}, {'sdk_library': None}]:
                invalid = copy.deepcopy(manifest)
                invalid['libraries'][name].update(change)
                with self.assertRaises(ValueError):
                    verify_package_provenance(root, invalid)
            receipt.write_text(json.dumps({**build, 'frame_contract': {}}))
            with self.assertRaises(ValueError):
                verify_package_provenance(root, manifest)

    def test_bundler_materialized_entry_must_equal_the_qualified_private_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'lib').mkdir()
            name = 'libgen2_nemo_nemo_speech_asr_c.1.dylib'
            native = root / 'lib' / name
            native.write_bytes(b'qualified runtime')
            expected = hashlib.sha256(native.read_bytes()).hexdigest()
            alias = root / 'lib/libnemo_speech_asr_c.dylib'
            alias.symlink_to(name)
            verify_entry(root, name, expected)
            alias.unlink()
            alias.write_bytes(native.read_bytes())
            verify_entry(root, name, expected)
            alias.write_bytes(b'wrong runtime')
            with self.assertRaisesRegex(ValueError, 'entry'):
                verify_entry(root, name, expected)
            alias.unlink()
            alias.symlink_to(native)
            with self.assertRaisesRegex(ValueError, 'entry'):
                verify_entry(root, name, expected)

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
