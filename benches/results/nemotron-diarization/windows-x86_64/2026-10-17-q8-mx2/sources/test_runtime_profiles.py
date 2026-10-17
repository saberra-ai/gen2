"""Fail closed when candidate source/semantic provenance is missing or inconsistent."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import runtime_profiles


class RuntimeProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tools = Path(temporary.name)
        original = Path(__file__).resolve().parent
        for name in ['runtime.lock.json', 'runtime-valid-mel.lock.json', 'patches/v3-finish-valid-mel.patch']:
            destination = self.tools / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / name, destination)
        context = patch.object(runtime_profiles, '__file__', str(self.tools / 'runtime_profiles.py'))
        context.start()
        self.addCleanup(context.stop)
        self.path = self.tools / 'runtime-valid-mel.lock.json'
        self.lock = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.lock))

    def test_current_profile_binds_new_semantics_to_the_unchanged_model(self):
        _, lock = runtime_profiles.load_profile()
        self.assertEqual(lock['profile'], runtime_profiles.PROFILE)
        self.assertEqual(lock['runtime']['frame_contract'], runtime_profiles.FRAME_CONTRACT)
        self.assertEqual(lock['runtime']['reported_version'], 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1')

    def test_a_changed_parent_lock_is_rejected(self):
        parent = self.tools / 'runtime.lock.json'
        parent.write_bytes(parent.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'historical runtime lock differs'):
            runtime_profiles.load_profile()

    def test_a_changed_patch_payload_is_rejected(self):
        patch_path = self.tools / 'patches/v3-finish-valid-mel.patch'
        patch_path.write_bytes(patch_path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'patch bytes differ'):
            runtime_profiles.load_profile()

    def test_model_and_upstream_pins_cannot_be_changed_with_the_frame_profile(self):
        for target in [('model', 'revision'), ('runtime', 'revision'), ('runtime', 'llama_revision')]:
            with self.subTest(target=target):
                self.lock = json.loads(self.path.read_text())
                old = self.lock[target[0]][target[1]]
                self.lock[target[0]][target[1]] = '0' * 40
                self.save()
                with self.assertRaises(ValueError):
                    runtime_profiles.load_profile()
                self.lock[target[0]][target[1]] = old
                self.save()

    def test_a_named_profile_cannot_keep_the_old_frame_rule(self):
        self.lock['runtime']['frame_contract']['final_frames'] = 'floor(real_input_samples / 160) + 1'
        self.save()
        with self.assertRaisesRegex(ValueError, 'semantic profile differs'):
            runtime_profiles.load_profile()

    def test_an_arbitrary_version_suffix_is_not_this_profile(self):
        self.lock['runtime']['reported_version'] += '.extra'
        self.save()
        with self.assertRaisesRegex(ValueError, 'semantic profile differs'):
            runtime_profiles.load_profile()

    def test_missing_patch_source_provenance_is_rejected(self):
        self.lock['runtime']['compatibility_patches'][0]['source_sha256_before'] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, 'source provenance is incomplete'):
            runtime_profiles.load_profile()


if __name__ == '__main__':
    unittest.main()
