import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pooled_runtime
import runtime_profiles


class PooledRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tools = Path(temporary.name)
        original = Path(__file__).resolve().parent
        for name in ['runtime.lock.json', 'runtime-valid-mel.lock.json', 'patches/v3-finish-valid-mel.patch',
                     'runtime-valid-mel-pool.lock.json', 'patches/persistent-cpu-pool.patch']:
            target = self.tools / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / name, target)
        for module in [runtime_profiles, pooled_runtime]:
            context = patch.object(module, '__file__', str(self.tools / (module.__name__ + '.py')))
            context.start()
            self.addCleanup(context.stop)
        self.path = self.tools / 'runtime-valid-mel-pool.lock.json'

    def test_pool_profile_is_explicit_and_preserves_frame_model_and_original_default(self):
        _, original = pooled_runtime.load_profile()
        _, pooled = pooled_runtime.load_profile(pooled_runtime.POOLED_PROFILE)
        self.assertEqual(original['profile'], 'gen2-valid-mel-v1')
        self.assertEqual(pooled['runtime']['reported_version'], 'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1')
        for key in ['model', 'fixture', 'artifacts']:
            self.assertEqual(pooled[key], original[key])
        self.assertEqual(pooled['runtime']['frame_contract'], original['runtime']['frame_contract'])

    def test_pool_profile_rejects_changed_patch_and_changed_parent(self):
        payload = self.tools / 'patches/persistent-cpu-pool.patch'
        previous = payload.read_bytes()
        payload.write_bytes(previous + b'\n')
        with self.assertRaisesRegex(ValueError, 'patch bytes differ'):
            pooled_runtime.load_profile(pooled_runtime.POOLED_PROFILE)
        payload.write_bytes(previous)
        parent = self.tools / 'runtime-valid-mel.lock.json'
        parent.write_bytes(parent.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'pool profile parent differs'):
            pooled_runtime.load_profile(pooled_runtime.POOLED_PROFILE)

    def test_pool_profile_cannot_change_version_or_unrelated_cmake_settings(self):
        original = self.path.read_bytes()
        for change in ['version', 'cmake']:
            lock = json.loads(original)
            if change == 'version':
                lock['runtime']['reported_version'] += '.extra'
            else:
                lock['runtime']['cmake_definitions']['GGML_OPENMP'] = 'ON'
            self.path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, 'unrelated runtime settings'):
                pooled_runtime.load_profile(pooled_runtime.POOLED_PROFILE)

    def test_pool_profile_rejects_missing_sources_and_unknown_profiles(self):
        lock = json.loads(self.path.read_text())
        lock['runtime']['compatibility_patches'][1]['source_sha256_before'] = {}
        self.path.write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError, 'source provenance is incomplete'):
            pooled_runtime.load_profile(pooled_runtime.POOLED_PROFILE)
        with self.assertRaisesRegex(ValueError, 'unsupported native runtime profile'):
            pooled_runtime.load_profile('gen2-valid-mel-pool-v1.extra')
