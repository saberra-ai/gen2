"""Keep the bounded candidate isolated from old artifact identities/policies."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import pooled_runtime
import runtime_profiles


class BoundedCpuRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tools = Path(temporary.name)
        original = Path(__file__).resolve().parent
        for name in ['runtime.lock.json','runtime-valid-mel.lock.json','runtime-valid-mel-pool.lock.json',
                     'runtime-valid-mel-pool-cpu.lock.json','patches/v3-finish-valid-mel.patch',
                     'patches/persistent-cpu-pool.patch','patches/cpu-thread-cap.patch']:
            target = self.tools / name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original / name,target)
        for module in [runtime_profiles,pooled_runtime]:
            context = patch.object(module,'__file__',str(self.tools / (module.__name__+'.py')))
            context.start()
            self.addCleanup(context.stop)
        self.path = self.tools / 'runtime-valid-mel-pool-cpu.lock.json'

    def test_new_identity_preserves_models_geometry_and_old_profile_defaults(self):
        _,original = pooled_runtime.load_profile()
        _,parent = pooled_runtime.load_profile(pooled_runtime.POOLED_PROFILE)
        _,bounded = pooled_runtime.load_profile(pooled_runtime.BOUNDED_PROFILE)
        self.assertEqual(original['profile'],'gen2-valid-mel-v1')
        self.assertEqual(parent['runtime']['reported_version'],'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1')
        self.assertEqual(bounded['runtime']['reported_version'],'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-cpu-v2')
        for key in ['model','fixture','artifacts']:
            self.assertEqual(bounded[key],parent[key])
        self.assertEqual(bounded['runtime']['frame_contract'],parent['runtime']['frame_contract'])

    def test_changed_patch_or_parent_and_unrelated_settings_are_rejected(self):
        path = self.tools / 'patches/cpu-thread-cap.patch'
        previous = path.read_bytes()
        path.write_bytes(previous+b'\n')
        with self.assertRaisesRegex(ValueError,'patch bytes differ'):
            pooled_runtime.load_profile(pooled_runtime.BOUNDED_PROFILE)
        path.write_bytes(previous)
        parent = self.tools / 'runtime-valid-mel-pool.lock.json'
        previous = parent.read_bytes()
        parent.write_bytes(previous+b'\n')
        with self.assertRaisesRegex(ValueError,'profile parent differs'):
            pooled_runtime.load_profile(pooled_runtime.BOUNDED_PROFILE)
        parent.write_bytes(previous)
        original = json.loads(self.path.read_text())
        for key in ['version','cmake','geometry','patch-order','missing-source']:
            changed = copy.deepcopy(original)
            if key == 'version': changed['runtime']['reported_version'] += '.extra'
            elif key == 'cmake': changed['runtime']['cmake_definitions']['GGML_OPENMP'] = 'ON'
            elif key == 'geometry': changed['runtime']['frame_contract']['output_channels'] = 4
            elif key == 'patch-order': changed['runtime']['compatibility_patches'].reverse()
            else: changed['runtime']['compatibility_patches'][1]['source_sha256_after'] = {}
            self.path.write_text(json.dumps(changed))
            with self.subTest(change=key),self.assertRaises(ValueError):
                pooled_runtime.load_profile(pooled_runtime.BOUNDED_PROFILE)
