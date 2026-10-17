"""Synthetic admission controls; these records are not Windows runtime evidence."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from bounded_platform_sdk_receipt import (PROTOCOL, copy_provenance, load_profile,
    validate_native_receipt, verify_provenance)
from build_valid_mel_candidate import descriptor
from platform_bounded_metrics import VERSION, compare_complete_60s, derive_native_probe
from pooled_runtime import profile_provenance
import test_stream_comparison_valid_mel as fixtures


class BoundedPlatformReceiptTests(unittest.TestCase):
    def fixture(self):
        tools = Path(__file__).resolve().parent
        lock_path, lock = load_profile()
        protocol = json.loads((tools / PROTOCOL).read_text())
        phases = [name for i in range(3) for name in ['inference-patch-check-' + str(i), 'inference-patch-' + str(i)]]
        phases += ['sdk-build', 'sdk-install', 'model-free-state-test', 'native-probe-build', 'windows-provenance',
                   'version-only', 'windows-default-loader-control', 'native-inference', 'public-inference', 'native-lifecycle-list']
        phases += ['native-lifecycle-' + str(i) for i in range(7)]
        derived = derive_native_probe((tools / 'native_stream_probe_valid_mel.cpp').read_bytes())
        return {**profile_provenance(lock_path, lock), 'schema_version': 2, 'backend': 'cpu',
            'development_native_gates_passed': True, 'functional_lifecycle_gates_passed': True,
            'realtime_gate_passed': True, 'model_free_state_test_completed': True, 'release_qualified': False,
            'platform': 'Windows-fixture', 'machine': 'AMD64', 'actual_reported_version': VERSION,
            'protocol_sha256': descriptor(tools / PROTOCOL)['sha256'],
            'runner_sha256': protocol['tools']['platform_bounded_windows_native.py']['sha256'],
            'completed_lifecycle_tests': protocol['required_lifecycle_tests'],
            'source_commit': 'dfdede8e74709a0c89be01d53aad92d346b50b99',
            'commands': [{'name': name, 'returncode': 0} for name in phases],
            'compiler_audit': {'bytes': 2, 'sha256': '0' * 64},
            'native_probe_source': {'original': descriptor(tools / 'native_stream_probe_valid_mel.cpp'),
                'derived': {'bytes': len(derived), 'sha256': hashlib.sha256(derived).hexdigest()}}}

    def test_stale_incomplete_slow_or_wrong_producer_cannot_be_relabelled(self):
        fixture = self.fixture()
        validate_native_receipt(fixture)
        for delta in [{'runtime_profile': 'gen2-valid-mel-v1'}, {'realtime_gate_passed': False},
                {'functional_lifecycle_gates_passed': False}, {'development_native_gates_passed': False},
                {'completed_lifecycle_tests': []}, {'actual_reported_version': VERSION + '.extra'},
                {'runner_sha256': '0' * 64}, {'source_commit': 'e99f59ac54abc4ce450793cb35a978734cdd7e53'},
                {'native_probe_source': {}}, {'release_qualified': True}, {'platform': 'Darwin-fixture'}]:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                validate_native_receipt({**fixture, **delta})

    def test_missing_failed_duplicate_or_reordered_patch_operations_are_rejected(self):
        fixture = self.fixture()
        reordered = copy.deepcopy(fixture['commands'])
        reordered[0], reordered[2] = reordered[2], reordered[0]
        variants = [fixture['commands'][1:], fixture['commands'] + [fixture['commands'][0]], reordered,
            [{**row, 'returncode': 1} if row['name'] == 'inference-patch-1' else row for row in fixture['commands']],
            [{**row, 'returncode': False} if row['name'] == 'sdk-build' else row for row in fixture['commands']]]
        for commands in variants:
            with self.subTest(commands=commands), self.assertRaises(ValueError):
                validate_native_receipt({**fixture, 'commands': commands})

    def evidence_fixture(self, root):
        build = self.fixture()
        tools = Path(__file__).resolve().parent
        audit = root / 'platform-llama-materialized-audit.json'
        audit.write_text('{}')
        build['compiler_audit'] = descriptor(audit)
        (root / 'receipt.json').write_text(json.dumps(build))
        (root / 'native_stream_probe_bounded.cpp').write_bytes(derive_native_probe((tools / 'native_stream_probe_valid_mel.cpp').read_bytes()))
        public, native, _ = fixtures.ValidMelReference().fixture()
        raw = struct.pack('<f', .5) * 6000 * 8
        pcm = json.loads((tools / PROTOCOL).read_text())['cases']['actual_pcm_sha256']
        shared = {'runtime_version': VERSION, 'input_pcm_f32le_sha256': pcm, 'input_repeats': 1,
                  'audio_duration_seconds': 60, 'native_frame_count': 6000, 'elapsed_micros': 59000000}
        public.update(**shared, accepted_samples=960000, backend='cpu', device=None,
            retained_probabilities={'first_frame': 0, 'rows': 6000}, retained_probability_sha256=hashlib.sha256(raw).hexdigest())
        native.update(**shared, audio_samples=960000, requested_gpu_index=-1, retained_first_frame=0, retained_rows=6000)
        (root / 'native-ami.json').write_text(json.dumps(native))
        (root / 'public.json').write_text(json.dumps(public))
        for name in ['native-ami.json.f32', 'public.f32']:
            (root / name).write_bytes(raw)
        (root / 'comparison.json').write_text(json.dumps(compare_complete_60s(public, native, raw, raw)))
        return build

    def test_package_retains_all_three_patches_derived_helper_and_replayed_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = self.evidence_fixture(root)
            package = root / 'package'
            package.mkdir()
            copy_provenance(root / 'receipt.json', package / 'provenance')
            manifest = {**build, 'build_receipt_sha256': descriptor(root / 'receipt.json')['sha256']}
            verify_provenance(package, manifest)
            for name in ['cpu-thread-cap.patch', 'persistent-cpu-pool.patch', 'v3-finish-valid-mel.patch',
                         'native-probe.cpp', 'native.f32', 'public.f32', 'compiler-audit.json']:
                path = package / 'provenance' / name
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    verify_provenance(package, manifest)
                path.write_bytes(original)
            (package / 'provenance/extra').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'unexpected bounded'):
                verify_provenance(package, manifest)

    def test_declared_pass_cannot_hide_actual_slow_timing_or_wrong_pcm(self):
        for change in ['slow', 'pcm', 'truncated']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.evidence_fixture(root)
                public = json.loads((root / 'public.json').read_text())
                if change == 'slow': public['elapsed_micros'] = 60000000
                elif change == 'pcm': public['input_pcm_f32le_sha256'] = '0' * 64
                else: (root / 'public.f32').write_bytes(b'truncated')
                (root / 'public.json').write_text(json.dumps(public))
                with self.assertRaises(ValueError):
                    copy_provenance(root / 'receipt.json', root / 'rejected')
                self.assertFalse((root / 'rejected').exists())


if __name__ == '__main__':
    unittest.main()
