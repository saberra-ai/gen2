"""Exercise composed source admission and independent Windows performance limits."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from platform_bounded_metrics import VERSION, compare_complete_60s, realtime_passed, derive_native_probe
from platform_bounded_sources import INFERENCE_FILES, verify_completed_source
from pooled_runtime import BOUNDED_PROFILE
from test_platform_valid_mel import commit, git, initialize
from windows_compat import WRITE, rewrite_materializer
import test_stream_comparison_valid_mel as fixtures


class BoundedWindowsTests(unittest.TestCase):
    def source_fixture(self, root):
        source = root / 'source'
        initialize(source)
        llama = source / 'llama.cpp'
        initialize(llama)
        (llama / 'tracked.h').write_bytes(b'pristine\n')
        llama_revision = commit(llama)
        (source / '.gitignore').write_text('build/\n')
        original = WRITE + b'\n'
        cmake = source / 'cmake/llama_cpp.cmake'
        cmake.parent.mkdir()
        cmake.write_bytes(original)
        for name in INFERENCE_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'before\n')
        revision = commit(source)
        before_hash = hashlib.sha256(b'before\n').hexdigest()
        after_hash = hashlib.sha256(b'after\n').hexdigest()
        for name in INFERENCE_FILES:
            (source / name).write_bytes(b'after\n')
        payload = git(source, 'diff', '--binary', '--no-ext-diff', '--', *sorted(INFERENCE_FILES))
        patch_path = root / 'inference.patch'
        patch_path.write_bytes(payload)
        lock = {'profile': BOUNDED_PROFILE, 'runtime': {'revision': revision, 'llama_revision': llama_revision,
            'compatibility_patches': [{'source_sha256_before': {name: before_hash for name in INFERENCE_FILES}}],
            'source_sha256': {name: after_hash for name in INFERENCE_FILES}}}
        return source, cmake, original, patch_path, payload, lock

    def test_exact_composition_before_and_after_windows_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cmake, original, recorded, payload, lock = self.source_fixture(root)
            with patch('platform_bounded_sources.compatibility_patch_bytes', return_value=payload), patch(
                    'platform_bounded_sources.ORIGINAL_SHA256', hashlib.sha256(original).hexdigest()):
                verify_completed_source(source, None, lock, recorded, True)
                build = source / 'build/gen2-cpu-diar'
                build.mkdir(parents=True)
                (build / 'generated.cpp').write_bytes(b'allowed isolated build output\n')
                cmake.write_bytes(rewrite_materializer(original))
                verify_completed_source(source, build, lock, recorded, True)
                for name in INFERENCE_FILES:
                    path = source / name
                    path.write_bytes(b'before\n')
                    with self.subTest(missing=name), self.assertRaisesRegex(ValueError, 'composition'):
                        verify_completed_source(source, build, lock, recorded, True)
                    path.write_bytes(b'after\n')
                cmake.write_bytes(cmake.read_bytes() + b'# extra\n')
                with self.assertRaisesRegex(ValueError, 'exact LF overlay'):
                    verify_completed_source(source, build, lock, recorded, True)

    def test_rejects_unbound_inputs_staging_submodule_or_before_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _, recorded, payload, lock = self.source_fixture(root)
            with patch('platform_bounded_sources.compatibility_patch_bytes', return_value=payload):
                extra = source / 'extra.h'
                extra.write_bytes(b'extra\n')
                with self.assertRaisesRegex(ValueError, 'unrecorded'):
                    verify_completed_source(source, None, lock, recorded, True)
                extra.unlink()
                git(source, 'add', sorted(INFERENCE_FILES)[0])
                with self.assertRaisesRegex(ValueError, 'staged'):
                    verify_completed_source(source, None, lock, recorded, True)
                git(source, 'reset', '-q')
                submodule = source / 'llama.cpp/tracked.h'
                submodule.write_bytes(b'changed\n')
                with self.assertRaisesRegex(ValueError, 'pristine llama'):
                    verify_completed_source(source, None, lock, recorded, True)
                submodule.write_bytes(b'pristine\n')
                invalid = copy.deepcopy(lock)
                invalid['runtime']['compatibility_patches'][0]['source_sha256_before'][sorted(INFERENCE_FILES)[0]] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'pre-patch'):
                    verify_completed_source(source, None, invalid, recorded, True)
                with self.assertRaisesRegex(ValueError, 'profile required'):
                    verify_completed_source(source, None, lock, recorded, False)
                with self.assertRaisesRegex(ValueError, 'isolated source'):
                    verify_completed_source(source, root / 'foreign-build', lock, recorded, True)

    def test_rejects_changed_series_or_post_patch_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _, recorded, payload, lock = self.source_fixture(Path(directory))
            with patch('platform_bounded_sources.compatibility_patch_bytes', return_value=payload):
                recorded.write_bytes(payload + b'\n')
                with self.assertRaisesRegex(ValueError, 'patch series differs'):
                    verify_completed_source(source, None, lock, recorded, True)
                recorded.write_bytes(payload)
                lock['runtime']['source_sha256'][sorted(INFERENCE_FILES)[0]] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'source bytes differ'):
                    verify_completed_source(source, None, lock, recorded, True)

    def test_complete_parity_cannot_hide_slow_or_stale_runtime(self):
        public, native, _ = fixtures.ValidMelReference().fixture()
        raw = struct.pack('<f', .5) * 6000 * 8
        public.update(accepted_samples=960000, audio_duration_seconds=60, native_frame_count=6000,
            runtime_version=VERSION, elapsed_micros=59000000, backend='cpu', device=None,
            retained_probabilities={'first_frame': 0, 'rows': 6000}, retained_probability_sha256=hashlib.sha256(raw).hexdigest())
        native.update(audio_samples=960000, audio_duration_seconds=60, native_frame_count=6000,
            runtime_version=VERSION, elapsed_micros=59000000, requested_gpu_index=-1,
            retained_first_frame=0, retained_rows=6000)
        result = compare_complete_60s(public, native, raw, raw)
        self.assertTrue(result['passed'])
        self.assertTrue(result['realtime_gate_passed'])
        native['elapsed_micros'] = 60000000
        result = compare_complete_60s(public, native, raw, raw)
        self.assertTrue(result['passed'])
        self.assertFalse(result['realtime_gate_passed'])
        native['runtime_version'] = 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
        with self.assertRaisesRegex(ValueError, 'exact bounded'):
            compare_complete_60s(public, native, raw, raw)
        native['runtime_version'] = VERSION
        public['retained_probabilities'] = {'first_frame': 5900, 'rows': 100}
        with self.assertRaisesRegex(ValueError, 'complete6000row'):
            compare_complete_60s(public, native, raw, raw)

    def test_both_rtf_observations_must_be_positive_finite_and_below_one(self):
        for bad in [True, False, 0, -1, 1, 1.01, float('inf'), float('nan'), '0.5']:
            for key in ['gen2_rtf', 'native_rtf']:
                values = {'gen2_rtf': .5, 'native_rtf': .5, key: bad}
                with self.subTest(key=key, value=bad):
                    self.assertFalse(realtime_passed(values))

    def test_actual_frozen_helper_changes_only_the_single_version_literal(self):
        original = (Path(__file__).resolve().parent / 'native_stream_probe_valid_mel.cpp').read_bytes()
        derived = derive_native_probe(original)
        self.assertIn(('std::strcmp(version,"' + VERSION + '")').encode(), derived)
        self.assertEqual(derived.replace(('"' + VERSION + '"').encode(),
            b'"nemo-speech-asr 0.2.0+gen2-valid-mel-v1"'), original)
        for invalid in [derived, original + original]:
            with self.assertRaisesRegex(ValueError, 'one exact historical'):
                derive_native_probe(invalid)

    def test_frozen_protocol_binds_all_current_source_bytes(self):
        tools = Path(__file__).resolve().parent
        protocol = json.loads((tools / 'bounded-windows-native.protocol.json').read_text())
        for field, root in [('tools', tools), ('repository_files', tools.parents[1])]:
            for name, expected in protocol[field].items():
                raw = (root / name).read_bytes()
                self.assertEqual({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, expected, name)
        self.assertEqual(protocol['rtf_limit_exclusive'], 1.0)
        self.assertEqual(protocol['runtime_provenance']['runtime_profile'], BOUNDED_PROFILE)


if __name__ == '__main__':
    unittest.main()
