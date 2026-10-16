"""Compaction parity cannot hide wrong duration, missing history or unbounded RSS."""
import copy
import hashlib
import struct
import unittest

import test_stream_comparison_valid_mel as fixtures
from valid_mel_long_metrics import audit_case, peak_rss


class LongSessionMetricsTests(unittest.TestCase):
    def fixture(self):
        public, native, raw = fixtures.ValidMelReference().fixture()
        raw = struct.pack('<f', .5) * 60000 * 8
        public['retained_probability_sha256'] = hashlib.sha256(raw).hexdigest()
        case = {'samples': 28800000, 'repeats': 30, 'pcm_sha256': '0' * 64}
        public.update(accepted_samples=case['samples'], audio_duration_seconds=1800,
                      native_frame_count=180000, backend='cpu', device=None,
                      input_repeats=30, input_pcm_f32le_sha256=case['pcm_sha256'],
                      elapsed_micros=100000000, backpressure_count=2,
                      retained_probabilities={'first_frame': 120000, 'rows': 60000},
                      snapshots=[{'revision': 1, 'wall_micros': 1, 'processed_audio_seconds': 5., 'turn_count': 1},
                                 {'revision': 2, 'wall_micros': 2, 'processed_audio_seconds': 1790., 'turn_count': 20}])
        native.update(audio_samples=case['samples'], audio_duration_seconds=1800,
                      native_frame_count=180000, requested_gpu_index=-1,
                      input_repeats=30, input_pcm_f32le_sha256=case['pcm_sha256'],
                      elapsed_micros=100000000, retained_first_frame=120000, retained_rows=60000)
        return public, native, raw, case

    def test_compacted_probability_tail_and_provisional_history_pass(self):
        public, native, raw, case = self.fixture()
        result = audit_case(public, native, raw, raw, '200000000 maximum resident set size\n',
                            '250000000 maximum resident set size\n', case)
        self.assertTrue(result['passed'])
        self.assertEqual(result['peak_rss_bytes']['public'], 250000000)
        for key, value in [('accepted_samples', 57600000), ('input_repeats', 60),
                           ('input_pcm_f32le_sha256', '1' * 64), ('elapsed_micros', 1800000000)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_case({**public, key: value}, native, raw, raw,
                           '1 maximum resident set size', '1 maximum resident set size', case)

    def test_metal_requires_explicit_backend_and_named_device(self):
        public, native, raw, case = self.fixture()
        public.update(backend='metal', device='Apple M4 Pro')
        native['requested_gpu_index'] = 0
        args = (public, native, raw, raw, '1 maximum resident set size', '1 maximum resident set size', case)
        self.assertTrue(audit_case(*args, backend='metal', device='Apple M4 Pro')['passed'])
        with self.assertRaisesRegex(ValueError, 'backend or device'):
            audit_case(*args)
        with self.assertRaisesRegex(ValueError, 'backend or device'):
            audit_case(*args, backend='metal', device='other device')
        native['requested_gpu_index'] = -1
        with self.assertRaisesRegex(ValueError, 'Metal placement'):
            audit_case(*args, backend='metal', device='Apple M4 Pro')

    def test_matching_short_tail_cannot_replace_prescribed_retention(self):
        public, native, raw, case = self.fixture()
        raw = raw[-3200:]
        public['retained_probabilities'] = {'first_frame': 179900, 'rows': 100}
        public['retained_probability_sha256'] = hashlib.sha256(raw).hexdigest()
        native.update(retained_first_frame=179900, retained_rows=100)
        with self.assertRaisesRegex(ValueError, 'ten-minute tail'):
            audit_case(public, native, raw, raw, '1 maximum resident set size',
                       '1 maximum resident set size', case)

    def test_missing_or_excessive_process_memory_evidence_fails(self):
        public, native, raw, case = self.fixture()
        for log in ['', '0 maximum resident set size', '1 maximum resident set size\n2 maximum resident set size',
                    '1073741825 maximum resident set size']:
            with self.subTest(log=log), self.assertRaises(ValueError):
                audit_case(public, native, raw, raw, '1 maximum resident set size', log, case)
        self.assertEqual(peak_rss('  1073741824 maximum resident set size\n'), 1073741824)

    def test_stale_missing_unordered_or_unbounded_snapshots_fail(self):
        public, native, raw, case = self.fixture()
        mutations = [[], [public['snapshots'][0]], list(reversed(public['snapshots']))]
        for key, value in [('revision', 1), ('processed_audio_seconds', 100.), ('wall_micros', 1),
                           ('turn_count', 100001), ('processed_audio_seconds', float('nan'))]:
            snapshots = copy.deepcopy(public['snapshots'])
            snapshots[-1][key] = value
            mutations.append(snapshots)
        for snapshots in mutations:
            with self.subTest(snapshots=snapshots), self.assertRaises(ValueError):
                audit_case({**public, 'snapshots': snapshots}, native, raw, raw,
                           '1 maximum resident set size', '1 maximum resident set size', case)


if __name__ == '__main__':
    unittest.main()
