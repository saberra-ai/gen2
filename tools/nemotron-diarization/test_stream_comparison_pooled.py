import copy
import hashlib
import struct
import unittest

from compare_stream_pooled import compare
from pooled_runtime import POOLED_PROFILE
from qualify_pooled_native import NEW_VERSION, OLD_VERSION
import test_stream_comparison_valid_mel as historical_fixture


class PooledStreamComparison(unittest.TestCase):
    def fixture(self):
        public, native, raw = historical_fixture.ValidMelReference().fixture()
        for record in [public, native]:
            record['runtime_version'] = NEW_VERSION
        public.update(backend='cpu', device=None)
        native['requested_gpu_index'] = -1
        return public, native, raw

    def test_exact_candidate_identity_full_history_and_turns_pass_without_rewriting_records(self):
        public, native, raw = self.fixture()
        saved = copy.deepcopy([public, native])
        result = compare(public, native, raw, raw)
        self.assertTrue(result['passed'])
        self.assertTrue(result['probabilities_bit_exact'])
        self.assertEqual(result['runtime_profile'], POOLED_PROFILE)
        self.assertEqual(result['runtime_version'], NEW_VERSION)
        self.assertEqual([public, native], saved)

    def test_old_suffix_and_future_identities_cannot_qualify_either_side(self):
        for version in [OLD_VERSION, NEW_VERSION + '.extra', NEW_VERSION.replace('pool-v1', 'pool-v2')]:
            for side in [0, 1]:
                public, native, raw = self.fixture()
                [public, native][side]['runtime_version'] = version
                with self.subTest(version=version, side=side), self.assertRaisesRegex(ValueError, 'version'):
                    compare(public, native, raw, raw)

    def test_final_probability_cannot_be_discarded_even_with_a_matching_digest(self):
        public, native, raw = self.fixture()
        changed = raw[:-4] + struct.pack('<f', .51)
        public['retained_probability_sha256'] = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, 'probability values differ'):
            compare(public, native, raw, changed)

    def test_incomplete_transport_history_and_turns_are_rejected(self):
        public, native, raw = self.fixture()
        for delta in [{'complete': False}, {'native_frame_count': 99},
                      {'retained_probabilities': {'first_frame': 1, 'rows': 99}}, {'turns': []}]:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                compare({**public, **delta}, native, raw, raw)
        with self.assertRaises(ValueError):
            compare(public, native, raw, raw[:-4])

    def test_unqualified_metal_and_wrong_cpu_device_cannot_qualify(self):
        public, native, raw = self.fixture()
        for actual, reference in [({**public, 'backend': 'metal'}, native),
                                  (public, {**native, 'requested_gpu_index': 0}),
                                  ({**public, 'device': 'unexpected GPU'}, native)]:
            with self.assertRaises(ValueError):
                compare(actual, reference, raw, raw)
