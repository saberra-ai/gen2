import copy
import hashlib
import struct
import unittest

from pooled_long_metrics import audit_case
from pooled_runtime import POOLED_PROFILE
from qualify_pooled_native import NEW_VERSION, OLD_VERSION
import test_valid_mel_long_metrics as fixtures


class PooledLongMetrics(unittest.TestCase):
    def fixture(self):
        public, native, raw, case = fixtures.LongSessionMetricsTests().fixture()
        public['runtime_version'] = native['runtime_version'] = NEW_VERSION
        return public, native, raw, case

    def test_complete_compacted_tail_and_original_limits_pass_without_rewriting_inputs(self):
        public, native, raw, case = self.fixture()
        original = copy.deepcopy([public, native])
        result = audit_case(public, native, raw, raw, '1 maximum resident set size', '1 maximum resident set size', case)
        self.assertTrue(result['passed'])
        self.assertTrue(result['probabilities_bit_exact'])
        self.assertEqual(result['runtime_profile'], POOLED_PROFILE)
        self.assertEqual(result['runtime_version'], NEW_VERSION)
        self.assertEqual([public, native], original)

    def test_legacy_suffix_and_other_backend_cannot_be_relabelled(self):
        for version in [OLD_VERSION, NEW_VERSION + '.extra']:
            for side in [0, 1]:
                public, native, raw, case = self.fixture()
                [public, native][side]['runtime_version'] = version
                with self.subTest(version=version, side=side), self.assertRaisesRegex(ValueError, 'version'):
                    audit_case(public, native, raw, raw, '1 maximum resident set size', '1 maximum resident set size', case)
        public, native, raw, case = self.fixture()
        with self.assertRaisesRegex(ValueError, 'CPU'):
            audit_case({**public, 'backend': 'metal'}, native, raw, raw,
                       '1 maximum resident set size', '1 maximum resident set size', case)

    def test_last_retained_value_and_existing_rtf_rss_snapshot_gates_still_apply(self):
        public, native, raw, case = self.fixture()
        changed = raw[:-4] + struct.pack('<f', .51)
        with self.assertRaisesRegex(ValueError, 'probability values differ'):
            audit_case({**public, 'retained_probability_sha256': hashlib.sha256(changed).hexdigest()},
                       native, raw, changed, '1 maximum resident set size', '1 maximum resident set size', case)
        for changed_public, time, message in [
            ({**public, 'elapsed_micros': 1800000000}, '1 maximum resident set size', 'real time'),
            (public, '1073741825 maximum resident set size', 'RSS'),
            ({**public, 'snapshots': []}, '1 maximum resident set size', 'revision history'),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                audit_case(changed_public, native, raw, raw, '1 maximum resident set size', time, case)
