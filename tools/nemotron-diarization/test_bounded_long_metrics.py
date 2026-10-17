import copy
import hashlib
import struct
import unittest

from bounded_cpu_metrics import VERSION
from bounded_long_metrics import audit_case
from pooled_runtime import BOUNDED_PROFILE
from qualify_pooled_native import NEW_VERSION, OLD_VERSION
import test_valid_mel_long_metrics as fixtures


class BoundedLongMetrics(unittest.TestCase):
    def fixture(self):
        public, native, raw, case = fixtures.LongSessionMetricsTests().fixture()
        public['runtime_version'] = native['runtime_version'] = VERSION
        return public, native, raw, case

    def audit(self, public, native, raw, actual_raw, case, *, public_time='1 maximum resident set size'):
        return audit_case(public, native, raw, actual_raw,
                          '1 maximum resident set size', public_time, case)

    def test_complete_compacted_tail_preserves_all_original_gates_without_mutation(self):
        public, native, raw, case = self.fixture()
        original = copy.deepcopy([public, native])
        result = self.audit(public, native, raw, raw, case)
        self.assertTrue(result['passed'])
        self.assertTrue(result['probabilities_bit_exact'])
        self.assertEqual(result['runtime_profile'], BOUNDED_PROFILE)
        self.assertEqual(result['runtime_version'], VERSION)
        self.assertEqual(result['retained_rows'], 60000)
        self.assertEqual([public, native], original)

    def test_parent_suffix_and_future_profiles_cannot_be_relabelled_on_either_side(self):
        for version in [OLD_VERSION, NEW_VERSION, VERSION + '.extra', VERSION.replace('cpu-v2', 'cpu-v3')]:
            for side in [0, 1]:
                public, native, raw, case = self.fixture()
                [public, native][side]['runtime_version'] = version
                with self.subTest(version=version, side=side), self.assertRaisesRegex(ValueError, 'version'):
                    self.audit(public, native, raw, raw, case)

    def test_backend_device_and_fixed_input_cannot_be_changed(self):
        public, native, raw, case = self.fixture()
        for left, right in [({**public, 'backend': 'metal'}, native),
                            ({**public, 'device': 'unexpected GPU'}, native),
                            (public, {**native, 'requested_gpu_index': 0}),
                            ({**public, 'input_repeats': 60}, native),
                            ({**public, 'input_pcm_f32le_sha256': '1' * 64}, native)]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                self.audit(left, right, raw, raw, case)

    def test_last_retained_value_and_required_ten_minute_history_cannot_be_discarded(self):
        public, native, raw, case = self.fixture()
        changed = raw[:-4] + struct.pack('<f', .51)
        public['retained_probability_sha256'] = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, 'probability values differ'):
            self.audit(public, native, raw, changed, case)
        raw = raw[-3200:]
        public.update(retained_probability_sha256=hashlib.sha256(raw).hexdigest(),
                      retained_probabilities={'first_frame': 179900, 'rows': 100})
        native.update(retained_first_frame=179900, retained_rows=100)
        with self.assertRaisesRegex(ValueError, 'ten-minute tail'):
            self.audit(public, native, raw, raw, case)

    def test_real_time_rss_and_revision_history_limits_still_apply(self):
        public, native, raw, case = self.fixture()
        for changed_public, log, message in [
            ({**public, 'elapsed_micros': 1800000000}, '1 maximum resident set size', 'real time'),
            (public, '1073741825 maximum resident set size', 'RSS'),
            ({**public, 'snapshots': []}, '1 maximum resident set size', 'revision history'),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.audit(changed_public, native, raw, raw, case, public_time=log)
