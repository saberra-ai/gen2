import unittest

import numpy as np

from frontend_metrics import difference, valid_prefix


class FrontendMetricsTests(unittest.TestCase):
    def test_valid_mask_excludes_padded_frame_but_keeps_all_real_frames(self):
        native = np.zeros((4, 128), dtype=np.float32)
        hf = native.copy()
        native[3] = 999
        native[2, 17] = .01
        left, right, count = valid_prefix(hf, np.array([1, 1, 1, 0]), native)
        report = difference(left, right, 1e-3)
        self.assertEqual(count, 3)
        self.assertEqual(report['first_frame_feature_over_tolerance'], [2, 17])
        self.assertEqual(report['worst_frame_feature'], [2, 17])
        self.assertEqual(report['values_over_tolerance'], 1)
        self.assertAlmostEqual(report['maximum_absolute_error'], .01)

    def test_equal_arrays_and_signed_zero_keep_bit_identity_distinct(self):
        left = np.zeros((2, 128), dtype=np.float32)
        right = left.copy()
        self.assertTrue(difference(left, right, 0)['bitwise_equal'])
        right[0, 0] = -0.0
        report = difference(left, right, 0)
        self.assertFalse(report['bitwise_equal'])
        self.assertEqual(report['maximum_absolute_error'], 0)
        self.assertIsNone(report['first_frame_feature_over_tolerance'])

    def test_nonfinite_mismatched_or_nonprefix_data_rejected(self):
        values = np.zeros((3, 128), dtype=np.float32)
        for mask in [np.array([1, 0, 1]), np.array([0, 0, 0]), np.array([1, 2, 0]), np.ones(2)]:
            with self.assertRaises(ValueError):
                valid_prefix(values, mask, values)
        with self.assertRaises(ValueError):
            valid_prefix(values, np.ones(3), values[:2])
        for other, tolerance in [(values[:2], 1e-3), (values + np.nan, 1e-3), (values, -1), (values, np.nan)]:
            with self.assertRaises(ValueError):
                difference(values, other, tolerance)


if __name__ == '__main__':
    unittest.main()
