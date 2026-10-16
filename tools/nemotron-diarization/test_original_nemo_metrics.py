"""Ensure diagnostics include the last valid row and reject legacy extra rows."""
import unittest

import numpy as np

from original_nemo_metrics import compare_original


class OriginalNemoMetrics(unittest.TestCase):
    def test_last_valid_row_is_scored(self):
        reference = np.zeros((111, 8), np.float32)
        actual = reference.copy()
        actual[-1, -1] = .5
        result = compare_original(actual, reference, 17872)
        self.assertEqual(result['maximum_absolute_error'], .5)
        self.assertEqual(result['worst_frame_feature'], [110, 7])
        self.assertEqual(result['values_over_tolerance'], 1)

    def test_extra_missing_or_invalid_rows_rejected(self):
        reference = np.zeros((111, 8), np.float32)
        for actual in [np.zeros((112, 8)), np.zeros((110, 8)),
                       np.full((111, 8), np.nan), np.full((111, 8), 1.1)]:
            with self.assertRaises(ValueError):
                compare_original(actual, reference, 17872)


if __name__ == '__main__':
    unittest.main()
