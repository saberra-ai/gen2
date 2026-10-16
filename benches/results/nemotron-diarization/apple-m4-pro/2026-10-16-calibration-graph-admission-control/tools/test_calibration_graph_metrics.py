import unittest

import numpy as np

from calibration_graph_metrics import probability_comparisons, variants


class CalibrationGraphTests(unittest.TestCase):
    def test_variants_change_only_the_declared_terminal_slot(self):
        first = np.ones((105, 128), np.float32)
        first[-1] = 0
        hf = np.full((39, 128), -3, np.float32)
        native = np.full((40, 128), -7, np.float32)
        value = variants(first, hf, native)
        np.testing.assert_array_equal(value['zero40'][:39], hf)
        np.testing.assert_array_equal(value['extra40'][:39], hf)
        self.assertFalse(value['zero40'][-1].any())
        np.testing.assert_array_equal(value['extra40'][-1], native[-1])
        value['valid39'][0, 0] = 5
        self.assertEqual(hf[0, 0], -3)
        self.assertEqual(value['zero40'][0, 0], -3)

    def test_full_and_returned_comparisons_keep_cache_and_terminal_separate(self):
        full = np.zeros((112, 8), np.float32)
        full[72:111] = .25
        native = {key: full.copy() for key in ['valid39', 'zero40', 'extra40']}
        native['first104'] = np.zeros((104, 8), np.float32)
        native['extra40'][111, 7] = 1
        report = probability_comparisons(native, native['first104'], full, full[72:111])
        self.assertEqual(report['extra40_full_native_hf']['worst_frame_feature'], [111, 7])
        self.assertEqual(report['extra40_returned_native_hf']['maximum_absolute_error'], 0)
        self.assertEqual(report['hf_full_returned_prefix']['maximum_absolute_error'], 0)

    def test_wrong_shape_nonfinite_and_probability_range_rejected(self):
        first = np.zeros((105, 128), np.float32)
        with self.assertRaises(ValueError):
            variants(first, np.zeros((40, 128), np.float32), np.zeros((40, 128), np.float32))
        native = {key: np.zeros((112, 8), np.float32) for key in ['valid39', 'zero40', 'extra40']}
        native['first104'] = np.zeros((104, 8), np.float32)
        for bad in [np.nan, 1.01, -.01]:
            native['valid39'][0, 0] = bad
            with self.assertRaises(ValueError):
                probability_comparisons(native, native['first104'], native['zero40'], native['zero40'][72:111])


if __name__ == '__main__':
    unittest.main()
