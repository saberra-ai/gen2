import unittest

import numpy as np

from native_f32_metrics import comparisons


class NativeF32MetricsTests(unittest.TestCase):
    def test_terminal_error_retained_without_misalignment_against_hf(self):
        q8 = np.zeros((4, 8), dtype=np.float32)
        f32 = q8.copy()
        hf = q8[:3].copy()
        f32[-1, 7] = .5
        result = comparisons(f32, q8, hf, 480)
        self.assertEqual(result['native_f32_vs_hf_fp32_common_valid']['maximum_absolute_error'], 0)
        self.assertEqual(result['native_f32_vs_native_q8_all_rows']['maximum_absolute_error'], .5)
        self.assertEqual(result['native_f32_vs_native_q8_all_rows']['worst_frame_speaker_channel'], [3, 7])
        self.assertEqual(result['native_terminal_frame_retained']['f32'][7], .5)

    def test_partial_pcm_tail_uses_floor_valid_rows_and_preserves_channels(self):
        native = np.zeros((3, 8), dtype=np.float32)
        hf = np.zeros((2, 8), dtype=np.float32)
        native[1, 3] = .25
        result = comparisons(native, native, hf, 321)
        self.assertEqual(result['hf_valid_rows'], 2)
        self.assertEqual(result['native_raw_rows'], 3)
        self.assertEqual(result['native_f32_vs_hf_fp32_common_valid']['worst_frame_speaker_channel'], [1, 3])

    def test_invalid_shapes_ranges_or_nonfinite_arrays_rejected(self):
        native = np.zeros((3, 8), dtype=np.float32)
        hf = native[:2].copy()
        for f32, q8, fp32, samples in [(native[:2], native, hf, 320), (native, native, native, 320),
                                       (native + np.nan, native, hf, 320), (native, native - .1, hf, 320),
                                       (native, native, hf + 1.1, 320), (native, native, hf, 319)]:
            with self.assertRaises(ValueError):
                comparisons(f32, q8, fp32, samples)


if __name__ == '__main__':
    unittest.main()
