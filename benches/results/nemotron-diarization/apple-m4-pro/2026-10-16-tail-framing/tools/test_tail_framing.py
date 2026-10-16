import unittest

import numpy as np

from tail_framing import native_schedule, stack_inputs


class TailFramingTests(unittest.TestCase):
    def test_partial_stack_preserves_real_negative_features_and_masks(self):
        features = -np.arange(39 * 128, dtype=np.float32).reshape(39, 128) - 1
        mask = np.ones(39, dtype=np.uint8)
        mask[32:] = 0
        stacked, encoder_mask, padding = stack_inputs(features, mask)
        self.assertEqual((stacked.shape, padding), ((5, 1024), 1))
        np.testing.assert_array_equal(stacked.reshape(40, 128)[:39], features)
        np.testing.assert_array_equal(stacked.reshape(40, 128)[39], np.zeros(128))
        np.testing.assert_array_equal(encoder_mask, [1, 1, 1, 1, 0])

    def test_terminal_real_row_changes_token_count_at_multiple_of_eight(self):
        hf, _, hf_padding = stack_inputs(np.ones((48, 128), np.float32), np.ones(48, np.uint8))
        native, _, native_padding = stack_inputs(np.ones((49, 128), np.float32), np.ones(49, np.uint8))
        self.assertEqual((hf.shape, hf_padding), ((6, 1024), 0))
        self.assertEqual((native.shape, native_padding), ((7, 1024), 7))
        np.testing.assert_array_equal(native[:6], hf)
        np.testing.assert_array_equal(native[6, :128], np.ones(128))
        self.assertFalse(native[6, 128:].any())

    def test_observed_readiness_conserves_every_native_output_once(self):
        # The source scheduler waits for 104 available rows. Transport changes
        # execution phase but must conserve each emitted global row exactly once.
        for available in [[103, 111], [0, 33, 104, 111], [111]]:
            rows = native_schedule(available, 112)
            emitted = [frame for row in rows for frame in range(row['start'], row['emitted_end'])]
            self.assertEqual(emitted, list(range(112)))
            self.assertEqual(rows[-1], {'start': 72, 'emitted_end': 112,
                                        'input_end': 112, 'phase': 'finish'})
        rows = native_schedule([11999], 12001)
        self.assertEqual(rows[-1]['start'], 11952)
        self.assertEqual(rows[-1]['input_end'], 12001)
        with self.assertRaises(ValueError):
            native_schedule([104, 103], 105)


if __name__ == '__main__':
    unittest.main()
