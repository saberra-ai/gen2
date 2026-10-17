"""Test exact retained frame edges and turn intersections for the prefix contrast."""
import unittest

import numpy as np

from ami_prefix_metrics import clip_turns, prefix_target


class PrefixMetricsTests(unittest.TestCase):
    def test_target_keeps_first_and_last_valid_rows_without_prefix_leakage(self):
        values = np.zeros((42000, 8), dtype='<f4')
        values[29999, 0] = .9
        values[30000, 1] = .8
        values[41999, 2] = .7
        actual = prefix_target(values.tobytes())
        self.assertEqual(actual.shape, (12000, 8))
        self.assertEqual(actual[:, 0].sum(), 0)
        self.assertEqual(actual[0, 1], values[30000, 1])
        self.assertEqual(actual[-1, 2], values[41999, 2])
        with self.assertRaisesRegex(ValueError, '42000'):
            prefix_target(values[:-1].tobytes())
        values[10, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'prefix probabilities'):
            prefix_target(values.tobytes())

    def test_crossing_turns_clip_and_shift_with_overlap_and_exact_edges(self):
        turn = lambda start, end, speaker: {'start_seconds': start, 'end_seconds': end, 'speaker': speaker}
        values = [turn(299., 301., 1), turn(300., 300.4, 2), turn(419.5, 421., 1),
                  turn(0., 300., 3), turn(420., 421., 4)]
        actual = clip_turns(values, 300., 420.)
        self.assertEqual(actual, [{'start': 0., 'end': 1., 'speaker': 1},
            {'start': 0., 'end': 300.4 - 300., 'speaker': 2},
            {'start': 119.5, 'end': 120., 'speaker': 1}])
        self.assertEqual(values[0]['start_seconds'], 299.)
        with self.assertRaisesRegex(ValueError, 'native turn'):
            clip_turns([turn(299., float('nan'), 1)], 300., 420.)
