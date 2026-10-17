"""Test exact retained frame edges and turn intersections for the prefix contrast."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ami_prefix_metrics import clip_turns, load_readonly_scorer, prefix_target


class PrefixMetricsTests(unittest.TestCase):
    def test_real_source_import_keeps_protected_directory_unchanged(self):
        def actual_import(path, _expected):
            spec = importlib.util.spec_from_file_location('fixed_scorer_fixture', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = False
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / 'scorer.py'
                source.write_text('value = 7\n')
                before = {p.relative_to(temporary).as_posix(): p.read_bytes()
                          for p in Path(temporary).rglob('*') if p.is_file()}
                with patch('ami_prefix_metrics.official_engine', side_effect=actual_import):
                    self.assertEqual(load_readonly_scorer(source, {}).value, 7)
                after = {p.relative_to(temporary).as_posix(): p.read_bytes()
                         for p in Path(temporary).rglob('*') if p.is_file()}
                self.assertEqual(after, before)
                self.assertFalse(sys.dont_write_bytecode)
                with patch('ami_prefix_metrics.official_engine', side_effect=ValueError('broken source')):
                    with self.assertRaisesRegex(ValueError, 'broken source'):
                        load_readonly_scorer(source, {})
                self.assertFalse(sys.dont_write_bytecode)
        finally:
            sys.dont_write_bytecode = previous

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
