"""New control/candidate isolation and unchanged numerical/timing predicates."""
import copy
import json
from pathlib import Path
import struct
import unittest

from q8_row_source import control_source, transform
from q8_row_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from windows_q8_model_sources import candidate_source, GUARD
from windows_q8_row_metrics import validate_output, timing_evidence, KERNEL_ORDER
from test_windows_q8_worker_timing import rows as parent_rows, cases as parent_cases
from q8_mx4_geometry import SHAPES, PRIMARY_SHAPES

TOOLS = Path(__file__).resolve().parent


def relabel(rows):
    for row in rows:
        if row['variant'] == 'mx4':
            row['variant'] = 'row1'
    return rows


def cases():
    values = parent_cases()
    for case in values:
        relabel(case['observed']['rows'])
    return values


class RowKernelTest(unittest.TestCase):
    def test_control_is_exact_current_production_kernel_without_embedded_define(self):
        original = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        self.assertEqual(GUARD + control_source(original), candidate_source())
        self.assertIn(b'gemmMx4<2>(0, me, 0, ne);', control_source(original))

    def test_only_one_row_instantiation_and_compile_guard_differ(self):
        original = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        candidate = transform(original)
        restored = candidate.replace(b'gemmMx4<1>(0, me, 0, ne);',
            b'gemmMx4<2>(0, me, 0, ne);', 1).replace(b'GEN2_Q8_ROW_EXPECTED', b'GEN2_Q8_MX4_EXPECTED', 1)
        self.assertEqual(restored, control_source(original))
        self.assertEqual(candidate.count(b'gemmMx4<1>(0, me, 0, ne);'), 1)
        self.assertIn(b'const int64_t me = m / 2 * 2;', candidate)
        self.assertIn(b'mnpack(me, m, 0, ne);', candidate)
        self.assertIn(b'mnpack(0, m, ne, n);', candidate)
        for bad in (original[:-1], b'changed' + original, original.replace(b'void matmul(', b'void changed(', 1)):
            with self.assertRaises(ValueError): transform(bad)

    def test_all_construction_support_clock_and_metric_changes_are_exact_derivations(self):
        for target, parent, derive in (
                ('windows_q8_row_probe.py', 'windows_q8_pair_probe.py', derive_probe),
                ('windows_q8_row_metrics.py', 'windows_q8_pair_metrics.py', derive_metrics),
                ('q8_row_benchmark.cmake', 'q8_pair_benchmark.cmake', derive_cmake),
                ('q8_row_benchmark.cpp', 'q8_pair_benchmark.cpp', derive_benchmark)):
            with self.subTest(target=target):
                self.assertEqual((TOOLS / target).read_bytes(), derive((TOOLS / parent).read_bytes()))
                with self.assertRaises(ValueError): derive(b'incomplete parent')

    def test_complete_phase_vectors_and_tails_reject_changed_or_missing_bytes(self):
        rows = relabel(parent_rows('on', 4))
        vectors = [struct.pack('<f', .25) * (m*n) for m, n, k in SHAPES]
        outputs = [raw for raw in vectors for _ in range(4)]
        text = '\n'.join(map(json.dumps, rows))
        self.assertEqual(len(validate_output(text, 'on', 4, outputs)['outputs']), 88)
        changed = outputs.copy(); changed[1] = b'\0\0\0\0' + changed[1][4:]
        with self.assertRaises(ValueError): validate_output(text, 'on', 4, changed)
        with self.assertRaises(ValueError): validate_output(text, 'on', 4, outputs[:-1])
        rows[1]['variant'] = 'mx4'
        with self.assertRaises(ValueError): validate_output('\n'.join(map(json.dumps, rows)), 'on', 4, outputs)

    def test_original_integer_speedup_predicate_applies_to_every_primary_process(self):
        self.assertEqual(KERNEL_ORDER, ['control', 'row1', 'row1', 'control'])
        self.assertTrue(timing_evidence(cases())['kernel_speedup_prediction_supported'])
        for shape in PRIMARY_SHAPES:
            for process in range(8, 12):
                value = cases()
                value[process]['observed']['rows'][shape*4+1]['timed_ns'] = [1901] * 8
                with self.subTest(shape=shape, process=process):
                    self.assertFalse(timing_evidence(value)['same_process_kernel_prediction_supported'])

    def test_observer_bounds_and_order_cannot_be_replaced_by_kernel_speed(self):
        value = cases()
        value[9]['observed']['rows'][4*4]['timed_ns'] = [2101] * 8
        result = timing_evidence(value)
        self.assertTrue(result['same_process_kernel_prediction_supported'])
        self.assertFalse(result['bounded_observer_interpretation_supported'])
        self.assertFalse(result['kernel_speedup_prediction_supported'])
        value = copy.deepcopy(cases()); value[0]['observed']['rows'][0]['worker_intervals_ns'] = [[{}]]
        with self.assertRaises(ValueError): timing_evidence(value)
        with self.assertRaises(ValueError): timing_evidence(cases()[1:])


if __name__ == '__main__': unittest.main()
