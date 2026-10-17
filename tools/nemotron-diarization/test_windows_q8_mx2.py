"""Private packed-B-scale 2x2 isolation and unchanged complete numerical/timing gates."""
import copy
import json
from pathlib import Path
import struct
import unittest

from q8_mx2_source import control_source, transform, mx2_kernel, OLD_DELTA, NEW_DELTA
from q8_mx2_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from windows_q8_model_sources import candidate_source, GUARD
from windows_q8_mx2_metrics import validate_output, timing_evidence, KERNEL_ORDER
from test_windows_q8_worker_timing import rows as parent_rows, cases as parent_cases
from q8_mx4_geometry import SHAPES, PRIMARY_SHAPES

TOOLS = Path(__file__).resolve().parent


def relabel(rows):
    for row in rows:
        if row['variant'] == 'mx4':
            row['variant'] = 'mx2'
    return rows


def cases():
    values = parent_cases()
    for case in values:
        relabel(case['observed']['rows'])
    return values


class Mx2KernelTest(unittest.TestCase):
    def test_control_is_exact_current_production_kernel_without_embedded_define(self):
        original = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        self.assertEqual(GUARD + control_source(original), candidate_source())
        self.assertIn(b'gemmMx4<2>(0, me, 0, ne);', control_source(original))

    def test_private_packed_helper_and_original_dispatch_region_are_isolated(self):
        original = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        control = control_source(original)
        candidate = transform(original)
        helper = mx2_kernel(control)
        restored = candidate.replace(helper, b'', 1).replace(
            b'gemmMx2<2>(0, me, 0, ne);', b'gemmMx4<2>(0, me, 0, ne);', 1).replace(
            b'GEN2_Q8_MX2_EXPECTED', b'GEN2_Q8_MX4_EXPECTED', 1)
        self.assertEqual(restored, control)
        self.assertEqual(helper.count(NEW_DELTA), 1)
        self.assertNotIn(OLD_DELTA, helper)
        self.assertIn(b'int64_t xtiles = (n - n0) / 2;', helper)
        self.assertIn(b'int64_t jj = n0 + job % xtiles * 2;', helper)
        self.assertIn(b'__m256 Cv[2][RM] = {};', helper)
        self.assertIn(b'for (int64_t j = 0; j < 2; ++j)', helper)
        self.assertIn(b'__m128 db = _mm_cvtph_ps(_mm_set_epi64x(0, b_delta));', helper)
        self.assertIn(b'__m128 da = _mm_set1_ps(unhalf((A[lda * (ii + i) + l].d)));', helper)
        # Both surviving complete ordered contributions must match canonical text.
        for column in (0, 1):
            start = f'                    Cv[{column}][i] = madd('.encode()
            end = f'                                    Cv[{column}][i]);'.encode()
            expected = control[control.index(start, control.index(b'NOINLINE void gemmMx4')):]
            expected = expected[:expected.index(end) + len(end)]
            self.assertIn(expected, helper)
        self.assertNotIn(b'bvec2', helper); self.assertNotIn(b'bvec3', helper)
        self.assertNotIn(b'Cv[2][i]', helper); self.assertNotIn(b'Cv[3][i]', helper)
        for line in (b'if (m >= 2 && n >= 4)', b'const int64_t me = m / 2 * 2;',
                b'const int64_t ne = n / 4 * 4;', b'mnpack(me, m, 0, ne);', b'mnpack(0, m, ne, n);'):
            self.assertIn(line, candidate)
        for bad in (original[:-1], b'changed' + original):
            with self.assertRaises(ValueError): transform(bad)

    def test_all_construction_support_clock_and_metric_changes_are_exact_derivations(self):
        for target, parent, derive in (
                ('windows_q8_mx2_probe.py', 'windows_q8_row_probe.py', derive_probe),
                ('windows_q8_mx2_metrics.py', 'windows_q8_row_metrics.py', derive_metrics),
                ('q8_mx2_benchmark.cmake', 'q8_row_benchmark.cmake', derive_cmake),
                ('q8_mx2_benchmark.cpp', 'q8_row_benchmark.cpp', derive_benchmark)):
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
        self.assertEqual(KERNEL_ORDER, ['control', 'mx2', 'mx2', 'control'])
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
