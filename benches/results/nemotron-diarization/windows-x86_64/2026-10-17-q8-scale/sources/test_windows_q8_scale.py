"""Exact scale-only source isolation, all half lanes, and unchanged timing gates."""
import copy
import json
from pathlib import Path
import struct
import unittest

from q8_scale_source import BROADCAST, OLD_SCALE, NEW_SCALE, control_source, scale_kernel, transform
from q8_scale_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from q8_scale_half_metrics import admit_scale_half_table
from windows_q8_model_sources import candidate_source, GUARD
from windows_q8_scale_metrics import validate_output, timing_evidence, KERNEL_ORDER
from test_windows_q8_worker_timing import rows as parent_rows, cases as parent_cases
from q8_mx4_geometry import SHAPES, PRIMARY_SHAPES

TOOLS = Path(__file__).resolve().parent


def relabel(rows):
    for row in rows:
        if row['variant'] == 'mx4': row['variant'] = 'scale'
    return rows


def cases():
    values = parent_cases()
    for case in values: relabel(case['observed']['rows'])
    return values


def half_outputs():
    return b''.join(struct.pack('<f', struct.unpack('<e', struct.pack('<H', i))[0]) * 4
        for i in range(65536))


class ScaleKernelTest(unittest.TestCase):
    def test_exact_production_control_and_private_helper_reconstruction(self):
        original = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        control = control_source(original)
        self.assertEqual(GUARD + control, candidate_source())
        helper = scale_kernel(control)
        canonical = control.split(b'    // Templated functions for gemm of dimensions Mx4\n', 1)[1].split(
            b'\n#endif\n\n    template <int RM, int RN>', 1)[0]
        guard = b'#if defined(_WIN32) && defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)\n'
        self.assertEqual(helper, guard + canonical.replace(b'void gemmMx4(', b'void gemmScaleMx4(', 1).replace(
            OLD_SCALE, NEW_SCALE, 1) + b'\n#endif\n\n')
        candidate = transform(original)
        restored = candidate.replace(guard + BROADCAST + b'#endif\n\n', b'', 1).replace(helper, b'', 1).replace(
            b'gemmScaleMx4<2>(0, me, 0, ne);', b'gemmMx4<2>(0, me, 0, ne);', 1).replace(
            b'GEN2_Q8_SCALE_EXPECTED', b'GEN2_Q8_MX4_EXPECTED', 1)
        self.assertEqual(restored, control)
        self.assertIn(b'mnpack(me, m, 0, ne);', candidate)
        self.assertIn(b'mnpack(0, m, ne, n);', candidate)
        for bad in (original[:-1], b'changed' + original):
            with self.assertRaises(ValueError): transform(bad)

    def test_all_experiment_edits_are_checked_derivations(self):
        for target, parent, derive in (
                ('windows_q8_scale_probe.py', 'windows_q8_row_probe.py', derive_probe),
                ('windows_q8_scale_metrics.py', 'windows_q8_row_metrics.py', derive_metrics),
                ('q8_scale_benchmark.cmake', 'q8_row_benchmark.cmake', derive_cmake),
                ('q8_scale_benchmark.cpp', 'q8_row_benchmark.cpp', derive_benchmark)):
            with self.subTest(target=target):
                self.assertEqual((TOOLS / target).read_bytes(), derive((TOOLS / parent).read_bytes()))
                with self.assertRaises(ValueError): derive(b'incomplete parent')

    def test_all_four_lanes_have_independent_complete_binary16_reference(self):
        raw = half_outputs()
        evidence = admit_scale_half_table(raw)
        self.assertEqual(evidence['entries'], 65536)
        self.assertEqual([lane['finite_entries_verified'] for lane in evidence['lanes']], [63488] * 4)
        self.assertFalse(evidence['nan_payload_equivalence_claimed'])
        for half in (0, 1, 0x3c00, 0x7bff, 0x7c00, 0x7e00, 0x8000, 0xfc00):
            for lane in range(4):
                changed = bytearray(raw)
                changed[half * 16 + lane * 4:half * 16 + lane * 4 + 4] = struct.pack('<f', .5)
                with self.subTest(half=half, lane=lane):
                    with self.assertRaises(ValueError): admit_scale_half_table(changed)
        with self.assertRaises(ValueError): admit_scale_half_table(raw[:-1])

    def test_nans_preserve_classification_and_broadcast_identity(self):
        raw = bytearray(half_outputs())
        raw[0x7e00 * 16:0x7e00 * 16 + 16] = struct.pack('<I', 0x7fc01234) * 4
        self.assertTrue(admit_scale_half_table(raw)['nan_classification_verified'])
        raw[0x7e00 * 16:0x7e00 * 16 + 4] = struct.pack('<I', 0x7fc05678)
        with self.assertRaises(ValueError): admit_scale_half_table(raw)

    def test_all_full_phase_vectors_and_tails_remain_required(self):
        rows = relabel(parent_rows('on', 4))
        outputs = [struct.pack('<f', .25) * (m*n) for m, n, k in SHAPES for _ in range(4)]
        text = '\n'.join(map(json.dumps, rows))
        self.assertEqual(len(validate_output(text, 'on', 4, outputs)['outputs']), 88)
        changed = outputs.copy(); changed[1] = b'\0\0\0\0' + changed[1][4:]
        with self.assertRaises(ValueError): validate_output(text, 'on', 4, changed)
        with self.assertRaises(ValueError): validate_output(text, 'on', 4, outputs[:-1])

    def test_every_original_primary_speed_predicate_remains_mandatory(self):
        self.assertEqual(KERNEL_ORDER, ['control', 'scale', 'scale', 'control'])
        self.assertTrue(timing_evidence(cases())['kernel_speedup_prediction_supported'])
        for shape in PRIMARY_SHAPES:
            for process in range(8, 12):
                value = cases()
                value[process]['observed']['rows'][shape*4+1]['timed_ns'] = [1901] * 8
                with self.subTest(shape=shape, process=process):
                    self.assertFalse(timing_evidence(value)['same_process_kernel_prediction_supported'])

    def test_observer_bound_and_complete_order_remain_separate(self):
        value = cases(); value[9]['observed']['rows'][4*4]['timed_ns'] = [2101] * 8
        result = timing_evidence(value)
        self.assertTrue(result['same_process_kernel_prediction_supported'])
        self.assertFalse(result['bounded_observer_interpretation_supported'])
        self.assertFalse(result['kernel_speedup_prediction_supported'])
        value = copy.deepcopy(cases()); value[0]['observed']['rows'][0]['worker_intervals_ns'] = [[{}]]
        with self.assertRaises(ValueError): timing_evidence(value)
        with self.assertRaises(ValueError): timing_evidence(cases()[1:])


if __name__ == '__main__': unittest.main()
