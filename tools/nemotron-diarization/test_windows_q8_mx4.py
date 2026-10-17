"""Captured model geometry, exact dispatch delta and strict surrounding controls."""
import copy
import json
from pathlib import Path
import struct
import unittest

from q8_mx4_geometry import SHAPES, PRIMARY_SHAPES, admit_geometry
from q8_mx4_source import CLASS, OLD, NEW, transform
from windows_q8_mx4_metrics import ORDER, REPETITIONS, timing_evidence, validate_output


TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
RESULT = REPO / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-matrix-batch-profile'


class Q8Mx4Test(unittest.TestCase):
    def test_actual_qualified_profiles_select_identical_geometries_on_every_worker(self):
        result = admit_geometry(json.loads((RESULT / 'collection-qualification.json').read_text()))
        self.assertEqual(result['primary_columns'], [13, 292, 535])
        self.assertEqual(result['profiled_workers_checked'], 8)
        self.assertEqual(len(SHAPES), 22)
        self.assertEqual(PRIMARY_SHAPES, list(range(4, 16)))

    def test_unbounded_incomplete_or_changed_model_geometry_rejects(self):
        original = json.loads((RESULT / 'collection-qualification.json').read_text())
        for failure in ('timing', 'worker', 'shape', 'roots', 'columns', 'index', 'types'):
            value = copy.deepcopy(original)
            if failure == 'timing':
                value['timing_comparison']['bounded_observer_interpretation_supported'] = False
            elif failure == 'worker':
                value['per_worker_matrix_groups'][0]['per_worker'].pop('3')
            elif failure == 'index':
                value['per_worker_matrix_groups'][0]['index'] = True
            else:
                rows = value['per_worker_matrix_groups'][0]['per_worker']['0']
                row = next(r for r in rows if r['route'] == 4 and r['a'] == [512, 512, 1, 1])
                if failure == 'shape': row['a'][0] = 256
                if failure == 'roots': row['roots'] += 1
                if failure == 'columns': row['b'][1] = 999
                if failure == 'types': row['types'][1] = False
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                admit_geometry(value)

    def test_transform_preserves_helpers_and_changes_only_q8_dispatch(self):
        raw = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        changed = transform(raw)
        prefix, body = raw.split(CLASS, 1)
        self.assertEqual(changed, prefix + CLASS + body.replace(OLD, NEW, 1))
        self.assertIn(b'gemmMx4<2>(0, me, 0, ne);', NEW)
        self.assertIn(b'defined(__F16C__)', NEW)
        for bad in (b'', CLASS + OLD, changed):
            with self.assertRaises(ValueError): transform(bad)

    def rows(self, variant='control', workers=4, time=1000):
        return [{'shape': i, 'm': m, 'n': n, 'k': k, 'workers': workers, 'workers_started': workers,
                 'half_table_entries': 65536, 'finite_half_entries_checked': 63488,
                 'variant': variant, 'bytes': m*n*4, 'warmups': 1, 'repetitions': REPETITIONS,
                 'timed_ns': [time]*REPETITIONS} for i, (m, n, k) in enumerate(SHAPES)]

    def test_complete_new_shape_vectors_and_variants_admit(self):
        outputs = [struct.pack('<f', .25)*(m*n) for m, n, _ in SHAPES]
        for variant, workers in [('control', 1), ('mx4', 4)]:
            result = validate_output('\n'.join(map(json.dumps, self.rows(variant, workers))), variant, workers, outputs)
            self.assertEqual(len(result['outputs']), 22)
        rows = self.rows()
        for changed in [rows[:-1], list(reversed(rows)), [{**rows[0], 'workers_started': True}, *rows[1:]]]:
            with self.assertRaises(ValueError):
                validate_output('\n'.join(map(json.dumps, changed)), 'control', 4, outputs)
        with self.assertRaises(ValueError):
            validate_output('\n'.join(map(json.dumps, rows)), 'control', 4, [outputs[0][:-1], *outputs[1:]])

    def cases(self, times):
        return [{'variant': v, 'workers': 4, 'observed': {'rows': self.rows(v, time=t)}} for v, t in zip(ORDER, times)]

    def test_original_integer_bounds_and_all_twelve_primary_shapes_are_required(self):
        self.assertTrue(timing_evidence(self.cases([1000, 950, 950, 1100]))['kernel_speedup_prediction_supported'])
        for times in ([1000, 951, 950, 1000], [1000, 900, 900, 1101], [1100, 1000, 1000, 1000]):
            self.assertFalse(timing_evidence(self.cases(times))['kernel_speedup_prediction_supported'])
        for shape in PRIMARY_SHAPES:
            cases = self.cases([1000, 900, 900, 1000])
            cases[1]['observed']['rows'][shape]['timed_ns'] = [1000]*REPETITIONS
            self.assertFalse(timing_evidence(cases)['kernel_speedup_prediction_supported'])
        cases = self.cases([1000, 900, 900, 1000]); cases[2]['workers'] = 2
        with self.assertRaises(ValueError): timing_evidence(cases)

    def test_unchanged_harness_retains_actual_guards_and_shape_order(self):
        from q8_mx4_geometry import derive_harness, derive_project
        self.assertEqual((TOOLS / 'q8_mx4_benchmark.cpp').read_bytes(), derive_harness((TOOLS / 'q8_tile_benchmark_v2.cpp').read_bytes()))
        self.assertEqual((TOOLS / 'q8_mx4_benchmark.cmake').read_bytes(), derive_project((TOOLS / 'q8_tile_benchmark_v2.cmake').read_bytes()))


if __name__ == '__main__': unittest.main()
