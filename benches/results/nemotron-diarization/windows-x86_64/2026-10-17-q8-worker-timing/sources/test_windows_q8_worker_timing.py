"""Reject incomplete worker evidence and keep observer/kernel decisions separate."""
import copy
import json
from pathlib import Path
import struct
import unittest

from q8_mx4_geometry import SHAPES, PRIMARY_SHAPES
from q8_worker_timing_source import derive_probe
from windows_q8_worker_timing_metrics import (
    ORDER, KERNEL_ORDER, REPETITIONS, timing_evidence, validate_rows, validate_output)

TOOLS = Path(__file__).resolve().parent


def rows(mode, workers, times=(2000, 1900, 1900, 2000)):
    result = []
    for shape, (m, n, k) in enumerate(SHAPES):
        for phase, variant in enumerate(KERNEL_ORDER):
            elapsed = times[phase]
            intervals = [[{'worker': ith, 'start_ns': 10 + ith, 'end_ns': elapsed - 10 - ith}
                          for ith in range(workers)] for _ in range(REPETITIONS)] if mode == 'on' else []
            result.append({'shape': shape, 'm': m, 'n': n, 'k': k, 'phase': phase, 'variant': variant,
                'telemetry_mode': mode, 'workers': workers, 'workers_started': workers,
                'half_table_entries': 65536, 'finite_half_entries_checked': 63488,
                'bytes': m*n*4, 'warmups': 1, 'repetitions': REPETITIONS,
                'timed_ns': [elapsed]*REPETITIONS, 'worker_intervals_ns': intervals})
    return result


def cases():
    return [{'workers': workers, 'index': index, 'telemetry_mode': mode,
             'observed': {'rows': rows(mode, workers)}}
            for workers in (1, 2, 4) for index, mode in enumerate(ORDER)]


class WorkerTimingTest(unittest.TestCase):
    def test_complete_vectors_admit_and_phase_byte_difference_rejects(self):
        vectors = [struct.pack('<f', .25)*(m*n) for m, n, k in SHAPES]
        outputs = [value for value in vectors for _ in range(4)]
        text = '\n'.join(map(json.dumps, rows('on', 4)))
        result = validate_output(text, 'on', 4, outputs)
        self.assertEqual(len(result['outputs']), 88)
        changed = outputs.copy()
        changed[1] = struct.pack('<f', -.25) + changed[1][4:]
        with self.assertRaises(ValueError): validate_output(text, 'on', 4, changed)
        with self.assertRaises(ValueError): validate_output(text, 'on', 4, outputs[:-1])

    def test_missing_reordered_unbounded_or_disabled_worker_intervals_reject(self):
        original = rows('on', 4)
        for failure in ('row', 'phase', 'mode', 'worker', 'boolean', 'start', 'end', 'zero', 'sample', 'extra', 'duration'):
            value = copy.deepcopy(original)
            row = value[0]
            if failure == 'row': value.pop()
            if failure == 'phase': row['phase'] = 1
            if failure == 'mode': row['telemetry_mode'] = 'off'
            if failure == 'worker': row['worker_intervals_ns'][0].reverse()
            if failure == 'boolean': row['worker_intervals_ns'][0][0]['start_ns'] = False
            if failure == 'start': row['worker_intervals_ns'][0][0]['start_ns'] = -1
            if failure == 'end': row['worker_intervals_ns'][0][0]['end_ns'] = 2001
            if failure == 'zero': row['worker_intervals_ns'][0][0]['end_ns'] = 9
            if failure == 'sample': row['worker_intervals_ns'].pop()
            if failure == 'extra': row['worker_intervals_ns'][0][0]['cpu_time'] = 1
            if failure == 'duration': row['timed_ns'][0] = 600_000_000_001
            with self.subTest(failure=failure), self.assertRaises(ValueError): validate_rows(value, 'on', 4)
        disabled = rows('off', 4)
        disabled[0]['worker_intervals_ns'] = original[0]['worker_intervals_ns']
        with self.assertRaises(ValueError): validate_rows(disabled, 'off', 4)
        below_resolution = rows('on', 4)
        below_resolution[0]['worker_intervals_ns'][0][3]['end_ns'] = 13
        validate_rows(below_resolution, 'on', 4)

    def test_original_speedup_and_drift_bounds_apply_to_every_primary_and_process(self):
        self.assertTrue(timing_evidence(cases())['kernel_speedup_prediction_supported'])
        for shape in PRIMARY_SHAPES:
            value = cases()
            value[8]['observed']['rows'][shape*4+1]['timed_ns'] = [1901]*REPETITIONS
            with self.subTest(shape=shape):
                self.assertFalse(timing_evidence(value)['same_process_kernel_prediction_supported'])
        for position in range(8, 12):
            value = cases()
            value[position]['observed']['rows'][4*4+3]['timed_ns'] = [2201]*REPETITIONS
            with self.subTest(position=position):
                self.assertFalse(timing_evidence(value)['same_process_kernel_prediction_supported'])

    def test_observer_bound_is_independent_and_exact_integer_boundary_is_inclusive(self):
        value = cases()
        for position in (9, 10):
            value[position]['observed']['rows'] = rows('on', 4, (2100, 1995, 1995, 2100))
        proof = timing_evidence(value)
        self.assertTrue(proof['bounded_observer_interpretation_supported'])
        self.assertTrue(proof['same_process_kernel_prediction_supported'])
        value[9]['observed']['rows'][4*4]['timed_ns'] = [2101]*REPETITIONS
        proof = timing_evidence(value)
        self.assertTrue(proof['same_process_kernel_prediction_supported'])
        self.assertFalse(proof['bounded_observer_interpretation_supported'])
        self.assertFalse(proof['kernel_speedup_prediction_supported'])
        value = cases()
        value[11]['observed']['rows'][4*4]['timed_ns'] = [2200]*REPETITIONS
        self.assertTrue(timing_evidence(value)['bounded_observer_interpretation_supported'])
        value[11]['observed']['rows'][4*4]['timed_ns'] = [2201]*REPETITIONS
        self.assertFalse(timing_evidence(value)['bounded_observer_interpretation_supported'])

    def test_wall_summaries_are_derived_from_complete_intervals_without_cpu_time_claim(self):
        proof = timing_evidence(cases())
        self.assertEqual(proof['processes'][0]['worker_observations'], [])
        sample = proof['processes'][9]['worker_observations'][0]['samples'][0]
        self.assertEqual(sample, {'whole_dispatch_ns': 2000, 'maximum_worker_wall_ns': 1980,
            'worker_start_spread_ns': 3, 'completion_tail_ns': 10, 'whole_minus_maximum_worker_ns': 20})
        value = cases(); value[9]['telemetry_mode'] = 'off'
        with self.assertRaises(ValueError): timing_evidence(value)
        value = cases(); value[0]['index'] = False
        with self.assertRaises(ValueError): timing_evidence(value)
        with self.assertRaises(ValueError): timing_evidence(cases()[:-1])

    def test_construction_and_support_sources_remain_exact_parent_derivations(self):
        self.assertEqual((TOOLS/'windows_q8_worker_timing_probe.py').read_bytes(),
                         derive_probe((TOOLS/'windows_q8_mx4_probe.py').read_bytes()))
        self.assertEqual((TOOLS/'q8_worker_timing_benchmark.cmake').read_bytes(),
                         (TOOLS/'q8_mx4_benchmark.cmake').read_bytes().replace(
                             b'project(gen2_q8_mx4_control', b'project(gen2_q8_worker_timing_control'))
        original = (TOOLS/'q8_mx4_benchmark.cpp').read_bytes()
        actual = (TOOLS/'q8_worker_timing_benchmark.cpp').read_bytes()
        self.assertEqual(original.split(b'using Clock =')[0].split(b'struct Work {')[0],
                         actual.split(b'using Clock =')[0])
        self.assertEqual(original.split(b'class Pool {')[1].split(b'int main(')[0],
                         actual.split(b'class Pool {')[1].split(b'int main(')[0])


if __name__ == '__main__': unittest.main()
