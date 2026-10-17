"""Strict same-process Q8 comparisons and separately bounded worker telemetry."""
import hashlib
import json

from q8_mx4_geometry import SHAPES, PRIMARY_SHAPES
from windows_q8_mx4_metrics import validate_output as validate_reference
from windows_q8_tile_metrics_v2 import GUARD_CODES, admit_guard, admit_half_table

ORDER = ['off', 'on', 'on', 'off']
KERNEL_ORDER = ['control', 'scale', 'scale', 'control']
REPETITIONS = 8
MAX_NS = 600_000_000_000


def validate_rows(rows, mode, workers):
    if mode not in ORDER or type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError('fixed observer mode and worker count required')
    if type(rows) is not list or len(rows) != len(SHAPES) * 4:
        raise ValueError('all 88 ordered shape/kernel phases required')
    for index, (m, n, k) in enumerate(SHAPES):
        for phase, variant in enumerate(KERNEL_ORDER):
            row = rows[index * 4 + phase]
            expected = {'shape': index, 'm': m, 'n': n, 'k': k, 'phase': phase,
                'variant': variant, 'telemetry_mode': mode, 'workers': workers,
                'workers_started': workers, 'half_table_entries': 65536,
                'finite_half_entries_checked': 63488, 'bytes': m*n*4,
                'warmups': 1, 'repetitions': REPETITIONS}
            if (type(row) is not dict or set(row) != set(expected) | {'timed_ns', 'worker_intervals_ns'}
                    or any(type(row[name]) is not type(value) or row[name] != value
                           for name, value in expected.items())):
                raise ValueError('exact phase/shape/kernel/observer schema required')
            times, intervals = row['timed_ns'], row['worker_intervals_ns']
            if (type(times) is not list or len(times) != REPETITIONS
                    or any(type(t) is not int or not 0 < t <= MAX_NS for t in times)):
                raise ValueError('eight positive bounded dispatch timings required')
            if type(intervals) is not list:
                raise ValueError('explicit worker interval list required')
            if mode == 'off':
                if intervals:
                    raise ValueError('disabled worker clocks must emit no intervals')
                continue
            if len(intervals) != REPETITIONS:
                raise ValueError('every timed dispatch needs its complete worker intervals')
            for elapsed, sample in zip(times, intervals):
                if type(sample) is not list or len(sample) != workers:
                    raise ValueError('complete ordered worker coverage required')
                for ith, interval in enumerate(sample):
                    if (type(interval) is not dict or set(interval) != {'worker', 'start_ns', 'end_ns'}
                            or any(type(value) is not int for value in interval.values())
                            or interval['worker'] != ith
                            or not 0 <= interval['start_ns'] <= interval['end_ns'] <= elapsed):
                        raise ValueError('worker identity/interval must fit whole dispatch')


def validate_output(text, mode, workers, outputs):
    if len(text.encode()) > 4 * 1024**2:
        raise ValueError('bounded worker timing stdout required')
    rows = [json.loads(line) for line in text.splitlines()]
    validate_rows(rows, mode, workers)
    if len(outputs) != len(rows):
        raise ValueError('all 88 full output vectors required')
    reference_rows, reference_outputs = [], []
    for index in range(len(SHAPES)):
        first = rows[index * 4]
        raw = outputs[index * 4]
        if any(value != raw for value in outputs[index*4:index*4+4]):
            raise ValueError('all same-process kernel phase bytes must match')
        reference_rows.append({name: value for name, value in first.items()
                               if name not in ('phase', 'telemetry_mode', 'worker_intervals_ns')})
        reference_outputs.append(raw)
    # Existing vector admission checks every logical float once per shape;
    # the other three complete vectors are independently compared byte for byte.
    validate_reference('\n'.join(map(json.dumps, reference_rows)), 'control', workers, reference_outputs)
    return {'rows': rows, 'outputs': [
        {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for raw in outputs]}


def kernel_comparison(rows):
    result = []
    for index in range(len(SHAPES)):
        totals = [sum(rows[index*4+phase]['timed_ns']) for phase in range(4)]
        drift = 10 * abs(totals[3] - totals[0]) <= totals[0]
        faster = 20 * max(totals[1:3]) <= 19 * min(totals[0], totals[3])
        result.append({'shape': index, 'eight_call_totals_ns': totals,
            'control_drift_at_most_10_percent': drift,
            'both_candidates_at_least_5_percent_faster_than_both_controls': faster,
            'candidate_repeat_drift_at_most_10_percent': 10 * abs(totals[2] - totals[1]) <= totals[1],
            'prediction_supported': drift and faster})
    return result


def worker_observations(rows):
    summaries = []
    for row in rows:
        samples = []
        for elapsed, intervals in zip(row['timed_ns'], row['worker_intervals_ns']):
            starts = [v['start_ns'] for v in intervals]
            ends = [v['end_ns'] for v in intervals]
            maximum = max(v['end_ns'] - v['start_ns'] for v in intervals)
            samples.append({'whole_dispatch_ns': elapsed, 'maximum_worker_wall_ns': maximum,
                'worker_start_spread_ns': max(starts) - min(starts),
                'completion_tail_ns': elapsed - max(ends),
                'whole_minus_maximum_worker_ns': elapsed - maximum})
        if samples:
            summaries.append({'shape': row['shape'], 'phase': row['phase'], 'samples': samples})
    return summaries


def timing_evidence(cases):
    if type(cases) is not list or len(cases) != 12:
        raise ValueError('all twelve ordered observer processes required')
    processes = []
    for position, case in enumerate(cases):
        workers, index = (1, 2, 4)[position // 4], position % 4
        for key, expected in [('workers', workers), ('index', index), ('telemetry_mode', ORDER[index])]:
            if type(case.get(key)) is not type(expected) or case[key] != expected:
                raise ValueError('exact fresh-process worker/mode order required')
        rows = case['observed']['rows']
        validate_rows(rows, ORDER[index], workers)
        processes.append({'workers': workers, 'index': index, 'telemetry_mode': ORDER[index],
            'kernel_comparison': kernel_comparison(rows), 'worker_observations': worker_observations(rows)})
    observer = []
    for shape in range(len(SHAPES)):
        for phase in range(4):
            totals = [sum(case['observed']['rows'][shape*4+phase]['timed_ns']) for case in cases[-4:]]
            drift = 10 * abs(totals[3] - totals[0]) <= totals[0]
            bounded = 20 * max(totals[1:3]) <= 21 * min(totals[0], totals[3])
            observer.append({'shape': shape, 'phase': phase, 'off_on_on_off_totals_ns': totals,
                'control_drift_at_most_10_percent': drift,
                'both_observers_at_most_5_percent_over_both_controls': bounded,
                'bounded_interpretation_supported': drift and bounded})
    bounded = all(row['bounded_interpretation_supported'] for row in observer if row['shape'] in PRIMARY_SHAPES)
    faster = all(process['kernel_comparison'][shape]['prediction_supported']
                 for process in processes[-4:] for shape in PRIMARY_SHAPES)
    return {'processes': processes, 'observer_comparison': observer, 'primary_shapes': PRIMARY_SHAPES,
        'same_process_kernel_prediction_supported': faster,
        'bounded_observer_interpretation_supported': bounded,
        'kernel_speedup_prediction_supported': faster and bounded,
        'limits': 'Worker wall intervals include preemption, stalls and observer cost. No CPU-time or causal classification; no SDK/model/RTF/production/release qualification.'}
