"""Bounded model-free Q8 tile outputs and integer surrounding-control decisions."""
import hashlib
import json
import math
import struct

ORDER = ['control', 'tile2', 'tile2', 'control']
SHAPES = [(1,2,32),(2,3,64),(5,7,96),(7,5,160),(512,33,512),(2048,33,512),(512,264,512)]
REPETITIONS = 8
PRIMARY_SHAPES = [4, 5, 6]


def validate_output(text, variant, workers, outputs):
    if variant not in ('control', 'tile2') or type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError('fixed tile variant and worker control required')
    if len(text.encode()) > 1024**2 or len(outputs) != len(SHAPES):
        raise ValueError('bounded complete tile output required')
    rows = [json.loads(line) for line in text.splitlines()]
    if len(rows) != len(SHAPES):
        raise ValueError('all seven ordered shape observations required')
    identities = []
    for index, ((m,n,k), row, raw) in enumerate(zip(SHAPES, rows, outputs)):
        expected = {'shape': index, 'm': m, 'n': n, 'k': k, 'workers': workers,
            'workers_started': workers, 'unsupported_support_controls': 3,
            'variant': variant, 'bytes': m*n*4, 'warmups': 1,
            'repetitions': REPETITIONS}
        if set(row) != set(expected) | {'timed_ns'} or any(row[name] != value or type(row[name]) is not type(value)
                for name, value in expected.items()):
            raise ValueError('exact actual shape/variant/worker/iteration evidence required')
        times = row['timed_ns']
        if (not isinstance(times, list) or len(times) != REPETITIONS
                or any(type(value) is not int or not 0 < value <= 600000000000 for value in times)):
            raise ValueError('eight positive bounded integer kernel timings required')
        if len(raw) != m*n*4:
            raise ValueError('complete compact binary32 tile vector required')
        values = [value for value, in struct.iter_unpack('<f', raw)]
        if not all(math.isfinite(value) and abs(value) < 2**70 for value in values) or not any(value != 0 for value in values):
            raise ValueError('finite nonzero tile outputs without unwritten guards required')
        identities.append({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return {'rows': rows, 'outputs': identities}


def timing_evidence(cases):
    if (len(cases) != 4 or [case['variant'] for case in cases] != ORDER
            or any(type(case['workers']) is not int or case['workers'] != 4 for case in cases)):
        raise ValueError('four ordered four-worker observations required')
    shapes = []
    for index in range(len(SHAPES)):
        totals = []
        for case in cases:
            rows = case['observed']['rows']
            if len(rows) != len(SHAPES) or rows[index]['shape'] != index:
                raise ValueError('complete ordered timing rows required')
            times = rows[index]['timed_ns']
            if len(times) != REPETITIONS or any(type(value) is not int or not 0 < value <= 600000000000 for value in times):
                raise ValueError('positive bounded integer timing rows required')
            totals.append(sum(times))
        drift = 10 * abs(totals[3] - totals[0]) <= totals[0]
        faster = 20 * max(totals[1:3]) <= 19 * min(totals[0], totals[3])
        shapes.append({'shape': index, 'eight_call_totals_ns': totals,
            'control_drift_fraction': abs(totals[3] - totals[0]) / totals[0],
            'control_drift_at_most_10_percent': drift,
            'both_candidates_at_least_5_percent_faster_than_both_controls': faster,
            'prediction_supported': drift and faster})
    return {'shapes': shapes, 'primary_shapes': PRIMARY_SHAPES,
        'kernel_speedup_prediction_supported': all(shapes[index]['prediction_supported'] for index in PRIMARY_SHAPES),
        'limits': 'Model-free fixed Q8 matrices and harness dispatch only; no model, RTF, public runtime or release qualification. Retain all timings; no outlier removal.'}
