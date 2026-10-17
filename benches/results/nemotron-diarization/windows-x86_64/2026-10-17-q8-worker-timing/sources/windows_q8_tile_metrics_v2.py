"""Versioned half-table and fail-closed support admission; original timing gates."""
import hashlib
import json
import math
import struct
from windows_q8_tile_metrics import ORDER, SHAPES, REPETITIONS, PRIMARY_SHAPES, timing_evidence

GUARD_CODES = {'barrier': 73, 'set': 74, 'add': 75}


def admit_guard(name, code, stdout, stderr):
    if (name not in GUARD_CODES or type(code) is not int or code != GUARD_CODES[name]
            or stdout != b'' or stderr.replace(b'\r\n', b'\n') != ('GEN2_Q8_UNSUPPORTED_' + name + '\n').encode()):
        raise ValueError('exact fail-closed support exit and diagnostic required')
    return {'name': name, 'returncode': code, 'rejection_observed': True}


def admit_half_table(raw):
    if len(raw) != 65536 * 4:
        raise ValueError('complete binary32 half table required')
    finite = 0
    for index, (value,) in enumerate(struct.iter_unpack('<f', raw)):
        expected, = struct.unpack('<e', struct.pack('<H', index))
        actual = raw[index*4:index*4+4]
        if math.isnan(expected):
            if not math.isnan(value): raise ValueError('half NaN classification changed')
        elif actual != struct.pack('<f', expected):
            raise ValueError('half conversion differs from independent binary16 reference')
        if math.isfinite(expected): finite += 1
    if finite != 63488: raise ValueError('complete finite half coverage required')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'entries': 65536, 'finite_entries_verified': finite}


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
            'workers_started': workers, 'half_table_entries': 65536, 'finite_half_entries_checked': 63488,
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
