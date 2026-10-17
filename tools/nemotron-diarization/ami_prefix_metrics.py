"""Fixed AMI prefix contrast; preserve the target, scorer and failed release gate."""
import math

import numpy as np

from compare import segments
from diagnose_ami_reference import official_der
from score import der


def clip_turns(turns, start, end):
    if not all(math.isfinite(x) for x in [start, end]) or not 0 <= start < end:
        raise ValueError('invalid diagnostic window')
    result = []
    for row in turns:
        left, right, speaker = row['start_seconds'], row['end_seconds'], row['speaker']
        if (not all(math.isfinite(x) for x in [left, right]) or left < 0 or right < left
                or type(speaker) is not int or not 1 <= speaker <= 8):
            raise ValueError('invalid diagnostic native turn')
        left, right = max(left, start), min(right, end)
        if left < right:
            result.append({'start': left - start, 'end': right - start, 'speaker': speaker})
    return result


def prefix_target(raw):
    if len(raw) != 42000 * 8 * 4:
        raise ValueError('prefix must retain all 42000 valid frames')
    values = np.frombuffer(raw, dtype='<f4').reshape(42000, 8)
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError('invalid prefix probabilities')
    # Exact full rows: [300,420) seconds on a 10 ms frame grid. No prefix labels.
    return values[30000:42000]


def contrast(engine, cold, warm, cold_raw, warm_raw, references):
    cold_values = np.frombuffer(cold_raw, dtype='<f4')
    if cold_values.size != 12000 * 8:
        raise ValueError('cold crop must retain every 12000 valid frames')
    cold_values = cold_values.reshape(12000, 8)
    if not np.isfinite(cold_values).all() or not ((0 <= cold_values) & (cold_values <= 1)).all():
        raise ValueError('invalid cold crop probabilities')
    warm_values = prefix_target(warm_raw)
    hypotheses = {
        'cold_shared': segments(cold_values),
        'cold_production': clip_turns(cold['turns'], 0., 120.),
        'prefix_shared': segments(warm_values),
        'prefix_production': clip_turns(warm['turns'], 300., 420.),
    }
    result = {}
    for policy, reference in references.items():
        values = {name: {'frame': der(reference, hyp, 120.),
                         'official': official_der(engine, reference, hyp, 120.)}
                  for name, hyp in hypotheses.items()}
        differences = {}
        for mode in ['shared', 'production']:
            differences[mode] = {}
            for scorer in ['frame', 'official']:
                left, right = values['cold_' + mode][scorer], values['prefix_' + mode][scorer]
                if abs(left['reference_speaker_seconds'] - right['reference_speaker_seconds']) > 1e-9:
                    raise ValueError('diagnostic changed reference denominator')
                differences[mode][scorer] = {key: right[key] - left[key] for key in
                    ['der', 'missed_speaker_seconds', 'false_alarm_speaker_seconds', 'confusion_speaker_seconds']}
        result[policy] = {'scores': values, 'prefix_minus_cold': differences}
    return {'target_seconds_in_source': [300, 420], 'scoring_uem_seconds': [0, 120],
            'target_probability_rows': len(warm_values), 'reference_policies': result,
            'original_gate_status_changed': False,
            'warm_fp32_or_q8_degradation_bound_evaluated': False}
