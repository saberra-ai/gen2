"""Saved-development replay of pinned native segmentation; no model execution."""
import math
import struct

import numpy as np


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


DEFAULT = {'onset': f32(.641), 'offset': f32(.561), 'pad_onset': .229,
           'pad_offset': .079, 'min_duration_on': .511, 'min_duration_off': .296}
CADENCE = f32(.01)
CONTRASTS = {
    'native_default': DEFAULT,
    'without_edge_padding': {**DEFAULT, 'pad_onset': 0., 'pad_offset': 0.},
    'without_gap_fill': {**DEFAULT, 'min_duration_off': 0.},
    'without_short_segment_filter': {**DEFAULT, 'min_duration_on': 0.},
    'shared_thresholds_only': {**DEFAULT, 'onset': .5, 'offset': .5},
}


def native_segments(probabilities, config):
    """Replay diar_segments_from_probs: strict hysteresis, pad, merge, filter.

    The native window_stride is float32 before promotion to double. Preserve
    that cadence, float32 thresholds, operation order and strict inequalities.
    This diagnostic covers full histories before compaction, not streaming
    compaction under alternative policies.
    """
    if (not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.dtype('<f4')
            or probabilities.ndim != 2 or probabilities.shape[1] != 8
            or not 0 < len(probabilities) <= 30000 or not np.isfinite(probabilities).all()
            or not ((0 <= probabilities) & (probabilities <= 1)).all()):
        raise ValueError('bounded finite N-by-eight F32 probabilities required')
    if (set(config) != set(DEFAULT) or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0
                                         for v in config.values())
            or not 0 <= config['offset'] <= config['onset'] <= 1):
        raise ValueError('invalid segmentation config')
    total = len(probabilities) * CADENCE
    output = []
    for speaker in range(8):
        active, start, intervals = False, 0, []
        for frame, probability in enumerate(probabilities[:,speaker]):
            if not active and probability > config['onset']:
                active, start = True, frame
            elif active and probability < config['offset']:
                active = False
                intervals.append((max(0.,start * CADENCE - config['pad_onset']),
                                  min(total,frame * CADENCE + config['pad_offset'])))
        if active:
            intervals.append((max(0.,start * CADENCE - config['pad_onset']),total))
        merged = []
        for left,right in intervals:
            if merged and left - merged[-1][1] < config['min_duration_off']:
                merged[-1] = (merged[-1][0],max(merged[-1][1],right))
            else:
                merged.append((left,right))
        output.extend({'start':left,'end':right,'speaker':speaker+1} for left,right in merged
                      if right-left >= config['min_duration_on'])
    return sorted(output,key=lambda s:(s['start'],s['end'],s['speaker']))


def compare_native_turns(actual, expected):
    if any(left['start_seconds'] > right['start_seconds'] for left,right in zip(expected,expected[1:])):
        raise ValueError('native turns are not in nondecreasing start order')
    expected = sorted(({'start':s['start_seconds'],'end':s['end_seconds'],'speaker':s['speaker']}
                       for s in expected),key=lambda s:(s['start'],s['end'],s['speaker']))
    if actual != expected:
        raise ValueError('default segmentation does not exactly reproduce native turns')
    return {'turns':len(actual),'all_boundaries_and_speakers_exact':True,
            'equal_start_order_canonicalized':True}
