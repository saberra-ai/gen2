"""Validate full-recording development turns before any rounding clamp."""
import math


def production_turns(turns):
    if not isinstance(turns, list) or len(turns) > 100000:
        raise ValueError('invalid development native turn list')
    result = []
    for turn in turns:
        start, end, speaker = turn['start_seconds'], turn['end_seconds'], turn['speaker']
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in [start, end])
                or type(speaker) is not int or not 1 <= speaker <= 8
                or not 0 <= start <= end <= 300. + 1e-4):
            raise ValueError('development native turn exceeds fixed recording bounds')
        left, right = min(start, 300.), min(end, 300.)
        if left < right:
            result.append({'start': left, 'end': right, 'speaker': speaker})
    return result
