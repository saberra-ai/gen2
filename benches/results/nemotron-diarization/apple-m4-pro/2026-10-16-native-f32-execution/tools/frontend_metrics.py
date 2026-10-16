"""Numerical frontend comparisons, separate from diarization quality scoring."""
import numpy as np


def difference(left, right, tolerance):
    if left.shape != right.shape or left.ndim != 2 or not left.size:
        raise ValueError('frontend tensors must have equal nonempty matrix shapes')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('frontend tensors must be finite')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('frontend tolerance must be finite and nonnegative')
    error = np.abs(left.astype(np.float64) - right.astype(np.float64))
    exceeds = np.argwhere(error > tolerance)
    worst = np.unravel_index(np.argmax(error), error.shape)
    return {'shape': list(left.shape), 'bitwise_equal': left.dtype == right.dtype and left.tobytes() == right.tobytes(),
            'maximum_absolute_error': float(error.max()), 'mean_absolute_error': float(error.mean()),
            'root_mean_square_error': float(np.sqrt(np.mean(error * error))),
            'p95_absolute_error': float(np.quantile(error, .95)), 'p99_absolute_error': float(np.quantile(error, .99)),
            'worst_frame_feature': [int(x) for x in worst],
            'first_frame_feature_over_tolerance': [int(x) for x in exceeds[0]] if len(exceeds) else None,
            'values_over_tolerance': len(exceeds), 'absolute_tolerance': tolerance}


def valid_prefix(features, mask, native):
    if features.ndim != 2 or features.shape[1] != 128 or mask.shape != features.shape[:1]:
        raise ValueError('unexpected HF frontend/mask shape')
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('frontend mask must contain only zero or one')
    valid = int(mask.sum())
    if not valid or not np.array_equal(mask, np.arange(len(mask)) < valid):
        raise ValueError('frontend mask must be a nonempty contiguous valid prefix')
    if native.shape != features.shape:
        raise ValueError('raw centered native/HF frame shapes differ')
    return native[:valid], features[:valid], valid
