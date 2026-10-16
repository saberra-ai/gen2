"""Fixed excluded-calibration graph comparisons, no neural execution."""
import numpy as np

from frontend_metrics import difference


def variants(first_raw, final_hf, final_native):
    arrays = [np.asarray(x) for x in [first_raw, final_hf, final_native]]
    for value, shape in zip(arrays, [(105, 128), (39, 128), (40, 128)]):
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('invalid fixed calibration features')
    first_raw, final_hf, final_native = arrays
    if first_raw[-1].any():
        raise ValueError('raw HF first terminal row is not masked zero')
    return {'first104': first_raw[:104].copy(), 'valid39': final_hf.copy(),
            'zero40': np.concatenate([final_hf, np.zeros((1, 128), np.float32)]),
            'extra40': np.concatenate([final_hf, final_native[-1:]])}


def probability_comparisons(native, hf_first, hf_final_full, hf_final_returned):
    expected = {'first104': 104, 'valid39': 112, 'zero40': 112, 'extra40': 112}
    if set(native) != set(expected):
        raise ValueError('fixed native cases differ')
    for value, rows in [(native[k], n) for k, n in expected.items()] + [
            (hf_first, 104), (hf_final_full, 112), (hf_final_returned, 39)]:
        value = np.asarray(value)
        if (value.dtype != np.float32 or value.shape != (rows, 8)
                or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any()):
            raise ValueError('invalid fixed graph probabilities')
    # Full-sequence predictions include nine cached encoder rows (72 logits).
    # Keep the synthetic terminal output row111, even when comparing returned39.
    pairs = {'first_full_native_hf': (native['first104'], hf_first),
             'valid39_full_native_hf': (native['valid39'], hf_final_full),
             'zero40_full_native_hf': (native['zero40'], hf_final_full),
             'extra40_full_native_hf': (native['extra40'], hf_final_full),
             'valid39_zero40_full_native': (native['valid39'], native['zero40']),
             'extra40_zero40_full_native': (native['extra40'], native['zero40']),
             'hf_full_returned_prefix': (hf_final_full[72:111], hf_final_returned)}
    for key in ['valid39', 'zero40', 'extra40']:
        pairs[key + '_returned_native_hf'] = (native[key][72:111], hf_final_returned)
    return {key: difference(left, right, 0) for key, (left, right) in pairs.items()}
