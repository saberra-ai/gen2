"""Three-way probability diagnostics with explicit native/HF terminal alignment."""
import numpy as np

from frontend_metrics import difference


def comparisons(f32, q8, hf, samples):
    if not isinstance(samples, int) or not 320 <= samples <= 1920000:
        raise ValueError('fixed CPU control requires 20ms–120s of PCM')
    valid = samples // 160
    for name, array, rows in [('native F32', f32, valid + 1), ('native Q8', q8, valid + 1), ('HF FP32', hf, valid)]:
        if array.shape != (rows, 8) or not np.isfinite(array).all() or not ((array >= 0) & (array <= 1)).all():
            raise ValueError('invalid ' + name + ' probabilities or terminal shape')

    def report(left, right):
        value = difference(left, right, 0)
        value['worst_frame_speaker_channel'] = value.pop('worst_frame_feature')
        value['first_unequal_frame_speaker_channel'] = value.pop('first_frame_feature_over_tolerance')
        return value

    return {'native_f32_vs_hf_fp32_common_valid': report(f32[:valid], hf),
            'native_q8_vs_hf_fp32_common_valid': report(q8[:valid], hf),
            'native_f32_vs_native_q8_all_rows': report(f32, q8),
            'native_terminal_frame_retained': {'frame': valid, 'f32': f32[-1].tolist(), 'q8': q8[-1].tolist()},
            'hf_valid_rows': valid, 'native_raw_rows': valid + 1,
            'rule': 'Zero numeric tolerance describes equality only; no quality/release acceptance threshold.'}
