"""Source-derived framing descriptions; no neural model or learned projection."""
import numpy as np


def native_schedule(available_after_push, final_frames):
    """Describe the fixed 9/4 native scheduler from observed FE row counts.

    This does not execute DiarStream or its cache. The probe observes FE row
    readiness; these intervals are a source-derived diagnostic description.
    """
    available = list(available_after_push)
    if not available or any(type(x) is not int or x < 0 for x in available):
        raise ValueError('invalid observed frontend readiness')
    if available != sorted(available) or type(final_frames) is not int or final_frames < available[-1]:
        raise ValueError('frontend readiness must be monotonic')
    cursor, rows = 0, []
    for frames in available:
        while cursor + 104 <= frames:
            rows.append({'start': cursor, 'emitted_end': cursor + 72,
                         'input_end': cursor + 104, 'phase': 'feed'})
            cursor += 72
    while cursor < final_frames:
        end = min(cursor + 72, final_frames)
        rows.append({'start': cursor, 'emitted_end': end,
                     'input_end': min(end + 32, final_frames), 'phase': 'finish'})
        cursor = end
    return rows


def stack_inputs(features, mask):
    """Capture HF/native feature-stack inputs before the learned projection.

    Both pinned embedders pad to eight Mel rows with numeric zero. HF downsamples
    the *unpadded* Mel mask with ::8; no synthetic extra encoder token is added.
    """
    features, mask = np.asarray(features), np.asarray(mask)
    if (features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 128
            or len(features) < 1 or not np.isfinite(features).all()
            or mask.shape != (len(features),) or not np.isin(mask, [0, 1]).all()):
        raise ValueError('invalid feature/mask input')
    padding = -len(features) % 8
    padded = np.pad(features, ((0, padding), (0, 0)))
    stacked = padded.reshape(-1, 8 * 128).astype('<f4', copy=False)
    encoder_mask = mask[::8].astype(np.uint8)
    if len(encoder_mask) != len(stacked):
        raise ValueError('stack/mask encoder shape differs')
    return stacked, encoder_mask, padding
