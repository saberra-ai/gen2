"""Score current valid-Mel outputs against unchanged frozen corpus/oracle gates.

This module does not run inference or replace historical qualification results.
The caller must bind cached oracle arrays/metadata to a reviewed frozen protocol.
"""
import numpy as np

from compare import segments
from compare_stream_valid_mel import compare
from qualify_corpus import GEOMETRY, MODEL_REVISION
from score import der


def score_slice(item, gen2, native, q8_raw, native_raw, fp32, metadata, reference, expected_pcm_digest):
    samples = item['audio_samples']
    duration = samples / 16000
    if type(samples) is not int or not 320 <= samples <= 600 * 16000 or item['uem_seconds'] != [0, duration]:
        raise ValueError('invalid frozen UEM')
    if gen2['accepted_samples'] != samples or metadata['audio_samples'] != samples:
        raise ValueError('inference input differs from frozen slice')
    for output in [gen2, native, metadata]:
        if output.get('input_pcm_f32le_sha256') != expected_pcm_digest or output.get('input_repeats') != 1:
            raise ValueError('actual input PCM digest/repeat semantics differ from frozen slice')
    if metadata['geometry'] != GEOMETRY or metadata['model_revision'] != MODEL_REVISION:
        raise ValueError('FP32 reference identity/geometry differs')
    if metadata['device'] != 'cpu' or metadata['dtype'] != 'float32':
        raise ValueError('reference must use original CPU FP32 weights')
    parity = compare(gen2, native, native_raw, q8_raw)
    if gen2['retained_probabilities']['first_frame'] != 0:
        raise ValueError('short corpus slice unexpectedly compacted')
    q8 = np.frombuffer(q8_raw, dtype='<f4').reshape(-1, 8)
    if fp32.dtype != np.dtype('float32') or fp32.shape != (samples // 160, 8) or metadata['shape'] != list(fp32.shape):
        raise ValueError('FP32 output differs from pinned valid-frame rule')
    if not np.isfinite(fp32).all() or not ((0 <= fp32) & (fp32 <= 1)).all():
        raise ValueError('invalid FP32 probabilities')
    if len({s['speaker'] for s in reference}) != item['statistics']['speakers']:
        raise ValueError('reference speaker count differs')
    # Score every floor-valid row, including the final row. The historical
    # floor-plus-one arrays and their quality results remain unchanged.
    original = der(reference, segments(fp32), duration)
    quantized = der(reference, segments(q8), duration)
    deployed = der(reference, [{'start': s['start_seconds'], 'end': s['end_seconds'], 'speaker': s['speaker']}
                              for s in gen2['turns']], duration)
    delta = quantized['der'] - original['der']
    return {'slice': item['id'], 'backend': gen2['backend'], 'device': gen2['device'],
            'scope': 'local tuning-heldout integration and quality, not guaranteed checkpoint-unseen',
            'wrapper_parity': parity, 'fp32': original, 'q8_shared_threshold': quantized,
            'q8_production_default': deployed, 'q8_minus_fp32_absolute_der': delta,
            'gates': {'wrapper_parity': True, 'q8_degradation_at_most_0_02': delta <= .02 + 1e-12,
                      'q8_shared_der_at_most_0_20': quantized['der'] <= .20,
                      'q8_production_der_at_most_0_20': deployed['der'] <= .20},
            'postprocessing': 'shared probability >0.5, no hysteresis/padding/minimum-duration; production default reported separately',
            'shape_difference': {'native': list(q8.shape), 'fp32': list(fp32.shape),
                                 'native_extra_terminal_frame': 0, 'frame_seconds': .01,
                                 'rule': 'native and FP32 floor(N/160); every valid row retained; full audio UEM'},
            'channel_mae_common_frames': float(np.abs(q8 - fp32).mean()),
            'channel_max_error_common_frames': float(np.abs(q8 - fp32).max())}

