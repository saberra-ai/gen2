"""Windows bounded identity, complete parity and unchanged below-one RTF gate."""
import math

from compare_stream_valid_mel import compare as compare_valid_mel
from pooled_runtime import BOUNDED_PROFILE


VERSION = 'nemo-speech-asr 0.2.0+' + BOUNDED_PROFILE


def derive_native_probe(original):
    before = b'"nemo-speech-asr 0.2.0+gen2-valid-mel-v1"'
    if original.count(before) != 1:
        raise ValueError('one exact historical probe version literal required')
    return original.replace(before, ('"' + VERSION + '"').encode())


def compare_complete_60s(public, native, raw, actual_raw):
    if (public.get('runtime_version') != VERSION or native.get('runtime_version') != VERSION
            or public.get('backend') != 'cpu' or native.get('requested_gpu_index') != -1):
        raise ValueError('exact bounded CPU runtime and placement required')
    if (public['accepted_samples'] != 960000 or native['audio_samples'] != 960000
            or public['retained_probabilities'] != {'first_frame': 0, 'rows': 6000}
            or native['retained_first_frame'] != 0 or native['retained_rows'] != 6000
            or len(raw) != 192000 or len(actual_raw) != 192000):
        raise ValueError('bounded Windows qualification requires complete6000row histories')
    old_version = 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
    result = compare_valid_mel({**public, 'runtime_version': old_version},
                               {**native, 'runtime_version': old_version}, raw, actual_raw)
    return {**result, 'runtime_profile': BOUNDED_PROFILE, 'runtime_version': VERSION,
            'probabilities_bit_exact': raw == actual_raw,
            'realtime_gate_passed': realtime_passed(result), 'rtf_limit_exclusive': 1.0}


def realtime_passed(comparison):
    values = [comparison[key] for key in ['gen2_rtf', 'native_rtf']]
    return all(type(value) in (float, int) and math.isfinite(value) and 0 < value < 1 for value in values)
