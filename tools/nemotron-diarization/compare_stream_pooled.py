"""Exact owned-pool CPU admission with unchanged valid-Mel numerical checks.

The historical comparator remains frozen. Validate the new identity first,
then adapt copies to reuse its unchanged frame, transport, placement, turn and
probability rules. Retained input records and reported identity stay pooled.
"""
from compare_stream_valid_mel import compare as compare_valid_mel
from pooled_runtime import POOLED_PROFILE
from qualify_pooled_native import NEW_VERSION, OLD_VERSION


def compare(gen2, native, raw, actual_raw):
    if any(record.get('runtime_version') != NEW_VERSION for record in [gen2, native]):
        raise ValueError('unexpected owned-pool runtime version')
    if gen2.get('backend') != 'cpu' or native.get('requested_gpu_index') != -1:
        raise ValueError('owned-pool CPU qualification required')
    result = compare_valid_mel({**gen2, 'runtime_version': OLD_VERSION},
                               {**native, 'runtime_version': OLD_VERSION}, raw, actual_raw)
    return {**result, 'runtime_profile': POOLED_PROFILE, 'runtime_version': NEW_VERSION,
            'probabilities_bit_exact': raw == actual_raw}
