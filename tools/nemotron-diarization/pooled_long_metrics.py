"""Owned-pool CPU long sessions with the unchanged valid-Mel acceptance gates."""
from pooled_runtime import POOLED_PROFILE
from qualify_pooled_native import NEW_VERSION, OLD_VERSION
from valid_mel_long_metrics import audit_case as audit_valid_mel


def audit_case(public, native, raw, actual_raw, native_time, public_time, case):
    if any(record.get('runtime_version') != NEW_VERSION for record in [public, native]):
        raise ValueError('unexpected owned-pool runtime version')
    if public.get('backend') != 'cpu' or native.get('requested_gpu_index') != -1:
        raise ValueError('owned-pool CPU long-session qualification required')
    result = audit_valid_mel({**public, 'runtime_version': OLD_VERSION},
                            {**native, 'runtime_version': OLD_VERSION}, raw, actual_raw,
                            native_time, public_time, case)
    return {**result, 'runtime_profile': POOLED_PROFILE, 'runtime_version': NEW_VERSION,
            'probabilities_bit_exact': raw == actual_raw}
