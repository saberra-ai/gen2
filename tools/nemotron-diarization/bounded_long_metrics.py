"""Bounded CPU identity with the original long-session acceptance criteria."""
from bounded_cpu_metrics import VERSION
from pooled_runtime import BOUNDED_PROFILE
from qualify_pooled_native import OLD_VERSION
from valid_mel_long_metrics import audit_case as audit_valid_mel


def audit_case(public, native, raw, actual_raw, native_time, public_time, case):
    if any(record.get('runtime_version') != VERSION for record in [public, native]):
        raise ValueError('exact bounded CPU runtime version required')
    if public.get('backend') != 'cpu' or native.get('requested_gpu_index') != -1:
        raise ValueError('bounded CPU long-session qualification required')
    result = audit_valid_mel({**public, 'runtime_version': OLD_VERSION},
                            {**native, 'runtime_version': OLD_VERSION}, raw, actual_raw,
                            native_time, public_time, case)
    return {**result, 'runtime_profile': BOUNDED_PROFILE, 'runtime_version': VERSION,
            'probabilities_bit_exact': raw == actual_raw}
