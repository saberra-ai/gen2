"""Exact bounded CPU identity admission with unchanged full-history math."""
from bounded_cpu_metrics import VERSION
from compare_stream_valid_mel import compare as compare_valid_mel
from pooled_runtime import BOUNDED_PROFILE
from qualify_pooled_native import OLD_VERSION


def compare(gen2,native,raw,actual_raw):
    if any(value.get('runtime_version') != VERSION for value in [gen2,native]):
        raise ValueError('exact bounded CPU runtime version required')
    if gen2.get('backend') != 'cpu' or native.get('requested_gpu_index') != -1:
        raise ValueError('bounded CPU placement required')
    result = compare_valid_mel({**gen2,'runtime_version':OLD_VERSION},
                               {**native,'runtime_version':OLD_VERSION},raw,actual_raw)
    return {**result,'runtime_profile':BOUNDED_PROFILE,'runtime_version':VERSION,
            'probabilities_bit_exact':raw==actual_raw}
