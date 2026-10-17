"""Strict identity admission for the bounded candidate against the pooled oracle."""
import json
from qualify_pooled_native import OLD_VERSION,NEW_VERSION,compare_history
from pooled_runtime import BOUNDED_PROFILE

VERSION = 'nemo-speech-asr 0.2.0+'+BOUNDED_PROFILE
REFERENCE_VERSION = NEW_VERSION


def derive_helper(original):
    before = json.dumps(OLD_VERSION)
    if original.count(before) != 1:
        raise ValueError('one exact original version literal required')
    return original.replace(before,json.dumps(VERSION))


def compare_bounded_history(value,raw,reference,golden):
    if value.get('runtime_version') != VERSION or reference.get('runtime_version') != REFERENCE_VERSION:
        raise ValueError('exact bounded candidate and pooled reference identities required')
    if (type(value.get('elapsed_micros')) is not int or value['elapsed_micros'] <= 0
            or type(value.get('cold_native_load_micros')) is not int or value['cold_native_load_micros'] <= 0):
        raise ValueError('positive native timing observations required')
    result = compare_history({**value,'runtime_version':NEW_VERSION},raw,
                             {**reference,'runtime_version':OLD_VERSION},golden)
    return {**result,'runtime_version':VERSION,'reference_runtime_version':REFERENCE_VERSION,
            'warm_rtf':value['elapsed_micros']/1e6/value['audio_duration_seconds']}


def validate_failed_loads(value):
    if (value['failed_model_construction_attempts'] != 101 or value['valid_models_loaded'] != 0
            or value['frontend_or_neural_calls'] != 0 or value['thread_growth_allowance'] != 2
            or value['maximum_requested_settling_wait_milliseconds'] != 100):
        raise ValueError('complete model-denied failed-load control required')
    immediate,settled,waits = [value[key] for key in ['immediate_thread_counts',
        'threads_after_each_failure','requested_settling_wait_milliseconds']]
    if (any(len(rows) != 101 for rows in [immediate,settled,waits])
            or any(type(v) is not int or v < 1 for v in immediate+settled)
            or any(type(v) is not int or v not in range(10,101,10) for v in waits)
            or max(settled) > settled[0]+2):
        raise ValueError('incomplete or growing failed-load thread history')
    return value
