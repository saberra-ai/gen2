"""Admission and fixed-order accounting for the owned-pool thread diagnostic."""
import shlex
from pathlib import Path
from profile_macos_pooled_cpu import compare_sampled_history

ORDER = [4,3,3,4]


def validate_self_test(value):
    if (value.get('complete') is not True or value['backend_lifetimes'] != 22 or value['model_constructed'] is not False
            or value['frontend_executed'] is not False or value['thread_growth_allowance'] != 2
            or type(value['warmed_thread_baseline']) is not int or value['warmed_thread_baseline'] < 1
            or [case['threads'] for case in value['cases']] != [4,3]):
        raise ValueError('complete owned-pool model-free lifetime control required')
    for case in value['cases']:
        if len(case['observations']) != 10:
            raise ValueError('ten post-warm backend lifetimes per thread count required')
        for attempt,row in enumerate(case['observations'],1):
            if (type(row['attempt']) is not int or row['attempt'] != attempt
                    or any(type(row[key]) is not int for key in ['immediate','settled','requested_wait_ms'])
                    or row['immediate'] < 1 or not 1 <= row['settled'] <= value['warmed_thread_baseline']+2
                    or row['requested_wait_ms'] not in range(10,101,10)):
                raise ValueError('invalid cleanup count or bounded settling wait')
    return value


def validate_failure_control(value):
    if value.get('complete') is not False or value.get('phase') != 'injected-failure-control':
        raise ValueError('retained injected-failure evidence required')
    validate_self_test({**value,'complete':True})
    return value


def used_dependencies(text):
    body = text.replace('\\\n',' ').partition(':')[2]
    paths = shlex.split(body)
    if not paths or any(not Path(path).is_absolute() for path in paths):
        raise ValueError('absolute compiler dependency records required')
    return set(paths)


def compare_thread_trial(value,raw,reference,golden,threads):
    if (threads not in [3,4] or type(value['threads']) is not int or value['threads'] != threads
            or type(value['elapsed_micros']) is not int or value['elapsed_micros'] <= 0
            or type(value['load_micros']) is not int or value['load_micros'] <= 0):
        raise ValueError('thread trial identity or clock differs')
    result = compare_sampled_history(value,raw,reference,golden)
    return {**result,'threads':threads,'warm_rtf':value['elapsed_micros']/12e6,
            'cold_model_load_seconds_excluding_backend':value['load_micros']/1e6}
