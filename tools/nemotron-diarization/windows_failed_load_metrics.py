"""Own-process failure-control admission, independent of native/model execution."""


def validate_failed_loads(value):
    constants = {'failed_model_construction_attempts': 101, 'valid_models_loaded': 0,
                 'frontend_or_neural_calls': 0, 'thread_growth_allowance': 2,
                 'maximum_requested_settling_wait_milliseconds': 100}
    if any(type(value.get(name)) is not int or value[name] != expected for name, expected in constants.items()):
        raise ValueError('complete model-free Windows failed-load control required')
    immediate, settled, waits = [value.get(key) for key in ['immediate_thread_counts',
        'threads_after_each_failure', 'requested_settling_wait_milliseconds']]
    if (any(not isinstance(rows, list) or len(rows) != 101 for rows in [immediate, settled, waits])
            or any(type(number) is not int or number < 1 for number in immediate + settled)
            or any(type(number) is not int or number not in range(10, 101, 10) for number in waits)
            or max(settled) > settled[0] + 2):
        raise ValueError('incomplete or growing Windows failed-load thread history')
    return value
