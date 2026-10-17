"""Admit the exact bounded CPU identity before unchanged complete-app checks."""
import copy

from bounded_cpu_metrics import VERSION
import pooled_pnola_metrics as original


def _native(value):
    if value['runtime_version'] != VERSION:
        raise ValueError('exact bounded CPU native reference required')
    return {**value, 'runtime_version': original.NEW_VERSION}


def _timeline(value):
    if (value['runtime_version'] != VERSION or value['backend'] != 'cpu'
            or value.get('device') is not None):
        raise ValueError('exact bounded CPU app timeline required')
    return {**value, 'runtime_version': original.NEW_VERSION}


def native_reference(value, duration):
    return original.native_reference(_native(value), duration)


def timeline_matches(actual, native, duration, *, compact=False):
    return original.timeline_matches(_timeline(actual), _native(native), duration, compact=compact)


def recorded_matches(actual, native, bark_text, wav_sha256, *, unavailable=False):
    if unavailable:
        return original.recorded_matches(actual, {}, bark_text, wav_sha256, unavailable=True)
    adapted = copy.deepcopy(actual)
    adapted['attribution']['timeline'] = _timeline(adapted['attribution']['timeline'])
    return original.recorded_matches(adapted, _native(native), bark_text, wav_sha256)


def live_matches(actual, native):
    adapted = copy.deepcopy(actual)
    adapted['speakers']['timeline'] = _timeline(adapted['speakers']['timeline'])
    return original.live_matches(adapted, _native(native))
