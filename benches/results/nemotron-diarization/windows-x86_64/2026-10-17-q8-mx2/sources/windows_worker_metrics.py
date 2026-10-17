"""Diagnostic-only worker transforms, telemetry and complete native parity."""
import hashlib
import re

from compare_stream_valid_mel import compare, finite

ORDER = [4, 2, 1, 4]
PCM_SHA256 = '96858fdcf0c8854f8fc659403c11ddc7565f538a7402803e4235e9e354d87ae6'
DIARIZER = 'src/asr/diar/diarizer.cpp'
BACKEND = 'src/runtime/ggml/backend.cpp'


def metadata(cap):
    if type(cap) is not int or cap not in (1, 2, 4):
        raise ValueError('diagnostic cap must be1,2,4')
    return 'gen2-diagnostic-cpu-threads-' + str(cap)


def version(cap):
    return 'nemo-speech-asr 0.2.0+' + metadata(cap)


def replace_once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('one exact diagnostic source site required')
    return raw.replace(old, new)


def transform(name, raw, cap):
    """Applied only after every bounded source hash has been checked."""
    metadata(cap)
    if name == DIARIZER:
        raw = replace_once(raw, b'#include <algorithm>\n', b'#include <algorithm>\n#include <cstdio>\n')
        return replace_once(raw,
            b'        params.n_threads = bounded_cpu_threads(params.n_threads, std::thread::hardware_concurrency());',
            ('        const unsigned available = std::thread::hardware_concurrency();\n'
             f'        params.n_threads = bounded_cpu_threads({cap}, available);\n'
             f'        std::fprintf(stderr, "[DEBUG-worker-v1] cap={cap} available=%u effective=%d\\n", available, params.n_threads);').encode())
    if name == BACKEND:
        raw = replace_once(raw, b'#include <cstdlib>\n', b'#include <cstdlib>\n#include <cstdio>\n')
        return replace_once(raw, b'        set_pool(backend, pool);',
            b'        set_pool(backend, pool);\n        std::fprintf(stderr, "[DEBUG-worker-v1] attached_pool_threads=%d\\n", threads);')
    return raw


def derive_probe(original, cap):
    return replace_once(original, b'"nemo-speech-asr 0.2.0+gen2-valid-mel-v1"',
                        ('"' + version(cap) + '"').encode())


def telemetry(log, cap):
    metadata(cap)
    configured = re.findall(r'^\[DEBUG-worker-v1\] cap=(\d+) available=(\d+) effective=(\d+)$', log, re.MULTILINE)
    pools = re.findall(r'^\[DEBUG-worker-v1\] attached_pool_threads=(\d+)$', log, re.MULTILINE)
    if len(configured) != 1 or len(pools) != 1:
        raise ValueError('one actual CPU configuration and attached pool required')
    requested, available, effective = map(int, configured[0])
    if requested != cap or effective != (min(cap, available) if available else 1) or int(pools[0]) != effective:
        raise ValueError('actual diagnostic worker count differs')
    return {'requested_cap': cap, 'hardware_concurrency': available,
            'effective_threads': effective, 'attached_pool_threads': int(pools[0])}


def compare_native(reference, actual, raw_reference, raw_actual, reference_cap, actual_cap):
    for result, cap in [(reference, reference_cap), (actual, actual_cap)]:
        if (result['runtime_version'] != version(cap) or result['input_pcm_f32le_sha256'] != PCM_SHA256
                or result['input_repeats'] != 1 or result['audio_samples'] != 960000
                or result['retained_first_frame'] != 0 or result['retained_rows'] != 6000
                or result['requested_gpu_index'] != -1):
            raise ValueError('complete60s diagnostic CPU identity required')
    if len(raw_reference) != 192000 or len(raw_actual) != 192000:
        raise ValueError('all48000 probabilities required')
    # Adapt native field names only after admitting both diagnostic identities.
    # Reuse the frozen full-vector/turn/geometry/time validators and1e-6 bounds.
    old = 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
    for turn in actual['turns']:
        start, end = finite(turn['start_seconds'], 'native start'), finite(turn['end_seconds'], 'native end')
        if start < 0 or end < start or end > 60.0001:
            raise ValueError('invalid diagnostic native turn bounds')
    mapped = {**actual, 'accepted_samples': actual['audio_samples'], 'complete': True,
              'model_revision': 'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
              'runtime_version': old, 'backend': 'cpu', 'device': None,
              'retained_probabilities': {'first_frame': 0, 'rows': 6000},
              'retained_probability_sha256': hashlib.sha256(raw_actual).hexdigest()}
    # Clip native binary floating-point tail boundaries as the public API does.
    mapped['turns'] = [{**t, 'start_seconds': min(t['start_seconds'], 60),
                       'end_seconds': min(t['end_seconds'], 60)} for t in actual['turns']]
    result = compare(mapped, {**reference, 'runtime_version': old}, raw_reference, raw_actual)
    reference_rtf, actual_rtf = result.pop('native_rtf'), result.pop('gen2_rtf')
    return {**result, 'scope': 'native diagnostic parity only; no public/lifecycle/package qualification',
            'runtime_profile': 'diagnostic-worker-matrix-v1',
            'reference_runtime_version': version(reference_cap), 'actual_runtime_version': version(actual_cap),
            'reference_cap': reference_cap, 'actual_cap': actual_cap,
            'probabilities_bit_exact': raw_reference == raw_actual,
            'reference_rtf': reference_rtf, 'actual_rtf': actual_rtf}
