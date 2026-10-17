"""Exact identities, compiler/source admissions and full-model timing predicates."""
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import struct

from build_valid_mel_candidate import descriptor
from windows_compiler_metrics import (admit_compiler_paths, command_compiler,
    compiler_environment as parent_environment, compiler_evidence as parent_evidence)
from windows_compiler_fp_metrics import FLAG, baseline, cache_value, normalized_commands, build_command as fp_build_command
from windows_cpu_profile_metrics import canonical_audit
from windows_q8_model_sources import PATH, Q8_PATCH, ORIGINAL, candidate_source, q8_patch, validate_variant
from windows_worker_metrics import (PCM_SHA256, DIARIZER, BACKEND, transform, telemetry,
    replace_once, version as worker_version, compare_native as worker_compare)

ORDER = ['control', 'mx4', 'mx4', 'control']


def metadata(cap, variant):
    validate_variant(variant)
    if type(cap) is not int or cap != 4:
        raise ValueError('fixed four-worker Q8 model identity required')
    return 'gen2-diagnostic-q8-model-' + variant + '-v1'


def version(cap, variant):
    return 'nemo-speech-asr 0.2.0+' + metadata(cap, variant)


def derive_probe(original, cap, variant):
    return replace_once(original, b'"nemo-speech-asr 0.2.0+gen2-valid-mel-v1"',
                        ('"' + version(cap, variant) + '"').encode())


def compare_native(reference, actual, raw_reference, raw_actual, variant):
    validate_variant(variant)
    if reference['runtime_version'] != worker_version(4) or actual['runtime_version'] != version(4, variant):
        raise ValueError('exact reference and variant Q8 model versions required')
    value = worker_compare(reference, {**actual, 'runtime_version': worker_version(4)}, raw_reference, raw_actual, 4, 4)
    if raw_reference != raw_actual or type(actual.get('push_count')) is not int or actual['push_count'] != 58:
        raise ValueError('byte-exact complete 58-push Q8 model history required')
    return {**value, 'actual_runtime_version': version(4, variant), 'runtime_profile': metadata(4, variant)}


def compiler_kind(kind):
    if kind != 'clang-cl':
        raise ValueError('exact contraction-off Clang model compiler required')
    return kind


def compiler_environment(original):
    return parent_environment(original)


def build_command(command, msvc, compiler, kind):
    compiler_kind(kind)
    return fp_build_command(command, msvc, compiler, kind)


def compiler_evidence(cache_text, database, compiler_files, compiler, kind, aliases, variant):
    validate_variant(variant)
    compiler_kind(kind)
    result = parent_evidence(cache_text, database, compiler_files, compiler, kind, aliases)
    prior_cache, prior_database, prior = baseline(kind)
    if (descriptor(Path(compiler)) != prior['compiler_binary']
            or descriptor(Path(result['linker'])) != prior['linker_binary']):
        raise ValueError('original compiler and linker binaries required')
    for language in ('C', 'CXX'):
        field = 'CMAKE_' + language + '_FLAGS'
        flags = cache_value(cache_text, field)
        if flags.count(FLAG) != 1 or cache_value(cache_text, field + '_INIT') != [FLAG]:
            raise ValueError('one explicit original contraction initializer required')
        flags.remove(FLAG)
        if (flags != cache_value(prior_cache, field)
                or cache_value(cache_text, field + '_RELEASE') != cache_value(prior_cache, field + '_RELEASE')):
            raise ValueError('original platform or Release flags changed')
    if cache_value(cache_text, 'NEMO_SPEECH_VERSION_METADATA') != [metadata(4, variant)]:
        raise ValueError('distinct Q8 model diagnostic metadata required')
    actual = normalized_commands(database, kind, True)
    expected = normalized_commands(prior_database, kind, False)
    if set(actual) != set(expected):
        raise ValueError('original compiler source identities required')
    metadata_tokens = 0
    for name, row in actual.items():
        tokens = row['arguments']
        if len(tokens) != len(expected[name]['arguments']):
            raise ValueError('original compiler argument count required')
        for index, token in enumerate(tokens):
            if token.startswith('-DNEMO_SPEECH_VERSION_STR='):
                prior_token = expected[name]['arguments'][index]
                old = worker_version(4).removeprefix('nemo-speech-asr ')
                new = version(4, variant).removeprefix('nemo-speech-asr ')
                if old not in prior_token or token != prior_token.replace(old, new):
                    raise ValueError('only exact diagnostic version token may change')
                tokens[index] = prior_token
                metadata_tokens += 1
    if actual != expected or metadata_tokens != len(database) or len(database) != 97:
        raise ValueError('compiler commands exceed contraction and diagnostic version deltas')
    result['q8_model_admission'] = {'contraction_flag': FLAG, 'compiler_commands': 97,
        'version_tokens_changed': metadata_tokens,
        'normalization': 'Selected compiler alias, isolated source root, one contraction flag and exact diagnostic version token only.'}
    return result


def admit_q8_audit(audit, variant):
    validate_variant(variant)
    original = canonical_audit()
    if original['files'][PATH] != ORIGINAL:
        raise ValueError('canonical original SGEMM descriptor required')
    files = original['files'].copy()
    patches = original['patch_series'].copy()
    if variant == 'mx4':
        raw, patch = candidate_source(), q8_patch()
        files[PATH] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        patches.append({'name': Q8_PATCH, 'bytes': len(patch), 'sha256': hashlib.sha256(patch).hexdigest()})
    def normalized_inputs(record):
        root = PureWindowsPath(record['compiled_tree_path']).parents[3]
        inputs = {}
        for path, value in record['all_compiler_sources'].items():
            path = PureWindowsPath(path)
            if not path.is_relative_to(root):
                raise ValueError('compiler source outside isolated native root')
            name = path.relative_to(root).as_posix()
            if name in inputs:
                raise ValueError('duplicate compiler source identity')
            inputs[name] = value
        return inputs
    expected_inputs = normalized_inputs(original)
    expected_ggml = original['actual_compiler_sources'].copy()
    if variant == 'mx4':
        expected_inputs['build/gen2-cpu-diar/_deps/llama.cpp/' + PATH] = files[PATH]
        expected_ggml[PATH] = files[PATH]
    if (audit['files'] != files or audit['patch_series'] != patches or len(files) != 3425
            or normalized_inputs(audit) != expected_inputs or len(expected_inputs) != 97
            or audit['actual_compiler_sources'] != expected_ggml
            or audit['archive'] != original['archive'] or audit['base_revision'] != original['base_revision']
            or audit['exact_independent_replay_match'] is not True):
        raise ValueError('GGML source differs beyond exact variant Q8 patch')
    return {'variant': variant, 'canonical_files': 3425, 'changed_files': [PATH] if variant == 'mx4' else [],
            'patches': len(patches), 'compiler_inputs': 97}


def graph_vectors(stdout, threads, outputs, reference=None):
    if type(threads) is not int or threads not in (1, 2, 4):
        raise ValueError('one/two/four model-free workers required')
    if stdout.splitlines() != [
            f'matrix-control threads={threads} shape={shape} quantized={shape%2} columns={32 if shape<2 else 1} repeat={repeat} bytes={8192 if shape<2 else 256}'
            for shape in range(4) for repeat in range(3)] or len(outputs) != 12:
        raise ValueError('all twelve model-free graph vectors required')
    expected = {} if reference is None else dict(reference)
    for index, raw in enumerate(outputs):
        shape = index // 3
        if len(raw) != (8192 if shape < 2 else 256):
            raise ValueError('complete model-free graph vector required')
        values = [v for v, in struct.iter_unpack('<f', raw)]
        if not all(math.isfinite(v) for v in values) or not any(v != 0 for v in values):
            raise ValueError('finite nonzero graph vector required')
        if shape not in expected:
            expected[shape] = raw
        if raw != expected[shape]:
            raise ValueError('per-shape graph output differs across repeats/workers/SDKs')
    if set(expected) != set(range(4)):
        raise ValueError('all four graph shapes required')
    return expected


def timing_evidence(cases):
    if (len(cases) != 4 or [case['variant'] for case in cases] != ORDER
            or [case['index'] for case in cases] != list(range(4))):
        raise ValueError('four ordered full-model observations required')
    times = [case['result']['elapsed_micros'] for case in cases]
    if any(type(value) is not int or value <= 0 for value in times):
        raise ValueError('positive integer full-model times required')
    drift = 10 * abs(times[3] - times[0]) <= times[0]
    speed = 20 * max(times[1:3]) <= 19 * min(times[0], times[3])
    return {'elapsed_micros': times, 'native_rtfs': [value / 60e6 for value in times],
        'control_drift_fraction': abs(times[3] - times[0]) / times[0],
        'control_drift_at_most_10_percent': drift,
        'both_candidates_at_least_5_percent_faster_than_both_controls': speed,
        'full_model_speedup_prediction_supported': drift and speed,
        'both_candidates_strictly_real_time': all(value < 60_000_000 for value in times[1:3]),
        'limits': 'Native diagnostic only. No production/public/lifecycle/package/release pass or repair of prior failed predictions.'}
