"""Diagnostic identity, exact compiler delta and bounded observer interpretation."""
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from ggml_matrix_profile import environment as profile_environment
from windows_compiler_metrics import (admit_compiler_paths, command_compiler,
    compiler_environment as parent_environment, compiler_evidence as parent_evidence)
from windows_compiler_fp_metrics import FLAG, baseline, cache_value, normalized_commands, build_command as fp_build_command
from windows_matrix_batch_profile_sources import CPU_SOURCE, PROFILE_PATCH, profile_patch
from windows_worker_metrics import (PCM_SHA256, DIARIZER, BACKEND, transform, telemetry,
    replace_once, version as worker_version, compare_native as worker_compare)


ORDER = [False, True, True, False]
REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-fp'


def metadata(cap):
    if type(cap) is not int or cap != 4:
        raise ValueError('fixed four-worker CPU profile identity required')
    return 'gen2-diagnostic-matrix-batch-v1'


def version(cap):
    return 'nemo-speech-asr 0.2.0+' + metadata(cap)


def derive_probe(original, cap):
    return replace_once(original, b'"nemo-speech-asr 0.2.0+gen2-valid-mel-v1"',
                        ('"' + version(cap) + '"').encode())


def compare_native(reference, actual, raw_reference, raw_actual, reference_cap=4, actual_cap=4):
    if (reference_cap != 4 or actual_cap != 4 or reference['runtime_version'] != worker_version(4)
            or actual['runtime_version'] != version(4)):
        raise ValueError('exact reference and diagnostic CPU profile versions required')
    value = worker_compare(reference, {**actual, 'runtime_version': worker_version(4)},
                           raw_reference, raw_actual, 4, 4)
    return {**value, 'actual_runtime_version': version(4), 'runtime_profile': metadata(4)}


def compiler_kind(kind):
    if kind != 'clang-cl':
        raise ValueError('exact contraction-off Clang profile compiler required')
    return kind


def compiler_environment(original):
    return profile_environment(parent_environment(original), False)


def build_command(command, msvc, compiler, kind):
    compiler_kind(kind)
    return fp_build_command(command, msvc, compiler, kind)


def compiler_evidence(cache_text, database, compiler_files, compiler, kind, aliases=()):
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
    if cache_value(cache_text, 'NEMO_SPEECH_VERSION_METADATA') != [metadata(4)]:
        raise ValueError('distinct CPU profiling diagnostic metadata required')
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
                new = version(4).removeprefix('nemo-speech-asr ')
                if old not in prior_token or token != prior_token.replace(old, new):
                    raise ValueError('only exact diagnostic version token may change')
                tokens[index] = prior_token
                metadata_tokens += 1
    if actual != expected or metadata_tokens != len(database) or len(database) != 97:
        raise ValueError('compiler commands exceed contraction and diagnostic version deltas')
    result['matrix_profile_admission'] = {'contraction_flag': FLAG, 'compiler_commands': 97,
        'version_tokens_changed': metadata_tokens,
        'normalization': 'Selected compiler alias, isolated source root, one contraction flag and exact diagnostic version token only.'}
    return result


def canonical_audit():
    root = Path(__file__).resolve().parents[2] / REFERENCE
    manifest = json.loads((root / 'collection.json').read_text())['files']
    name = 'evidence-clang-cl/platform-llama-materialized-audit.json'
    if descriptor(root / name) != manifest[name]:
        raise ValueError('canonical full-SDK source audit changed')
    return json.loads((root / name).read_text())


def admit_profile_audit(audit):
    from ggml_matrix_profile_source import PATH
    original = canonical_audit()
    files = original['files'].copy()
    files[PATH] = CPU_SOURCE
    patch = profile_patch()
    import hashlib
    added = {'name': PROFILE_PATCH, 'bytes': len(patch), 'sha256': hashlib.sha256(patch).hexdigest()}
    if (audit['files'] != files or audit['patch_series'] != original['patch_series'] + [added]
            or len(files) != 3425 or len(audit['all_compiler_sources']) != 97):
        raise ValueError('GGML source differs beyond the exact CPU timing patch')
    return {'canonical_files': 3425, 'changed_files': [PATH], 'patches': 20, 'compiler_inputs': 97}


def profile_signature(profile):
    return [{key: graph['workers'][0][key] for key in ('nodes', 'roots', 'fused', 'skipped', 'topology', 'execution')}
            | {'operations': [[op['id'], op['roots'], op['fused']] for op in graph['workers'][0]['operations']],
               'matrix_metadata': [{key: (row[key].copy() if isinstance(row[key],list) else row[key]) for key in ('node','types','names','a','b','c','sa','sb','sc')}
                                   for row in graph['workers'][0]['matrices']]}
            for graph in profile['graphs']]


def timing_evidence(cases):
    if (len(cases) != 4 or any(type(case['enabled']) is not bool for case in cases)
            or [case['enabled'] for case in cases] != ORDER):
        raise ValueError('all four ordered CPU profiling observations required')
    times = [case['result']['elapsed_micros'] for case in cases]
    if any(type(value) is not int or value <= 0 for value in times):
        raise ValueError('positive integer CPU profile timings required')
    drift = 10 * abs(times[3] - times[0]) <= times[0]
    overhead = 10 * max(times[1:3]) <= 11 * min(times[0], times[3])
    return {'elapsed_micros': times, 'native_rtfs': [value / 60e6 for value in times],
        'control_drift_fraction': abs(times[3] - times[0]) / times[0],
        'control_drift_at_most_10_percent': drift,
        'both_instrumented_cases_at_most_10_percent_above_both_controls': overhead,
        'bounded_observer_interpretation_supported': drift and overhead,
        'limits': 'Diagnostic observation only. No release pass; kernel wall includes internal waits, scheduling and observer effects. Never sum overlapping workers.'}
