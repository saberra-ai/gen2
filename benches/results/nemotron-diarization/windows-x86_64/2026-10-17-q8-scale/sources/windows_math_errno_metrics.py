"""Admit only removal of math errno requirements; retain contraction and outputs."""
import re
from pathlib import Path

from build_valid_mel_candidate import descriptor
from platform_native import powershell_literal
from windows_worker_metrics import PCM_SHA256, telemetry, compare_native as worker_compare
from windows_compiler_metrics import (admit_compiler_paths, command_compiler, compiler_environment,
    compiler_evidence as parent_evidence)
from windows_compiler_fp_metrics import (FLAG, baseline, cache_value, normalized_commands,
    build_command as contraction_build_command)

ORDER = ['control', 'no-errno', 'no-errno', 'control']
ERRNO_FLAG = '/clang:-fno-math-errno'
FLAGS = {'control': [FLAG], 'no-errno': [FLAG, ERRNO_FLAG]}


def validate_variant(variant):
    if variant not in FLAGS:
        raise ValueError('fixed math-errno control or candidate required')
    return variant


def compiler_kind(kind):
    if kind != 'clang-cl':
        raise ValueError('exact original Clang compiler required')
    return kind


def build_command(command, msvc, compiler, kind, variant):
    compiler_kind(kind)
    validate_variant(variant)
    result = contraction_build_command(command, msvc, compiler, kind)
    text = result[4]
    for language in ('C', 'CXX'):
        old = powershell_literal('-DCMAKE_' + language + '_FLAGS_INIT:STRING=' + FLAG)
        new = powershell_literal('-DCMAKE_' + language + '_FLAGS_INIT:STRING=' + ' '.join(FLAGS[variant]))
        if text.count(old) != 1:
            raise ValueError('one unchanged contraction initializer per language required')
        text = text.replace(old, new)
    return result[:4] + [text]


def admit_math_driver(text, variant):
    validate_variant(variant)
    lines = [line for line in text.splitlines() if '"-cc1"' in line]
    if len(lines) != 1:
        raise ValueError('one actual compiler driver command required')
    tokens = re.findall(r'"([^"\r\n]*)"', lines[0])
    if tokens.count('-target-cpu') != 1:
        raise ValueError('one effective target CPU required')
    cpu_index = tokens.index('-target-cpu')
    if cpu_index + 1 >= len(tokens):
        raise ValueError('explicit target CPU value required')
    if (tokens.count('-cc1') != 1 or [token for token in tokens if re.fullmatch(r'-O(?:[0-3szg]|fast)', token)] != ['-O3']
            or tokens.count('-ffp-contract=off') != 1
            or tokens.count('-vectorize-loops') != 1 or tokens.count('-vectorize-slp') != 1
            or tokens[cpu_index + 1] != 'haswell'):
        raise ValueError('original effective optimization, ISA and contraction required')
    forbidden = {'-ffast-math', '-menable-unsafe-fp-math', '-funsafe-math-optimizations',
        '-freciprocal-math', '-fapprox-func', '-ffinite-math-only', '-fno-signed-zeros', '-ffp-contract=fast'}
    if forbidden.intersection(tokens) or any(token.startswith('-ffp-contract=') and token != '-ffp-contract=off' for token in tokens):
        raise ValueError('unrelated math semantics changed')
    if (tokens.count('-fmath-errno') != (1 if variant == 'control' else 0)
            or tokens.count('-fno-math-errno') > (0 if variant == 'control' else 1)):
        raise ValueError('actual math errno state differs')
    return {'effective_optimization': '-O3', 'contraction': 'off', 'target_cpu': 'haswell',
        'math_errno': variant == 'control', 'loop_and_slp_vectorization': True,
        'scope': 'Driver observation for exact compiler/flags; no model or speedup evidence.'}


def compiler_evidence(cache_text, database, compiler_files, compiler, kind, aliases=(), variant='control'):
    compiler_kind(kind)
    validate_variant(variant)
    result = parent_evidence(cache_text, database, compiler_files, compiler, kind, aliases)
    prior_cache, prior_database, prior = baseline(kind)
    if descriptor(Path(compiler)) != prior['compiler_binary'] or descriptor(Path(result['linker'])) != prior['linker_binary']:
        raise ValueError('original exact compiler and linker required')
    for language in ('C', 'CXX'):
        field = 'CMAKE_' + language + '_FLAGS'
        flags = cache_value(cache_text, field)
        if cache_value(cache_text, field + '_INIT') != FLAGS[variant]:
            raise ValueError('exact math-errno initializers required')
        for flag in FLAGS[variant]:
            if flags.count(flag) != 1:
                raise ValueError('one declared global flag required')
            flags.remove(flag)
        if flags != cache_value(prior_cache, field):
            raise ValueError('original platform flags changed')
        if cache_value(cache_text, field + '_RELEASE') != cache_value(prior_cache, field + '_RELEASE'):
            raise ValueError('original Release flags changed')
    actual = normalized_commands(database, kind, True)
    expected = normalized_commands(prior_database, kind, False)
    if variant == 'no-errno':
        for row in actual.values():
            if row['arguments'].count(ERRNO_FLAG) != 1:
                raise ValueError('exactly one candidate flag per compiler command required')
            row['arguments'].remove(ERRNO_FLAG)
    if actual != expected or len(database) != 97:
        raise ValueError('compiler commands exceed the single math-errno delta')
    result['math_errno_admission'] = {'variant': variant, 'flags': FLAGS[variant],
        'compiler_commands_matching_original': 97,
        'normalization': 'Selected compiler alias, isolated source root, one contraction flag and candidate math-errno flag only.'}
    return result


def timing_evidence(cases):
    if len(cases) != 4 or [case['compiler'] for case in cases] != ORDER:
        raise ValueError('all four ordered same-compiler math-errno cases required')
    times = [case['result']['elapsed_micros'] for case in cases]
    if any(type(value) is not int or value <= 0 for value in times):
        raise ValueError('positive integer timings required')
    drift = 10 * abs(times[3] - times[0]) <= times[0]
    faster = 20 * max(times[1:3]) <= 19 * min(times[0], times[3])
    return {'elapsed_micros': times, 'native_rtfs': [value / 60e6 for value in times],
        'control_drift_fraction': abs(times[3] - times[0]) / times[0],
        'control_drift_at_most_10_percent': drift,
        'both_candidates_at_least_5_percent_faster_than_both_controls': faster,
        'speedup_prediction_supported': drift and faster,
        'both_candidates_strictly_real_time': all(value < 60000000 for value in times[1:3]),
        'limits': 'Native diagnostic only; all original public/lifecycle/performance/quality/release gates remain.'}


def compare_native(reference, actual, raw_reference, raw_actual, reference_cap, actual_cap):
    result = worker_compare(reference, actual, raw_reference, raw_actual, reference_cap, actual_cap)
    if raw_actual != raw_reference:
        raise ValueError('byte-exact complete math-errno output required')
    return result
