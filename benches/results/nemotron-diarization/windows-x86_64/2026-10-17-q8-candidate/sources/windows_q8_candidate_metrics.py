"""Original bounded public/ABI semantics with a pinned Windows kernel/compiler recipe."""
import hashlib
import json
from pathlib import Path, PureWindowsPath

from build_valid_mel_candidate import descriptor
from platform_bounded_metrics import VERSION, compare_complete_60s, realtime_passed
from pooled_runtime import BOUNDED_PROFILE
from windows_compiler_metrics import (admit_compiler_paths, command_compiler,
    compiler_environment as parent_environment, compiler_evidence as parent_evidence)
from windows_compiler_fp_metrics import FLAG, baseline, cache_value, normalized_commands, build_command as fp_build_command
from windows_q8_candidate_sources import RECIPE_ID
from windows_q8_model_sources import PATH, Q8_PATCH, ORIGINAL, candidate_source, q8_patch
from windows_worker_metrics import PCM_SHA256, version as worker_version, compare_native as worker_compare

BOUNDED_RESULT = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-bounded-native-failure'
BOUNDED_MANIFEST = {'bytes': 25158, 'sha256': '88d208be33dd75cdf51f5a49ed0d6dd1eab40e5da3d3e6e5876faa9b5dac5b21'}
REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'


def compiler_kind(kind):
    if kind != 'clang-cl':
        raise ValueError('exact contraction-off Clang model compiler required')
    return kind


def compiler_environment(original):
    return parent_environment(original)


def build_command(command, msvc, compiler, kind):
    compiler_kind(kind)
    return fp_build_command(command, msvc, compiler, kind)


def compiler_evidence(cache_text, database, compiler_files, compiler, kind, aliases):
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
    if cache_value(cache_text, 'NEMO_SPEECH_VERSION_METADATA') != [BOUNDED_PROFILE]:
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
                new = VERSION.removeprefix('nemo-speech-asr ')
                if old not in prior_token or token != prior_token.replace(old, new):
                    raise ValueError('only exact diagnostic version token may change')
                tokens[index] = prior_token
                metadata_tokens += 1
    if actual != expected or metadata_tokens != len(database) or len(database) != 97:
        raise ValueError('compiler commands exceed contraction and diagnostic version deltas')
    result['candidate_compiler_admission'] = {'contraction_flag': FLAG, 'compiler_commands': 97,
        'version_tokens_changed': metadata_tokens,
        'normalization': 'Selected compiler alias, isolated source root, one contraction flag and exact diagnostic version token only.'}
    return result


def bounded_audit():
    root = Path(__file__).resolve().parents[2] / BOUNDED_RESULT
    if descriptor(root / 'collection.json') != BOUNDED_MANIFEST:
        raise ValueError('original bounded source collection required')
    manifest = json.loads((root / 'collection.json').read_text())['files']
    name = 'platform-llama-materialized-audit.json'
    if descriptor(root / name) != manifest[name]:
        raise ValueError('original bounded source audit changed')
    return json.loads((root / name).read_text())


def normalized_inputs(record):
    root = PureWindowsPath(record['compiled_tree_path']).parents[3]
    result = {}
    for name, identity in record['all_compiler_sources'].items():
        path = PureWindowsPath(name)
        if not path.is_relative_to(root):
            raise ValueError('compiler source outside isolated native root')
        name = path.relative_to(root).as_posix()
        if name in result:
            raise ValueError('duplicate compiler source')
        result[name] = identity
    return result


def admit_candidate_audit(audit):
    original = bounded_audit()
    if original['files'][PATH] != ORIGINAL:
        raise ValueError('canonical bounded SGEMM required')
    files, inputs, ggml = original['files'].copy(), normalized_inputs(original), original['actual_compiler_sources'].copy()
    raw, patch = candidate_source(), q8_patch()
    identity = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    files[PATH] = ggml[PATH] = identity
    inputs['build/gen2-cpu-diar/_deps/llama.cpp/' + PATH] = identity
    added = {'name': Q8_PATCH, 'bytes': len(patch), 'sha256': hashlib.sha256(patch).hexdigest()}
    if (audit['files'] != files or audit['patch_series'] != original['patch_series'] + [added]
            or len(files) != 3425 or len(inputs) != 97 or normalized_inputs(audit) != inputs
            or audit['actual_compiler_sources'] != ggml or audit['archive'] != original['archive']
            or audit['base_revision'] != original['base_revision'] or audit['exact_independent_replay_match'] is not True):
        raise ValueError('candidate source differs beyond bounded inference and exact SGEMM delta')
    return {'recipe_id': RECIPE_ID, 'canonical_files': 3425, 'changed_ggml_files': [PATH],
            'ggml_patches': 20, 'compiler_inputs': 97, 'diagnostic_worker_telemetry': False}


def compare_candidate_pair(public, native, raw_native, raw_public):
    value = compare_complete_60s(public, native, raw_native, raw_public)
    if (value.get('passed') is not True or value['probabilities_bit_exact'] is not True
            or any(record.get('input_pcm_f32le_sha256') != PCM_SHA256 or record.get('input_repeats') != 1
                   for record in (public, native))
            or type(native.get('push_count')) is not int or native['push_count'] != 58):
        raise ValueError('complete byte-exact native/public candidate histories required')
    path = Path(__file__).resolve().parents[2] / REFERENCE
    reference, raw = json.loads(path.read_text()), Path(str(path) + '.f32').read_bytes()
    if reference['runtime_version'] != worker_version(4):
        raise ValueError('exact frozen four-worker reference required')
    # Admit the unchanged bounded identity before mapping the tag for the
    # historical full-history comparator. No numerical tolerance is relaxed.
    check = worker_compare(reference, {**native, 'runtime_version': worker_version(4)}, raw, raw_native, 4, 4)
    if raw != raw_native:
        raise ValueError('candidate must match every frozen probability byte')
    return {**value, 'frozen_reference_comparison': check, 'build_recipe_id': RECIPE_ID}


def both_pairs_realtime(comparisons):
    if len(comparisons) != 2:
        raise ValueError('both complete native/public pairs required')
    return all(record.get('passed') is True and record.get('probabilities_bit_exact') is True
               and realtime_passed(record) for record in comparisons)
