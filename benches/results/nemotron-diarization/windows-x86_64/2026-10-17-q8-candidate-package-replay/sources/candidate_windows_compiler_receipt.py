"""Replay compiler records on a receiver without invoking Windows build tools."""
from pooled_runtime import BOUNDED_PROFILE
from windows_compiler_metrics import compiler_evidence as inspect_compiler_records
from windows_compiler_fp_metrics import FLAG, baseline, cache_value, normalized_commands
from platform_bounded_metrics import VERSION
from windows_worker_metrics import version as worker_version


def verify_compiler_records(cache_text, database, language_files, record, recipe):
    """Compiler binary/alias identities are CI observations, not local rehashes."""
    aliases = record.get('compiler_path_aliases', {})
    expected_binary = recipe['compiler']['binary']
    if (record.get('compiler_binary') != expected_binary or not aliases
            or any(value != expected_binary for value in aliases.values())
            or record.get('linker_binary') != recipe['compiler']['linker_binary']):
        raise ValueError('original candidate compiler/linker identities required')
    result = inspect_compiler_records(cache_text, database, language_files,
                                     record['compiler'], 'clang-cl', aliases)
    prior_cache, prior_database, prior = baseline('clang-cl')
    if (expected_binary != prior['compiler_binary']
            or record['linker_binary'] != prior['linker_binary']
            or recipe['compiler']['probe_compiler_binary'] != baseline('msvc')[2]['compiler_binary']):
        raise ValueError('original scalar-tested compiler and linker baselines required')
    for language in ('C', 'CXX'):
        field = 'CMAKE_' + language + '_FLAGS'
        flags = cache_value(cache_text, field)
        if flags.count(FLAG) != 1 or cache_value(cache_text, field + '_INIT') != [FLAG]:
            raise ValueError('one explicit contraction-off initializer required')
        flags.remove(FLAG)
        if (flags != cache_value(prior_cache, field)
                or cache_value(cache_text, field + '_RELEASE') != cache_value(prior_cache, field + '_RELEASE')):
            raise ValueError('original platform and Release compiler flags required')
    if cache_value(cache_text, 'NEMO_SPEECH_VERSION_METADATA') != [BOUNDED_PROFILE]:
        raise ValueError('unchanged bounded runtime metadata required')
    actual = normalized_commands(database, 'clang-cl', True)
    expected = normalized_commands(prior_database, 'clang-cl', False)
    if set(actual) != set(expected) or len(database) != 97:
        raise ValueError('all97 original compiler sources required')
    count = 0
    for name, row in actual.items():
        if len(row['arguments']) != len(expected[name]['arguments']):
            raise ValueError('original compiler argument count required')
        for index, token in enumerate(row['arguments']):
            if token.startswith('-DNEMO_SPEECH_VERSION_STR='):
                previous = expected[name]['arguments'][index]
                old, new = worker_version(4).removeprefix('nemo-speech-asr '), VERSION.removeprefix('nemo-speech-asr ')
                if old not in previous or token != previous.replace(old, new):
                    raise ValueError('only original bounded version token delta allowed')
                row['arguments'][index] = previous
                count += 1
    if actual != expected or count != 97:
        raise ValueError('compiler commands exceed original contraction/version deltas')
    result['candidate_compiler_admission'] = {'contraction_flag': FLAG, 'compiler_commands': 97,
        'version_tokens_changed': count,
        'normalization': 'Selected compiler alias, isolated source root, one contraction flag and exact diagnostic version token only.'}
    result.update(compiler_binary=expected_binary, compiler_path_aliases=aliases,
                  linker_binary=recipe['compiler']['linker_binary'])
    if result != record:
        raise ValueError('recorded compiler admission differs from complete replay')
    return result
