"""Replay the actual completed math-errno diagnostic using saved data only."""
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
import struct
import subprocess
import sys
import tarfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'windows-math-errno-ci-v1'
admitted = root / 'delivery-admission-v1'
delivered = admitted / 'delivered'
source = 'e76a3199a468f9207c8bc70ab861f2fe918c39c1'
run_id, job_id = 39039873138, 117416985932


def descriptor(path):
    value, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            size += len(block)
            value.update(block)
    return {'bytes': size, 'sha256': value.hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


identity = descriptor(Path(__file__))
env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}


def git_bytes(name):
    return subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo, env=env)


admission = read(admitted / 'qualification.json')
assert admission['source_commit'] == source and admission['run_id'] == run_id
assert admission['job_id'] == job_id
artifact_id = admission['artifact_id']
assert descriptor(root / 'artifact.zip') == admission['archive']
assert descriptor(admitted / 'artifact-members.json') == admission['artifact_members']
assert descriptor(cache / 'admit-windows-math-errno-delivery-v1.py') == admission['collector']
supervision = read(root / 'supervision-replay-v1.json')
assert supervision['source_commit'] == source and supervision['run_id'] == run_id
assert supervision['artifact_id'] == artifact_id and supervision['archive'] == admission['archive']
assert descriptor(cache / 'replay-windows-math-errno-supervision-v1.py') == supervision['collector']
assert supervision['delivery_admission'] == descriptor(admitted / 'qualification.json')
assert all(row['returncode'] == 0 and row['error'] is None for row in supervision['commands'])
assert all(row['observed_exit_preserved'] is True and row['owned_tree_reaped_and_job_closed'] is True
           for row in supervision['commands'])
bound = {name: descriptor(path) for name, path in {
    **{name: root / name for name in admission['metadata']},
    'delivery-admission.json': admitted / 'qualification.json',
    'artifact-members.json': admitted / 'artifact-members.json',
    'supervision-replay.json': root / 'supervision-replay-v1.json',
    'admit-windows-math-errno-delivery-v1.py': cache / 'admit-windows-math-errno-delivery-v1.py',
    'replay-windows-math-errno-supervision-v1.py': cache / 'replay-windows-math-errno-supervision-v1.py'}.items()}
assert all(descriptor(root / name) == value for name, value in admission['metadata'].items())
p = read(delivered / 'protocol.json')
assert len(p['tools']) == 86 and len(p['repository_files']) == 1411
assert (delivered / 'protocol.json').read_bytes() == git_bytes('tools/nemotron-diarization/windows-math-errno.protocol.json')
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = git_bytes(prefix + name)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import source_inventory
from ctest_output import require_single_test_pass
from pooled_runtime import BOUNDED_PROFILE, compatibility_patch_bytes, load_profile
from windows_compiler_metrics import compiler_evidence
from windows_math_errno_metrics import ORDER, FLAGS, ERRNO_FLAG, admit_math_driver, compare_native, timing_evidence
from windows_compiler_fp_metrics import FLAG, baseline, cache_value, normalized_commands
from windows_naming import verify_private_outputs
from windows_session_timing import file_inventory, summarize
from windows_worker_metrics import derive_probe, telemetry, version
from windows_worker_sources_v2 import verify_completed_source

members = read(admitted / 'artifact-members.json')
assert file_inventory(delivered) == members
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source and q['protocol'] == descriptor(delivered / 'protocol.json')
assert q['order'] == p['order'] == ORDER and q['limits'] == p['limits']
assert q['diagnostic_completed'] is True
assert all(q[key] is False for key in ['release_qualified', 'package_qualified', 'public_or_lifecycle_qualified'])
assert 'error' not in q
assert set(q['builds']) == set(q['commands']) == {'control', 'no-errno'}
assert supervision['commands_verified'] == sum(len(rows) for rows in q['commands'].values())
assert [case['compiler'] for case in q['cases']] == ORDER
assert 'timing_comparison' in q
assert q['host'] == read(delivered / 'evidence-control/receiver-host.stdout.log')
assert q['host']['avx2'] is True and q['host']['fma'] is True
_, lock = load_profile(BOUNDED_PROFILE)
assert q['assets'] == {item['name']: {'bytes': item['size'], 'sha256': item['sha256']}
                       for item in lock['artifacts'] if item['role'] != 'reference'}
work = root / 'math-errno-replay-v1'
work.mkdir()
replay = work / 'ggml'
replay.mkdir()
pristine = cache / 'NeMo-Speech.cpp'
archive = work / 'independent-pristine-llama.tar'
subprocess.run(['git', 'archive', '--format=tar', '-o', str(archive), lock['runtime']['llama_revision'],
                '--', '.', ':(exclude)models', ':(exclude)docs', ':(exclude)media'],
               cwd=pristine / 'llama.cpp', env=env, check=True)
archive_identity = descriptor(archive)
with tarfile.open(archive) as bundle:
    bundle.extractall(replay, filter='data')
first_audit = read(delivered / 'evidence-control/platform-llama-materialized-audit.json')
series_raw = subprocess.check_output(['git', 'show', lock['runtime']['revision'] + ':patches/series'], cwd=pristine, env=env)
series = [line.split('#', 1)[0].strip() for line in series_raw.decode().splitlines()]
series = [name for name in series if name]
assert len(series) == 19 and series == [item['name'] for item in first_audit['patch_series']]
patch_commands = []
for index, item in enumerate(first_audit['patch_series']):
    raw = subprocess.check_output(['git', 'show', lock['runtime']['revision'] + ':patches/' + item['name']], cwd=pristine, env=env)
    assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == {key: item[key] for key in ['bytes', 'sha256']}
    patch = work / ('patch-' + str(index) + '.diff')
    patch.write_bytes(raw.replace(b'\r\n', b'\n'))
    command = ['/usr/bin/git', 'apply', '--whitespace=nowarn', str(patch)]
    with (work / ('patch-' + str(index) + '.stdout.log')).open('xb') as out, (work / ('patch-' + str(index) + '.stderr.log')).open('xb') as err:
        result = subprocess.run(command, cwd=replay, env={**env, 'GIT_CEILING_DIRECTORIES': str(replay.parent)}, stdout=out, stderr=err)
    patch_commands.append({'argv': command, 'cwd': str(replay), 'returncode': result.returncode})
    result.check_returncode()
replayed = source_inventory(replay)
assert len(replayed) == 3425
native = cache / 'windows-worker-source-admission-v1/source-4'
verify_completed_source(native, native / 'build/gen2-cpu-diar', lock,
    cache / 'windows-worker-source-admission-v1/evidence-4/bounded-inference.patch', True,
    p['variants']['4']['source_sha256'])
builds = {}
for kind in ('control', 'no-errno'):
    evidence, sdk = delivered / ('evidence-' + kind), delivered / ('sdk-' + kind)
    build = q['builds'][kind]
    assert build['construction_completed'] is True and build['reported_version'] == version(4)
    assert file_inventory(sdk) == build['installed_files'] and len(build['installed_files']) == 66
    commands = q['commands'][kind]
    assert commands == read(evidence / 'commands.json') and 0 < len(commands) <= 128
    assert all(row['returncode'] == 0 and 'error' not in row for row in commands)
    assert (evidence / 'bounded-inference.patch').read_bytes() == compatibility_patch_bytes(lock)
    for patch in lock['runtime']['compatibility_patches']:
        assert descriptor(evidence / Path(patch['path']).name) == {key: patch[key] for key in ['bytes', 'sha256']}
    prior = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4'
    assert (evidence / 'diagnostic-inference.diff').read_bytes() == (prior / 'diagnostic-inference.diff').read_bytes()
    require_single_test_pass((evidence / 'model-free-state-test.stdout.log').read_text(), 'diar_state')
    assert (evidence / 'version-only.stdout.log').read_text().strip() == version(4)
    original, derived = tools / 'native_stream_probe_valid_mel.cpp', evidence / 'native_stream_probe_bounded.cpp'
    assert descriptor(original) == build['native_probe_source']['original']
    assert descriptor(derived) == build['native_probe_source']['derived']
    assert derived.read_bytes() == derive_probe(original.read_bytes(), 4)
    pe = work / ('pe-' + kind)
    pe.mkdir()
    verify_private_outputs(sdk, pe)
    assert read(pe / 'windows-private-pe.json') == read(evidence / 'windows-private-pe.json')
    audit = read(evidence / 'platform-llama-materialized-audit.json')
    assert descriptor(evidence / 'platform-llama-materialized-audit.json') == build['compiler_audit']
    assert audit['base_revision'] == lock['runtime']['llama_revision'] and audit['exact_independent_replay_match'] is True
    assert descriptor(evidence / 'platform-pristine-llama.tar') == audit['archive'] == archive_identity
    assert audit['files'] == replayed and audit['patch_series'] == first_audit['patch_series']
    assert descriptor(evidence / 'compile_commands.json') == audit['compile_database']
    database = read(evidence / 'compile_commands.json')
    materialized = PureWindowsPath(audit['compiled_tree_path'])
    native_root = materialized.parents[3]
    assert {PureWindowsPath(row['file']) for row in database} == {PureWindowsPath(name) for name in audit['all_compiler_sources']}
    for name, expected in audit['actual_compiler_sources'].items():
        assert replayed[name] == expected
    for name, expected in audit['all_compiler_sources'].items():
        path = PureWindowsPath(name)
        if path.is_relative_to(materialized):
            assert replayed[path.relative_to(materialized).as_posix()] == expected
        else:
            assert path.is_relative_to(native_root)
            assert descriptor(native / path.relative_to(native_root).as_posix()) == expected
    assert all(descriptor(evidence / name) == expected for name, expected in build['compiler_evidence_files'].items())
    aliases = build['compiler_admission']['compiler_path_aliases']
    assert aliases and all(value == build['compiler_binary'] for value in aliases.values())
    compiler = compiler_evidence((evidence / 'CMakeCache.txt').read_text(), database,
        {language: (evidence / ('compiler-' + language + '.cmake')).read_text() for language in ('C', 'CXX')},
        build['compiler'], 'clang-cl', aliases)
    # Replay flag/source/target admission from saved text only. Recorded binary
    # descriptors were measured in CI; do not stat Windows executables on macOS.
    prior_cache, prior_database, prior_compiler = baseline('clang-cl')
    assert build['compiler_binary'] == prior_compiler['compiler_binary']
    assert build['probe_compiler'] == q['builds']['control']['probe_compiler']
    assert {key: build['probe_compiler'][key] for key in ('bytes', 'sha256')} == baseline('msvc')[2]['compiler_binary']
    assert build['compiler_admission']['linker_binary'] == prior_compiler['linker_binary']
    cache_text = (evidence / 'CMakeCache.txt').read_text()
    for language in ('C', 'CXX'):
        field = 'CMAKE_' + language + '_FLAGS'
        flags = cache_value(cache_text, field)
        assert cache_value(cache_text, field + '_INIT') == FLAGS[kind]
        for flag in FLAGS[kind]:
            assert flags.count(flag) == 1
            flags.remove(flag)
        assert flags == cache_value(prior_cache, field)
        assert cache_value(cache_text, field + '_RELEASE') == cache_value(prior_cache, field + '_RELEASE')
    actual_commands = normalized_commands(database, 'clang-cl', True)
    if kind == 'no-errno':
        for row in actual_commands.values():
            assert row['arguments'].count(ERRNO_FLAG) == 1
            row['arguments'].remove(ERRNO_FLAG)
    assert actual_commands == normalized_commands(prior_database, 'clang-cl', False) and len(database) == 97
    compiler['math_errno_admission'] = {'variant': kind, 'flags': FLAGS[kind],
        'compiler_commands_matching_original': 97,
        'normalization': 'Selected compiler alias, isolated source root, one contraction flag and candidate math-errno flag only.'}
    compiler.update(compiler_binary=build['compiler_binary'], compiler_path_aliases=aliases,
                    linker_binary=build['compiler_admission']['linker_binary'])
    assert compiler == build['compiler_admission'] == read(evidence / 'compiler-admission.json')
    driver = admit_math_driver((evidence / 'compiler-math-driver.log').read_text(), kind)
    assert driver == build['math_driver']
    builds[kind] = {'sdk_compiler': 'clang-cl', 'sdk_files': 66, 'commands': len(commands), 'compiler': compiler, 'math_driver': driver,
                    'ggml_files': len(replayed), 'compiler_inputs': len(database)}
assert q['builds']['control']['probe_compiler'] == q['builds']['no-errno']['probe_compiler']
assert q['builds']['control']['native_probe_source'] == q['builds']['no-errno']['native_probe_source']
reference_path = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'
reference, reference_raw = read(reference_path), Path(str(reference_path) + '.f32').read_bytes()
cases, vectors = [], []
for index, kind in enumerate(ORDER):
    case = q['cases'][index]
    evidence = delivered / ('evidence-' + kind)
    phase = 'case-' + str(index) + '-' + kind
    path = evidence / (phase + '.json')
    actual, raw = read(path), Path(str(path) + '.f32').read_bytes()
    assert case['index'] == index and case['compiler'] == kind and case['command_started'] is True
    assert case['result'] == actual and case['result_file'] == descriptor(path)
    assert case['probabilities'] == descriptor(Path(str(path) + '.f32'))
    assert actual['push_count'] == 58 and actual['geometry'] == reference['geometry']
    workers = telemetry((evidence / (phase + '.log')).read_text(), 4)
    assert workers == case['workers'] and workers['effective_threads'] == 4
    assert summarize((evidence / (phase + '.log')).read_text(), 'control', actual['elapsed_micros']) == case['timing_control']
    assert compare_native(reference, actual, reference_raw, raw, 4, 4) == case['comparison']
    assert raw == reference_raw
    assert len(raw) == 192000
    vector = struct.unpack('<48000f', raw)
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in vector)
    vectors.append(vector)
    cases.append({'index': index, 'variant': kind, 'sdk_compiler': 'clang-cl', 'elapsed_micros': actual['elapsed_micros'],
                  'native_rtf': actual['elapsed_micros'] / 60e6, 'turns': len(actual['turns']),
                  'raw_probabilities': case['probabilities'], 'parity_passed': True})
timing = timing_evidence(q['cases'])
assert timing == q['timing_comparison']
errors = [abs(left - right) for left, right in zip(vectors[0], vectors[1])]
maximum = max(range(len(errors)), key=errors.__getitem__)
numeric = {'values_compared': 48000, 'maximum_absolute_probability_error': errors[maximum],
           'maximum_error_frame': maximum // 8, 'maximum_error_slot': maximum % 8,
           'maximum_error_values': [vector[maximum] for vector in vectors],
           'values_above_original_1e_6_tolerance': sum(error > 1e-6 for error in errors),
           'nonidentical_values': sum(error != 0 for error in errors),
           'mean_absolute_error': sum(errors) / len(errors),
           'half_probability_threshold_disagreements': sum((left >= .5) != (right >= .5) for left, right in zip(vectors[0], vectors[1]))}
assert file_inventory(delivered) == members and descriptor(root / 'artifact.zip') == admission['archive']
assert descriptor(Path(__file__)) == identity
bound_paths = {**{name: root / name for name in admission['metadata']},
    'delivery-admission.json': admitted / 'qualification.json',
    'artifact-members.json': admitted / 'artifact-members.json', 'supervision-replay.json': root / 'supervision-replay-v1.json',
    'admit-windows-math-errno-delivery-v1.py': cache / 'admit-windows-math-errno-delivery-v1.py',
    'replay-windows-math-errno-supervision-v1.py': cache / 'replay-windows-math-errno-supervision-v1.py'}
assert all(descriptor(path) == bound[name] for name, path in bound_paths.items())
dest = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-math-errno'
assert not dest.exists()
dest.mkdir()
copied = {}
for name, expected in members.items():
    if name in ('protocol.json', 'qualification.json') or (name.startswith('evidence-') and Path(name).suffix.lower() not in ('.tar', '.exe', '.dll', '.lib', '.obj', '.pdb')):
        assert Path(name).suffix.lower() not in ('.exe', '.dll', '.lib', '.obj', '.pdb')
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(delivered / name, target)
        assert descriptor(target) == expected
        copied[name] = expected
for name, path in bound_paths.items():
    shutil.copyfile(path, dest / name)
    assert descriptor(dest / name) == bound[name]
shutil.copyfile(Path(__file__), dest / 'collector.py')
assert descriptor(dest / 'collector.py') == identity
for name, expected in p['tools'].items():
    target = dest / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(git_bytes('tools/nemotron-diarization/' + name))
    assert descriptor(target) == expected
summary = {'schema_version': 1, 'collection_completed': True, 'source_commit': source,
    'run_id': run_id, 'job_id': job_id, 'artifact_id': artifact_id, 'archive': admission['archive'],
    'protocol': descriptor(delivered / 'protocol.json'), 'tools_verified': 86, 'repository_files_verified': 1411,
    'artifact_members_verified': len(members), 'artifact_evidence_files_copied': len(copied),
    'supervised_zero_exit_and_reaped_commands': supervision['commands_verified'], 'host': q['host'], 'builds': builds, 'cases': cases,
    'full_sequence_bit_exact': True, 'timing_comparison': timing, 'numerical_difference': numeric, 'independent_pristine_archive': archive_identity,
    'ggml_replay_commands': patch_commands, 'release_qualified': False, 'package_qualified': False,
    'public_or_lifecycle_qualified': False, 'new_compiler_sdk_model_calls_during_collection': 0,
    'limits': ['Both full-output variants pass byte identity for this fixed clip; broader quality/lifecycle qualification remains open.',
        'The 5-percent speedup and 10-percent drift predicates are diagnostic; strict native/public and all production requirements remain.',
        'CI bound physical compiler aliases to selected binaries; saved replay verifies recorded hashes/paths but does not rerun Windows samefile.',
        'SDK binaries and pristine archives remain local; Git retains full raw arrays, source/compiler evidence and all payload descriptors.',
        'Timing comparisons are limited to this same-host sequence, without combining prior CPUs or instances.']}
assert all(descriptor(dest / name) == expected for name, expected in copied.items())
assert all(descriptor(dest / name) == expected for name, expected in bound.items())
assert descriptor(dest / 'collector.py') == identity
assert file_inventory(delivered) == members
(dest / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
(dest / 'collection.json').write_text(json.dumps({'schema_version': 1, 'files': file_inventory(dest)}, indent=2) + '\n')
print(json.dumps({'receipt': str(dest), 'files': len(file_inventory(dest)), 'cases': cases, 'difference': numeric}))
