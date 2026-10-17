"""Replay full-model Q8 SDK, source/object, graph and numerical evidence using saved data only."""
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
root = cache / 'windows-q8-model-ci-v1'
admitted = root / 'delivery-admission-v1'
delivered = admitted / 'delivered'
source = '919ddf1fcf20f7b16d2e5143cffda14f83fb48df'
run_id, job_id = 39059902777, 117475526989


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
assert descriptor(cache / 'admit-windows-q8-model-delivery-v1.py') == admission['collector']
supervision = read(root / 'supervision-replay-v2.json')
assert supervision['source_commit'] == source and supervision['run_id'] == run_id
assert supervision['artifact_id'] == artifact_id and supervision['archive'] == admission['archive']
assert descriptor(cache / 'replay-windows-q8-model-supervision-v2.py') == supervision['collector']
assert supervision['delivery_admission'] == descriptor(admitted / 'qualification.json')
assert all(row['returncode'] == 0 and row['error'] is None for row in supervision['commands'])
assert all(row['observed_exit_preserved'] is True and row['owned_tree_reaped_and_job_closed'] is True
           for row in supervision['commands'])
bound = {name: descriptor(path) for name, path in {
    **{name: root / name for name in admission['metadata']},
    'delivery-admission.json': admitted / 'qualification.json',
    'artifact-members.json': admitted / 'artifact-members.json',
    'supervision-replay.json': root / 'supervision-replay-v2.json',
    'admit-windows-q8-model-delivery-v1.py': cache / 'admit-windows-q8-model-delivery-v1.py',
    'replay-windows-q8-model-supervision-v2.py': cache / 'replay-windows-q8-model-supervision-v2.py'}.items()}
assert all(descriptor(root / name) == value for name, value in admission['metadata'].items())
p = read(delivered / 'protocol.json')
assert len(p['tools']) == 101 and len(p['repository_files']) == 821
assert (delivered / 'protocol.json').read_bytes() == git_bytes('tools/nemotron-diarization/windows-q8-model.protocol.json')
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = git_bytes(prefix + name)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import source_inventory
from ctest_output import require_single_test_pass
from pooled_runtime import BOUNDED_PROFILE, compatibility_patch_bytes, load_profile
from windows_compiler_metrics import compiler_evidence as parent_compiler_evidence
from windows_compiler_fp_metrics import FLAG, baseline, cache_value, normalized_commands
from windows_q8_model_metrics import ORDER, metadata, version, derive_probe, compare_native, timing_evidence, admit_q8_audit, graph_vectors, telemetry
from windows_q8_model_sources import PATH, q8_patch, Q8_PATCH
from windows_q8_model_verify import verify_completed_source
from windows_naming import verify_private_outputs
from windows_session_timing import file_inventory, summarize

members = read(admitted / 'artifact-members.json')
assert file_inventory(delivered) == members
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source and q['protocol'] == descriptor(delivered / 'protocol.json')
assert q['order'] == p['order'] == ORDER and q['limits'] == p['limits']
assert q['diagnostic_completed'] is True and 'error' not in q
assert all(q[key] is False for key in ['release_qualified', 'package_qualified', 'public_or_lifecycle_qualified'])
assert set(q['builds']) == set(q['commands']) == {'control', 'mx4'}
assert supervision['commands_verified'] == sum(len(rows) for rows in q['commands'].values())
assert [case['variant'] for case in q['cases']] == ORDER
assert q['host'] == read(delivered / 'evidence-control/receiver-host.stdout.log')
assert q['host']['avx2'] is True and q['host']['fma'] is True
_, lock = load_profile(BOUNDED_PROFILE)
assert q['assets'] == {item['name']: {'bytes': item['size'], 'sha256': item['sha256']}
                       for item in lock['artifacts'] if item['role'] != 'reference'}
work = root / 'result-replay-v2'; work.mkdir()
proof_root = cache / 'windows-q8-model-source-admission-v2'
proof_path = proof_root / 'qualification.json'
proof_identity = descriptor(proof_path); proof = read(proof_path)
assert proof['completed'] is True and proof['compiled_sdk_or_model'] is False
assert len(proof['commands']) == 22 and all(row['returncode'] == 0 and row['process_tree_reaped'] is True for row in proof['commands'])
assert all(descriptor(Path(name)) == expected for name, expected in proof['inputs'].items())
proof_inputs = {name: descriptor(Path(name)) for name in proof['inputs']}
first_clone = q['commands']['control'][0]
assert first_clone['name'] == 'clone-native'
ci_root = PureWindowsPath(first_clone['command'][-1]).parent
builds, graph_reference, graph_count = {}, None, 0
unavailable_graph_libraries = {}
kernel_objects = 0

for kind in ('control', 'mx4'):
    evidence, sdk = delivered / ('evidence-' + kind), delivered / ('sdk-' + kind)
    ci_evidence = ci_root / ('evidence-' + kind)
    ci_native = ci_root / ('source-' + kind)
    ci_build = ci_native / 'build/gen2-cpu-diar'
    build = q['builds'][kind]
    assert build['construction_completed'] is True and build['reported_version'] == version(4, kind)
    assert file_inventory(sdk) == build['installed_files'] and len(build['installed_files']) == 66
    commands = q['commands'][kind]
    assert commands == read(evidence / 'commands.json') and 0 < len(commands) <= 128
    assert all(row['returncode'] == 0 and 'error' not in row for row in commands)
    assert PureWindowsPath(commands[0]['command'][-1]) == ci_native
    assert (evidence / 'bounded-inference.patch').read_bytes() == compatibility_patch_bytes(lock)
    for item in lock['runtime']['compatibility_patches']:
        assert descriptor(evidence / Path(item['path']).name) == {key: item[key] for key in ('bytes', 'sha256')}
    native = proof_root / ('source-' + kind)
    verify_completed_source(native, native / 'build/gen2-cpu-diar', lock,
        proof_root / ('evidence-' + kind) / 'bounded-inference.patch', True,
        p['variants'][kind]['source_sha256'], kind)
    materialized = native / 'build/gen2-cpu-diar/_deps/llama.cpp'
    replayed = source_inventory(materialized)
    assert replayed == proof['variants'][kind]['materialized_files'] and len(replayed) == 3425
    audit = read(evidence / 'platform-llama-materialized-audit.json')
    assert descriptor(evidence / 'platform-llama-materialized-audit.json') == build['compiler_audit']
    assert audit['files'] == replayed
    assert admit_q8_audit(audit, kind) == build['exact_source']
    assert descriptor(evidence / 'platform-pristine-llama.tar') == audit['archive']
    assert descriptor(evidence / 'compile_commands.json') == audit['compile_database']
    if kind == 'mx4': assert (evidence / Q8_PATCH).read_bytes() == q8_patch()
    database = read(evidence / 'compile_commands.json')
    audit_inputs = {PureWindowsPath(name): value for name, value in audit['all_compiler_sources'].items()}
    assert {PureWindowsPath(row['file']) for row in database} == set(audit_inputs)
    for name, expected in audit['all_compiler_sources'].items():
        path = PureWindowsPath(name)
        assert path.is_relative_to(ci_native)
        assert descriptor(native / path.relative_to(ci_native).as_posix()) == expected
    require_single_test_pass((evidence / 'model-free-state-test.stdout.log').read_text(), 'diar_state')
    assert (evidence / 'version-only.stdout.log').read_text().strip() == version(4, kind)
    original, derived = tools / 'native_stream_probe_valid_mel.cpp', evidence / 'native_stream_probe_bounded.cpp'
    assert descriptor(original) == build['native_probe_source']['original']
    assert descriptor(derived) == build['native_probe_source']['derived']
    assert derived.read_bytes() == derive_probe(original.read_bytes(), 4, kind)
    pe = work / ('pe-' + kind); pe.mkdir()
    verify_private_outputs(sdk, pe)
    assert read(pe / 'windows-private-pe.json') == read(evidence / 'windows-private-pe.json')
    assert all(descriptor(evidence / name) == expected for name, expected in build['compiler_evidence_files'].items())
    aliases = build['compiler_admission']['compiler_path_aliases']
    assert aliases and all(value == build['compiler_binary'] for value in aliases.values())
    cache_text = (evidence / 'CMakeCache.txt').read_text()
    compiler = parent_compiler_evidence(cache_text, database,
        {language: (evidence / ('compiler-' + language + '.cmake')).read_text() for language in ('C', 'CXX')},
        build['compiler'], 'clang-cl', aliases)
    prior_cache, prior_database, prior_compiler = baseline('clang-cl')
    assert build['compiler_binary'] == prior_compiler['compiler_binary']
    assert build['probe_compiler'] == q['builds']['control']['probe_compiler']
    assert {key: build['probe_compiler'][key] for key in ('bytes', 'sha256')} == baseline('msvc')[2]['compiler_binary']
    assert build['compiler_admission']['linker_binary'] == prior_compiler['linker_binary']
    for language in ('C', 'CXX'):
        field = 'CMAKE_' + language + '_FLAGS'; flags = cache_value(cache_text, field)
        assert flags.count(FLAG) == 1 and cache_value(cache_text, field + '_INIT') == [FLAG]
        flags.remove(FLAG)
        assert flags == cache_value(prior_cache, field)
        assert cache_value(cache_text, field + '_RELEASE') == cache_value(prior_cache, field + '_RELEASE')
    assert cache_value(cache_text, 'NEMO_SPEECH_VERSION_METADATA') == [metadata(4, kind)]
    actual_commands = normalized_commands(database, 'clang-cl', True)
    expected_commands = normalized_commands(prior_database, 'clang-cl', False)
    version_tokens = 0
    for name, row in actual_commands.items():
        for index, token in enumerate(row['arguments']):
            if token.startswith('-DNEMO_SPEECH_VERSION_STR='):
                expected = expected_commands[name]['arguments'][index]
                assert token == expected.replace('0.2.0+gen2-diagnostic-cpu-threads-4', version(4, kind).removeprefix('nemo-speech-asr '))
                row['arguments'][index] = expected; version_tokens += 1
    assert actual_commands == expected_commands and len(database) == version_tokens == 97
    compiler['q8_model_admission'] = {'contraction_flag': FLAG, 'compiler_commands': 97,
        'version_tokens_changed': 97,
        'normalization': 'Selected compiler alias, isolated source root, one contraction flag and exact diagnostic version token only.'}
    compiler.update(compiler_binary=build['compiler_binary'], compiler_path_aliases=aliases,
                    linker_binary=build['compiler_admission']['linker_binary'])
    assert compiler == build['compiler_admission'] == read(evidence / 'compiler-admission.json')
    kernel = build['kernel']; units = kernel['translation_units']
    assert len(units) == 97
    bindings, objects = {}, set()
    for unit, row in zip(units, database):
        assert unit['command'] == row
        source_path, obj = PureWindowsPath(row['file']), PureWindowsPath(row['output'])
        assert PureWindowsPath(row['directory']) == ci_build
        if not obj.is_absolute(): obj = ci_build / obj
        assert obj.is_relative_to(ci_build) and obj not in objects
        objects.add(obj)
        retained = 'compiler-objects/' + obj.relative_to(ci_build).as_posix()
        assert unit['retained_object'] == retained
        assert descriptor(evidence / retained) == unit['object']
        assert unit['source'] == audit_inputs[PureWindowsPath(row['file'])]
        bindings.update({str(source_path): unit['source'], str(obj): unit['object'], str(ci_evidence / retained): unit['object']})
        kernel_objects += 1
    sg_rows = [row for row in database if PureWindowsPath(row['file']) == ci_build / '_deps/llama.cpp' / PATH]
    sg, = sg_rows
    assert kernel['compiler_command'] == sg
    sg_unit, = [unit for unit in units if unit['command'] == sg]
    assert descriptor(evidence / 'kernel.obj') == sg_unit['object']
    dump, = [row for row in commands if row['name'] == 'kernel-disassembly']
    assert dump['command'][1:] == ['/nologo', '/disasm', '/symbols', str(ci_evidence / 'kernel.obj')]
    bindings[str(PureWindowsPath(dump['command'][0]))] = kernel['bindings'][str(PureWindowsPath(dump['command'][0]))]
    for name in ('kernel.obj', 'kernel-disassembly.stdout.log', 'kernel-disassembly.log'):
        bindings[str(ci_evidence / name)] = descriptor(evidence / name)
    assert 'llamafile_sgemm' in (evidence / 'kernel-disassembly.stdout.log').read_text()
    assert bindings == kernel['bindings']
    controls = build['model_free_controls']; control_bindings = {}
    unavailable_libraries = {}
    assert controls['vectors'] == 36 and [case['threads'] for case in controls['cases']] == [1, 2, 4]
    binary = 'matrix-profile-control.exe'
    for name, expected in controls['bindings'].items():
        path = PureWindowsPath(name)
        if path.is_relative_to(ci_evidence): actual_path = evidence / path.relative_to(ci_evidence).as_posix()
        elif path.name == 'ggml_matrix_profile_control.cpp': actual_path = tools / path.name
        else:
            assert path.is_relative_to(ci_build) and path.name in {
                'gen2_nemo_windows_ggml.lib', 'gen2_nemo_windows_ggml-base.lib', 'gen2_nemo_windows_ggml-cpu.lib'}
            # Build-only import libraries were hashed/rechecked in CI but not
            # delivered by this producer. Preserve this missing-payload limit.
            assert type(expected['bytes']) is int and expected['bytes'] > 0
            assert len(expected['sha256']) == 64 and all(c in '0123456789abcdef' for c in expected['sha256'])
            unavailable_libraries[name] = expected
            control_bindings[name] = expected
            continue
        assert descriptor(actual_path) == expected
        control_bindings[name] = expected
    for case in controls['cases']:
        threads = case['threads']; phase = f'matrix-control-{threads}'
        paths = [evidence / (phase + f'.{shape}.{repeat}.f32') for shape in range(4) for repeat in range(3)]
        graph_reference = graph_vectors((evidence / (phase + '.stdout.log')).read_text(), threads,
                                        [path.read_bytes() for path in paths], graph_reference)
        expected = {str(ci_evidence / path.name): descriptor(path) for path in
                    [*paths, evidence / (phase + '.stdout.log'), evidence / (phase + '.log')]}
        assert expected == case['outputs_and_logs']
        assert all(control_bindings[name] == value for name, value in expected.items())
        call, = [row for row in commands if row['name'] == phase]
        assert call['command'] == [str(ci_evidence / binary), str(threads), str(ci_evidence / phase)]
        graph_count += 12
    assert len(control_bindings) == 47 and len(unavailable_libraries) == 3
    unavailable_graph_libraries[kind] = unavailable_libraries
    builds[kind] = {'sdk_compiler': 'clang-cl', 'sdk_files': 66, 'commands': len(commands), 'compiler': compiler,
                    'ggml_files': len(replayed), 'compiler_inputs': 97, 'retained_compiler_objects': len(units),
                    'model_free_graph_processes': 3, 'model_free_vectors': 36, 'source_admission': build['exact_source'],
                    'graph_import_library_payloads_unavailable': unavailable_libraries}

reference_path = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'
reference, reference_raw = read(reference_path), Path(str(reference_path) + '.f32').read_bytes()
cases, model_bindings = [], {}
for index, kind in enumerate(ORDER):
    case = q['cases'][index]; evidence = delivered / ('evidence-' + kind)
    ci_evidence = ci_root / ('evidence-' + kind)
    phase = f'case-{index}-{kind}'; path = evidence / (phase + '.json')
    actual, raw = read(path), Path(str(path) + '.f32').read_bytes()
    assert type(case['index']) is int and case['index'] == index and case['variant'] == kind and case['command_started'] is True
    assert case['result'] == actual and case['result_file'] == descriptor(path)
    assert case['probabilities'] == descriptor(Path(str(path) + '.f32'))
    assert actual['push_count'] == 58 and actual['geometry'] == reference['geometry']
    log = (evidence / (phase + '.log')).read_text()
    assert telemetry(log, 4) == case['workers'] and case['workers']['effective_threads'] == 4
    assert summarize(log, 'control', actual['elapsed_micros']) == case['timing_control']
    assert compare_native(reference, actual, reference_raw, raw, kind) == case['comparison']
    assert raw == reference_raw and len(raw) == 192000
    for path in (path, Path(str(path) + '.f32'), evidence / (phase + '.log'), evidence / (phase + '.stdout.log')):
        model_bindings[str(ci_evidence / path.name)] = descriptor(path)
    cases.append({'index': index, 'variant': kind, 'elapsed_micros': actual['elapsed_micros'],
                  'native_rtf': actual['elapsed_micros'] / 60e6, 'turns': len(actual['turns']),
                  'raw_probabilities': case['probabilities'], 'parity_passed': True})
assert model_bindings == q['model_output_bindings']
assert timing_evidence(q['cases']) == q['timing_comparison']
assert graph_count == 72 and kernel_objects == 194
assert file_inventory(delivered) == members
assert descriptor(root / 'artifact.zip') == admission['archive']
assert descriptor(proof_path) == proof_identity and all(descriptor(Path(name)) == expected for name, expected in proof_inputs.items())
assert descriptor(Path(__file__)) == identity
bound_paths = {**{name: root / name for name in admission['metadata']},
    'delivery-admission.json': admitted / 'qualification.json', 'artifact-members.json': admitted / 'artifact-members.json',
    'supervision-replay.json': root / 'supervision-replay-v2.json',
    'admit-windows-q8-model-delivery-v1.py': cache / 'admit-windows-q8-model-delivery-v1.py',
    'replay-windows-q8-model-supervision-v2.py': cache / 'replay-windows-q8-model-supervision-v2.py'}
assert all(descriptor(path) == bound[name] for name, path in bound_paths.items())
result = {'schema_version': 1, 'collection_completed': True, 'source_commit': source, 'run_id': run_id, 'job_id': job_id,
    'artifact_id': artifact_id, 'archive': admission['archive'], 'protocol': descriptor(delivered / 'protocol.json'),
    'collector': identity, 'delivery_admission': descriptor(admitted / 'qualification.json'),
    'supervision_replay': descriptor(root / 'supervision-replay-v2.json'), 'receiver_inputs': bound,
    'independent_source_only_proof': {'path': str(proof_path), **proof_identity},
    'tools_verified': 101, 'repository_files_verified': 821, 'artifact_members_verified': len(members),
    'supervised_zero_exit_and_reaped_commands': supervision['commands_verified'], 'host': q['host'], 'builds': builds, 'cases': cases,
    'per_shape_graph_vectors_bit_exact': True, 'graph_vectors_verified': graph_count, 'retained_compiler_objects_verified': kernel_objects,
    'full_sequence_bit_exact': True, 'timing_comparison': q['timing_comparison'],
    'graph_input_payloads_fully_rehashed': False, 'all_preregistered_evidence_retained': False,
    'unavailable_graph_import_library_payloads': unavailable_graph_libraries,
    'release_qualified': False, 'package_qualified': False, 'public_or_lifecycle_qualified': False,
    'new_compiler_sdk_model_calls_during_collection': 0,
    'limits': ['Native diagnostic only; original quality/performance/production/public/release requirements remain.',
        'Three build-only GGML import libraries per SDK were bound and rechecked by CI but not delivered. Their recorded hashes/command associations are preserved; no independent payload rehash or complete preregistered retention pass is claimed.',
        'Physical compiler aliases/binary hashes and source/object execution were observed in CI; replay checks records and retained bytes without executing Windows tools.',
        'Independent original19/candidate20 patch materializations were executed in the pinned local source-only preparation; complete current maps and inputs are re-admitted here.',
        'Earlier failed microkernel and worker-observer predictions remain failed; no timing averaging, rerun or automatic promotion.']}
with (root / 'result-replay-v2.json').open('x') as stream:
    stream.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({'complete': True, 'commands': supervision['commands_verified'], 'graphs': graph_count,
                  'retained_objects': kernel_objects, 'cases': cases, 'timing': q['timing_comparison']}))
