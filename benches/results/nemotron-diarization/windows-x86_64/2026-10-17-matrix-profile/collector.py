"""Success-only full matrix evidence replay; unsuccessful runs use the separate failure collector. No native calls."""
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
root = cache / 'windows-matrix-profile-ci-v1'
admitted = root / 'delivery-admission-v1'
delivered = admitted / 'delivered'
source = '59d1ff1a34dcc3d6bd1e29dda33ee47e3bd8a832'
run_id, job_id = 39049459015, 117444894388


def descriptor(path):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            size += len(block)
            digest.update(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


identity = descriptor(Path(__file__))
env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}


def git_bytes(name):
    return subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo, env=env)


admission = read(admitted / 'qualification.json')
assert admission['source_commit'] == source and admission['run_id'] == run_id and admission['job_id'] == job_id
artifact_id = admission['artifact_id']
assert descriptor(root / 'artifact.zip') == admission['archive']
assert descriptor(admitted / 'artifact-members.json') == admission['artifact_members']
assert descriptor(cache / 'admit-windows-matrix-profile-delivery-v1.py') == admission['collector']
supervision = read(root / 'supervision-replay-v1.json')
assert supervision['source_commit'] == source and supervision['run_id'] == run_id
assert supervision['artifact_id'] == artifact_id and supervision['archive'] == admission['archive']
assert descriptor(cache / 'replay-windows-matrix-profile-supervision-v1.py') == supervision['collector']
assert supervision['delivery_admission'] == descriptor(admitted / 'qualification.json')
assert all(row['returncode'] == 0 and row['error'] is None for row in supervision['commands'])
assert all(row['observed_exit_preserved'] is True and row['owned_tree_reaped_and_job_closed'] is True
           for row in supervision['commands'])
bound_paths = {**{name: root / name for name in admission['metadata']},
    'delivery-admission.json': admitted / 'qualification.json',
    'artifact-members.json': admitted / 'artifact-members.json',
    'supervision-replay.json': root / 'supervision-replay-v1.json',
    'admit-windows-matrix-profile-delivery-v1.py': cache / 'admit-windows-matrix-profile-delivery-v1.py',
    'replay-windows-matrix-profile-supervision-v1.py': cache / 'replay-windows-matrix-profile-supervision-v1.py'}
bound = {name: descriptor(path) for name, path in bound_paths.items()}
assert all(descriptor(root / name) == value for name, value in admission['metadata'].items())
p = read(delivered / 'protocol.json')
assert (delivered / 'protocol.json').read_bytes() == git_bytes('tools/nemotron-diarization/windows-matrix-profile.protocol.json')
assert len(p['tools']) == 110 and len(p['repository_files']) == 1804
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = git_bytes(prefix + name)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import source_inventory
from ctest_output import require_single_test_pass
from ggml_matrix_profile import MAX_LOG_BYTES, summarize
from windows_matrix_profile_report import admit_saved_profile
from pooled_runtime import BOUNDED_PROFILE, compatibility_patch_bytes, load_profile
from windows_compiler_metrics import compiler_evidence as parent_compiler_evidence
from windows_compiler_fp_metrics import FLAG, baseline, cache_value, normalized_commands
from windows_matrix_profile_metrics import (ORDER, admit_profile_audit, compare_native, derive_probe,
    metadata, profile_signature, telemetry, timing_evidence, version)
from windows_matrix_profile_sources import PROFILE_PATCH, profile_patch
from windows_matrix_profile_verify import verify_completed_source
from windows_naming import verify_private_outputs
from windows_session_timing import file_inventory
from windows_worker_metrics import version as worker_version

members = read(admitted / 'artifact-members.json')
assert file_inventory(delivered) == members
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source and q['protocol'] == descriptor(delivered / 'protocol.json')
assert q['order'] == p['order'] == ORDER and q['limits'] == p['limits']
assert all(q[key] is False for key in ['release_qualified', 'package_qualified', 'public_or_lifecycle_qualified'])
assert q['diagnostic_completed'] is True and 'error' not in q
assert q['commands'] == read(delivered / 'evidence/commands.json')
assert supervision['commands_verified'] == len(q['commands'])
assert all(row['returncode'] == 0 and 'error' not in row for row in q['commands'])
assert q['host'] == read(delivered / 'evidence/receiver-host.stdout.log')
assert q['host']['avx2'] is True and q['host']['fma'] is True
_, lock = load_profile(BOUNDED_PROFILE)
assert q['assets'] == {item['name']: {'bytes': item['size'], 'sha256': item['sha256']}
                       for item in lock['artifacts'] if item['role'] != 'reference'}
work = root / 'cpu-profile-replay-v1'
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
evidence, sdk = delivered / 'evidence', delivered / 'sdk'
audit = read(evidence / 'platform-llama-materialized-audit.json')
series_raw = subprocess.check_output(['git', 'show', lock['runtime']['revision'] + ':patches/series'], cwd=pristine, env=env)
series = [line.split('#', 1)[0].strip() for line in series_raw.decode().splitlines()]
series = [name for name in series if name]
assert len(series) == 19 and series + [PROFILE_PATCH] == [item['name'] for item in audit['patch_series']]
patch_commands = []
for index, item in enumerate(audit['patch_series']):
    raw = profile_patch() if index == 19 else subprocess.check_output(
        ['git', 'show', lock['runtime']['revision'] + ':patches/' + item['name']], cwd=pristine, env=env)
    assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == {key: item[key] for key in ['bytes', 'sha256']}
    patch = work / ('patch-' + str(index) + '.diff')
    patch.write_bytes(raw.replace(b'\r\n', b'\n'))
    command = ['/usr/bin/git', 'apply', '--whitespace=nowarn', str(patch)]
    with (work / ('patch-' + str(index) + '.stdout.log')).open('xb') as out, (work / ('patch-' + str(index) + '.stderr.log')).open('xb') as err:
        result = subprocess.run(command, cwd=replay, env={**env, 'GIT_CEILING_DIRECTORIES': str(replay.parent)}, stdout=out, stderr=err)
    patch_commands.append({'argv': command, 'cwd': str(replay), 'returncode': result.returncode})
    result.check_returncode()
replayed = source_inventory(replay)
assert audit['files'] == replayed and len(replayed) == 3425
assert audit['base_revision'] == lock['runtime']['llama_revision'] and audit['exact_independent_replay_match'] is True
assert descriptor(evidence / 'platform-pristine-llama.tar') == audit['archive'] == archive_identity
build = q['build']
assert build['construction_completed'] is True and build['reported_version'] == version(4)
assert build['exact_profile_source'] == admit_profile_audit(audit)
assert file_inventory(sdk) == build['installed_files'] and len(build['installed_files']) == 66
assert (evidence / 'bounded-inference.patch').read_bytes() == compatibility_patch_bytes(lock)
for patch in lock['runtime']['compatibility_patches']:
    assert descriptor(evidence / Path(patch['path']).name) == {key: patch[key] for key in ['bytes', 'sha256']}
assert (evidence / PROFILE_PATCH).read_bytes() == profile_patch()
require_single_test_pass((evidence / 'model-free-state-test.stdout.log').read_text(), 'diar_state')
assert (evidence / 'version-only.stdout.log').read_text().strip() == version(4)
original, derived = tools / 'native_stream_probe_valid_mel.cpp', evidence / 'native_stream_probe_bounded.cpp'
assert descriptor(original) == build['native_probe_source']['original']
assert descriptor(derived) == build['native_probe_source']['derived']
assert derived.read_bytes() == derive_probe(original.read_bytes(), 4)
pe = work / 'pe'
pe.mkdir()
verify_private_outputs(sdk, pe)
assert read(pe / 'windows-private-pe.json') == read(evidence / 'windows-private-pe.json')
assert descriptor(evidence / 'platform-llama-materialized-audit.json') == build['compiler_audit']
assert descriptor(evidence / 'compile_commands.json') == audit['compile_database']
database = read(evidence / 'compile_commands.json')
assert len(database) == 97
materialized = PureWindowsPath(audit['compiled_tree_path'])
native_root = materialized.parents[3]
native = cache / 'windows-matrix-profile-source-admission-v1/source'
verify_completed_source(native, native / 'build/gen2-cpu-diar', lock,
    cache / 'windows-matrix-profile-source-admission-v1/evidence/bounded-inference.patch', True,
    p['variants']['4']['source_sha256'])
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
compiler = parent_compiler_evidence((evidence / 'CMakeCache.txt').read_text(), database,
    {language: (evidence / ('compiler-' + language + '.cmake')).read_text() for language in ('C', 'CXX')},
    build['compiler'], 'clang-cl', aliases)
# Saved Windows binary/physical-alias observations are retained; no Windows stat on macOS.
prior_cache, prior_database, prior = baseline('clang-cl')
assert build['compiler_binary'] == prior['compiler_binary']
assert {key: build['probe_compiler'][key] for key in ('bytes', 'sha256')} == baseline('msvc')[2]['compiler_binary']
assert build['compiler_admission']['linker_binary'] == prior['linker_binary']
cache_text = (evidence / 'CMakeCache.txt').read_text()
for language in ('C', 'CXX'):
    field = 'CMAKE_' + language + '_FLAGS'
    flags = cache_value(cache_text, field)
    assert flags.count(FLAG) == 1 and cache_value(cache_text, field + '_INIT') == [FLAG]
    flags.remove(FLAG)
    assert flags == cache_value(prior_cache, field)
    assert cache_value(cache_text, field + '_RELEASE') == cache_value(prior_cache, field + '_RELEASE')
assert cache_value(cache_text, 'NEMO_SPEECH_VERSION_METADATA') == [metadata(4)]
actual_commands = normalized_commands(database, 'clang-cl', True)
expected_commands = normalized_commands(prior_database, 'clang-cl', False)
assert set(actual_commands) == set(expected_commands)
metadata_tokens = 0
for name, row in actual_commands.items():
    assert len(row['arguments']) == len(expected_commands[name]['arguments'])
    for index, token in enumerate(row['arguments']):
        if token.startswith('-DNEMO_SPEECH_VERSION_STR='):
            prior_token = expected_commands[name]['arguments'][index]
            old, new = worker_version(4).removeprefix('nemo-speech-asr '), version(4).removeprefix('nemo-speech-asr ')
            assert old in prior_token and token == prior_token.replace(old, new)
            row['arguments'][index] = prior_token
            metadata_tokens += 1
assert actual_commands == expected_commands and metadata_tokens == 97
compiler['matrix_profile_admission'] = {'contraction_flag': FLAG, 'compiler_commands': 97,
    'version_tokens_changed': 97,
    'normalization': 'Selected compiler alias, isolated source root, one contraction flag and exact diagnostic version token only.'}
compiler.update(compiler_binary=build['compiler_binary'], compiler_path_aliases=aliases,
                linker_binary=build['compiler_admission']['linker_binary'])
assert compiler == build['compiler_admission'] == read(evidence / 'compiler-admission.json')

control = q['model_free_controls']
assert len(control['cases']) == 12
assert descriptor(evidence / 'matrix-profile-control.exe') == control['binary']
assert len(control['inputs']) == 4
assert {PureWindowsPath(name).name for name in control['inputs']} == {
    'ggml_matrix_profile_control.cpp','gen2_nemo_windows_ggml.lib',
    'gen2_nemo_windows_ggml-base.lib','gen2_nemo_windows_ggml-cpu.lib'}
for name,value in control['inputs'].items():
    basename=PureWindowsPath(name).name
    if basename=='ggml_matrix_profile_control.cpp':assert descriptor(tools/basename)==value
    else:
        # Build-tree GGML import libraries are recorded/rechecked on CI, not installed in the SDK.
        # Preserve those actual identities without inventing unavailable local payload verification.
        assert PureWindowsPath(name).is_absolute()
        assert type(value['bytes'])is int and value['bytes']>0
        assert len(value['sha256'])==64 and all(c in '0123456789abcdef'for c in value['sha256'])
control_reference={};control_files={}
for position,case in enumerate(control['cases']):
    threads,index=(1,2,4)[position//4],position%4;enabled=ORDER[index]
    assert case['threads']==threads and case['index']==index and case['enabled']is enabled
    phase=f'matrix-control-{threads}-{index}'
    assert (evidence/(phase+'.stdout.log')).read_text().splitlines()==[
        f'matrix-control threads={threads} shape={shape} quantized={shape%2} columns={32 if shape<2 else 1} repeat={repeat} bytes={8192 if shape<2 else 256}'
        for shape in range(4)for repeat in range(3)]
    log_path=evidence/(phase+'.log');assert log_path.stat().st_size<=MAX_LOG_BYTES
    observed=summarize(log_path.read_text(),enabled,expected_graphs=12,expected_threads=threads,expected_matrices=1)
    observed.pop('cpu');assert observed==case['profile']
    command,=[row for row in q['commands']if row['name']==phase]
    win_evidence=PureWindowsPath(command['command'][0]).parent
    assert PureWindowsPath(command['command'][0]).name=='matrix-profile-control.exe'
    assert command['command'][1]==str(threads) and PureWindowsPath(command['command'][2])==win_evidence/phase
    if enabled:
        assert len({g['pool']for g in observed['graphs']})==1
        for graph_index,graph in enumerate(observed['graphs']):
            shape=graph_index//3
            for worker in graph['workers']:
                matrix,=worker['matrices']
                assert worker['fused']==1 and matrix['route']==[2,4,5,5][shape]
                assert matrix['types']==[8 if shape%2 else 0,0,0]
                assert matrix['a']==[64,64,1,1]and matrix['b']==[64,32 if shape<2 else 1,1,1]
                assert matrix['first'][:3]==([1,1,0]if shape==0 else[1,0,1])
                assert matrix['second'][:3]==([1,1,0]if shape==1 else[1,0,1]if shape==3 else[0,0,0])
                assert matrix['conversion'][0]==shape%2
    assert set(case['outputs'])=={phase+f'.{shape}.{repeat}.f32'for shape in range(4)for repeat in range(3)}
    for shape in range(4):
        for repeat in range(3):
            name=phase+f'.{shape}.{repeat}.f32';path=evidence/name;raw=path.read_bytes()
            assert descriptor(path)==case['outputs'][name] and len(raw)==(8192 if shape<2 else 256)
            values=[v for v,in struct.iter_unpack('<f',raw)]
            assert all(math.isfinite(v)for v in values)and any(v!=0 for v in values)
            if shape not in control_reference:control_reference[shape]=raw
            assert raw==control_reference[shape];control_files[name]=descriptor(path)
    for suffix in ('.stdout.log','.log'):control_files[phase+suffix]=descriptor(evidence/(phase+suffix))
assert len(control_files)==168 and len(control['outputs_and_logs'])==168
assert {PureWindowsPath(name).name:value for name,value in control['outputs_and_logs'].items()}==control_files
assert all(PureWindowsPath(name).parent==win_evidence for name in control['outputs_and_logs'])

reference_path = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'
reference, reference_raw = read(reference_path), Path(str(reference_path) + '.f32').read_bytes()
assert len(q['cases']) == 4
cases, signatures, operations, matrix_groups = [], [], [], []
model_bindings = {}
for index, enabled in enumerate(ORDER):
    case = q['cases'][index]
    phase = 'case-' + str(index)
    path = evidence / (phase + '.json')
    actual, raw = read(path), Path(str(path) + '.f32').read_bytes()
    assert case['index'] == index and case['enabled'] is enabled and case['command_started'] is True
    assert case['result'] == actual and case['result_file'] == descriptor(path)
    assert case['probabilities'] == descriptor(Path(str(path) + '.f32'))
    assert actual['push_count'] == 58 and actual['geometry'] == reference['geometry']
    log_path = evidence / (phase + '.log')
    assert log_path.stat().st_size <= MAX_LOG_BYTES
    log = log_path.read_text()
    workers = telemetry(log, 4)
    assert workers == case['workers'] and workers['effective_threads'] == 4
    assert compare_native(reference, actual, reference_raw, raw) == case['comparison']
    assert raw == reference_raw and len(raw) == 192000
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in struct.unpack('<48000f', raw))
    profile = summarize(log, enabled, expected_graphs=p['expected_graphs'], expected_threads=4, expected_matrices=p['expected_matrices'])
    saved_profile=evidence/(phase+'.matrix-profile.json')
    assert admit_saved_profile(saved_profile,case['profile'],profile)==case['profile']
    for item in (path,Path(str(path)+'.f32'),log_path,evidence/(phase+'.stdout.log'),saved_profile):
        model_bindings[item.name]=descriptor(item)
    if enabled:
        assert len({graph['pool'] for graph in profile['graphs']}) == 1
        signatures.append(profile_signature(profile))
        # Each worker is reported separately: worker intervals overlap in real time.
        by_worker = {}
        for graph in profile['graphs']:
            for worker in graph['workers']:
                key = str(worker['thread'])
                totals = by_worker.setdefault(key, {})
                for op in worker['operations']:
                    slot = totals.setdefault(str(op['id']), {'id': op['id'], 'name': op['name'], 'roots': 0,
                        'fused': 0, 'kernel_us': 0, 'barrier_us': 0})
                    for field in ('roots', 'fused', 'kernel_us', 'barrier_us'):
                        slot[field] += op[field]
        operations.append({'index': index, 'per_worker': by_worker})
        grouped={}
        fields=('types','names','a','b','c','sa','sb','sc','dot','route')
        for graph in profile['graphs']:
            for worker in graph['workers']:
                worker_groups=grouped.setdefault(str(worker['thread']),{})
                for matrix in worker['matrices']:
                    key=json.dumps({name:matrix[name]for name in fields},sort_keys=True,separators=(',',':'))
                    slot=worker_groups.setdefault(key,{**{name:matrix[name]for name in fields},'roots':0,'kernel_us':0,
                        'first':[0,0,0,0],'second':[0,0,0,0],'conversion':[0,0],'inner_wait_us':0,'chunks':[0,0]})
                    slot['roots']+=1
                    for name in ('kernel_us','inner_wait_us'):slot[name]+=matrix[name]
                    for name in ('first','second','conversion','chunks'):
                        slot[name]=[a+b for a,b in zip(slot[name],matrix[name])]
        for thread,groups in grouped.items():
            mul_mat=next(op for op in by_worker[thread].values()if op['name']=='MUL_MAT')
            assert sum(value['roots']for value in groups.values())==84*129
            assert sum(value['kernel_us']for value in groups.values())==mul_mat['kernel_us']
        matrix_groups.append({'index':index,'per_worker':{thread:list(groups.values())for thread,groups in grouped.items()}})

    cases.append({'index': index, 'enabled': enabled, 'elapsed_micros': actual['elapsed_micros'],
        'native_rtf': actual['elapsed_micros'] / 60e6, 'turns': len(actual['turns']),
        'raw_probabilities': case['probabilities'], 'parity_passed': True})
assert signatures[0] == signatures[1]
assert len(q['model_output_bindings'])==20 and all(PureWindowsPath(name).parent==win_evidence for name in q['model_output_bindings'])
assert {PureWindowsPath(name).name:value for name,value in q['model_output_bindings'].items()}==model_bindings
timing = timing_evidence(q['cases'])
assert timing == q['timing_comparison']
assert file_inventory(delivered) == members and descriptor(root / 'artifact.zip') == admission['archive']
assert descriptor(Path(__file__)) == identity
assert all(descriptor(path) == bound[name] for name, path in bound_paths.items())
dest = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-matrix-profile'
assert not dest.exists()
dest.mkdir()
copied = {}
for name, expected in members.items():
    if name in ('protocol.json', 'qualification.json') or (name.startswith('evidence/') and Path(name).suffix.lower() not in ('.tar', '.exe', '.dll', '.lib', '.obj', '.pdb')):
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
    'protocol': descriptor(delivered / 'protocol.json'), 'tools_verified': 110, 'repository_files_verified': 1804,
    'artifact_members_verified': len(members), 'artifact_evidence_files_copied': len(copied),
    'supervised_zero_exit_and_reaped_commands': len(q['commands']), 'host': q['host'],
    'sdk_files': 66, 'compiler': compiler, 'ggml_files': 3425, 'compiler_inputs': 97,
    'model_free_processes': 12, 'model_free_identical_outputs_per_shape': 144, 'cases': cases,
    'full_sequence_bit_exact': True, 'timing_comparison':timing,'per_worker_operations':operations,'per_worker_matrix_groups':matrix_groups,
    'enabled_matrix_records_each':84*4*129,'complete_compact_profiles_replayed':4,
    'independent_pristine_archive': archive_identity, 'ggml_replay_commands': patch_commands,
    'release_qualified': False, 'package_qualified': False, 'public_or_lifecycle_qualified': False,
    'new_compiler_sdk_model_calls_during_collection': 0,
    'limits': ['Per-worker matrix and operation walls include internal waits, scheduling and observer effects; never sum overlapping workers.',
        'Timing interpretation requires the preregistered control drift and observer bounds; passing this diagnostic does not qualify release performance.',
        'Physical compiler aliases and model-free build-tree import-library hashes are actual CI observations; saved replay does not stat absent Windows inputs.',
        'SDK binaries and pristine archives remain local; Git retains complete raw vectors, source/compiler evidence and all payload descriptors.',
        'Diagnostic SDK identity is distinct from production; public/lifecycle/package, broader quality and desktop release qualification remain open.']}
assert all(descriptor(dest / name) == expected for name, expected in copied.items())
assert all(descriptor(dest / name) == expected for name, expected in bound.items())
assert descriptor(dest / 'collector.py') == identity and file_inventory(delivered) == members
(dest / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
(dest / 'collection.json').write_text(json.dumps({'schema_version': 1, 'files': file_inventory(dest)}, indent=2) + '\n')
print(json.dumps({'receipt': str(dest), 'files': len(file_inventory(dest)), 'cases': cases, 'timing': timing}))
