"""Admit and replay saved scalar arithmetic evidence without running binaries."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-fp-contract-ci-v1')
source = 'aa70983a0ad741d92c9b29c3326ab2846da6e6af'
run_id, job_id, artifact_id = 39032867461, 117396700802, 12100149564


def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


identity = descriptor(Path(__file__))
metadata = {name: descriptor(root / name) for name in ['producer-run.json', 'artifacts.json', 'job.json', 'job.log']}
run, job = read(root / 'producer-run.json'), read(root / 'job.json')
assert run['id'] == run_id and run['head_sha'] == source and run['conclusion'] == 'success'
assert run['status'] == 'completed' and run['repository']['full_name'] == 'saberra-ai/gen2'
assert job['id'] == job_id and job['run_id'] == run_id and job['head_sha'] == source
assert job['status'] == 'completed' and job['conclusion'] == 'success'
for name in ['Scalar interpretation controls', 'Actual owned-process controls',
             'Observe exact compilers without SDK or model', 'Preserve successful or failed arithmetic observation']:
    step, = [row for row in job['steps'] if row['name'] == name]
    assert step['conclusion'] == 'success'
artifact, = read(root / 'artifacts.json')['artifacts']
assert artifact['id'] == artifact_id and artifact['expired'] is False
assert artifact['name'] == 'nemotron-windows-fp-contract-probe'
assert artifact['workflow_run']['id'] == run_id and artifact['workflow_run']['head_sha'] == source
archive_identity = {'bytes': 41444, 'sha256': 'db9a4ffe9f57105d8437cba39bb548831bca65835b73e0d26b8e6aadf48f67bc'}
assert descriptor(root / 'artifact.zip') == archive_identity
assert artifact['size_in_bytes'] == archive_identity['bytes'] and artifact['digest'] == 'sha256:' + archive_identity['sha256']
env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}


def git_bytes(name):
    return subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo, env=env)


protocol = git_bytes('tools/nemotron-diarization/windows-fp-contract-probe.protocol.json')
p = json.loads(protocol)
assert {'bytes': len(protocol), 'sha256': hashlib.sha256(protocol).hexdigest()} == {
    'bytes': 3174, 'sha256': '9f35f7de9a93aaf95b6dfc07d9067c938e0a6ef34dc5242b3357d124540beff7'}
assert len(p['tools']) == 17 and len(p['repository_files']) == 3
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = git_bytes(prefix + name)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
# Full bounded ZIP admission precedes imports or extraction.
members, payloads, folded, total = {}, {}, set(), 0
with zipfile.ZipFile(root / 'artifact.zip') as bundle:
    entries = bundle.infolist()
    assert len(entries) == 44
    for item in entries:
        path = PureWindowsPath(item.filename)
        assert not path.is_absolute() and not path.drive and '..' not in path.parts
        assert '\\' not in item.filename and str(PurePosixPath(item.filename)) == item.filename
        assert item.filename.casefold() not in folded
        folded.add(item.filename.casefold())
        assert not item.is_dir() and not item.flag_bits & 1
        assert stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
        assert item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        assert 0 <= item.file_size <= 16 * 1024**2
        total += item.file_size
        assert total <= 64 * 1024**2
        with bundle.open(item) as stream:
            raw = stream.read(item.file_size + 1)
        assert len(raw) == item.file_size
        payloads[item.filename] = raw
        members[item.filename] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
sys.path.insert(0, str(tools))
from windows_fp_contract_probe import (BOOTSTRAP, FUSED, HOST, REFERENCE, VARIANTS,
                                     compile_command, observation)

work = root / 'collection-replay-v1'
work.mkdir()
delivered = work / 'delivered'
delivered.mkdir()
for name, raw in payloads.items():
    target = delivered / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)
    assert descriptor(target) == members[name]
assert (delivered / 'protocol.json').read_bytes() == protocol
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source and q['protocol'] == descriptor(delivered / 'protocol.json')
assert q['probe_completed'] is True and 'error' not in q and q['model_or_sdk_calls'] == 0
assert all(q[key] is False for key in ['release_qualified', 'package_qualified', 'public_or_lifecycle_qualified'])
assert q['host'] == read(delivered / 'host.stdout.log') and q['host']['avx2'] is True and q['host']['fma'] is True
reference = read(repo / REFERENCE)
assert set(q['compilers']) == {'msvc', 'clang-cl'} and set(q['variants']) == set(VARIANTS)
for kind, compiler in q['compilers'].items():
    assert {key: compiler[key] for key in ('bytes', 'sha256')} == reference['builds'][kind]['compiler']['compiler_binary']
expected_names = ['host', 'compiler-environment']
binary_names, observations, disassembly = set(), {}, {}
for variant in VARIANTS:
    if variant != 'msvc-default':
        expected_names.append(variant + '-driver')
    expected_names += [variant + '-build', variant + '-disassembly'] + [variant + '-run-' + str(i) for i in range(3)]
assert [row['name'] for row in q['commands']] == expected_names and len(expected_names) == 19
rows = {row['name']: row for row in q['commands']}
assert rows['host']['command'] == ['pwsh', '-NoProfile', '-NonInteractive', '-Command', HOST]
assert rows['compiler-environment']['command'][1:] == ['-B', '-c', BOOTSTRAP]
for variant in VARIANTS:
    saved = q['variants'][variant]
    assert saved['repetitions'] == 3 and set(saved['binaries']) == {'probe.exe', 'probe.obj'}
    for name, expected in saved['binaries'].items():
        relative = variant + '/' + name
        assert members[relative] == expected
        binary_names.add(relative)
    command = rows[variant + '-build']['command']
    directory = PureWindowsPath(command[-1][3:]).parent
    kind = 'msvc' if variant == 'msvc-default' else 'clang-cl'
    original_source = PureWindowsPath(command[-3])
    assert original_source.is_absolute() and original_source.name == 'fp_contract_probe.cpp'
    assert command == compile_command(q['compilers'][kind]['path'], original_source, directory, variant)
    assert rows[variant + '-disassembly']['command'] == [q['dumpbin']['path'], '/nologo', '/disasm', '/symbols', str(directory / 'probe.obj')]
    for repeat in range(3):
        phase = variant + '-run-' + str(repeat)
        assert rows[phase]['command'] == [str(directory / 'probe.exe')]
        value = observation((delivered / (phase + '.stdout.log')).read_text(), variant)
        assert value == saved['observation']
    observations[variant] = saved['observation']
    text = (delivered / (variant + '-disassembly.stdout.log')).read_text()
    body, = re.findall(r'^contract_ordinary:\n(.*?)^contract_separate:\n', text, re.M | re.S)
    mnemonics = re.findall(r'\b(vmulss|vaddss|vfmadd213ss)\b', body)
    expected = ['vfmadd213ss'] if variant == 'clang-default' else ['vmulss', 'vaddss']
    assert mnemonics == expected
    disassembly[variant] = mnemonics
    if kind == 'clang-cl':
        assert rows[variant + '-driver']['command'] == command + ['-###']
        modes = re.findall(r'"-ffp-contract=(on|off)"', (delivered / (variant + '-driver.stderr.log')).read_text())
        assert modes == ['off' if variant == 'clang-no-contract' else 'on']
for row in q['commands']:
    assert row['returncode'] == 0 and 'error' not in row and row['process_tree_reaped'] is True
    value = row['job_observation']
    assert value['worker_assigned'] is True and value['normal_worker_exit_observed'] is True
    assert value['observed_worker_exit_code'] == value['reaped_worker_exit_code'] == 0
    assert value['process_tree_reaped'] is True and value['job_handle_closed'] is True
    assert value['cleanup_errors'] == [] and 'snapshot_error' not in value
assert len({value['mxcsr'] & 0xffc0 for value in observations.values()}) == 1
supported = (observations['msvc-default']['ordinary'] == 0
             and observations['clang-default']['ordinary'] == FUSED
             and observations['clang-no-contract']['ordinary'] == 0)
assert supported is True and q['scalar_contraction_difference_observed'] == supported
expected_members = {'qualification.json', 'protocol.json'} | binary_names
expected_members |= {name + suffix for name in expected_names if name != 'compiler-environment'
                     for suffix in ('.stdout.log', '.stderr.log')}
assert set(members) == expected_members
log = (root / 'job.log').read_text()
assert 'Ran 4 tests' in log and 'Ran 10 tests' in log
assert all(descriptor(root / name) == value for name, value in metadata.items())
assert descriptor(root / 'artifact.zip') == archive_identity and descriptor(Path(__file__)) == identity
dest = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-fp-contract-probe'
assert not dest.exists()
dest.mkdir()
for name, expected in members.items():
    if name not in binary_names:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(delivered / name, target)
        assert descriptor(target) == expected
for name, expected in metadata.items():
    shutil.copyfile(root / name, dest / name)
    assert descriptor(dest / name) == expected
for name, expected in p['tools'].items():
    target = dest / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(git_bytes('tools/nemotron-diarization/' + name))
    assert descriptor(target) == expected
shutil.copyfile(Path(__file__), dest / 'collector.py')
assert descriptor(dest / 'collector.py') == identity
(dest / 'artifact-members.json').write_text(json.dumps(members, indent=2) + '\n')
result = {'schema_version': 1, 'collection_completed': True, 'source_commit': source,
    'run_id': run_id, 'job_id': job_id, 'artifact_id': artifact_id, 'archive': archive_identity,
    'protocol': descriptor(delivered / 'protocol.json'), 'tools_verified': 17, 'repository_files_verified': 3,
    'artifact_members_verified': 44, 'zero_status_and_reaped_commands': 19,
    'observations': observations, 'ordinary_function_mnemonics': disassembly,
    'scalar_contraction_difference_observed': True, 'new_compiler_sdk_model_calls_during_collection': 0,
    'release_qualified': False, 'package_qualified': False, 'public_or_lifecycle_qualified': False,
    'limits': ['Scalar expression result only; no speech-model causality or fix established.',
        'No SDK/model execution or timing comparison occurred in this experiment.',
        'Exact same compiler hashes as prior failed diagnostic; host CPU differs, so no cross-run timing comparison.',
        'Private environment output is absent; probe executables/objects remain local, with all hashes retained.']}
# Rebind every actual copy and source before manifest publication.
for name, expected in members.items():
    assert descriptor(delivered / name) == expected
    if name not in binary_names:
        assert descriptor(dest / name) == expected
assert all(descriptor(dest / name) == expected for name, expected in metadata.items())
assert all(descriptor(dest / 'sources' / name) == expected for name, expected in p['tools'].items())
assert descriptor(dest / 'collector.py') == identity
(dest / 'collection-qualification.json').write_text(json.dumps(result, indent=2) + '\n')
files = {path.relative_to(dest).as_posix(): descriptor(path) for path in sorted(dest.rglob('*')) if path.is_file()}
(dest / 'collection.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
print(json.dumps({'receipt': str(dest), 'files': len(files) + 1, 'scalar_prediction_supported': supported}))
