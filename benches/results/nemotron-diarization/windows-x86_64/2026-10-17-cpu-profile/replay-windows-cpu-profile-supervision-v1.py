"""Replay only actual saved command supervision; do not qualify SDKs or timings."""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-cpu-profile-ci-v1')
work = root / 'delivery-admission-v1'
delivered = work / 'delivered'
source = 'a705db243ab134c8f0ed9aa375731a17057d145b'


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
admission = read(work / 'qualification.json')
admission_identity = descriptor(work / 'qualification.json')
assert admission['source_commit'] == source and admission['run_id'] == 39037988751
assert admission['job_id'] == 117411575626
assert descriptor(root / 'artifact.zip') == admission['archive']
assert descriptor(root.parent / 'admit-windows-cpu-profile-delivery-v1.py') == admission['collector']
assert descriptor(work / 'artifact-members.json') == admission['artifact_members']
assert all(descriptor(root / name) == value for name, value in admission['metadata'].items())
p = read(delivered / 'protocol.json')
env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}
assert (delivered / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/windows-cpu-profile.protocol.json'], cwd=repo, env=env)
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = subprocess.check_output(['git', 'show', source + ':' + prefix + name], cwd=repo, env=env)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
sys.path.insert(0, str(tools))
from windows_session_timing import file_inventory

members = read(work / 'artifact-members.json')
assert file_inventory(delivered) == members
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source
summaries = []
commands = q['commands']
assert isinstance(commands, list) and 0 < len(commands) <= 256
assert commands == read(delivered / 'evidence/commands.json')
assert len(commands) == len({row['name'] for row in commands})
for row in commands:
    assert row['process_supervision'] == 'Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup'
    assert row['process_tree_reaped'] is True
    value = row['job_observation']
    assert value['worker_assigned'] is True and value['normal_worker_exit_observed'] is True
    assert value['process_tree_reaped'] is True and value['job_handle_closed'] is True
    assert value['cleanup_errors'] == [] and 'snapshot_error' not in value
    status = row['returncode']
    assert type(status) is int and status == value['observed_worker_exit_code'] == value['reaped_worker_exit_code']
    assert type(value['worker_pid']) is int and 0 < value['worker_pid'] <= 0xffffffff
    snapshot = value['remaining_before_cleanup']
    records = snapshot['processes']
    assert type(snapshot['assigned_count']) is int and snapshot['assigned_count'] == len(records) <= 4096
    assert len(records) == len({record['pid'] for record in records})
    for record in records:
        assert type(record['pid']) is int and 0 < record['pid'] <= 0xffffffff
        assert type(record['membership_confirmed']) is bool
        if 'image' in record:
            assert record['membership_confirmed'] is True and PureWindowsPath(record['image']).is_absolute()
    for suffix in ('.stdout.log', '.log'):
        assert ('evidence/' + row['name'] + suffix) in members
    summaries.append({'sdk_compiler': 'clang-cl', 'name': row['name'], 'returncode': status,
                      'error': row.get('error'), 'elapsed_seconds': row['elapsed_seconds'],
                      'remaining_before_cleanup': snapshot,
                      'observed_exit_preserved': True, 'owned_tree_reaped_and_job_closed': True})
assert descriptor(Path(__file__)) == identity
assert descriptor(work / 'qualification.json') == admission_identity
assert file_inventory(delivered) == members
result = {'schema_version': 1, 'source_commit': source, 'run_id': 39037988751,
          'artifact_id': admission['artifact_id'], 'archive': admission['archive'],
          'delivery_admission': admission_identity, 'collector': identity,
          'commands_verified': len(summaries), 'commands': summaries,
          'new_compiler_sdk_model_calls': 0,
          'scope': 'Saved command-supervision replay only; compiler/source/numerical/timing admission remains separate.'}
target = root / 'supervision-replay-v1.json'
with target.open('x') as stream:
    stream.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({'commands_verified': len(summaries), 'nonempty_snapshots': [row for row in summaries
                  if row['remaining_before_cleanup']['assigned_count']]}))
