"""Replay only actual saved command supervision; do not qualify SDKs or timings."""
import hashlib
import json
import os
import stat
from pathlib import Path, PureWindowsPath
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-q8-apack-ci-v1')
work = root / 'delivery-admission-v1'
delivered = work / 'delivered'
source = '3fd41caa862c34134d008d18b0b6b830b736c864'


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
assert admission['source_commit'] == source and admission['run_id'] == 39077373013
assert admission['job_id'] == 117527216794
assert descriptor(root / 'artifact.zip') == admission['archive']
assert descriptor(root.parent / 'admit-windows-q8-apack-delivery-v1.py') == admission['collector']
assert descriptor(work / 'artifact-members.json') == admission['artifact_members']
assert all(descriptor(root / name) == value for name, value in admission['metadata'].items())
p = read(delivered / 'protocol.json')
env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}
assert (delivered / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/windows-q8-apack.protocol.json'], cwd=repo, env=env)
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = subprocess.check_output(['git', 'show', source + ':' + prefix + name], cwd=repo, env=env)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
sys.path.insert(0, str(tools))
def file_inventory(root):
    result, folded, count, total = {}, set(), 0, 0
    assert root.is_dir() and not root.is_symlink()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            count += 1
            assert count <= 10000 and relative.casefold() not in folded
            folded.add(relative.casefold())
            mode = path.lstat().st_mode
            assert not stat.S_ISLNK(mode)
            if stat.S_ISDIR(mode):
                continue
            assert stat.S_ISREG(mode)
            value = descriptor(path)
            assert 0 <= value['bytes'] <= 256 * 1024**2
            total += value['bytes']
            assert total <= 1024**3
            result[relative] = value
    return result




members = read(work / 'artifact-members.json')
assert file_inventory(delivered) == members
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source
summaries = []
assert isinstance(q['commands'], list) and len(q['commands']) <= 128
if q['commands'] or q['diagnostic_completed'] is True:
    assert q['commands'] and q['commands'] == read(delivered / 'evidence/commands.json')
else:
    assert not (delivered / 'evidence/commands.json').exists() or read(delivered / 'evidence/commands.json') == []
assert len(q['commands']) == len({row['name'] for row in q['commands']})
guard_codes={'barrier':73,'set':74,'add':75}
expected_order=['receiver-host','native-revision','llama-revision','pristine-archive']
expected_codes={}
for index in range(19): expected_order.extend([f'patch-{index}-read',f'patch-{index}-apply'])
expected_order.extend(['configure','build','disasm-control','disasm-apack','binary-dependencies'])
for workers in (1,2,4):
    for index in range(4):
        phase=f'case-{workers}-{index}'
        for mode,code in guard_codes.items():
            name=phase+'-guard-'+mode
            expected_order.append(name);expected_codes[name]=code
        expected_order.append(phase)
assert [row['name']for row in q['commands']]==expected_order[:len(q['commands'])]
if q['diagnostic_completed'] is True: assert len(q['commands'])==len(expected_order)==95
for row in q['commands']:
    expected=expected_codes.get(row['name'],0)
    assert row['expected_returncode']==expected
    if expected:
        mode=row['name'].rsplit('-guard-',1)[1]
        assert row['command'][1:]==['--support-control',mode]
    assert row['process_supervision'] == 'Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup'
    logs={suffix:('evidence/'+row['name']+suffix)in members for suffix in ('.stdout.log','.log')}
    if q['diagnostic_completed']is True:assert all(logs.values())
    value = row.get('job_observation') or {}
    status = row.get('returncode')
    snapshot = value.get('remaining_before_cleanup')
    records = snapshot.get('processes', []) if isinstance(snapshot, dict) else []
    snapshot_valid = (isinstance(snapshot, dict) and 'processes'in snapshot and isinstance(snapshot['processes'],list) and type(snapshot.get('assigned_count')) is int
                      and snapshot['assigned_count'] == len(records) <= 4096
                      and len(records) == len({record.get('pid') for record in records}))
    for record in records:
        snapshot_valid = snapshot_valid and type(record.get('pid')) is int and 0 < record['pid'] <= 0xffffffff
        snapshot_valid = snapshot_valid and type(record.get('membership_confirmed')) is bool
        if 'image' in record:
            snapshot_valid = snapshot_valid and record.get('membership_confirmed') is True and PureWindowsPath(record['image']).is_absolute()
    clean = (row.get('process_tree_reaped') is True and value.get('worker_assigned') is True
             and value.get('normal_worker_exit_observed') is True and value.get('process_tree_reaped') is True
             and value.get('job_handle_closed') is True and value.get('cleanup_errors') == []
             and 'snapshot_error' not in value and snapshot_valid
             and type(status) is int and type(value.get('observed_worker_exit_code')) is int
             and type(value.get('reaped_worker_exit_code')) is int
             and status == value.get('observed_worker_exit_code') == value.get('reaped_worker_exit_code')
             and type(value.get('worker_pid')) is int and 0 < value['worker_pid'] <= 0xffffffff)
    if q['diagnostic_completed'] is True:
        assert clean and status == expected and 'error' not in row
    summaries.append({'name': row['name'], 'returncode': status, 'error': row.get('error'),
                      'elapsed_seconds': row.get('elapsed_seconds'), 'remaining_before_cleanup': snapshot, 'logs_available':logs,
                      'normal_exit_and_cleanup_admitted': clean, 'saved_command': row})
assert descriptor(Path(__file__)) == identity
assert descriptor(work / 'qualification.json') == admission_identity
assert file_inventory(delivered) == members
result = {'schema_version': 1, 'source_commit': source, 'run_id': 39077373013,
          'artifact_id': admission['artifact_id'], 'archive': admission['archive'],
          'delivery_admission': admission_identity, 'collector': identity,
          'commands_verified': len(summaries), 'commands': summaries,
          'supervision_observed': bool(summaries), 'all_observed_commands_cleanup_admitted': bool(summaries) and all(row['normal_exit_and_cleanup_admitted'] for row in summaries),
          'new_compiler_sdk_model_calls': 0,
          'scope': 'Saved command-supervision replay only; compiler/source/numerical/timing admission remains separate.'}
target = root / 'supervision-replay-v1.json'
with target.open('x') as stream:
    stream.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({'commands_verified': len(summaries), 'cleanup_not_admitted': [row['name'] for row in summaries if not row['normal_exit_and_cleanup_admitted']]}))
