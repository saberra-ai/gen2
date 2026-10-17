"""Replay only actual saved command supervision; do not qualify SDKs or timings."""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-q8-model-ci-v1')
work = root / 'delivery-admission-v1'
delivered = work / 'delivered'
source = '919ddf1fcf20f7b16d2e5143cffda14f83fb48df'


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
assert admission['source_commit'] == source and admission['run_id'] == 39059902777
assert admission['job_id'] == 117475526989
assert descriptor(root / 'artifact.zip') == admission['archive']
assert descriptor(root.parent / 'admit-windows-q8-model-delivery-v1.py') == admission['collector']
assert descriptor(work / 'artifact-members.json') == admission['artifact_members']
assert all(descriptor(root / name) == value for name, value in admission['metadata'].items())
p = read(delivered / 'protocol.json')
env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}
assert (delivered / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/windows-q8-model.protocol.json'], cwd=repo, env=env)
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
groups=q.get('commands',{})
assert isinstance(groups,dict) and set(groups)<= {'control','mx4'}
if q['diagnostic_completed']is True:assert set(groups)=={'control','mx4'}
for kind,commands in groups.items():
    assert isinstance(commands,list) and len(commands)<=128
    command_file=delivered/('evidence-'+kind)/'commands.json'
    if commands or q['diagnostic_completed']is True:
        assert commands and commands==read(command_file)
    else:
        assert not command_file.exists()or read(command_file)==[]
    assert len(commands)==len({row['name']for row in commands})
    for row in commands:
        assert row['process_supervision']=='Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup'
        value=row.get('job_observation')or {};status=row.get('returncode')
        snapshot=value.get('remaining_before_cleanup')
        records=snapshot.get('processes',[])if isinstance(snapshot,dict)else[]
        snapshot_valid=(isinstance(snapshot,dict)and 'processes'in snapshot and isinstance(snapshot['processes'],list)and type(snapshot.get('assigned_count'))is int
                        and snapshot['assigned_count']==len(records)<=4096
                        and len(records)==len({r.get('pid')for r in records}))
        for record in records:
            snapshot_valid=snapshot_valid and type(record.get('pid'))is int and 0<record['pid']<=0xffffffff
            snapshot_valid=snapshot_valid and type(record.get('membership_confirmed'))is bool
            if 'image'in record:
                snapshot_valid=snapshot_valid and record.get('membership_confirmed')is True and PureWindowsPath(record['image']).is_absolute()
        exit_preserved=(type(status)is int and status==value.get('observed_worker_exit_code')==value.get('reaped_worker_exit_code'))
        clean=(row.get('process_tree_reaped')is True and value.get('worker_assigned')is True
               and value.get('normal_worker_exit_observed')is True and value.get('process_tree_reaped')is True
               and value.get('job_handle_closed')is True and value.get('cleanup_errors')==[]
               and 'snapshot_error'not in value and snapshot_valid and exit_preserved
               and type(value.get('worker_pid'))is int and 0<value['worker_pid']<=0xffffffff)
        logs={suffix:('evidence-'+kind+'/'+row['name']+suffix)in members for suffix in ('.stdout.log','.log')}
        if q['diagnostic_completed']is True:assert clean and status==0 and 'error'not in row and all(logs.values())
        summaries.append({'variant':kind,'sdk_compiler':'clang-cl','name':row['name'],'returncode':status,'error':row.get('error'),
            'elapsed_seconds':row.get('elapsed_seconds'),'remaining_before_cleanup':snapshot,'logs_available':logs,
            'observed_exit_preserved':exit_preserved,'owned_tree_reaped_and_job_closed':clean,'saved_command':row})
assert descriptor(Path(__file__)) == identity
assert descriptor(work / 'qualification.json') == admission_identity
assert file_inventory(delivered) == members
result = {'schema_version': 1, 'source_commit': source, 'run_id': 39059902777,
          'artifact_id': admission['artifact_id'], 'archive': admission['archive'],
          'delivery_admission': admission_identity, 'collector': identity,
          'commands_verified':len(summaries),'commands':summaries,
          'supervision_observed':bool(summaries),
          'all_observed_commands_cleanup_admitted':bool(summaries)and all(row['owned_tree_reaped_and_job_closed']for row in summaries),
          'new_compiler_sdk_model_calls': 0,
          'scope': 'Saved command-supervision replay only; compiler/source/numerical/timing admission remains separate.'}
target = root / 'supervision-replay-v2.json'
with target.open('x') as stream:
    stream.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({'commands_verified': len(summaries), 'nonempty_snapshots': [row for row in summaries
                  if isinstance(row['remaining_before_cleanup'],dict)and row['remaining_before_cleanup'].get('assigned_count')]}))
