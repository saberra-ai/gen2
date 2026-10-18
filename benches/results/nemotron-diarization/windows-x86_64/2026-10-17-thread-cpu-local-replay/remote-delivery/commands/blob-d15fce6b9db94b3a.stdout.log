"""Admit the completed matrix path profile artifact and exact source inputs, without native execution."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-thread-cpu-profile-ci-v1')
source = '15c905d8815190a54f25452e7e51378c24f85914'
run_id, job_id = 39083120058, 117544022272


def descriptor(path):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            size += len(block)
            digest.update(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}


def git_bytes(name):
    return subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo, env=env)


identity = descriptor(Path(__file__))
metadata = {name: descriptor(root / name) for name in
            ['producer-run.json', 'artifacts.json', 'job.json', 'job.log']}
run, job = read(root / 'producer-run.json'), read(root / 'job.json')
assert run['id'] == run_id and run['head_sha'] == source
assert run['status'] == 'completed' and run['conclusion'] in ('success', 'failure')
assert run['repository']['full_name'] == 'saberra-ai/gen2'
assert job['id'] == job_id and job['run_id'] == run_id and job['head_sha'] == source
assert job['status'] == 'completed' and job['conclusion'] == run['conclusion']
for name in ['Original and contraction compiler controls',
             'Matrix source,coverage and saved report controls',
             'Batched matrix source, compiler and evidence controls',
             'Thread clock API adapter and complete counter traces',
             'Actual owned-background timeout and assignment controls',
             'Preserve successful or failed matrix profile evidence']:
    step, = [row for row in job['steps'] if row['name'] == name]
    assert step['conclusion'] == 'success'
artifact, = read(root / 'artifacts.json')['artifacts']
assert artifact['name'] == 'nemotron-windows-thread-cpu-profile'
assert artifact['expired'] is False
assert artifact['workflow_run']['id'] == run_id and artifact['workflow_run']['head_sha'] == source
archive_identity = descriptor(root / 'artifact.zip')
assert archive_identity == {'bytes': artifact['size_in_bytes'],
                            'sha256': artifact['digest'].removeprefix('sha256:')}
protocol = git_bytes('tools/nemotron-diarization/windows-thread-cpu-profile.protocol.json')
p = json.loads(protocol)
assert {'bytes': len(protocol), 'sha256': hashlib.sha256(protocol).hexdigest()} == {
    'bytes': 670257, 'sha256': 'e9809ffe025bbe127c27daa5f8dfa07916cab7fe24a40d074651103b8c3cfc66'}
assert len(p['tools']) == 135 and len(p['repository_files']) == 2880
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = git_bytes(prefix + name)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
# Bound and hash every archive payload before local imports or extraction.
members, folded, total = {}, set(), 0
with zipfile.ZipFile(root / 'artifact.zip') as bundle:
    entries = bundle.infolist()
    assert 0 < len(entries) <= 10000
    for item in entries:
        path = PureWindowsPath(item.filename)
        assert not path.is_absolute() and not path.drive and '..' not in path.parts
        assert '\\' not in item.filename and str(PurePosixPath(item.filename)) == item.filename
        assert item.filename.casefold() not in folded
        folded.add(item.filename.casefold())
        assert not item.is_dir() and not item.flag_bits & 1
        assert stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
        assert item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        assert 0 <= item.file_size <= 256 * 1024**2
        total += item.file_size
        assert total <= 1024**3
        digest, size = hashlib.sha256(), 0
        with bundle.open(item) as stream:
            while block := stream.read(1024**2):
                size += len(block)
                assert size <= item.file_size
                digest.update(block)
        assert size == item.file_size
        members[item.filename] = {'bytes': size, 'sha256': digest.hexdigest()}
sys.path.insert(0, str(tools))
from windows_session_timing import extract_delivery, file_inventory

work = root / 'delivery-admission-v1'
work.mkdir()
delivered = work / 'delivered'
extract_delivery(root / 'artifact.zip', archive_identity, members, delivered)
assert (delivered / 'protocol.json').read_bytes() == protocol
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source and q['protocol'] == descriptor(delivered / 'protocol.json')
assert q['order'] == p['order'] and q['limits'] == p['limits']
assert all(q[key] is False for key in ['release_qualified', 'package_qualified', 'public_or_lifecycle_qualified'])
assert (q['diagnostic_completed'] is True) == (run['conclusion'] == 'success')
log = (root / 'job.log').read_text(encoding='utf-8-sig')
assert log.count('Ran 15 tests') == 1 and log.count('Ran 21 tests') == 1 and log.count('Ran 10 tests') == 2 and log.count('Ran 8 tests') == 1
assert file_inventory(delivered) == members
assert descriptor(root / 'artifact.zip') == archive_identity
assert descriptor(Path(__file__)) == identity
assert all(descriptor(root / name) == value for name, value in metadata.items())
(work / 'artifact-members.json').write_text(json.dumps(members, indent=2) + '\n')
result = {'schema_version': 1, 'source_commit': source, 'run_id': run_id, 'job_id': job_id,
          'artifact_id': artifact['id'], 'archive': archive_identity, 'metadata': metadata,
          'collector': identity, 'tools_verified': 135, 'repository_files_verified': 2880,
          'artifact_members': descriptor(work / 'artifact-members.json'),
          'artifact_members_verified': len(members), 'actual_windows_control_tests': 10,
          'actual_compiler_control_tests': 15, 'actual_matrix_profile_control_tests': 21, 'actual_batch_profile_control_tests':10, 'actual_thread_clock_tests':8,
          'new_compiler_sdk_model_calls': 0,
          'scope': 'Delivery integrity and source admission only; runtime result replay remains separate.'}
(work / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
