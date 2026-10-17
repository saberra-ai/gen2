"""Collect saved checkout failure with stdlib only; no helper/build/model calls."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
root=Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-worker-diagnostic-ci-v1')
source='614aaa0dc5998a219806702e9a82128714b6fcd2'
def descriptor(path):
    raw=path.read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(path):
    return json.loads(path.read_text())
def git_bytes(name):
    return subprocess.check_output(['git','show',source+':'+name],cwd=repo,
        env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'})
run=read(root/'producer-run.json')
assert run['id']==39024922313 and run['head_sha']==source and run['conclusion']=='failure'
assert run['repository']['full_name']=='saberra-ai/gen2'
artifact,=read(root/'artifacts.json')['artifacts']
assert artifact['id']==12097908311 and artifact['workflow_run']['id']==run['id']
assert descriptor(root/'artifact.zip')=={'bytes':artifact['size_in_bytes'],'sha256':artifact['digest'].removeprefix('sha256:')}
members={}
with zipfile.ZipFile(root/'artifact.zip') as archive:
    assert len(archive.infolist())==7
    for info in archive.infolist():
        path=Path(info.filename)
        assert not info.is_dir() and not path.is_absolute() and '..' not in path.parts
        raw=archive.read(info)
        want={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        assert descriptor(root/'unpacked'/path)==want
        members[info.filename]=want
protocol=read(root/'unpacked/protocol.json')
assert (root/'unpacked/protocol.json').read_bytes()==git_bytes('tools/nemotron-diarization/windows-worker-diagnostic.protocol.json')
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
    for name,want in protocol[group].items():
        raw=git_bytes(prefix+name)
        assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==want
        assert descriptor(repo/(prefix+name))==want
q=read(root/'unpacked/qualification.json')
assert q['source_commit']==source and q['protocol']==descriptor(root/'unpacked/protocol.json')
assert all(q[key] is False for key in ['diagnostic_completed','release_qualified','package_qualified','development_native_gates_passed'])
assert q['builds']=={} and q['cases']==[]
assert set(q['commands'])=={'4'}
commands=q['commands']['4']
assert [row['name'] for row in commands]==['clone-native','checkout-native']
assert [row['returncode'] for row in commands]==[0,128]
assert all(row['command'][0]=='git' for row in commands)
assert commands==read(root/'unpacked/evidence-4/commands.json')
log=(root/'unpacked/evidence-4/checkout-native.log').read_text()
assert 'remote missing object' in log and 'smudge filter lfs failed' in log
assert 'src/tts/tokenizer/mandarin_data/hmm_model.utf8' in log
assert 'GIT_LFS_SKIP_SMUDGE' not in commands[1]['environment_overrides']
dest=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-checkout-lfs-failure'
dest.mkdir()
for name in members:
    target=dest/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'unpacked'/name,target)
for path in [root/'producer-run.json',root/'artifacts.json',Path(__file__)]:
    shutil.copy2(path,dest/path.name)
for name in protocol['tools']:
    target=dest/'sources'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(git_bytes('tools/nemotron-diarization/'+name))
(dest/'artifact-members.json').write_text(json.dumps(members,indent=2)+'\n')
summary={'schema_version':1,'source_commit':source,'run_id':run['id'],'artifact_id':artifact['id'],
    'artifact':descriptor(root/'artifact.zip'),'artifact_members_rebound':len(members),
    'tool_sources_verified_against_git_and_current':len(protocol['tools']),
    'repository_sources_verified_against_git_and_current':len(protocol['repository_files']),
    'commands':2,'clone_returncode':0,'checkout_returncode':128,
    'diagnostic_completed':False,'compiler_or_neural_calls':0,'release_qualified':False,
    'observed_failure':'Fresh local native checkout attempted unavailable unrelated TTS Git LFS payload.',
    'fix_candidate':'Set GIT_LFS_SKIP_SMUDGE=1 in diagnostic child mappings, as the existing source preparer does.',
    'collection_local_module_imports':0,'collection_neural_or_binary_execution':0}
(dest/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
files={p.relative_to(dest).as_posix():descriptor(p) for p in sorted(dest.rglob('*')) if p.is_file()}
(dest/'collection.json').write_text(json.dumps({'schema_version':1,'files':files},indent=2)+'\n')
print(json.dumps({'collection':str(dest),'members':len(files)+1}))
