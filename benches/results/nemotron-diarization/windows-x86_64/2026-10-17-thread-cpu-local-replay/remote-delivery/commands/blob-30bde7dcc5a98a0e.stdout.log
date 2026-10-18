"""Retain reviewed receivers and terminal API metadata; no archive consumption."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
root=cache/'windows-q8-apack-delivery-status-v1'
target=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-apack-admission-preparation'
source='3fd41caa862c34134d008d18b0b6b830b736c864'
bound={}


def desc(path):
    raw=path.read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def bind(path,expected=None):
    raw=path.read_bytes()
    value={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    key=str(path)
    if key in bound:assert bound[key]==value
    if expected is not None:assert value==expected
    bound.setdefault(key,value)
    return raw


def read(path):return json.loads(bind(path))


bind(Path(__file__))
pre_path=cache/'windows-q8-apack-publication-pre-execution-v1.json'
preexecution=read(pre_path)
assert preexecution['source_commit']==source and preexecution['run_id']==39077373013 and preexecution['job_id']==117527216794
assert len(preexecution['files'])==16 and preexecution['publisher_invoked'] is False
for path,expected in preexecution['files'].items():
    p=Path(path);assert p.parent==cache
    bind(p,expected)
for name in ['q8-apack-loop-inspector-pre-execution-v1.json','q8-apack-static-publisher-preparation-v1.json']:
    extra=read(cache/name)
    for path,expected in extra['files'].items():
        p=Path(path);assert p.parent==cache
        bind(p,expected)
controls=read(cache/'q8-apack-receiver-controls-local-v1.json')
assert controls['tests_run']==8 and controls['passed'] is True
assert controls['collector']==bound[str(cache/'collect-windows-q8-apack-v1.py')]
for name,publisher,count in [('q8-apack-publication-green-v1.json','publish-windows-q8-apack-v1.py',4),
        ('q8-apack-static-publication-green-v1.json','publish-q8-apack-static-v1.py',3)]:
    test=read(cache/name)
    assert test['tests_run']==count and test['passed'] is True and test['failures']==test['errors']==0
    assert test['publisher']==bound[str(cache/publisher)]
retention_control=read(cache/'q8-apack-preparation-retention-green-v2.json')
assert retention_control['tests_run']==3 and retention_control['passed'] is True
assert retention_control['publisher']==bound[str(Path(__file__))]
for name in ['test-q8-apack-preparation-retention-v2.py',
        'q8-apack-preparation-retention-green-v2.stdout.log','q8-apack-preparation-retention-green-v2.stderr.log']:
    bind(cache/name)
status=read(root/'status.json')
run=read(root/'producer-run.json');job=read(root/'job.json')
artifact,=read(root/'artifacts.json')['artifacts']
log=bind(root/'job.log').decode('utf-8-sig')
assert run['id']==39077373013 and run['head_sha']==source and run['repository']['full_name']=='saberra-ai/gen2'
assert run['status']=='completed' and run['conclusion']=='success'
assert job['id']==117527216794 and job['run_id']==run['id'] and job['head_sha']==source
assert job['status']=='completed' and job['conclusion']=='success' and status['completed_at']==job['completed_at']
assert artifact['id']==status['artifact_id'] and artifact['name']=='nemotron-windows-q8-apack'
assert artifact['expired'] is False and artifact['workflow_run']['id']==run['id'] and artifact['workflow_run']['head_sha']==source
assert 0<artifact['size_in_bytes']<=1024**3 and artifact['size_in_bytes']==status['api_reported_archive_bytes']
assert artifact['digest']==status['api_reported_archive_digest'] and re.fullmatch('sha256:[0-9a-f]{64}',artifact['digest'])
assert status['source_commit']==source and status['run_id']==run['id'] and status['job_id']==job['id']
assert all(status[name] is False for name in ['archive_downloaded','archive_digest_independently_verified',
    'artifact_members_admitted','numerical_or_timing_replay_completed','main_promoted','release_qualified'])
assert status['observed_free_bytes']<status['original_minimum_start_free_bytes']==8*1024**3
assert status['storage_admitted'] is False and status['new_native_calls']==0
assert log.count('Ran 15 tests')==1 and log.count('Ran 7 tests')==3 and log.count('Ran 6 tests')==2
assert log.count('Ran 10 tests')==1 and log.count('OK (skipped=1)')==1
files={'publisher.py':Path(__file__)}
for path in bound:
    p=Path(path)
    if p==Path(__file__):continue
    name=('api/' if p.parent==root else 'prepared/')+p.name
    assert name not in files
    files[name]=p
assert len({n.casefold()for n in files})==len(files)
inputs={str(path):bound[str(path)]for path in files.values()}
assert all(p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 for p in files.values())
assert all(d['bytes']<=8*1024**2 for d in inputs.values()) and sum(d['bytes']for d in inputs.values())<=16*1024**2
assert not target.exists() and not target.is_symlink()
target.mkdir()
for name,path in files.items():
    dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
    raw=path.read_bytes()
    assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==inputs[str(path)]
    with dest.open('xb')as f:f.write(raw)
    assert desc(dest)==inputs[str(path)]
result={'source_commit':source,'run_id':run['id'],'job_id':job['id'],'artifact_id':artifact['id'],
    'api_reported_archive_bytes':artifact['size_in_bytes'],'api_reported_archive_digest':artifact['digest'],
    'receiver_inputs':inputs,'receiver_pins':11,'publication_pins':16,'inspector_pins':3,
    'receiver_controls':8,'publication_controls':4,'static_publication_controls':3,
    'archive_consumed':False,'numerical_or_timing_replay_completed':False,'release_qualified':False,'main_promoted':False,
    'new_compiler_sdk_model_calls':0,
    'scope':'Prepared/reviewed receiver bytes and actual terminal GitHub API metadata only. Artifact size/digest are API claims, not independently verified archive identities. No numerical, timing, full-model, SDK or release acceptance; storage admission remains unmet at recorded observation.'}
with(target/'qualification.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
for path,expected in inputs.items():assert desc(Path(path))==expected
observed={};folded=set()
for parent,directories,names in os.walk(target,followlinks=False):
    for name in [*directories,*names]:
        p=Path(parent)/name;rel=p.relative_to(target).as_posix()
        assert rel.casefold()not in folded;folded.add(rel.casefold())
        mode=p.lstat().st_mode;assert not stat.S_ISLNK(mode)
        if stat.S_ISDIR(mode):continue
        assert stat.S_ISREG(mode) and p.stat().st_nlink==1
        observed[rel]=desc(p)
assert set(observed)==set(files)|{'qualification.json'}
assert all(observed[n]==inputs[str(p)]for n,p in files.items())
with(target/'collection.json').open('x')as f:json.dump({'schema_version':1,'files':observed},f,indent=2);f.write('\n')
print(json.dumps({'receipt':str(target),'members':len(observed)+1,'bytes_without_manifest':sum(d['bytes']for d in observed.values()),'artifact_consumed':False}))
