"""Preserve independently replayed Q8 V2 evidence without native execution."""
import hashlib
import json
from pathlib import Path

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'windows-q8-mx2-ci-v1'
work=root/'delivery-admission-v1'
delivered=work/'delivered'
target=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-mx2'
source='ebdf051a1be03df766a28e03e901e7dc3513d500'

def desc(path):
    h=hashlib.sha256();size=0
    with path.open('rb')as stream:
        while block:=stream.read(1024**2):h.update(block);size+=len(block)
    return {'bytes':size,'sha256':h.hexdigest()}

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

admission=read(work/'qualification.json')
replay=read(root/'collection-replay-v1.json')
supervision=read(root/'supervision-replay-v1.json')
q=read(delivered/'qualification.json')
p=read(delivered/'protocol.json')
assert admission['source_commit']==replay['source_commit']==supervision['source_commit']==q['source_commit']==source
assert admission['run_id']==replay['run_id']==supervision['run_id']==39075328381
assert admission['job_id']==replay['job_id']==117521191434
artifact,=read(root/'artifacts.json')['artifacts']
assert admission['artifact_id']==artifact['id'] and artifact['workflow_run']['id']==39075328381
assert admission['archive']=={'bytes':artifact['size_in_bytes'],'sha256':artifact['digest'].removeprefix('sha256:')}
assert desc(root/'artifact.zip')==admission['archive']
assert replay['delivery_admission']==desc(work/'qualification.json')
assert replay['supervision_replay']==desc(root/'supervision-replay-v1.json')
assert q['diagnostic_completed']==replay['diagnostic_completed'] and q['new_sdk_or_model_calls']==0
assert all(q[name] is False for name in ['release_qualified','package_qualified','public_or_lifecycle_qualified','sdk_or_model_qualified'])
assert admission['collector']==desc(cache/'admit-windows-q8-mx2-delivery-v1.py')
assert supervision['collector']==desc(cache/'replay-windows-q8-mx2-supervision-v1.py')
assert replay['collector']==desc(cache/'collect-windows-q8-mx2-v1.py')
assert supervision['commands_verified']==len(q['commands'])
if q['diagnostic_completed']:
    assert len(q['commands'])==95 and len(q['cases'])==12
    assert supervision['all_observed_commands_cleanup_admitted'] is True
    for name in ['all_1056_outputs_bit_exact','all_12_half_tables_bit_exact','all_36_support_rejections_verified']:
        assert replay[name] is True and q[name] is True
    assert replay['timing_comparison']==q['timing_comparison']
else:
    assert q.get('error') and replay.get('error')==q['error']
assert all(desc(root/name)==expected for name,expected in admission['metadata'].items())
assert desc(work/'artifact-members.json')==admission['artifact_members']
members=read(work/'artifact-members.json')
assert len(members)==admission['artifact_members_verified']
for name,expected in members.items():assert desc(delivered/name)==expected,name
files={}
retained=[]
for name in members:
    if not name.startswith('canonical/') and name!='pristine.tar' and Path(name).suffix.lower() not in ('.exe','.obj'):
        files[name]=delivered/name;retained.append(name)
for name in ['producer-run.json','artifacts.json','job.json','job.log','supervision-replay-v1.json','collection-replay-v1.json',
             'delivery.stdout.log','delivery.stderr.log','supervision.stdout.log','supervision.stderr.log',
             'collection.stdout.log','collection.stderr.log']:
    files['receiver/'+name]=root/name
for name in ['qualification.json','artifact-members.json']:files['receiver/delivery-'+name]=work/name
for name in ['admit-windows-q8-mx2-delivery-v1.py','replay-windows-q8-mx2-supervision-v1.py','collect-windows-q8-mx2-v1.py']:
    files['receiver/'+name]=cache/name
files['receiver/'+Path(__file__).name]=Path(__file__)
for name in ['fetch-windows-q8-mx2-artifact-v1.py','windows-q8-mx2-receivers-pre-execution-v1.json','windows-q8-mx2-receivers-preparation-v1.json',
        'prepare-windows-q8-mx2-receivers-v1.py','test-q8-mx2-receiver-controls-v1.py','q8-mx2-receiver-controls-local-v1.json']:
    files['receiver/'+name]=cache/name
for name in ['artifact-download-v2.json','artifact-download-v2.stderr.log']:
    files['receiver/'+name]=root/name
for name,expected in p['tools'].items():
    path=repo/'tools/nemotron-diarization'/name
    assert desc(path)==expected
    files['sources/'+name]=path
for name,expected in p['repository_files'].items():assert desc(repo/name)==expected
inputs={str(path):desc(path)for path in files.values()}
inputs[str(root/'artifact.zip')]=admission['archive']
assert not target.exists() and not target.is_symlink()
target.mkdir()
for name,path in files.items():
    dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
    value=path.read_bytes()
    assert {'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}==inputs[str(path)]
    with dest.open('xb')as stream:stream.write(value)
    assert desc(dest)==inputs[str(path)]
completed=q['diagnostic_completed'] is True
result={'schema_version':1,'source_commit':source,'run_id':39075328381,'job_id':117521191434,
        'artifact_id':admission['artifact_id'],'archive':admission['archive'],'publisher':desc(Path(__file__)),
        'diagnostic_completed':completed,'new_compiler_sdk_model_calls':0,
        'canonical_files_verified':replay['canonical_files_verified'],'source_checks':replay['source_checks'],
        'commands_verified':len(q['commands']),'commands_replay':desc(root/'supervision-replay-v1.json'),
        'all_observed_command_cleanup_admitted':supervision['all_observed_commands_cleanup_admitted'],
        'kernel_cases_recorded':len(q['cases']),'kernel_cases_verified':12 if completed else 0,
        'synthetic_kernel_output_parity_verified':completed,'full_phase_vectors_verified':1056 if completed else 0,'worker_interval_schema_verified':completed,
        'all_12_half_tables_bit_exact':completed,'all_36_support_rejections_verified':completed,
        'timing_comparison':replay.get('timing_comparison'),'model_geometry_admission':replay.get('model_geometry_admission'),
        'artifact_members_verified':len(members),'artifact_copies_retained':len(retained),'retained_artifact_names':retained,
        'available_cached_payloads':{'canonical_source_files':sum(name.startswith('canonical/') for name in members),
            'pristine_archive':'pristine.tar' in members,'objects':[name for name in members if Path(name).suffix.lower()=='.obj'],
            'executables':[name for name in members if Path(name).suffix.lower()=='.exe']},
        'frozen_tools_retained':len(p['tools']),'frozen_repository_references_verified':len(p['repository_files']),
        'release_qualified':False,'model_rtf_qualified':False,'production_sdk_qualified':False,
        'scope':'Saved-only replay of same-process Q8 kernels and separate worker wall observer at captured model geometries with synthesized values. Wall intervals include preemption and observer costs; no CPU-time or causal classification. No model RTF, SDK/public/lifecycle/package/desktop/release qualification. Available source/build artifacts excluded from copies remain in admitted local delivery; availability and full member map retained.',
        'receiver_inputs':inputs}
if not completed:result['error']=q['error']
with(target/'collection-qualification.json').open('x')as stream:stream.write(json.dumps(result,indent=2)+'\n')
for name,expected in inputs.items():assert desc(Path(name))==expected,name
for name,expected in p['repository_files'].items():assert desc(repo/name)==expected
manifest={path.relative_to(target).as_posix():desc(path)for path in sorted(target.rglob('*'))if path.is_file()}
with(target/'collection.json').open('x')as stream:stream.write(json.dumps({'schema_version':1,'files':manifest},indent=2)+'\n')
print(json.dumps({'receipt':str(target),'files':len(manifest)+1,'payloads':len(manifest),'diagnostic_completed':completed,'kernel_prediction':None if not completed else replay['timing_comparison']['kernel_speedup_prediction_supported']}))
