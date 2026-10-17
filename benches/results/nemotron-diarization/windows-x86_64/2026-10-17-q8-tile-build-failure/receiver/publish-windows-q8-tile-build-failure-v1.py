"""Preserve admitted failed-build facts and receiver provenance; no native calls."""
import hashlib
import json
from pathlib import Path

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'windows-q8-tile-ci-v1'
work=root/'delivery-admission-v3'
delivered=work/'delivered'
target=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-tile-build-failure'
source='72bcb3d28487d803048122af03501895fe384871'

def desc(path):
    value=path.read_bytes()
    return {'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

admission=read(work/'qualification.json')
replay=read(root/'collection-replay-v3.json')
supervision=read(root/'supervision-replay-v3.json')
q=read(delivered/'qualification.json')
p=read(delivered/'protocol.json')
assert admission['source_commit']==replay['source_commit']==supervision['source_commit']==q['source_commit']==source
assert admission['run_id']==replay['run_id']==39043559355
assert admission['job_id']==replay['job_id']==117427696242
assert admission['artifact_id']==12104713750
assert admission['archive']=={'bytes':25328962,'sha256':'67a4fbb97a6320015a46703768f324652f2a93890167df73f35b04c7269d4ea6'}
assert desc(root/'artifact.zip')==admission['archive']
assert replay['delivery_admission']==desc(work/'qualification.json')
assert replay['supervision_replay']==desc(root/'supervision-replay-v3.json')
assert q['diagnostic_completed'] is False and q['cases']==[] and q['new_sdk_or_model_calls']==0
assert replay['diagnostic_completed'] is False and replay['canonical_files_verified']==3425
assert all(replay['source_checks'][name] is True for name in ['canonical_complete_and_exact','pristine_archive_matches','original_sgemm_matches','candidate_sgemm_matches'])
assert replay['source_checks']['original_patch_reads_matching']==[True]*19
assert supervision['commands_verified']==len(q['commands'])==44
assert supervision['all_observed_commands_cleanup_admitted'] is True
assert [row['returncode']for row in q['commands']]==[0]*43+[4294967295]
assert q['commands'][-1]['name']=='build'
assert 'undefined symbol: ggml_table_f32_f16' in (delivered/'evidence/build.stdout.log').read_text()
assert admission['collector']==desc(cache/'admit-windows-q8-tile-delivery-v3.py')
assert supervision['collector']==desc(cache/'replay-windows-q8-tile-supervision-v3.py')
assert replay['collector']==desc(cache/'collect-windows-q8-tile-v3.py')
assert all(desc(root/name)==expected for name,expected in admission['metadata'].items())
for name,expected in admission['preceding_unadmitted_collection'].items():
    assert desc((cache if name.endswith('.py') else root)/name)==expected,name
assert desc(work/'artifact-members.json')==admission['artifact_members']
members=read(work/'artifact-members.json')
assert len(members)==3519
for name,expected in members.items():assert desc(delivered/name)==expected,name
files={}
for name in members:
    if not name.startswith('canonical/') and name!='pristine.tar': files[name]=delivered/name
for name in ['producer-run.json','artifacts.json','job.json','job.log','supervision-replay-v3.json','collection-replay-v3.json',
             'delivery.stdout.log','delivery.stderr.log','delivery-v2.stdout.log','delivery-v2.stderr.log',
             'delivery-v3.stdout.log','delivery-v3.stderr.log','supervision-v3.stdout.log','supervision-v3.stderr.log',
             'collection-v3.stdout.log','collection-v3.stderr.log']:
    files['receiver/'+name]=root/name
for name in ['qualification.json','artifact-members.json']:files['receiver/delivery-'+name]=work/name
for name in ['admit-windows-q8-tile-delivery-v1.py','admit-windows-q8-tile-delivery-v2.py','admit-windows-q8-tile-delivery-v3.py',
             'replay-windows-q8-tile-supervision-v3.py','collect-windows-q8-tile-v3.py']:
    files['receiver/'+name]=cache/name
files['receiver/'+Path(__file__).name]=Path(__file__)
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
result={'schema_version':1,'source_commit':source,'run_id':39043559355,'job_id':117427696242,
        'artifact_id':12104713750,'archive':admission['archive'],'publisher':desc(Path(__file__)),
        'diagnostic_completed':False,'kernel_cases_executed':0,'new_compiler_sdk_model_calls':0,
        'canonical_files_verified':3425,'original_patches_verified':19,'commands_verified':44,
        'zero_exit_commands':43,'build_returncode':4294967295,'all_observed_command_cleanup_admitted':True,
        'artifact_members_verified':3519,'artifact_copies_retained':sum(not name.startswith('canonical/') and name!='pristine.tar'for name in members),
        'frozen_tools_retained':len(p['tools']),'frozen_repository_references_verified':len(p['repository_files']),
        'release_qualified':False,'model_rtf_qualified':False,'kernel_accuracy_or_timing_qualified':False,
        'scope':'Failed standalone Windows link only. Complete canonical tree/pristine archive stay in admitted local delivery; full member map retained. No benchmark, SDK or model ran.',
        'receiver_inputs':inputs}
with(target/'collection-qualification.json').open('x')as stream:stream.write(json.dumps(result,indent=2)+'\n')
for name,expected in inputs.items():assert desc(Path(name))==expected,name
for name,expected in p['repository_files'].items():assert desc(repo/name)==expected
manifest={path.relative_to(target).as_posix():desc(path)for path in sorted(target.rglob('*'))if path.is_file()}
with(target/'collection.json').open('x')as stream:stream.write(json.dumps({'schema_version':1,'files':manifest},indent=2)+'\n')
print(json.dumps({'receipt':str(target),'files':len(manifest)+1,'payloads':len(manifest),'scope':result['scope']}))
