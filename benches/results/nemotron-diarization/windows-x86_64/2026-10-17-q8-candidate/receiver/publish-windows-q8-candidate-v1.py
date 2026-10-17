"""Publish an independently admitted Q8 full-model receipt; execute no native code."""
import hashlib,json,os,subprocess
from pathlib import Path
import shutil
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'windows-q8-candidate-ci-v2';admitted=root/'delivery-admission-v1';delivered=admitted/'delivered'
source='fee2177adcd7b75b488c67bea5a593ea17df65de';run_id=39063644421;job_id=117486629205

def descriptor(path):
    digest=hashlib.sha256();size=0
    with path.open('rb')as stream:
        while block:=stream.read(1024**2):size+=len(block);digest.update(block)
    return {'bytes':size,'sha256':digest.hexdigest()}
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
identity=descriptor(Path(__file__))
admission=read(admitted/'qualification.json');result=read(root/'result-replay-v2.json');supervision=read(root/'supervision-replay-v1.json')
members=read(admitted/'artifact-members.json');p=read(delivered/'protocol.json')
assert result['collection_completed']is True and result['source_commit']==admission['source_commit']==supervision['source_commit']==source
assert result['run_id']==admission['run_id']==run_id and result['job_id']==admission['job_id']==job_id
assert result['archive']==admission['archive']==supervision['archive']==descriptor(root/'artifact.zip')
assert result['collector']==descriptor(cache/'replay-windows-q8-candidate-result-v2.py')
assert result['delivery_admission']==descriptor(admitted/'qualification.json')
assert result['supervision_replay']==descriptor(root/'supervision-replay-v1.json')
assert admission['collector']==descriptor(cache/'admit-windows-q8-candidate-delivery-v2.py')
assert supervision['collector']==descriptor(cache/'replay-windows-q8-candidate-supervision-v1.py')
assert all(result[key]is False for key in ('release_qualified','package_qualified'))
assert result['graph_vectors_verified']==36 and result['retained_compiler_objects_verified']==97
assert result['graph_input_payloads_fully_rehashed']is True and result['all_preregistered_evidence_retained']is True
assert result['retained_graph_import_libraries_verified']==3
assert len(result['completed_lifecycle_tests'])==7 and result['functional_lifecycle_gates_passed']is True
assert result['production_candidate_sdk_gates_passed']is result['realtime_gate_passed']
assert result['full_sequence_bit_exact']is True
assert admission['artifact_members']==descriptor(admitted/'artifact-members.json')
assert all(descriptor(delivered/name)==expected for name,expected in members.items())
assert all(descriptor(root/name)==expected for name,expected in admission['metadata'].items())
helper_names=['admit-windows-q8-candidate-delivery-v2.py','replay-windows-q8-candidate-supervision-v1.py',
 'fetch-windows-q8-candidate-artifact-v2.py','replay-windows-q8-candidate-result-v2.py','publish-windows-q8-candidate-v1.py']
helpers={name:descriptor(cache/name)for name in helper_names}
files={name:delivered/name for name in members if name in('qualification.json','protocol.json')or
 (name.startswith('evidence/')and Path(name).suffix.lower()not in('.tar','.exe','.dll','.lib','.obj','.pdb'))}
for name in admission['metadata']:files['receiver/'+name]=root/name
for name,path in [('delivery-admission.json',admitted/'qualification.json'),('artifact-members.json',admitted/'artifact-members.json'),
 ('supervision-replay.json',root/'supervision-replay-v1.json'),('result-replay.json',root/'result-replay-v2.json'),
 ('artifact-download.json',root/'artifact-download-v2.json'),('artifact-download.stderr.log',root/'artifact-download-v2.stderr.log')]:files['receiver/'+name]=path
for name in helper_names:files['receiver/'+name]=cache/name
for name in ['q8-candidate-receivers-pre-execution-v2.json','test-q8-candidate-receiver-controls-v2.py','q8-candidate-receiver-controls-local-v2.json','run-q8-candidate-source-proof-v7.py']:
    files['receiver/'+name]=cache/name
proof=Path(result['independent_source_only_proof']['path'])
assert descriptor(proof)=={key:result['independent_source_only_proof'][key]for key in('bytes','sha256')}
files['receiver/independent-source-only-qualification.json']=proof
bindings={str(path):descriptor(path)for path in files.values()}
env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
 for name,expected in p[group].items():
    raw=subprocess.check_output(['git','show',source+':'+prefix+name],cwd=repo,env=env)
    assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==expected==descriptor(repo/(prefix+name))
dest=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-candidate'
assert not dest.exists();dest.mkdir()
for name,path in files.items():
    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    assert descriptor(target)==bindings[str(path)]
for name,expected in p['tools'].items():
    target=dest/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
    raw=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo,env=env)
    with target.open('xb')as stream:stream.write(raw)
    assert descriptor(target)==expected
summary={**result,'artifact_payloads_copied':len(files),'receiver_scripts':helpers,
 'publication_collector':identity,'result_replay':descriptor(root/'result-replay-v2.json'),
 'publication_inputs':bindings,'publication_limits':'Git retains all raw graph/model arrays, text and source evidence. SDK, object, graph-library, public executable and archive binaries remain in the admitted local/GitHub artifact with complete payload descriptors; the receiver rehashed all required retained inputs.'}
assert all(descriptor(Path(name))==value for name,value in bindings.items())
assert all(descriptor(cache/name)==value for name,value in helpers.items())
assert all(descriptor(delivered/name)==expected for name,expected in members.items())
with(dest/'collection-qualification.json').open('x')as stream:stream.write(json.dumps(summary,indent=2)+'\n')
manifest={path.relative_to(dest).as_posix():descriptor(path)for path in dest.rglob('*')if path.is_file()}
with(dest/'collection.json').open('x')as stream:stream.write(json.dumps({'schema_version':1,'files':manifest},indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'files':len(manifest)+1,'artifact_payloads_copied':len(files),'cases':result['cases'],'realtime_gate_passed':result['realtime_gate_passed'],'lifecycle_tests':len(result['completed_lifecycle_tests'])}))
