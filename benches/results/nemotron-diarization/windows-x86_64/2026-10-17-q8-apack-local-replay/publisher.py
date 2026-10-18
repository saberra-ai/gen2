"""Retain bounded return and local replay proof alongside byte-exact CI roots."""
from collections import Counter
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import sys

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
helper=repo/'tools/nemotron-diarization/remote_q8_apack_replay.py'
protocol=repo/'tools/nemotron-diarization/remote-q8-apack-replay.protocol.json'
protocol_raw=protocol.read_bytes()
protocol_identity={'bytes':len(protocol_raw),'sha256':hashlib.sha256(protocol_raw).hexdigest()}
assert protocol_identity=={'bytes':18183,'sha256':'6053c9cd14e21a5810f3fde0d47a9ba44da2e325e7127ace881f59b0a13c1a5d'}
helper_raw=helper.read_bytes();helper_identity={'bytes':len(helper_raw),'sha256':hashlib.sha256(helper_raw).hexdigest()}
assert helper_identity==json.loads(protocol_raw)['files']['tools/nemotron-diarization/remote_q8_apack_replay.py']
sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from remote_q8_apack_replay import descriptor,verify_collection,RESULTS
local=cache/'q8-apack-local-bundle-admission-v1';imports=cache/'q8-apack-result-import-v1'
target=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-apack-local-replay'
assert not target.exists()
bound={str(helper):helper_identity,str(protocol):protocol_identity}
def first_read(path,expected=None):
    raw=path.read_bytes();value={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};key=str(path)
    if key in bound:assert value==bound[key],key
    if expected is not None:assert value==expected,key
    bound.setdefault(key,value)
    return raw
first_read(Path(__file__))
controls_stderr=first_read(cache/'q8-apack-local-proof-first-binding-green-v3.stderr.log')
assert b'Ran 3 tests'in controls_stderr and b'\nOK\n'in controls_stderr
for root in RESULTS:first_read(repo/root/'collection.json')
admission=json.loads(first_read(local/'admission.json'))
assert {name:verify_collection(repo/name)for name in RESULTS}==admission['collections']
replay=json.loads(first_read(cache/'q8-apack-returned-evidence-replay-v2.json'))
assert replay['input_collections']==admission['collections']and replay['full_vectors_verified']==1056
ledger=json.loads(first_read(local/'commands.json'))
sidecars=list((local/'commands').glob('*.process.json'))+list((local/'blob-stage').glob('*.process.json'))+list(local.glob('*.process.json'))
reports={str(p.relative_to(local)):json.loads(first_read(p))for p in sidecars}
canonical=lambda row:json.dumps(row,sort_keys=True,separators=(',',':'))
assert Counter(map(canonical,ledger))==Counter(map(canonical,reports.values()))
assert len(ledger)==1542 and all(row['completed']and row['leader_reaped']and row['group_gone']and row['returncode']==0 for row in ledger)
inventory=cache/'q8-apack-local-process-inventory-v3.json'
with inventory.open('x')as f:json.dump({'reports':{name:bound[str(local/name)]for name in reports},
    'all_reports_equal_retained_ledger_multiset':True,'owned_commands_completed_reaped_group_gone':len(ledger)},f,indent=2);f.write('\n')
files={'publisher.py':Path(__file__),'source/raw-replay.py':cache/'replay-q8-apack-returned-evidence-v2.py',
    'source/local-process-helper.py':helper,'source/remote-replay.protocol.json':protocol,
    'source/raw-replay-pre-execution.json':cache/'q8-apack-returned-replay-pre-execution-v2.json',
    'controls/test-first-binding.py':cache/'test-q8-apack-local-proof-first-binding-v3.py',
    'controls/first-binding.stdout.log':cache/'q8-apack-local-proof-first-binding-green-v3.stdout.log',
    'controls/first-binding.stderr.log':cache/'q8-apack-local-proof-first-binding-green-v3.stderr.log',
    'source/local-receiver.py':repo/'tools/nemotron-diarization/receive_q8_apack_bundle.py',
    'source/local-receiver.protocol.json':repo/'tools/nemotron-diarization/q8-apack-bundle-receive.protocol.json',
    'local-admission.json':local/'admission.json','local-commands.json':local/'commands.json',
    'local-process-inventory.json':inventory,'pre-download.json':local/'pre-download.json',
    'raw-replay.json':cache/'q8-apack-returned-evidence-replay-v2.json',
    'raw-replay.stdout.log':cache/'q8-apack-returned-replay-v2.stdout.log',
    'raw-replay.stderr.log':cache/'q8-apack-returned-replay-v2.stderr.log',
    'raw-replay.process.json':cache/'q8-apack-returned-replay-v2.stdout.log.process.json',
    'delivery/delivery.zip':local/'delivery.zip','delivery/receiver-result.bundle':local/'delivery/receiver-result.bundle',
    'delivery/bundle.json':local/'delivery/bundle.json','delivery/storage-admission.json':local/'delivery/storage-admission.json',
    'delivery/bundle-process.json':local/'delivery/receiver-result.bundle.process.json'}
for path in sorted((local/'commands').iterdir()):
    if path.is_file()and(path.stat().st_size or path.name.endswith('.stdout')):
        files['local-command-output/'+path.name]=path
for path in sorted(imports.iterdir()):
    if path.is_file():files['import/'+path.name]=path
assert len(files)==len({name.casefold()for name in files})
for path in files.values():first_read(path)
assert bound[str(local/'delivery.zip')]=={'bytes':17392754,'sha256':'0a31a38d6e136bdf48a66b7b106745ab1c175e601e6f44b2af438cbe207bcd16'}
assert bound[str(local/'delivery/receiver-result.bundle')]=={'bytes':17070275,'sha256':'0f702b4ae0b54691f9122ab336083cbac1f116d72d37f3d6e4ae0360b656156e'}
pre=cache/'q8-apack-local-proof-pre-execution-v3.json'
with pre.open('x')as f:json.dump({'files':bound,'original_native_collections':admission['collections'],
    'local_result_only_commit':'ba5cce755ab7c4673892ccfc9b0bf4992e6fc26f','main_promoted':False},f,indent=2);f.write('\n')
files['publication-pre-execution.json']=pre;bound[str(pre)]=descriptor(pre)
target.mkdir();manifest={}
for name,path in files.items():
    assert not PurePosixPath(name).is_absolute()and '..'not in PurePosixPath(name).parts
    expected=bound[str(path)];raw=first_read(path)
    assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==expected
    dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb')as f:f.write(raw)
    assert descriptor(dest)==expected;manifest[name]=expected
for path,wanted in bound.items():first_read(Path(path),wanted)
assert {name:verify_collection(repo/name)for name in RESULTS}==admission['collections']
qualification={'schema_version':1,'native_source':replay['source_commit'],'native_run_id':replay['run_id'],
    'native_job_id':replay['job_id'],'remote_receiver_source':admission['remote_source_commit'],
    'remote_receiver_run_id':admission['remote_run_id'],'remote_receiver_job_id':admission['remote_job_id'],
    'remote_result_commit':admission['remote_result_commit'],'native_collections':admission['collections'],
    'all_consumed_inputs_first_bound_before_validation':True,'first_read_binding_controls_passed':3,
    'local_process_reports_consolidated':len(ledger),'local_raw_evidence_replay':descriptor(target/'raw-replay.json'),
    'native_original_zip_rehashed_locally':False,'local_source_state_unchanged_during_delivery_admission':True,
    'result_index_staged_after_independent_admission_and_replay':True,'new_compiler_kernel_sdk_model_calls':0,
    'production_or_release_qualified':False,
    'scope':'Exact original CI collections retained separately, bounded resultZIP and incrementalGitbundle retained here. Local command ledger equals all1542sidecar records; every nonempty command-output file retained, blob outputs live in the two exact result collections. Original full nativeZIP admitted remotely; complete returned vectors/half/packing/guards/95Job observations/timing/static replayed locally. All original failed speed/observer and quality/desktop/release gates preserved.'}
raw=(json.dumps(qualification,indent=2)+'\n').encode();(target/'qualification.json').write_bytes(raw);manifest['qualification.json']=descriptor(target/'qualification.json')
(target/'collection.json').write_text(json.dumps({'schema_version':1,'files':manifest},indent=2)+'\n')
print(json.dumps(verify_collection(target)))
