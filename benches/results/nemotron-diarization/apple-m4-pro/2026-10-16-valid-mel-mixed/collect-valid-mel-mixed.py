"""Collect saved Mac mixed smoke outputs; no helper/native/model execution."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from macos_auxiliary_loaded_libraries import audit_with_auxiliary
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
read=lambda path:json.loads(path.read_text())
source=cache/'valid-mel-mixed-runtime-v1';p=read(source/'protocol.json');r=read(source/'qualification.json');commands=read(source/'commands.json');work=Path(p['work_path']);build=read(Path(p['build_record_path']));compiled=cache/'valid-mel-mixed-helper-build-v1';compile_record=read(compiled/'command.json');target=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-mixed'
assert not target.exists()
assert r['mixed_residency_smoke_passed']and not r['release_qualified']and len(r['cases'])==4 and len(commands)==5 and all(c['returncode']==0 for c in commands)
assert descriptor(source/'protocol.json')==r['protocol']and(source/'protocol.json').read_bytes()==(tools/'valid-mel-mixed.protocol.json').read_bytes()
assert descriptor(Path(p['build_record_path']))==p['build_record'] and descriptor(work/'bin/mixed-native-tests')==compile_record['helper']==descriptor(compiled/'mixed-native-tests')
assert compile_record['compiled']and compile_record['returncode']==0 and not compile_record['models_loaded']and compile_record['native_or_neural_test_cases_executed']==0
assert compile_record['repository_files']==p['repository_files']
for path,expected in p['repository_files'].items():
 assert descriptor(repo/path)==expected
 for commit in [compile_record['source_commit'],r['source_commit']]:
  data=subprocess.check_output(['git','show',commit+':'+path],cwd=repo);assert{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==expected
for name,expected in p['tools'].items():assert descriptor(tools/name)==expected
for name,expected in p['files'].items():assert descriptor(work/name)==expected
assert {str(f.relative_to(work)):str(f.readlink())for f in work.rglob('*')if f.is_symlink()}==p['aliases']
assert {str(f.relative_to(work))for f in work.rglob('*')if f.is_file()and not f.is_symlink()}==set(p['files'])
manifests={backend:verify(work/backend/'nemo')for backend in ['cpu','metal']}
helper=work/'bin/mixed-native-tests'
assert audit_loaded_libraries((source/'list.stderr.log').read_text(),work/'cpu/nemo',manifests['cpu'],helper,require_runtime=False)==read(source/'list-images.json')
assert {line.removesuffix(': test')for line in(source/'list.stdout.log').read_text().splitlines()if line.endswith(': test')}==set(p['compiled_test_names'])
counts={};elapsed={}
for case,expected,command in zip(r['cases'],p['cases'],commands[1:]):
 backend=expected['backend'];name=backend+'-'+expected['test'];assert case['name']==command['name']==name and case['passed']
 trace=(source/(name+'.stderr.log')).read_text();images=audit_with_auxiliary(trace,work/backend/'nemo',manifests[backend],helper,{str(work/'providers/libonnxruntime.dylib'):p['files']['providers/libonnxruntime.dylib']})
 assert images==read(source/(name+'.images.json'))and descriptor(source/(name+'.images.json'))==case['loaded_images']
 assert images['runtime']['private_library_count']==p['expected_runtime_images'][backend]and images['auxiliary_library_count']==1
 stdout=(source/(name+'.stdout.log')).read_text();assert'1 passed; 0 failed; 0 ignored;'in stdout
 placement={}
 for line in stdout.splitlines():
  if line.startswith('chat-placement '):
   phase,value=line.removeprefix('chat-placement ').split(': ',1);placement[phase]=json.loads(value)
 assert placement==case['chat_placement']and set(placement)=={'initial','restored'}
 assert all(v['layers']==29 and v['backend']=='MTL'and v['device']==p['expected_chat_device']for v in placement.values())
 counts[name]=images['runtime']['private_library_count'];elapsed[name]=command['elapsed_seconds']
assert r['storage']['released']and r['storage']['minimum_observed_free_bytes']>=r['storage']['running_minimum_bytes']
target.mkdir(parents=True);shutil.copytree(source,target/'raw')
for name in ['command.json','protocol.json','cargo.stdout.jsonl','cargo.stderr.log','compile-only.sb']:
 dest=target/'helper-build'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(compiled/name,dest)
for name in ['build.json','commands.json']:
 dest=target/'input-preparation'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(cache/'valid-mel-mixed-inputs-v1'/name,dest)
for name in ['build-valid-mel-mixed-helper.py','prepare-valid-mel-mixed-inputs.py','freeze-valid-mel-mixed.py','cleanup-stale-gen2-objects-mixed-v1.py','compiler-object-cleanup-mixed-v1.json','compiler-object-cleanup-mixed-v1.events.jsonl','cleanup-stale-gen2-objects-mixed-v2.py','compiler-object-cleanup-mixed-v2.json','compiler-object-cleanup-mixed-v2.events.jsonl']:
 shutil.copy2(cache/name,target/name)
for name in p['tools']:
 dest=target/'execution-tools'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(tools/name,dest)
shutil.copy2(Path(__file__),target/Path(__file__).name)
record={'schema_version':1,'mixed_residency_smoke_passed':True,'release_qualified':False,'scope':r['scope'],'helper_build_source_commit':compile_record['source_commit'],'neural_run_source_commit':r['source_commit'],'collection_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'protocol':r['protocol'],'frozen_tool_count':len(p['tools']),'frozen_compiled_repository_files':len(p['repository_files']),'staged_regular_files':len(p['files']),'staged_aliases':len(p['aliases']),'compiled_helper':compile_record['helper'],'commands_passed':len(commands),'cases':r['cases'],'elapsed_seconds':elapsed,'loaded_private_image_counts':counts,'total_private_runtime_image_records':sum(counts.values()),'auxiliary_provider_records':4,'chat_initial_and_restored_layers':29,'storage':r['storage'],'new_native_or_neural_calls_during_collection':0,'limits':['Sequential residency/admission smoke only: 1s+20ms silent diarization, 20ms silent Bark and one Laya decision.','Nonempty initial/restored chat and actual placement pass; no claim of unchanged isolated text, real-speech parity, simultaneous neural calls, heldout accuracy or desktop release.','Helper compiled once at the recorded source; all314compiled repository inputs match the later neural-run source.','Raw model/helper/library payloads stay in the bound local staging cache; their descriptors and complete actual loaded-image audits are retained.'],'files':{str(path.relative_to(target)):descriptor(path)for path in sorted(target.rglob('*'))if path.is_file()and '__pycache__'not in path.parts}}
(target/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'target':str(target),'members':len(record['files'])+1,'images':counts,'elapsed':elapsed}))
