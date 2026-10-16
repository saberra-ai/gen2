"""Receiver-only evidence collection; no Linux binaries or models executed."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from linux_chat_audit import audit_results,ggml_bindings
from qualify_linux_package import RUNTIME_PATH,loader_paths
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
base=Path(sys.argv[1]);out=Path(sys.argv[2]);chat=base/'nemo-chat-evidence';offline=base/'nemo-container-evidence'
r=json.loads((chat/'receipt.json').read_text());f=json.loads((chat/'frozen-inputs.json').read_text())
prior=json.loads((offline/'receipt.json').read_text());ci=json.loads((base/'ci.json').read_text());bf=json.loads((offline/'frozen-inputs.json').read_text())
assert ci['conclusion']=='success' and ci['headSha']==r['source_commit']==f['source_commit']==prior['source_commit']
assert r['chat_diarization_gates_passed'] is True and r['release_qualified'] is False
assert sha(chat/'frozen-inputs.json')==r['frozen_inputs_sha256']
assert sha(offline/'receipt.json')==f['offline_receipt_sha256'] and sha(offline/'frozen-inputs.json')==f['offline_frozen_inputs_sha256']
assert prior['offline_container_gates_passed'] is True
assert r['image_id']==f['image_id']==prior['image_id'] and r['image_repo_digests']==f['image_repo_digests']==prior['image_repo_digests']
for name,expected in f['tools'].items():
 data=subprocess.check_output(['git','show',ci['headSha']+':tools/nemotron-diarization/'+name],cwd=repo)
 assert hashlib.sha256(data).hexdigest()==expected,name
for name in ['linux_chat_audit.py','qualify_linux_package.py','compare_stream.py']:
 assert sha(repo/'tools/nemotron-diarization'/name)==f['tools'][name],name
example=subprocess.check_output(['git','show',ci['headSha']+':examples/nemotron_chat_coexistence.rs'],cwd=repo)
assert hashlib.sha256(example).hexdigest()==f['example_sha256']
gitlink=subprocess.check_output(['git','ls-tree',ci['headSha'],'vendor/llama-cpp-sys-2/llama.cpp'],cwd=repo,text=True).split()[2]
assert gitlink==f['chat_llama_revision']
assert sha(chat/'independent-native.json')==f['native_reference_sha256']==sha(offline/'results/native.json')
assert sha(chat/'independent-native.f32')==f['native_probabilities_sha256']==sha(offline/'results/native.json.f32')
for name,record in bf['inputs'].items():assert f['files'][name]==record,name
package=json.loads((base/'independently verified delivered runtime ü/package.json').read_text())
assert f['package_manifest_sha256']==sha(base/'independently verified delivered runtime ü/package.json')==prior['package_manifest_sha256']
lock=json.loads((repo/'tools/nemotron-diarization/linux-chat.lock.json').read_text())
for item in lock['artifacts']:
 record=f['files']['chat/'+item['name']]
 assert record['bytes']==item['size'] and record['sha256']==item['sha256']
assert f['fingerprints']['chat_model']==next(x['sha256'] for x in lock['artifacts'] if x['name'].endswith('.gguf'))
assert f['fingerprints']['executable']==f['files']['bin/chat-coexistence']['sha256']
assert f['fingerprints']['library']==f['files']['relocated runtime ü/lib/'+package['entry_library']]['sha256']
for key,name in [('diarization_model','Nemotron-3-Diarization.q8_0.gguf'),('wav','ami_en2002d_2132.wav')]:
 assert f['fingerprints'][key]==f['files']['assets/'+name]['sha256']
gomp=f['openmp_test_runtime'];gcc=package['components']['libgcc-s1']['source_bundle']
assert gomp['shared_gcc_source_bundle']==gcc
for record in [gomp['package'],gomp['copyright_owner']]:
 assert (record['source:Package'],record['source:Version'])==(gcc['source_package'],gcc['source_version'])
assert gomp['sha256']==f['files']['chat/lib/libgomp.so.1']['sha256']
native=json.loads((chat/'independent-native.json').read_text());raw=(chat/'independent-native.f32').read_bytes()
comparison=audit_results(chat/'results',native,raw,f['fingerprints'])
assert comparison==r['comparisons']==json.loads((chat/'comparison.json').read_text())
loaded=json.loads((chat/'loaded-libraries.json').read_text());bindings=json.loads((chat/'ggml-bindings.json').read_text());overlaps={}
for order in ['isolated','chat-first','diar-first']:
 trace=(chat/(order+'.stderr.log')).read_text()
 gcc_paths={RUNTIME_PATH+'/lib/libstdc++.so.6',RUNTIME_PATH+'/lib/libgcc_s.so.1'}
 extra={'/work/chat/lib/libgomp.so.1'}|(gcc_paths if order=='isolated' else set())
 assert loader_paths(trace,[] if order=='isolated' else package['libraries'],extra)==loaded[order]
 if order!='isolated':
  assert ggml_bindings(trace,package['libraries'])==bindings[order]
  assert len(bindings[order])==r['native_ggml_binding_counts'][order]
  summary=json.loads((chat/'results'/order/'summary.json').read_text());i=summary['concurrent_intervals'];a=i['chat_micros'];b=i['diarization_micros']
  overlaps[order]=min(a[1],b[1])-max(a[0],b[0])
commands=json.loads((chat/'commands.json').read_text());executed=[]
for command in commands:
 assert command['returncode']==0,command['name']
 args=command['argv']
 if args[:2]!=['docker','run']:continue
 assert args[args.index('--network')+1]=='none' and '--read-only' in args and '--user' in args
 assert args[args.index('--pull')+1]=='never' and f['image_id'] in args
 assert args[args.index('--cap-drop')+1]=='ALL'
 executed.append(command['name'])
assert executed==['isolated','chat-first','diar-first']
shutil.copytree(chat,out/'nemo-chat-evidence')
base_check=json.loads((out/'qualification.json').read_text())
qualification={**r,'offline_container_gates_passed':True,'offline_container_receiver_verification':base_check['receiver_verification'],
 'receiver_verification':{'historical_tool_and_example_sources_verified':True,'chat_llama_gitlink_verified':True,'same_source_offline_prerequisite_verified':True,
  'all_six_output_comparisons_recomputed':True,'all_three_loader_traces_reparsed':True,'both_ggml_binding_traces_reparsed':True,'actual_api_overlap_microseconds':overlaps,
  'scope':'Read-only macOS verification of delivered archive/source bytes and evidence logs/results. Model/helper/OpenMP test-payload identities are recorded by CI; those separate test binaries/assets were not delivered for receiver rehashing. Actual Linux execution occurred only in the named CI run.'}}
(out/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
shutil.copy2(Path(__file__),out/'collect-linux-chat.py')
files={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='collection.json'}
(out/'collection.json').write_text(json.dumps({'schema_version':1,'release_qualified':False,'files':files},indent=2)+'\n')
print('PASS Linux chat collection',len(files),'members; overlaps',overlaps)
