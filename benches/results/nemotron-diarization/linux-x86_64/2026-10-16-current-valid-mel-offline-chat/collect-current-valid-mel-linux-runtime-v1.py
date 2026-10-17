"""Reverify saved Linux offline/chat CI evidence; run no native code or neural calls."""
import hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');base=cache/'current-valid-mel-linux-runtime-ci-v1/evidence';tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from platform_valid_mel_native import compare_complete_60s as compare
from linux_chat_audit import audit_results,ggml_bindings
from qualify_linux_package import loader_paths,require_lifecycle,RUNTIME_PATH
from runtime_profiles import validate_provenance
from sdk_archive import inventory
from frozen_git_sources import verify_git_sources
out=repo/'benches/results/nemotron-diarization/linux-x86_64/2026-10-16-current-valid-mel-offline-chat';assert not out.exists()
read=lambda p:json.loads(p.read_text())
sha=lambda p:descriptor(p)['sha256']
r=read(base/'qualification.json');p=read(base/'protocol.json');f=read(base/'frozen-inputs.json');offline=base/'offline';chat=base/'chat';prior=read(offline/'receipt.json');cr=read(chat/'receipt.json');bf=read(offline/'frozen-inputs.json');cf=read(chat/'frozen-inputs.json')
run=read(base.parent/'run.json');run['id']=run['databaseId'];run['head_sha']=run['headSha'];jobs=run['jobs'];jobs[0]['id']=jobs[0]['databaseId'];artifact=read(base.parent/'download-identity.json')
assert run['id']==38990890735 and run['conclusion']=='success' and len(jobs)==1 and jobs[0]['conclusion']=='success'
assert artifact['id']==12085348684 and not artifact['expired'] and artifact['workflow_run']['head_sha']==run['head_sha']==r['source_commit']==f['source_commit']=='4083013d56c59f67ad766ade1c331c3dd274eb67'
archive=base.parent/'artifact.zip';assert 'sha256:'+sha(archive)==artifact['digest'] and archive.stat().st_size==artifact['size_in_bytes']
with zipfile.ZipFile(archive)as z:
 members={item.filename for item in z.infolist() if not item.is_dir()}
 assert members=={path.relative_to(base).as_posix()for path in base.rglob('*')if path.is_file()}
 for item in z.infolist():
  if not item.is_dir():assert (base/item.filename).read_bytes()==z.read(item)
assert (base/'protocol.json').read_bytes()==subprocess.check_output(['git','show',run['head_sha']+':tools/nemotron-diarization/current-valid-mel-linux-runtime.protocol.json'],cwd=repo) and descriptor(base/'protocol.json')==r['protocol']
for name in ['tools','repository_files']:assert p[name]==f[name]
pins={'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()};pins.update(p['repository_files'])
verify_git_sources(repo,run['head_sha'],pins)
for name,revision in p['gitlinks'].items():verify_git_sources(repo/name,revision,p['vendor_sources'])
assert f['gitlinks']==p['gitlinks']
for path,expected in p['gitlinks'].items():assert subprocess.check_output(['git','ls-tree',run['head_sha'],path],cwd=repo,text=True).split()[2]==expected
assert r['offline_container_gates_passed']is True and r['chat_diarization_gates_passed']is True and r['release_qualified']is False
assert descriptor(offline/'receipt.json')==r['offline_receipt'] and descriptor(chat/'receipt.json')==r['chat_receipt']
for receipt,frozen in [(prior,offline/'frozen-inputs.json'),(cr,chat/'frozen-inputs.json')]:
 validate_provenance(receipt);assert receipt['source_commit']==r['source_commit'] and not receipt['release_qualified'];assert sha(frozen)==receipt['frozen_inputs_sha256']
assert prior['offline_container_gates_passed']and prior['negative_control_passed']and cr['chat_diarization_gates_passed']
assert cf['offline_receipt_sha256']==sha(offline/'receipt.json') and cf['offline_frozen_inputs_sha256']==sha(offline/'frozen-inputs.json')
assert cr['image_id']==prior['image_id']==cf['image_id']==bf['image']['Id'] and cr['image_repo_digests']==prior['image_repo_digests']==cf['image_repo_digests']==bf['image']['RepoDigests']
package_root=cache/'valid-mel-linux-delivered-runtime-v1';package=read(package_root/'package.json')
assert sha(package_root/'package.json')==prior['package_manifest_sha256']==cf['package_manifest_sha256']==bf['package_manifest_sha256']
assert {k:v for k,v in bf['inputs'].items()if k.startswith('relocated runtime ü/')}=={'relocated runtime ü/'+k:v for k,v in inventory(package_root).items()if k!='.'}
for path,record in bf['inputs'].items():assert cf['files'][path]==record,path
for frozen in [bf,cf]:
 for name,expected in frozen['tools'].items():assert sha(tools/name)==expected,name
native=read(offline/'results/native.json');public=read(offline/'results/public.json');raw=(offline/'results/native.json.f32').read_bytes()
comparison=compare(public,native,raw,(offline/'results/public.f32').read_bytes());assert comparison==prior['native_public_comparison']==read(offline/'comparison.json')
assert public['input_pcm_f32le_sha256']==native['input_pcm_f32le_sha256']==read(tools/'valid-mel-platform-native.protocol.json')['cases']['actual_pcm_sha256']
assert raw==(offline/'results/public.f32').read_bytes();require_lifecycle((offline/'packaged-lifecycle-list.stdout.log').read_text());require_lifecycle((offline/'packaged-lifecycle.stdout.log').read_text(),completed=True)
loaded={name:loader_paths((offline/(name+'.stderr.log')).read_text(),package['libraries'])for name in ['packaged-native','packaged-public','packaged-lifecycle']};assert loaded==read(offline/'loaded-libraries.json')
assert 'libgen2_nemo_linux_ggml-base.so.0.25.1: cannot open shared object file'in(offline/'missing-library-control.stderr.log').read_text() and not(offline/'results/negative.f32').exists()
assert sha(chat/'independent-native.json')==cf['native_reference_sha256']==sha(offline/'results/native.json') and sha(offline/'results/native.json.f32')==cf['native_probabilities_sha256']
chat_comparison=audit_results(chat/'results',native,raw,cf['fingerprints'],comparator=compare);assert chat_comparison==cr['comparisons']==read(chat/'comparison.json')
chat_loaded={};bindings={};overlaps={}
for order in ['isolated','chat-first','diar-first']:
 trace=(chat/(order+'.stderr.log')).read_text();extra={'/work/chat/lib/libgomp.so.1'}|({RUNTIME_PATH+'/lib/libstdc++.so.6',RUNTIME_PATH+'/lib/libgcc_s.so.1'}if order=='isolated'else set())
 chat_loaded[order]=loader_paths(trace,[]if order=='isolated'else package['libraries'],extra)
 if order!='isolated':
  bindings[order]=ggml_bindings(trace,package['libraries']);assert len(bindings[order])==cr['native_ggml_binding_counts'][order]
  i=read(chat/'results'/order/'summary.json')['concurrent_intervals'];a=i['chat_micros'];b=i['diarization_micros'];overlaps[order]=min(a[1],b[1])-max(a[0],b[0]);assert overlaps[order]>0
assert chat_loaded==read(chat/'loaded-libraries.json') and bindings==read(chat/'ggml-bindings.json')
gomp=cf['openmp_test_runtime'];gcc=package['components']['libgcc-s1']['source_bundle'];assert gomp['shared_gcc_source_bundle']==gcc
for record in [gomp['package'],gomp['copyright_owner']]:assert(record['source:Package'],record['source:Version'])==(gcc['source_package'],gcc['source_version'])
assert gomp['sha256']==cf['files']['chat/lib/libgomp.so.1']['sha256']
containers={}
for stage in ['offline','chat']:
 commands=read(base/stage/'commands.json');containers[stage]=[]
 for c in commands:
  assert c['returncode']!=0 if c['name']=='missing-library-control'else c['returncode']==0,c['name']
  argv=c['argv']
  if argv[:2]!=['docker','run']:continue
  assert argv[argv.index('--network')+1]=='none' and '--read-only'in argv and '--user'in argv and argv[argv.index('--pull')+1]=='never' and prior['image_id']in argv and argv[argv.index('--cap-drop')+1]=='ALL'
  assert all('readonly'in argv[i+1]for i,a in enumerate(argv)if a=='--mount'and'dst=/results'not in argv[i+1])
  containers[stage].append(c['name'])
assert containers=={'offline':['container-controls','packaged-native','packaged-public','packaged-lifecycle-list','packaged-lifecycle','platform-library-hashes','missing-library-control'],'chat':['isolated','chat-first','diar-first']}
assert read(base/'commands.json')==r['commands'] and len(r['commands'])==2 and all(c['returncode']==0 for c in r['commands'])
out.mkdir(parents=True);shutil.copytree(base,out/'raw')
for name in ['run.json','download-identity.json','delivery.json']:shutil.copy2(base.parent/name,out/name)
for name,expected in p['tools'].items():
 assert descriptor(tools/name)==expected
 dest=out/'execution-tools'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(tools/name,dest)
shutil.copy2(Path(__file__),out/Path(__file__).name)
record={'schema_version':1,'offline_container_gates_passed':True,'chat_diarization_gates_passed':True,'release_qualified':False,'scope':'Saved current-profile Ubuntu24.04 AVX2/FMA/F16C offline runtime and actual CPU chat coexistence evidence. No new model calls during collection.','producer_source_commit':r['source_commit'],'collection_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'run_id':run['id'],'job_id':jobs[0]['id'],'artifact_id':artifact['id'],'artifact_zip':descriptor(archive),'artifact_zip_verified_against_github_digest':True,'protocol':r['protocol'],'source_pins':{'tools':len(p['tools']),'repository_files':len(p['repository_files']),'gitlinks':p['gitlinks'],'vendor_files':len(p['vendor_sources'])},'offline_comparison':comparison,'chat_comparisons':chat_comparison,'actual_api_overlap_microseconds':overlaps,'native_ggml_binding_counts':cr['native_ggml_binding_counts'],'offline_loaded_private_images':sum(len(x['bundled'])for x in loaded.values()),'containers':containers,'corresponding_openmp_source_verified':True,'new_native_or_neural_calls_during_collection':0,'limits':['Actual Linux execution occurred only in the named CI run. Model/helper/OpenMP test binaries were not delivered in this evidence artifact; their frozen producer identities are recorded, not rehashed on the receiver.','No Bark/Laya, physical desktop, generic Linux baseline, heldout quality, uncontended live performance or release qualification.','Loader bindings cover these binaries/calls, not general ELF namespace isolation.'],'files':{str(path.relative_to(out)):descriptor(path)for path in sorted(out.rglob('*'))if path.is_file()}}
(out/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'target':str(out),'members':len(record['files'])+1,'overlaps':overlaps,'bindings':cr['native_ggml_binding_counts']}))
