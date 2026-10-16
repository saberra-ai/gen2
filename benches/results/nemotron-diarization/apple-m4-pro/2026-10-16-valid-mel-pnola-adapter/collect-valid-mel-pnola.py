"""Collect saved Pnola adapter evidence; execute no test helper, native code or model."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from qualify_valid_mel_pnola import timeline_matches
from macos_auxiliary_loaded_libraries import audit_with_auxiliary
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
read=lambda p:json.loads(p.read_text())
source=cache/'valid-mel-pnola-runtime-v2';r=read(source/'qualification.json');p=read(source/'protocol.json');commands=read(source/'commands.json');work=Path(p['work_path']);out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-pnola-adapter';assert not out.exists()
assert r['adapter_gates_passed']is True and r['release_qualified']is False and not r['microphone_used']and len(r['cases'])==10 and len(commands)==11 and all(c['returncode']==0 for c in commands)
assert descriptor(source/'protocol.json')==r['protocol']and(source/'protocol.json').read_bytes()==(tools/'valid-mel-pnola.protocol.json').read_bytes()
for name,expected in p['preparation_records'].items():assert descriptor(Path(name))==expected
compiled=read(cache/'valid-mel-pnola-helper-build-v2/protocol.json');build=read(cache/'valid-mel-pnola-helper-build-v2/build.json');assert build['compiled']and not build['models_loaded']and all(c['returncode']==0 for c in build['commands'])
assert compiled['repository_files']==p['repository_files']and compiled['pnola_files']==p['pnola_files']
for records,root in [(p['tools'],tools),(p['repository_files'],repo),(p['pnola_files'],Path(p['pnola_root'])),(p['files'],work)]:
 for name,expected in records.items():assert descriptor(root/name)==expected,name
for name,expected in p['repository_files'].items():
 for commit in [build['gen2_source_commit'],r['source_commit']]:
  data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo);assert{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==expected,name
for name,expected in p['tools'].items():
 data=subprocess.check_output(['git','show',r['source_commit']+':tools/nemotron-diarization/'+name],cwd=repo);assert{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==expected,name
assert {str(f.relative_to(work))for f in work.rglob('*')if f.is_file()and not f.is_symlink()}==set(p['files'])
assert {str(f.relative_to(work)):str(f.readlink())for f in work.rglob('*')if f.is_symlink()}==p['aliases']
assert descriptor(work/'bin/pnola-tests')=={k:build['tests'][k]for k in ['bytes','sha256']}and descriptor(work/'cli/Contents/MacOS/pnola-cli')=={k:build['cli'][k]for k in ['bytes','sha256']}
manifests={backend:verify(work/backend/'nemo')for backend in ['cpu','metal']}
listed={line.removesuffix(': test')for line in(source/'list.stdout.log').read_text().splitlines()if line.endswith(': test')};assert listed==set(p['ignored_test_names'])
assert audit_loaded_libraries((source/'list.stderr.log').read_text(),work/'cpu/nemo',manifests['cpu'],work/'bin/pnola-tests',require_runtime=False)==read(source/'list-images.json')
counts={};providers={};comparisons={};live={};bark=read(work/'assets/bark-reference.json')['fulltext']
for case,saved,command in zip(p['cases'],r['cases'],commands[1:]):
 name=case['name'];backend=case['backend'];assert name==saved['name']==command['name']and saved['passed']
 executable=work/('bin/pnola-tests'if case['kind']=='test'else'cli/Contents/MacOS/pnola-cli');provider='providers/libonnxruntime.dylib'if case['kind']=='test'else'cli/Contents/Frameworks/libonnxruntime.dylib'
 auxiliary={}if case.get('missing')=='GEN2_BARK_BUNDLE'else{str(work/provider):p['files'][provider]}
 manifest=dict(manifests[backend],libraries={})if case.get('missing')=='GEN2_NEMO_LIBRARY'else manifests[backend]
 images=audit_with_auxiliary((source/(name+'.stderr.log')).read_text(),work/backend/'nemo',manifest,executable,auxiliary)
 assert images==read(source/(name+'.images.json'))and descriptor(source/(name+'.images.json'))==saved['loaded_images']
 counts[name]=images['runtime']['private_library_count'];providers[name]=images['auxiliary_library_count']
 stdout=(source/(name+'.stdout.log')).read_text()
 if case['kind']=='test':assert'test '+case['test']+' ... ok'in stdout and'1 passed; 0 failed; 0 ignored;'in stdout
 result=source/(name+'.result.json')
 if case.get('shutdown'):assert not result.exists();continue
 assert descriptor(result)==saved['result'];value=read(result);reference=read(work/'references'/(backend+'.json'))
 if case.get('missing')=='GEN2_NEMO_LIBRARY':assert value['text']==bark and value['asr_error']is None and value['attribution']['status']=='unavailable'
 elif case['kind']=='live':
  assert value['passed']and not value['microphone_used']and value['speaker_qualification']and value['sections']==11 and value['queued']==0 and abs(value['duration_seconds']-80.04)<=1e-6 and value['provisional_revisions']and value['speakers']['status']=='complete'
  assert value['speakers']['timeline']['runtime_version']==reference['runtime_version']and value['speakers']['timeline']['backend']==backend
  live[backend]={'sections':value['sections'],'words':value['words'],'duration_seconds':value['duration_seconds'],'provisional_revisions':len(value['provisional_revisions']),'final_turns':len(value['speakers']['timeline']['turns']),'persisted_turn_tolerance_seconds':value['persisted_turn_tolerance_seconds'],'export_bytes':value['export_bytes']}
 elif case.get('capture'):
  assert value['passed']and not value['microphone_used']and value['final_matches_recording']and value['saved_pcm_matches_input']and value['provisional_revisions']
  comparisons[name]=timeline_matches(value['final']['timeline'],reference,60.,backend,compact=True);assert comparisons[name]==saved['turn_comparison']
 else:
  assert value['attribution']['status']=='complete'and value['attribution']['complete']and value['audio_sha256']==p['input_wav_sha256']
  comparisons[name]=timeline_matches(value['attribution']['timeline'],reference,60.,backend);assert comparisons[name]==saved['turn_comparison']
  if case.get('missing')=='GEN2_BARK_BUNDLE':assert value['text']==''and isinstance(value['asr_error'],str)and value['attribution']['words_complete']is False
  else:assert value['text']==bark and value['asr_error']is None and value['attribution']['words_complete']and len(value['attribution']['words'])==117
assert sum(counts.values())==832 and sum(providers.values())==9 and len(comparisons)==6 and set(live)=={'cpu','metal'}
assert r['storage']['released']and r['storage']['reserve_reported_allocated_bytes']==268435456
out.mkdir(parents=True);shutil.copytree(source,out/'raw')
for name,expected in p['tools'].items():
 dest=out/'execution-tools'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(tools/name,dest)
shutil.copy2(Path(__file__),out/Path(__file__).name)
record={'schema_version':1,'adapter_gates_passed':True,'release_qualified':False,'scope':r['scope'],'helper_build_source_commit':build['gen2_source_commit'],'neural_run_source_commit':r['source_commit'],'collection_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'protocol':r['protocol'],'frozen_gen2_compilation_inputs':len(p['repository_files']),'frozen_pnola_compilation_inputs':len(p['pnola_files']),'frozen_staged_regular_files':len(p['files']),'staged_aliases':len(p['aliases']),'commands_passed':11,'model_cases_passed':10,'native_turn_comparisons':comparisons,'actual_loaded_private_image_counts':counts,'total_private_runtime_image_records':sum(counts.values()),'actual_auxiliary_provider_counts':providers,'total_auxiliary_provider_image_records':sum(providers.values()),'paced_live_results':live,'microphone_used':False,'storage':r['storage'],'new_native_or_neural_calls_during_collection':0,'preparation_and_interruption_receipt':'../2026-10-16-valid-mel-pnola-interruption/qualification.json','limits':['Explicit relocated artifact overrides in debug CLI/test helpers. Tauri resource staging was disabled at compilation. No signed bundle, automatic bundled resolver, GUI, physical microphone, generic Mac or release qualification.','Recorded timelines match the separately qualified same-backend reference turns. Pnola compact capture timelines do not expose raw frame counts; their comparison explicitly leaves frames unobserved. No Pnola frame-probability parity claim.','Live80.04s parity/export/recovery checks execute inside the pinned Rust helper against a separate recording API call. Saved receipt contains its final snapshot and revisions; that separate recording tensor is not delivered.','Original quality failures remain unchanged. No heldout attribution accuracy or uncontended live-performance claim.','Original interrupted attempt remains failed. This collection reuses saved successful outputs and invokes no helper or neural call.'],'files':{str(path.relative_to(out)):descriptor(path)for path in sorted(out.rglob('*'))if path.is_file()and '__pycache__'not in path.parts}}
(out/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'target':str(out),'members':len(record['files'])+1,'images':sum(counts.values()),'providers':sum(providers.values()),'live':live}))
