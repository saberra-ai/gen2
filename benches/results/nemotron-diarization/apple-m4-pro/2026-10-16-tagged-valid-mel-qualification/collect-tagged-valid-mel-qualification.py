"""Replay fixed taggedCPU evidence from saved arrays, without helper/model execution."""
import hashlib,json,pathlib,shutil,subprocess,sys,wave
import numpy as np
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');base=cache/'tagged-valid-mel-qualification-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-tagged-valid-mel-qualification'
def read(path):return json.loads(path.read_text())
def rec(path):return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def blob(source,path):return subprocess.check_output(['git','show',source+':'+path],cwd=repo)
p=read(base/'protocol.json');r=read(base/'result.json');source=r['source_commit'];assert source=='b5693710a14853bfe9a23444bbb93a79d488dc49'
assert blob(source,'tools/nemotron-diarization/tagged-valid-mel-qualification.protocol.json')==(base/'protocol.json').read_bytes()
assert rec(base/'protocol.json')['sha256']==r['protocol_sha256']=='c6c1be84d777ddcb77fc2829e2df27aa29f9e1670f51b291e8765f4b51379dec'
for name,value in r['files'].items():assert rec(base/name)==value
for name,value in p['inputs'].items():assert rec(pathlib.Path(name))==value
for group,prefix in [('tools','tools/nemotron-diarization/'),('rust_sources','')]:
 for name,value in p[group].items():
  data=blob(source,prefix+name);assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==value==rec(repo/(prefix+name))
assert rec(cache/'run-tagged-valid-mel-qualification.py')==p['external_launcher']
profile=pathlib.Path(str(base)+'.sb');assert profile.read_text()=='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(x)+')'for x in p['denied_paths'])+')\n'
for name,count in [('comparison-preflight',8),('profile-preflight',7),('qualification',None)]:
 command=read(pathlib.Path(str(base)+'.'+name+'.command.json'));assert command['source_commit']==source and command['returncode']==0
 assert command['environment']=={'PATH':'/usr/bin:/bin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}and command['argv'][:4]==['/usr/bin/sandbox-exec','-f',str(profile),str(cache/'evidence-tools-env/bin/python')]
 if count:assert 'Ran '+str(count)+' tests'in pathlib.Path(str(base)+'.'+name+'.stderr.log').read_text()and'OK'in pathlib.Path(str(base)+'.'+name+'.stderr.log').read_text()
 else:assert command['argv'][-1]==str(tools/'qualify_tagged_valid_mel.py')
metadata=read(pathlib.Path(p['sdk_metadata_path']));assert metadata['runtime_reported_version']==p['runtime_version']=='nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
for name,value in metadata['installed_library_paths'].items():
 path=pathlib.Path(p['sdk_path'])/'lib'/name;assert rec(path)=={k:value[k]for k in ['bytes','sha256']}and(str(path.readlink())if path.is_symlink()else None)==value['symlink_target']
a=read(pathlib.Path(p['materialized_audit_path']))
for name,value in p['materialized_headers'].items():assert a['files'][name]==value==rec(pathlib.Path(a['compiled_tree_path'])/name)
probes=pathlib.Path(p['probe_directory']);build=read(probes/'build.json');assert read(base/'probe-build.json')==build and build['neural_calls']==0 and build['probes_executed']is False
assert all(c['returncode']==0 for c in build['commands'])and [c['name']for c in build['commands']]==['finish-probe','c-api-probe','rust-release']
for name,value in build['source_files'].items():assert rec(repo/name)==value
for name,value in build['binaries'].items():
 assert rec(probes/name)==value
 assert (probes/(name+'.load-commands.txt')).read_bytes()==subprocess.check_output(['/usr/bin/otool','-l',str(probes/name)])
sys.path.insert(0,str(tools));from valid_mel_finish_metrics import audit_outputs
from frontend_metrics import difference
from compare_stream_valid_mel import compare
from package_macos import discover,qualified_entry
sdk=pathlib.Path(p['sdk_path']);actual_closure=discover(qualified_entry(sdk,metadata),sdk,metadata);closure=read(probes/'loader-closure.json')
assert {str(path):{**rec(path),'dependencies':{name:str(target)for name,target in info['edges'].items()}}for path,info in actual_closure.items()}==closure and len(closure)==92
summary={'fixed_candidate_finish_regression_completed':True,'stream_sessions':28,'direct_graph_calls':4,'boundary_cases':21,'tolerance':0.0001}
mel=np.fromfile(p['mel_path'],dtype='<f4').reshape(112,128)[:111]
for name,model in [('f32',p['f32_model_path']),('q8',p['q8_model_path'])]:
 directory=base/name;assert audit_outputs(directory/'probe')==r['suites'][name]
 assert len(r['suites'][name]['sessions'])==28 and all(value['bitwise_equal']and value['maximum_absolute_error']==0 for value in r['suites'][name]['probability_comparisons'].values())
 assert read(directory/'suite.stdout.log')==summary
 assert read(directory/'suite.command.json')=={'argv':[str(probes/'valid-mel-finish-probe'),model,p['pcm_path'],p['mel_path'],str(directory/'probe')],'environment':{'PATH':'/usr/bin:/bin','LC_ALL':'C'},'returncode':0}
 for left in [0,1]:
  for stage,rows in [('first',mel[:104]),('final',mel[72-left*8:])]:assert (directory/'probe'/('direct-left'+str(left)+'-'+stage+'.mel.f32')).read_bytes()==rows.tobytes()
previous=np.fromfile(p['previous_f32_whole_path'],dtype='<f4').reshape(111,8);current=np.fromfile(base/'f32/probe/whole.f32',dtype='<f4').reshape(111,8);assert difference(current,previous,0.0001)==r['f32_rebuild']==read(base/'f32-rebuild-comparison.json')and r['f32_rebuild']['bitwise_equal']
public=[]
for case in p['public_cases']:
 directory=base/('public-'+case['name']);native=read(directory/'native.json');gen2=read(directory/'gen2.json');raw=pathlib.Path(str(directory/'native.json')+'.f32').read_bytes();actual=(directory/'gen2.f32').read_bytes()
 assert read(directory/'gen2.stdout.log')==gen2 and raw==actual
 assert read(directory/'native.command.json')=={'argv':[str(probes/'native-stream-probe-valid-mel'),p['q8_model_path'],case['wav_path'],'1','-1',str(directory/'native.json')],'environment':{'PATH':'/usr/bin:/bin','LC_ALL':'C'},'returncode':0}
 assert read(directory/'gen2.command.json')=={'argv':[str(probes/'nemotron-valid-mel-stream'),p['assets_path'],str(sdk/'lib/libnemo_speech_asr_c.dylib'),case['wav_path'],'1',str(directory/'gen2.f32')],'environment':{'PATH':'/usr/bin:/bin','LC_ALL':'C'},'returncode':0}
 with wave.open(case['wav_path'],'rb')as reader:
  assert(reader.getnchannels(),reader.getsampwidth(),reader.getframerate(),reader.getnframes())==(1,2,16000,case['samples']);audio=np.frombuffer(reader.readframes(case['samples']),dtype='<i2').astype('<f4')/32768
 assert hashlib.sha256(audio.tobytes()).hexdigest()==case['pcm_sha256']==gen2['input_pcm_f32le_sha256']==native['input_pcm_f32le_sha256']
 assert gen2['input_repeats']==native['input_repeats']==1 and gen2['accepted_samples']==case['samples']
 comparison=compare(gen2,native,raw,actual);pipeline=np.fromfile(base/'q8/probe'/case['cpp_reference'],dtype='<f4').reshape(case['samples']//160,8);comparison['c_api_vs_cpp_pipeline']=difference(np.frombuffer(raw,dtype='<f4').reshape(case['samples']//160,8),pipeline,0.000001)
 public.append({'case':case['name'],**comparison});assert public[-1]==read(directory/'comparison.json')
assert public==r['public_native_comparisons']and all(c['maximum_probability_error']==0 and c['maximum_turn_error_seconds']==0 and c['c_api_vs_cpp_pipeline']['bitwise_equal']for c in public)
assert r['native_finish_logical_cases']==56 and r['explicit_direct_graph_calls']==8 and r['public_rust_stream_runs']==r['independent_c_api_stream_runs']==2 and r['models_instantiated']==6
assert r['neural_inference_performed']is True
for key in ['hf_original_nemo_or_ami_target_executed','der_scoring_performed','original_quality_gate_changed','all_platform_or_release_qualified']:assert r[key]is False
assert not out.exists();shutil.copytree(base,out)
for suffix in ['sb',*[name+'.'+kind for name in ['comparison-preflight','profile-preflight','qualification']for kind in ['command.json','stdout.log','stderr.log']]]:shutil.copy2(pathlib.Path(str(base)+'.'+suffix),out/('outer.'+suffix))
(out/'probe-construction').mkdir()
for path in probes.iterdir():
 if path.is_file()and path.name not in ['valid-mel-finish-probe','native-stream-probe-valid-mel','nemotron-valid-mel-stream','calibration.wav']:shutil.copy2(path,out/'probe-construction'/path.name)
for name in ['run-tagged-valid-mel-qualification.py','build-tagged-valid-mel-probes.py','collect-tagged-valid-mel-qualification.py']:shutil.copy2(cache/name,out/name)
q={'schema_version':1,'source_commit':source,'scope':p['scope'],'fixed_tagged_cpu_f32_q8_and_public_rust_qualified':True,'f32_and_q8_native_finish_logical_cases':56,'explicit_direct_graph_calls':8,'models_instantiated':6,'all_twelve_native_pipeline_direct_comparisons_bitwise_equal':True,'tagged_vs_untagged_f32_calibration_bitwise_equal':True,'both_public_rust_vs_c_api_and_cpp_probabilities_bitwise_equal':True,'public_frames':{'minimum320':2,'calibration17872':111},'public_turn_counts':{'minimum320':0,'calibration17872':1},'wrapper_native_turns_exact':True,'all_saved_stable_prefixes_reset_and_provisional_checks_passed':True,'preflight_model_free_tests':15,'static_non_system_loader_libraries_bound':92,'sdk_alias_paths_bound':15,'materialized_headers_bound':22,'rust_source_manifest_files_bound':257,'neural_inference_performed':True,'hf_original_nemo_or_ami_target_executed':False,'der_scoring_performed':False,'original_quality_gate_changed':False,'all_platform_or_release_qualified':False,'raw_calibration_silence_tensors_retained_in_git':True,'weights_audio_sdk_and_probe_binaries_not_in_git':True,'limits':p['limitations']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={path.relative_to(out).as_posix():rec(path)for path in out.rglob('*')if path.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'files':len(files),'native_cases':56,'direct_calls':8,'public_runs':2,'c_api_runs':2,'all_compared_probabilities_exact':True}))
