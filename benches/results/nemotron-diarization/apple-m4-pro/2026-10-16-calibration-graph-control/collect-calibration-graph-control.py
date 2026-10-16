"""Recompute saved calibration graph results; no model/FE execution."""
import hashlib,json,pathlib,shutil,subprocess,sys
import numpy as np
import torch
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');base=cache/'calibration-graph-v2';tools=repo/'tools/nemotron-diarization';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-calibration-graph-control'
def read(p):return json.loads(p.read_text())
def rec(p):return{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def array(name,shape,dtype='<f4'):return np.fromfile(base/name,dtype=dtype).reshape(shape)
result=read(base/'result.json');protocol=read(base/'protocol.json');source=result['source_commit'];assert source.startswith('f8f7d71')
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/calibration-graph-v2.protocol.json'],cwd=repo)==(base/'protocol.json').read_bytes()
assert rec(base/'protocol.json')['sha256']==result['protocol_sha256']
for name,expected in result['files'].items():assert rec(base/name)==expected
for group,root in [('tools',tools),('native_source_files',cache/'NeMo-Speech.cpp'),('sdk_libraries',pathlib.Path(protocol['linked_sdk_path'])/'lib'),('assets',cache/'assets'),('inputs',cache/'tail-framing-v1'),('evidence_receipts',repo)]:
 for name,expected in protocol[group].items():assert rec(root/name)==expected
for name,expected in protocol['tools'].items():
 blob=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo);assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==expected
assert rec(cache/'native-f32-conversion-v1/Nemotron-3-Diarization.f32-control.gguf')==protocol['native_f32_model']
assert rec(cache/'calibration-graph-probe-v2/calibration-graph-probe')==protocol['probe']
assert rec(base/'probe-build.json')==rec(cache/'calibration-graph-probe-v2/build.json')==protocol['probe_build_manifest']
assert rec(cache/'run-calibration-graph-control-v2.py')==protocol['external_launcher']
profile=pathlib.Path(str(base)+'.sb');assert profile.read_text()=='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(p)+')'for p in protocol['denied_paths'])+')\n'
for kind in ['preflight','serialization-preflight','execution']:
 command=read(pathlib.Path(str(base)+'.'+kind+'.command.json'));assert command['source_commit']==source and command['returncode']==0
 assert command['argv'][:4]==['/usr/bin/sandbox-exec','-f',str(profile),str(cache/'python-reference/bin/python')]
 if kind!='execution':assert 'Ran 3 tests' in pathlib.Path(str(base)+'.'+kind+'.stderr.log').read_text()and 'OK' in pathlib.Path(str(base)+'.'+kind+'.stderr.log').read_text()
load=subprocess.check_output(['/usr/bin/otool','-l',str(cache/'calibration-graph-probe-v2/calibration-graph-probe')],text=True);assert load==(base/'helper-load-commands.txt').read_text()
lines=load.splitlines();assert [lines[i+2].strip().removeprefix('path ').split(' (offset ')[0]for i,line in enumerate(lines)if line.strip()=='cmd LC_RPATH']==[str(pathlib.Path(protocol['linked_sdk_path'])/'lib')]
loading=read(base/'hf-loading-info.json');assert set(loading)=={'missing_keys','unexpected_keys','mismatched_keys','error_msgs'}and all(value==[]for value in loading.values())
assert read(base/'hf-load.completed.json')['device']=='cpu'and read(base/'hf-load.completed.json')['dtype']=='float32'
for name in ['hf-first.started.json','hf-first.completed.json','hf-final.started.json','hf-final.completed.json']:assert (base/name).is_file()
assert read(base/'hf-first.completed.json')=={'returned_logit_rows':72,'speaker_cache_rows':0,'fifo_rows':9,'compressed':False}
assert read(base/'hf-final.completed.json')=={'returned_logit_rows':39,'full_logit_rows':112,'prefix_rows':72,'valid_end':111,'speaker_cache_rows':0,'fifo_rows_after':14}
sys.path.insert(0,str(tools))
from calibration_graph_metrics import probability_comparisons,variants
from frontend_metrics import difference
original=cache/'tail-framing-v1/calibration'
expected=variants(np.fromfile(original/'hf-first-raw.f32',dtype='<f4').reshape(105,128),np.fromfile(original/'hf-final.f32',dtype='<f4').reshape(39,128),np.fromfile(original/'native-final.f32',dtype='<f4').reshape(40,128))
for name,value in expected.items():assert (base/(name+'.mel.f32')).read_bytes()==value.tobytes()
assert (base/'empty.fifo.f32').stat().st_size==0
hf_first=array('hf-first.full-probabilities.f32',(104,8));hf_final=array('hf-final.full-probabilities.f32',(112,8));hf_returned=array('hf-final.returned-probabilities.f32',(39,8));fifo=array('hf-first.fifo.f32',(9,512))
hf_embeds={}
for label,tokens,chunk_rows in [('first',13,13),('final',14,5)]:
 logits=array('hf-'+label+'.full-logits.f32',(tokens*8,8));prob=array('hf-'+label+'.full-probabilities.f32',(tokens*8,8));hf_embeds[label]=array('hf-'+label+'.embeddings.f32',(chunk_rows,512));graph=array('hf-'+label+'.graph-input.f32',(tokens,512))
 assert torch.from_numpy(logits).sigmoid().numpy().tobytes()==prob.tobytes()
 assert (base/('hf-'+label+'.graph-mask.u8')).read_bytes()==np.ones(tokens,np.uint8).tobytes()
 assert (base/('hf-'+label+'.positions.i64')).read_bytes()==np.arange(tokens,dtype='<i8').tobytes()
 if label=='first':assert graph.tobytes()==hf_embeds[label].tobytes()and fifo.tobytes()==hf_embeds[label][:9].tobytes()
 else:assert graph[:9].tobytes()==fifo.tobytes()and graph[9:].tobytes()==hf_embeds[label].tobytes()
assert hf_final[72:111].tobytes()==hf_returned.tobytes()
native,embeddings={},{}
for name in ['first104','valid39','zero40','extra40']:
 directory=base/name;prefix=directory/'native';first=name=='first104';rows=104 if first else 112;chunk_rows=13 if first else 5
 command=read(directory/'native.command.json');assert command['returncode']==0 and command['environment']=={'PATH':'/usr/bin:/bin','LC_ALL':'C'}
 assert command['argv']==[str(cache/'calibration-graph-probe-v2/calibration-graph-probe'),str(cache/'native-f32-conversion-v1/Nemotron-3-Diarization.f32-control.gguf'),str(base/(name+'.mel.f32')),str(base/('empty.fifo.f32'if first else'hf-first.fifo.f32')),str(prefix)]
 assert read(directory/'native.json')=={'mel_rows':len(expected[name]),'fifo_rows':0 if first else 9,'speaker_cache_rows':0,'chunk_encoder_rows':chunk_rows,'total_encoder_rows':13 if first else 14,'native_probability_rows':rows,'speakers':8,'hidden_size':512,'cpu_threads':4,'gpu_requested':False,'batching':False}
 native[name]=array(name+'/native.predictions.f32',(rows,8));embeddings[name]=array(name+'/native.embeddings.f32',(chunk_rows,512))
assert probability_comparisons(native,hf_first,hf_final,hf_returned)==result['probability_comparisons']
assert {name:difference(value,hf_embeds['first'if name=='first104'else'final'],0)for name,value in embeddings.items()}==result['pre_encoder_embedding_comparisons']
assert result['new_hf_forward_calls']==2 and result['new_native_graph_calls']==4 and result['neural_model_inference_performed']is True
assert all(result[key]is False for key in ['native_aosc_executed','frontend_or_processor_executed','original_nemo_executed','target_ami_inputs_executed','der_scoring_performed','original_gate_status_changed','release_qualified','numeric_equivalence_claimed'])
assert not out.exists();shutil.copytree(base,out)
for suffix in ['sb','preflight.command.json','preflight.stdout.log','preflight.stderr.log','serialization-preflight.command.json','serialization-preflight.stdout.log','serialization-preflight.stderr.log','execution.command.json','execution.stdout.log','execution.stderr.log']:shutil.copy2(pathlib.Path(str(base)+'.'+suffix),out/('outer.'+suffix))
for name in ['run-calibration-graph-control-v2.py','collect-calibration-graph-control.py']:shutil.copy2(cache/name,out/name)
(out/'tools').mkdir()
for name in protocol['tools']:shutil.copy2(tools/name,out/'tools'/name)
q={'schema_version':1,'source_commit':source,'fixed_calibration_graph_control_completed':True,'new_hf_forward_calls':2,'new_native_graph_calls':4,'preflight_tests_passed':6,'all_local_and_collected_file_hashes_verified':True,'all_ten_probability_and_four_embedding_reports_recomputed':True,'hf_saved_logits_sigmoid_recomputed_without_model':True,'hf_final_graph_prefix_matches_pre_mutation_fifo':True,'hf_final_current_input_matches_current_embedder_output':True,'same_first_39_mel_rows_and_fifo_in_all_final_native_cases':True,'native_valid39_and_explicit_zero40_full_outputs_bitwise_equal':True,'neural_inference_performed':True,'frontend_or_native_aosc_executed':False,'target_ami_or_original_nemo_executed':False,'der_scoring_performed':False,'original_gate_status_changed':False,'release_qualified':False,'calibration_raw_tensors_retained_in_git':True,'weights_audio_sdk_or_probe_binary_in_git':False,'linked_source_sdk_library_paths':len(protocol['sdk_libraries']),'native_source_files':len(protocol['native_source_files']),'limits':protocol['limitations']+['The controlled extraMel content intervention changes common native probabilities by0.241053342819 atfullframe107/channel0. This demonstrates its effect on this excludedcalibration with teacher-forcedHFstate; it is not universal causal attribution for originalfree-running differences or a quality fix.','Identical valid39 graph input still retains maximum probability difference0.000422716141 versusHF; no bitwise numerical-equivalence or new threshold pass is claimed.','Receiver replays only saved-array metrics and elementwise sigmoid, without frontend execution, model construction or another neural forward. Small calibration tensors are retained; no weights/audio/runtime binaries are included.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={p.relative_to(out).as_posix():rec(p)for p in out.rglob('*')if p.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'files':len(files),'NN_HF_calls':2,'NN_native_calls':4,'reports_recomputed':14,'source_sdk_paths':len(protocol['sdk_libraries'])}))
