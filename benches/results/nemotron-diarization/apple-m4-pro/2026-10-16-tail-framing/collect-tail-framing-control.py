"""Receiver-side framing replay; no frontend or neural execution."""
import hashlib,json,pathlib,shutil,subprocess,sys
import numpy as np
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');base=cache/'tail-framing-v1';tools=repo/'tools/nemotron-diarization';output=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-tail-framing'
def read(p):return json.loads(p.read_text())
def record(p):return{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
result=read(base/'result.json');protocol=read(base/'protocol.json');source=result['source_commit']
assert source.startswith('5204386')
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/tail-framing.protocol.json'],cwd=repo)==(base/'protocol.json').read_bytes()
assert record(base/'protocol.json')['sha256']==result['protocol_sha256']
for name,expected in protocol['tools'].items():
 assert record(tools/name)==expected
 blob=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo);assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==expected
for name,expected in result['files'].items():assert record(base/name)==expected
for group,root in [('native_source_files',cache/'NeMo-Speech.cpp'),('sdk_libraries',pathlib.Path(protocol['linked_sdk_path'])/'lib'),('assets',cache/'assets'),('frontend_inputs',cache/'frontend-diagnostic-ec88564')]:
 for name,expected in protocol[group].items():assert record(root/name)==expected
assert record(cache/'frontend-tail-probe-v1/frontend-tail-probe')==protocol['probe']
assert record(base/'probe-build.json')==record(cache/'frontend-tail-probe-v1/build.json')==protocol['probe_build_manifest']
build=read(base/'probe-build.json');assert build['returncode']==0 and build['argv']==protocol['probe_build_command'] and build['source']==protocol['tools']['frontend_tail_probe.cpp']and build['binary']==protocol['probe']
assert record(cache/'run-tail-framing-control.py')==protocol['external_launcher']
profile=pathlib.Path(str(base)+'.sb');assert profile.read_text()=='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(p)+')'for p in protocol['denied_paths'])+')\n'
for kind in ['preflight','execution']:
 command=read(pathlib.Path(str(base)+'.'+kind+'.command.json'));assert command['source_commit']==source and command['returncode']==0
 assert command['argv'][:5]==['/usr/bin/sandbox-exec','-f',str(profile),str(cache/'python-reference/bin/python'),'-m'if kind=='preflight'else str(tools/'diagnose_tail_framing.py')]
 assert pathlib.Path(str(base)+'.'+kind+'.stdout.log').is_file()and pathlib.Path(str(base)+'.'+kind+'.stderr.log').is_file()
assert '-p' in read(pathlib.Path(str(base)+'.preflight.command.json'))['argv']
assert 'Ran 3 tests' in pathlib.Path(str(base)+'.preflight.stderr.log').read_text()
assert 'OK' in pathlib.Path(str(base)+'.preflight.stderr.log').read_text()
load=subprocess.check_output(['/usr/bin/otool','-l',str(cache/'frontend-tail-probe-v1/frontend-tail-probe')],text=True);assert load==(base/'helper-load-commands.txt').read_text()
lines=load.splitlines();assert [lines[i+2].strip().removeprefix('path ').split(' (offset ')[0]for i,line in enumerate(lines)if line.strip()=='cmd LC_RPATH']==[str(pathlib.Path(protocol['linked_sdk_path'])/'lib')]
sys.path.insert(0,str(tools))
from tail_framing import native_schedule,stack_inputs
from frontend_metrics import difference
assert len(result['results'])==5
assert result['neural_model_instantiation_or_inference']is False and result['native_scheduler_executed']is False and result['native_scheduler_source_replayed']is True and result['learned_projection_or_cache_executed']is False and result['checkpoint_deserialization_performed']is False
for row in result['results']:
 case=base/row['case'];assert read(case/'framing.json')==row
 assert record(case/'audio.f32')==row['pcm']
 native=np.fromfile(case/'native-stream.f32',dtype='<f4').reshape(-1,128);metadata=row['native_frontend'];assert native.shape==(metadata['raw_frames'],128)and np.isfinite(native).all()
 command=read(case/'native-stream.command.json');assert command['returncode']==0 and command['environment']=={'PATH':'/usr/bin:/bin','LC_ALL':'C'}
 assert command['argv']==[str(cache/'frontend-tail-probe-v1/frontend-tail-probe'),str(case/'audio.f32'),str(cache/'frontend-diagnostic-ec88564/native-bank.f32'),str(case/'native-stream.f32')]
 assert json.loads((case/'native-stream.stdout.log').read_text())==metadata
 schedule=native_schedule(metadata['available_after_push'],len(native));assert schedule==row['native_source_derived_schedule']
 nf=schedule[-1];hf=row['hf_actual_calls'][-1];native_final=native[nf['start']:nf['input_end']]
 assert np.array_equal(native_final,np.fromfile(case/'native-final.f32',dtype='<f4').reshape(-1,128))
 for label in ['native','hf']:
  features=np.fromfile(case/(label+'-final.f32'),dtype='<f4').reshape(-1,128);mask=np.fromfile(case/(label+'-final.mask.u8'),dtype=np.uint8)
  stacked,encoder_mask,padding=stack_inputs(features,mask)
  assert stacked.tobytes()==(case/(label+'-final-stack.f32')).read_bytes()
  assert encoder_mask.tobytes()==(case/(label+'-final-encoder.mask.u8')).read_bytes()
  assert row['final_stack_inputs'][label]=={'real_mel_rows':len(features),'zero_feature_padding_rows':padding,'encoder_tokens':len(stacked),'all_encoder_tokens_valid':bool(encoder_mask.all())}
 features=np.fromfile(case/'hf-final.f32',dtype='<f4').reshape(-1,128);start=row['comparison_global_start'];end=row['comparison_global_end'];assert start==max(nf['start'],hf['start'])and end==min(nf['input_end'],hf['input_end'])
 assert difference(native[start:end],features[start-hf['start']:end-hf['start']],protocol['absolute_tolerance'])==row['final_common_mel_comparison']
 assert row['native_extra_computed_rows']==1 and metadata['raw_frames']==metadata['audio_samples']//160+1 and hf['emitted_end']==metadata['audio_samples']//160
for case,row in zip(protocol['cases'],result['results'][2:]):
 assert case['id']==row['case']and case['expected_final_stacks']==row['final_stack_inputs']
 root=cache/('quality qualification 125f907'if case['calibration']else'quality-corpus-v1');assert record(root/case['wav'])==case['wav_identity']and row['pcm']['sha256']==case['pcm_sha256']
assert not output.exists();output.mkdir()
# Keep all command outcomes and complete framing metadata, with raw tensor hashes.
for path in base.rglob('*'):
 if path.is_file()and path.suffix in ['.json','.log','.txt']:
  destination=output/path.relative_to(base);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
for suffix in ['sb','preflight.command.json','preflight.stdout.log','preflight.stderr.log','execution.command.json','execution.stdout.log','execution.stderr.log']:shutil.copy2(pathlib.Path(str(base)+'.'+suffix),output/('outer.'+suffix))
for name in ['run-tail-framing-control.py','collect-tail-framing-control.py']:shutil.copy2(cache/name,output/name)
(output/'tools').mkdir()
for name in protocol['tools']:shutil.copy2(tools/name,output/'tools'/name)
(output/'build').mkdir()
for name in ['build.json','build.stdout.log','build.stderr.log']:shutil.copy2(cache/'frontend-tail-probe-v1'/name,output/'build'/name)
qualification={'schema_version':1,'source_commit':source,'fixed_tail_frontend_control_completed':True,'frontend_cases':5,'native_cpu_frontend_processes':5,'synthetic_structural_contracts_passed_before_real_wav_decoding':True,'preflight_grouping_tests':3,'all_raw_tensor_hashes_reverified':True,'all_five_framing_reports_replayed_without_frontend_execution':True,'all_ten_final_stacks_and_encoder_masks_recomputed':True,'final_common_mel_metrics_recomputed':5,'native_streaming_frontend_and_hf_processor_executed':True,'native_scheduler_executed':False,'native_scheduler_source_replayed':True,'neural_model_instantiation_or_inference':False,'learned_projection_or_cache_executed':False,'checkpoint_deserialization_performed':False,'original_gate_status_changed':False,'release_qualified':False,'raw_tensors_in_git':False,'limits':protocol['limitations']+['All final common Mel comparisons have zero values over the retained .001 absolute descriptive flag; this is not an all-utterance streaming feature or neural equivalence claim.','Calibration final stack has40native versus39HF real rows with oneHFzero slot; each120s final stack has49native versus48HF real rows, yielding7native versus6HFencoder tokens.','The short synthetic silence has2source-derived native calls versus1actualHFprocessor call; complete domains and global coordinates are retained separately.']}
(output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n');files={p.relative_to(output).as_posix():record(p)for p in output.rglob('*')if p.is_file()};(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'receipt_files':len(files),'local_raw_files':len(result['files']),'framing_reports_recomputed':5,'final_stacks':10,'neural_inference':False}))
