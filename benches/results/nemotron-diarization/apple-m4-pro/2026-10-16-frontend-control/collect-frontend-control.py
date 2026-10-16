"""Rehash/recompute retained frontend tensors; publish source-bound metadata only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
base=cache/'frontend-diagnostic-ec88564'
output=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-frontend-control'
read=lambda p:json.loads(p.read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=lambda p:{'bytes':p.stat().st_size,'sha256':digest(p)}
result=read(base/'result.json');protocol=read(base/'protocol.json');command=read(Path(str(base)+'.command.json'))
source='ec88564ec7f7b62e92b73156481630413b9928a9'
assert result['source_commit']==command['source_commit']==source and command['returncode']==0
assert all(result[name] is False for name in ['release_qualified','original_gate_status_changed','neural_model_inference_performed','checkpoint_deserialization_performed'])
assert result['protocol_sha256']==digest(base/'protocol.json')
for name,expected in protocol['tools'].items():
 data=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)
 assert hashlib.sha256(data).hexdigest()==expected==digest(repo/'tools/nemotron-diarization'/name)
for name,expected in result['files'].items():assert record(base/name)==expected,name
assert {p.relative_to(base).as_posix()for p in base.rglob('*')if p.is_file()}==set(result['files'])|{'result.json'}
module_path=repo/'tools/nemotron-diarization/frontend_metrics.py'
spec=importlib.util.spec_from_file_location('frozen_frontend_metrics',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
banks={name:np.fromfile(base/(name+'-bank.f32'),dtype='<f4').reshape(128,257)for name in ['native','hf']}
assert module.difference(banks['native'],banks['hf'],0)==result['bank_comparison']
recomputed=[]
for row in result['results']:
 case=base/row['case'];audio=np.fromfile(case/'audio.f32',dtype='<f4');outputs={}
 assert record(case/'audio.f32')==row['pcm']
 if 'pcm_sha256'in row['origin']:assert digest(case/'audio.f32')==row['origin']['pcm_sha256']
 for name in ['native','hf']:
  native=np.fromfile(case/('native-'+name+'.f32'),dtype='<f4').reshape(-1,128)
  hf=np.fromfile(case/('hf-'+name+'.f32'),dtype='<f4').reshape(-1,128)
  mask=np.fromfile(case/('hf-'+name+'.mask.u8'),dtype=np.uint8)
  n,h,valid=module.valid_prefix(hf,mask,native)
  assert valid==len(audio)//160 and len(mask)==valid+1 and row['masks'][name]=={'valid':valid,'raw':len(mask),'masked':1}
  for suffix in ['stdout.log','stderr.log','command.json']:assert (case/('native-'+name+'.'+suffix)).is_file()
  native_command=read(case/('native-'+name+'.command.json'));assert native_command['returncode']==0
  assert native_command['environment']=={'PATH':'/usr/bin:/bin','LC_ALL':'C'}
  assert read(case/('native-'+name+'.stdout.log'))=={key:value for key,value in row['native_observations'][name].items()if key!='argv'}
  assert native_command['argv']==row['native_observations'][name]['argv']
  assert native_command['argv']==[str(cache/'frontend-control-build/frontend-probe'),str(case/'audio.f32'),str(base/(name+'-bank.f32')),str(case/('native-'+name+'.f32'))]
  outputs['native-'+name],outputs['hf-'+name]=n,h
 comparisons={label:module.difference(outputs[left],outputs[right],protocol['absolute_tolerance'])for label,left,right in [('own_banks','native-native','hf-hf'),('same_native_bank','native-native','hf-native'),('same_hf_bank','native-hf','hf-hf')]}
 assert comparisons==row['comparisons']
 recomputed.append({'case':row['case'],'comparisons_recomputed':True})
 if row['origin'].get('synthetic'):
  for label in ['same_native_bank','same_hf_bank']:
   assert comparisons[label]['maximum_absolute_error']<=protocol['absolute_tolerance'] and comparisons[label]['mean_absolute_error']<=protocol['mean_absolute_tolerance']
assert [row['case']for row in result['results']]==['silence','fixed_tones_and_impulses','ami-ES2004a-Mix-Headset','ami-ES2004a-Array1-01']
assert record(cache/'frontend-control-build/frontend-probe')==protocol['probe']
for name,expected in protocol['sdk_libraries'].items():assert record(cache/'sdk-cpu/lib'/name)==expected
profile=cache/'frontend-diagnostic-ec88564.sb';assert digest(profile)==command['sandbox_sha256']
assert '(deny network*)'in profile.read_text() and str(cache/'assets/model.safetensors')in profile.read_text()
output.mkdir(exist_ok=False)
for path in base.rglob('*'):
 if path.is_file()and path.suffix in ['.json','.log']:
  target=output/path.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
(output/'tools').mkdir()
for name in protocol['tools']:shutil.copy2(repo/'tools/nemotron-diarization'/name,output/'tools'/name)
for suffix in ['command.json','stdout.log','stderr.log']:shutil.copy2(Path(str(base)+'.'+suffix),output/('execution.'+suffix))
shutil.copy2(profile,output/'offline-no-weights.sb')
shutil.copy2(cache/'run-frontend-control.py',output/'run-frontend-control.py')
shutil.copy2(__file__,output/'collect-frontend-control.py')
probe=cache/'frontend-control-build/frontend-probe'
for flag,name in [('-l','probe-load-commands.txt'),('-L','probe-linked-libraries.txt')]:
 (output/name).write_bytes(subprocess.check_output(['/usr/bin/otool',flag,str(probe)]))
qualification={'schema_version':1,'source_commit':source,'release_qualified':False,'original_gate_status_changed':False,
 'neural_model_inference_performed':False,'checkpoint_deserialization_performed':False,'frontend_execution_passed':True,
 'all_synthetic_same_bank_prerequisites_passed':True,'all_four_cases_twelve_comparisons_recomputed':True,
 'all_raw_features_masks_banks_rehashed':True,'native_probe_bound_to_exact_frozen_sdk_path':True,
 'raw_tensor_files_retained_outside_git':sum(p.suffix in ['.f32','.u8']for p in base.rglob('*')if p.is_file()),
 'native_probe_commands_reparsed':8,'valid_frames_per_meeting':12000,'extra_native_raw_terminal_rows_per_case':1,
 'same_bank_close_mic_values_above_fixed_absolute_tolerance':1,
 'same_bank_close_mic_worst_absolute_error':result['results'][2]['comparisons']['same_hf_bank']['maximum_absolute_error'],
 'own_bank_difference_measured_not_selected_as_a_model_fix':True,
 'execution_returncode':0,'execution_elapsed_seconds':command['elapsed_seconds'],'comparisons_recomputed':recomputed,
 'limitations':protocol['limitations']+['Receiver rehashed all local raw arrays and recomputed recorded metrics without rerunning frontend computation. Repository receipt includes metadata/logs/source, not probe/SDK/model/PCM/feature/bank binary payloads.','Synthetic same-bank controls pass. Close-mic same-bank comparison retains one over-tolerance feature value under each bank; no claim all target values pass the fixed numeric flag or that frontend equivalence/quality gates are proven.']}
(output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
files={path.relative_to(output).as_posix():record(path)for path in output.rglob('*')if path.is_file()}
(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'metadata_evidence_files':len(files),'raw_tensor_files_outside_git':qualification['raw_tensor_files_retained_outside_git'],'comparisons_recomputed':12,'close_same_bank_outlier_retained':True}))
