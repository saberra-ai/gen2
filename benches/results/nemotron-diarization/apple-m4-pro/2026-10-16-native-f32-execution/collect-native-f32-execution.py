"""Recompute fixed native F32/Q8/HF comparisons and bind retained model calls."""
import hashlib,importlib.util,json,shutil,subprocess
from pathlib import Path
import numpy as np
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');base=cache/'native-f32-execution-v1';old=cache/'quality qualification 125f907';corpus=cache/'quality-corpus-v1'
read=lambda p:json.loads(p.read_text());record=lambda p:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
source='03bc8a09a7fadd3d6ecfef3b2408d0d089a9e8bc';result=read(base/'result.json');protocol=read(base/'protocol.json');outer=read(Path(str(base)+'.command.json'))
assert result['source_commit']==outer['source_commit']==source and outer['returncode']==0
assert result['native_f32_neural_inference_performed']is True and result['original_gate_status_changed']is False and result['release_qualified']is False
assert record(base/'protocol.json')['sha256']==result['protocol_sha256']
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/native-f32-execution.protocol.json'],cwd=repo)==(base/'protocol.json').read_bytes()
for name,expected in protocol['tools'].items():
 assert record(repo/'tools/nemotron-diarization'/name)['sha256']==expected
 assert hashlib.sha256(subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)).hexdigest()==expected
for name,expected in protocol['original_inputs'].items():assert record(old/name)==expected
construction=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-native-f32-construction'
for name,expected in protocol['construction_files'].items():assert record(construction/name)==expected
assert record(cache/'native-f32-conversion-v1/Nemotron-3-Diarization.f32-control.gguf')==protocol['model']
assert record(old/'native-cpu')==protocol['helper']
sdk=Path(protocol['linked_sdk_path']);assert record(sdk/'package.json')==protocol['package_manifest']
for name,expected in protocol['sdk_libraries'].items():assert record(sdk/'lib'/name)==expected
# Frozen numerical module imports only the equally pinned metric helper.
import sys
sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from native_f32_metrics import comparisons
assert len(result['results'])==3
for case,row in zip(protocol['cases'],result['results']):
 assert case['id']==row['case'] and case['calibration']==row['calibration']
 root=old if case['calibration']else corpus;wav=root/case['wav'];assert record(wav)==case['wav_identity']
 directory=base/case['id'];metadata=read(directory/'native.json')
 assert record(directory/'native.json')==row['native_output'] and record(directory/'native.json.f32')==row['native_probabilities']
 assert metadata['audio_samples']==case['samples']and metadata['input_pcm_f32le_sha256']==case['pcm_sha256']and metadata['input_repeats']==1
 assert metadata['geometry']==protocol['geometry']and metadata['requested_gpu_index']==-1
 assert metadata['native_frame_count']==metadata['retained_rows']==case['samples']//160+1 and metadata['retained_first_frame']==0
 command=read(directory/'native.command.json');assert command['returncode']==0 and command['environment']=={'PATH':'/usr/bin:/bin'}
 assert command['argv']==['/usr/bin/sandbox-exec','-f',str(base/'offline.sb'),str(old/'native-cpu'),str(cache/'native-f32-conversion-v1/Nemotron-3-Diarization.f32-control.gguf'),str(wav),'1','-1',str(directory/'native.json')]
 assert (directory/'native.stdout.log').is_file()and(directory/'native.stderr.log').is_file()
 f32=np.fromfile(directory/'native.json.f32',dtype='<f4').reshape(-1,8);q8=np.fromfile(old/case['q8_probabilities'],dtype='<f4').reshape(-1,8);hf=np.load(old/case['hf_probabilities'],allow_pickle=False)
 assert comparisons(f32,q8,hf,case['samples'])==row['comparison']==read(directory/'comparison.json')
profile=base/'offline.sb';expected='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(path)+')'for path in protocol['denied_paths'])+')\n';assert profile.read_text()==expected
load_commands=subprocess.check_output(['/usr/bin/otool','-l',str(old/'native-cpu')],text=True);assert (base/'helper-load-commands.txt').read_text()==load_commands
lines=load_commands.splitlines();rpaths=[lines[i+2].strip().removeprefix('path ').split(' (offset ')[0]for i,line in enumerate(lines)if line.strip()=='cmd LC_RPATH'];assert rpaths==[str(sdk/'lib')]
output=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-native-f32-execution';output.mkdir(exist_ok=False)
shutil.copytree(base,output,dirs_exist_ok=True)
(output/'tools').mkdir()
for name in protocol['tools']:shutil.copy2(repo/'tools/nemotron-diarization'/name,output/'tools'/name)
for suffix in ['command.json','stdout.log','stderr.log']:shutil.copy2(Path(str(base)+'.'+suffix),output/('execution.'+suffix))
for name in ['run-native-f32-execution.py','collect-native-f32-execution.py']:shutil.copy2(cache/name,output/name)
qualification={'schema_version':1,'source_commit':source,'fixed_native_f32_execution_completed':True,'new_native_cpu_model_calls':3,'excluded_partial_tail_calibration_first':True,'all_actual_input_config_frame_range_contracts_reparsed':True,'all_three_probability_comparisons_recomputed_per_case':True,'all_original_q8_hf_inputs_unchanged':True,'f32_artifact_and_93_library_paths_rehashed':True,'native_helper_exact_original_cpu_package_rpath':True,'native_model_artifact':protocol['model'],'native_f32_neural_inference_performed':True,'hf_or_original_nemo_model_rerun':False,'der_scoring_performed':False,'original_gate_status_changed':False,'release_qualified':False,'both_target_raw_frame_counts':12001,'both_target_hf_common_valid_rows':12000,'calibration_native_raw_rows':112,'calibration_hf_common_valid_rows':111,'native_terminal_vectors_preserved':True,'numeric_equivalence_claimed':False,'outer_execution_seconds':outer['elapsed_seconds'],'limits':protocol['limitations']+['Validity/shape/PCM calibration passes, but nativeF32/HF calibration maximum difference is0.240976810455; no numerical-equality calibration pass is claimed.','Native F32/HF target maximum differences remain0.0444959998close and0.0059737670distant. Smaller means do not establish universal model equivalence or fix the original failed quality gate.','Receiver recomputed comparisons from retained new native vectors and original hash-locked local HF/Q8 arrays without rerunning inference. Timings are incidental on this host, not uncontended-performance qualification.']}
(output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
files={p.relative_to(output).as_posix():record(p)for p in output.rglob('*')if p.is_file()};(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'evidence_files':len(files),'native_model_calls':3,'probability_reports_recomputed':9,'native_f32_inference':True,'original_gate_changed':False}))
