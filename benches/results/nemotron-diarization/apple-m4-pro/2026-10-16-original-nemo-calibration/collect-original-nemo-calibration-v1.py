import hashlib,json,pathlib,shutil,sys
import numpy as np
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');work=cache/'original-nemo-calibration-runtime-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-original-nemo-calibration';sys.path.insert(0,str(tools))
from original_nemo_metrics import compare_original
from frontend_metrics import difference
def desc(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
p=json.loads((work/'protocol.json').read_text());q=json.loads((work/'qualification.json').read_text());assert desc(work/'protocol.json')==q['protocol']and (tools/'original-nemo-calibration.protocol.json').read_bytes()==(work/'protocol.json').read_bytes()
assert q['fixed_calibration_execution_passed']is True and q['returncode']==0 and q['release_qualified']is False and q['quality_qualified']is False and q['numerical_agreement_promoted_to_quality_gate']is False
assert q['source_commit']=='af4cfdbe75d0c031115076a65a081fd2f6d9631f'
for name,expected in p['tools'].items():assert desc(tools/name)==expected
for name,expected in p['inputs'].items():assert desc(pathlib.Path(name))==expected
for name,expected in p['environment_files'].items():assert desc(pathlib.Path(p['environment_root'])/name)==expected
for name,expected in p['installed_nemo_files'].items():assert desc(pathlib.Path(p['environment_root'])/'venv/lib/python3.12/site-packages'/name)==expected
forward=json.loads((work/'forward-completed.json').read_text());restore=json.loads((work/'restore-completed.json').read_text());assert forward['calls']==1 and forward['shapes']=={'probabilities':[1,111,8],'logits':[1,111,8],'activity':[1,111,3]}and forward['frontend_valid_lengths']==[111]and forward['frontend_physical_shape']==[1,128,112]
assert restore['strict']is True and restore['missing_keys']==[]and restore['unexpected_keys']==[]and restore['parameter_tensors']==361 and restore['async_streaming']is False
parameters=json.loads((work/'restored-parameters.json').read_text());assert len(parameters)==361 and all(v['dtype']=='torch.float32'for v in parameters.values())
actual=np.fromfile(work/'probabilities.f32',dtype='<f4').reshape(111,8);ref=np.load(p['reference_path'],allow_pickle=False);assert compare_original(actual,ref,17872)==q['full_valid_probability_comparison']
mel=np.fromfile(work/'frontend-valid.f32',dtype='<f4').reshape(111,128);physical=np.fromfile(work/'frontend-physical.f32',dtype='<f4').reshape(1,128,112);assert np.array_equal(mel,physical[0,:,:111].T)and (work/'frontend-capture-1.f32').read_bytes()==(work/'frontend-physical.f32').read_bytes()
for key,path,shape,rows in [('first_chunk_104_valid_rows',p['hf_first_mel_path'],(105,128),slice(0,104)),('final_chunk_39_valid_rows',p['hf_final_mel_path'],(39,128),slice(72,None))]:
 hf=np.fromfile(path,dtype='<f4').reshape(shape);assert difference(mel[rows],hf[:104]if key.startswith('first')else hf,1e-3)==q['frontend_comparisons'][key]
for name,channels in [('logits',8),('activity',3)]:assert np.isfinite(np.fromfile(work/(name+'.f32'),dtype='<f4').reshape(111,channels)).all()
assert q['storage']['released']is True and q['storage']['reserve_reported_allocated_bytes']==256*1024**2 and not(work/'diagnostic-reserve.bin').exists()
assert all(str(path)in(work/'runtime.sb').read_text()for path in p['denied_paths'])and '(deny network*)'in(work/'runtime.sb').read_text()
assert not out.exists();out.mkdir()
for path in work.iterdir():
 if path.is_file():shutil.copy2(path,out/path.name)
for name in p['tools']:shutil.copy2(tools/name,out/name)
for key in ['reference_path','reference_metadata_path','hf_first_mel_path','hf_final_mel_path']:shutil.copy2(p[key],out/('saved-'+pathlib.Path(p[key]).name))
for name in ['collect-original-nemo-calibration-v1.py','cleanup-stale-owned-incremental-original-nemo-v1.py','owned-incremental-original-nemo-cleanup-v1.json','cleanup-stale-rust-objects-pnola-original-nemo-v1.py','compiler-object-cleanup-pnola-original-nemo-v1.json','compiler-object-cleanup-pnola-original-nemo-v1.events.jsonl']:shutil.copy2(cache/name,out/name)
for prefix in ['compiler-object-cleanup-pnola-original-nemo-v1']:
 cleanup=json.loads((cache/(prefix+'.json')).read_text());events=[json.loads(s)for s in(cache/(prefix+'.events.jsonl')).read_text().splitlines()];assert len(events)==cleanup['removed_files']==508 and sum(e['bytes']for e in events)==cleanup['removed_apparent_bytes']==183464184 and all(not pathlib.Path(e['path']).exists()for e in events)
r={'schema_version':1,'source_commit':q['source_commit'],'strict_original_cpu_restore_and_one_calibration_forward_passed':True,'all_111_valid_probability_rows_compared':True,'maximum_probability_error':q['full_valid_probability_comparison']['maximum_absolute_error'],'mean_probability_error':q['full_valid_probability_comparison']['mean_absolute_error'],'values_above_1e_3':0,'frontend_equal':False,'original_corpus_parity_qualified':False,'quality_qualified':False,'release_qualified':False,'limits':['One excluded1.117s calibration; later official3.1 source, not identified training3.0 environment. No held-out or whole-corpus numerical-parity claim.','All111valid/eight learned channels compare without relabeling, row removal or new HF/native inference. Difference statistics are descriptive; original quality gates remain failed.','Original whole-wave frontend physical112/valid111 and HF chunked frontend differ; max Mel error0.0297584534. Do not infer all frontend features equal from small output differences.','Ad-hoc app CLI/GUI/physical microphone/clean performance/public signing/fresh-machine release scope is not expanded by this Python diagnostic.','Storage maintenance removed508old single-link Mach-O compiler objects in the owned Pnola target only; it preserved sources, models, evidence, SDKs and executables. Incremental cleanup found no eligible directories.']}
(out/'receiver-qualification.json').write_text(json.dumps(r,indent=2)+'\n');records={f.name:desc(f)for f in out.iterdir()if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':records},indent=2)+'\n');print(json.dumps({'members':len(records)+1,'max':r['maximum_probability_error'],'mean':r['mean_probability_error']}))
