"""Recompute candidate finish receipt from saved arrays only; never run the helper."""
import hashlib,json,pathlib,shutil,subprocess,sys
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');base=cache/'valid-mel-finish-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-finish'
def read(p):return json.loads(p.read_text())
def rec(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
r=read(base/'result.json');p=read(base/'protocol.json');source=r['source_commit'];assert source=='0ec641aea9ec66a3549d2db1aafa2649febf09df'
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-finish.protocol.json'],cwd=repo)==(base/'protocol.json').read_bytes()
assert rec(base/'protocol.json')['sha256']==r['protocol_sha256']=='fc7d7900b04fc7bb41ffea593dfedc1f1bbea38af92e40eed602945e99667eb6'
for name,expected in r['files'].items():assert rec(base/name)==expected
for path,expected in p['bound_paths'].items():assert rec(pathlib.Path(path))==expected
for name,expected in p['tools'].items():
 blob=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo);assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==rec(tools/name)==expected
assert rec(cache/'run-valid-mel-finish.py')==p['external_launcher']
root=pathlib.Path(p['candidate_source_path']);assert subprocess.check_output(['git','diff','--binary','--no-ext-diff','--ignore-submodules=all'],cwd=root)==(tools/'patches/v3-finish-valid-mel.patch').read_bytes()
a=read(pathlib.Path(p['materialized_audit_path']))
for name,expected in p['materialized_compiler_headers'].items():assert a['files'][name]==expected==rec(pathlib.Path(a['compiled_tree_path'])/name)
build=read(pathlib.Path(p['probe_build_manifest_path']));assert read(base/'helper-build.json')==build and build['returncode']==0 and build['probe_executed']is False
assert build['source']==p['tools']['valid_mel_finish_probe.cpp']and build['binary']==rec(pathlib.Path(p['probe_path']))and build['argv']==p['probe_build_command']
assert (base/'helper-load-commands.txt').read_bytes()==subprocess.check_output(['/usr/bin/otool','-l',p['probe_path']])
command=read(base/'candidate.command.json');assert command=={'argv':[p['probe_path'],p['model_path'],p['audio_path'],p['mel_path'],str(base/'probe')],'environment':{'PATH':'/usr/bin:/bin','LC_ALL':'C'},'returncode':0}
outer=read(pathlib.Path(str(base)+'.command.json'));assert outer['returncode']==0 and outer['source_commit']==source and outer['argv']==['/usr/bin/sandbox-exec','-f',str(base)+'.sb',str(cache/'evidence-tools-env/bin/python'),str(tools/'diagnose_valid_mel_finish.py')]
assert pathlib.Path(str(base)+'.sb').read_text()=='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(x)+')'for x in p['denied_paths'])+')\n'
sys.path.insert(0,str(tools));from valid_mel_finish_metrics import audit_outputs
replay=audit_outputs(base/'probe');assert all(r[k]==value for k,value in replay.items());assert len(r['sessions'])==28 and all(m['bitwise_equal']and m['maximum_absolute_error']==0 for m in r['probability_comparisons'].values())
import numpy as np
mel=np.fromfile(p['mel_path'],dtype='<f4').reshape(112,128)[:111]
for left in [0,1]:
 for stage,rows in [('first',mel[:104]),('final',mel[72-left*8:])]:assert (base/'probe'/('direct-left'+str(left)+'-'+stage+'.mel.f32')).read_bytes()==rows.tobytes()
expected={'fixed_candidate_finish_regression_completed':True,'stream_sessions':28,'direct_graph_calls':4,'boundary_cases':21,'tolerance':0.0001}
assert read(base/'candidate.stdout.log')==expected
for flag in ['candidate_finish_executed','neural_model_instantiated','stream_frontend_and_aosc_executed']:assert r[flag]is True
for flag in ['hf_or_original_nemo_executed','target_ami_executed','der_scoring_performed','production_runtime_changed','original_gate_status_changed','release_qualified']:assert r[flag]is False
assert not out.exists();shutil.copytree(base,out)
for suffix in ['sb','command.json','stdout.log','stderr.log']:shutil.copy2(pathlib.Path(str(base)+'.'+suffix),out/('outer.'+suffix))
for name in ['run-valid-mel-finish.py','build-valid-mel-finish-probe.py','collect-valid-mel-finish.py']:shutil.copy2(cache/name,out/name)
shutil.copytree(cache/'valid-mel-finish-probe-v1',out/'helper-construction',ignore=shutil.ignore_patterns('valid-mel-finish-probe'))
q={'schema_version':1,'source_commit':source,**expected,'candidate_cpu_finish_fixed_cases_qualified':True,'all_six_direct_probability_comparisons_bitwise_equal':True,'reset_replay_bitwise_equal':True,'all_28_final_frame_counts_and_stable_prefixes_verified':True,'actual_provisional_output_verified':True,'candidate_f32_model_only':True,'sdk_library_paths':15,'materialized_compiler_headers':len(p['materialized_compiler_headers']),'candidate_core_frame_rule':'floor(real_input_samples/160)','neural_model_instantiated':True,'stream_frontend_and_aosc_executed':True,'hf_or_original_nemo_executed':False,'target_ami_executed':False,'der_scoring_performed':False,'production_runtime_changed':False,'original_gate_status_changed':False,'release_qualified':False,'raw_calibration_and_silence_tensors_retained_in_git':True,'weights_audio_sdk_and_helper_binary_not_in_git':True,'limits':p['limitations']+['This result qualifies only fixed candidate CPU F32-model cases. It does not admit a silent substitution into Rust, whose current finished-count validator still binds the historicalfloor+1 contract; productionmigration must explicitly select andpin the newruntime contract. Q8model, otherplatforms/backends, long streams and unchangedquality gates require separatequalification.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={x.relative_to(out).as_posix():rec(x)for x in out.rglob('*')if x.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'files':len(files),'stream_sessions':28,'direct_graph_calls':4,'exact_probability_comparisons':6}))
