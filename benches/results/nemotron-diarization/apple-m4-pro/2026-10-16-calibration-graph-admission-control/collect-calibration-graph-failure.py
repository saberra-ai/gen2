"""Preserve the first graph-control admission failure, without executing a model."""
import hashlib,json,pathlib,shutil,subprocess
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');base=cache/'calibration-graph-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-calibration-graph-admission-control'
def rec(p):return{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(p.read_text())
command=read(pathlib.Path(str(base)+'.execution.command.json'));source=command['source_commit'];assert source.startswith('dc02371')and command['returncode']==1
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/calibration-graph.protocol.json'],cwd=repo)==(base/'protocol.json').read_bytes()
assert read(pathlib.Path(str(base)+'.preflight.command.json'))['returncode']==0
assert (base/'hf-load.started.json').is_file()and not(base/'hf-first.started.json').exists()and not(base/'hf-final.started.json').exists()
assert not(base/'result.json').exists()and not any((base/name).exists()for name in ['first104','valid39','zero40','extra40'])
stderr=pathlib.Path(str(base)+'.execution.stderr.log').read_text();assert 'TypeError: Object of type set is not JSON serializable'in stderr and "write_json('hf-loading-info.json', loading_info)"in stderr
protocol=read(base/'protocol.json')
for name,expected in protocol['tools'].items():assert rec(repo/'tools/nemotron-diarization'/name)==expected
assert not out.exists();out.mkdir()
for p in base.rglob('*'):
 if p.is_file()and p.suffix in ['.json','.txt','.log']:shutil.copy2(p,out/p.name)
for suffix in ['sb','preflight.command.json','preflight.stdout.log','preflight.stderr.log','execution.command.json','execution.stdout.log','execution.stderr.log']:shutil.copy2(pathlib.Path(str(base)+'.'+suffix),out/('outer.'+suffix))
(out/'tools').mkdir()
for name in protocol['tools']:shutil.copy2(repo/'tools/nemotron-diarization'/name,out/'tools'/name)
for version in ['v1','v2']:
 dest=out/('build-'+version);dest.mkdir()
 for name in ['build.json','build.stdout.log','build.stderr.log']:shutil.copy2(cache/('calibration-graph-probe-'+version)/name,dest/name)
for name in ['run-calibration-graph-control.py','collect-calibration-graph-failure.py','build-calibration-graph-probe.py','build-calibration-graph-probe-v2.py']:shutil.copy2(cache/name,out/name)
q={'schema_version':1,'source_commit':source,'control_completed':False,'failure_stage':'HF loading-info serialization after local model construction and before first forward','failure':'TypeError: Object of type set is not JSON serializable','hf_published_weights_loaded':True,'hf_forward_calls_completed_or_started':0,'native_graph_calls_started':0,'original_or_ami_target_inference':False,'preflight_tests_passed':3,'source_and_input_configuration_changed_after_failure':False,'raw_local_files':{p.relative_to(base).as_posix():rec(p)for p in base.rglob('*')if p.is_file()},'original_gate_status_changed':False,'release_qualified':False,'limits':['No graph probability result exists; no numerical-control pass or failure can be inferred.','Initial probev1 link failure and successful probev2 compile are separate build-only records.','A separate reviewed/preregistered revision is required for serialization repair and a fresh attempt; this prefix remains immutable.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={p.relative_to(out).as_posix():rec(p)for p in out.rglob('*')if p.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'files':len(files),'HF_forwards':0,'native_graph_calls':0,'failure_retained':True}))
