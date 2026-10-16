"""Audit saved candidate construction; no compilation, test or model execution."""
import hashlib,json,pathlib,re,shutil,subprocess,sys
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');base=cache/'valid-mel-candidate-build-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-candidate-build'
def read(p):return json.loads(p.read_text())
def rec(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def run(argv,cwd=repo):return subprocess.check_output(argv,cwd=cwd)
r=read(base/'result.json');p=read(base/'protocol.json');source=r['source_commit'];assert source=='1bc12ce2291743a9d5b423db5fb1e00eeb2719ce'
assert run(['git','show',source+':tools/nemotron-diarization/valid-mel-candidate-build.protocol.json'])==(base/'protocol.json').read_bytes()
assert rec(base/'protocol.json')['sha256']==r['protocol_sha256']
for name,value in r['files'].items():assert rec(base/name)==value
for name,value in p['tools'].items():
 blob=run(['git','show',source+':tools/nemotron-diarization/'+name]);assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==value==rec(tools/name)
root=pathlib.Path(p['candidate_source_path']);prefix=pathlib.Path(p['candidate_prefix_path'])
for name,value in p['source_files'].items():assert rec(root/name)==value
assert rec(tools/'patches/v3-finish-valid-mel.patch')==p['patch']==r['patch']
assert run(['git','diff','--binary','--no-ext-diff','--ignore-submodules=all'],root)==(base/'candidate-source.patch').read_bytes()==(tools/'patches/v3-finish-valid-mel.patch').read_bytes()
# Diff-payload blank context is required syntax; the applied source has no whitespace errors.
assert not run(['git','diff','--check'],root)
assert run(['git','rev-parse','HEAD'],root).decode().strip()==p['base_revision']
assert run(['git','rev-parse','HEAD'],root/'llama.cpp').decode().strip()==p['llama_revision']
assert not run(['git','status','--porcelain','--untracked-files=no'],root/'llama.cpp')
assert rec(tools/'runtime.lock.json')==p['original_runtime_lock']
assert rec(cache/'run-valid-mel-candidate-build.py')==p['external_launcher']
assert rec(base/'source-preparation.json')==p['source_preparation']==rec(cache/'valid-mel-source-v1/prepared.json')
assert rec(root/'src/asr/diar/diar_pipeline.cpp')==r['patched_file']
outer=read(pathlib.Path(str(base)+'.command.json'));assert outer['returncode']==0 and outer['source_commit']==source
assert outer['argv'][1]==str(tools/'build_valid_mel_candidate.py')
assert len(r['commands'])==24
for c in r['commands']:
 name=c['phase'];assert c==read(base/(name+'.completed.json')) and c['returncode']==0
 assert c['argv'][:3]==['/usr/bin/sandbox-exec','-f',str(base/'offline.sb')]
 started=read(base/(name+'.started.json'));assert started=={k:v for k,v in c.items()if k not in ['returncode','elapsed_seconds']}
log=(base/'model-free-state-test.stdout.log').read_text();assert re.search(r'1/1 Test #\d+: diar_state\s+.*Passed',log)and'100% tests passed, 0 tests failed out of 1'in log
assert (base/'offline.sb').read_text()=='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(x)+')'for x in p['denied_paths'])+')\n'
sys.path.insert(0,str(tools));from build_valid_mel_candidate import source_inventory
a=read(base/'llama-materialized-audit.json');assert a['exact_independent_replay_match']is True and a['base_revision']==p['llama_revision']
assert source_inventory(pathlib.Path(a['compiled_tree_path']))==a['files']
assert rec(base/'pristine-llama.tar')==a['archive']
series=[line.split('#',1)[0].strip()for line in (root/'patches/series').read_text().splitlines()];series=[x for x in series if x]
assert len(series)==19 and a['patch_series']==[{'name':name,**rec(root/'patches'/name)}for name in series]
assert len(r['libraries'])==15
for name,value in r['libraries'].items():
 path=prefix/'lib'/name;assert rec(path)=={k:value[k]for k in ['bytes','sha256']}
 assert (str(path.readlink())if path.is_symlink()else None)==value['symlink_target']
 assert run(['/usr/bin/otool','-L',str(path)]).decode()==value['dependencies']
arc=r['runtime_archive'];assert rec(pathlib.Path(arc['path']))=={k:arc[k]for k in ['bytes','sha256']}
for name in ['CMakeCache.txt','compile_commands.json']:assert (base/name).read_bytes()==(root/'build/cpu-diar'/name).read_bytes()
compile_db=read(base/'compile_commands.json');assert any(x['file']==str(root/'src/asr/diar/diar_pipeline.cpp')for x in compile_db)
assert any(x['file'].startswith(a['compiled_tree_path']+'/ggml/')for x in compile_db)
for flag in ['neural_model_instantiation_or_inference','candidate_finish_executed','artifact_admitted_for_inference','production_runtime_changed','release_qualified']:assert r[flag]is False
assert r['model_free_diar_state_test_completed']is True
assert not out.exists();out.mkdir(parents=True)
# Large archive/SDK remain external and hash-bound. All small build records/logs are retained.
for path in base.iterdir():
 if path.is_file()and path.name!='pristine-llama.tar':shutil.copy2(path,out/path.name)
for suffix in ['command.json','stdout.log','stderr.log']:shutil.copy2(pathlib.Path(str(base)+'.'+suffix),out/('outer.'+suffix))
shutil.copytree(cache/'valid-mel-source-v1',out/'preparation')
for name in ['prepare-valid-mel-source.py','run-valid-mel-candidate-build.py','collect-valid-mel-candidate-build.py']:shutil.copy2(cache/name,out/name)
q={'schema_version':1,'source_commit':source,'candidate_cpu_build_completed':True,'model_free_diar_state_test_completed':True,'materialized_ggml_input_files':len(a['files']),'independent_patch_replay_matched_before_compile_and_after_build':True,'sdk_library_paths':15,'runtime_static_archive_bound':True,'neural_model_instantiation_or_inference':False,'candidate_finish_executed':False,'artifact_admitted_for_inference':False,'production_runtime_changed':False,'release_qualified':False,'large_archive_sdk_models_audio_not_in_git':True,'limits':p['limitations']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={x.relative_to(out).as_posix():rec(x)for x in out.rglob('*')if x.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'files':len(files),'materialized_source_files':len(a['files']),'sdk_paths':15,'neural_calls':0}))
