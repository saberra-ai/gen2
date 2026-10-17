"""Frozen source-only composition/materialization; no compiler, SDK or model invocation."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import shutil
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'windows-q8-candidate-source-admission-v7';root.mkdir();evidence=root/'evidence';evidence.mkdir();source=root/'source'
def desc(path):
 raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p_path=tools/'windows-q8-candidate.protocol.json';p_raw=p_path.read_bytes();p=json.loads(p_raw)
fixed={str(tools/name):value for name,value in p['tools'].items()}|{str(repo/name):value for name,value in p['repository_files'].items()}
fixed.update({str(path):desc(path) for path in tools.glob('windows_q8_candidate*.py')});fixed.update({str(tools/name):desc(tools/name) for name in ['patches/q8-model-mx4.patch','fixtures/q8-model-original-series.txt','test_windows_q8_candidate.py','windows-q8-candidate.recipe.json']});fixed[str(Path(__file__))]=desc(Path(__file__));fixed[str(p_path)]=desc(p_path)
q={'completed':False,'scope':__doc__,'inputs':fixed,'commands':[],'compiled_sdk_or_model':False}
def persist():(root/'qualification.json').write_text(json.dumps(q,indent=2)+'\n')
def frozen():assert all(desc(Path(name))==value for name,value in fixed.items())
def alive(pid):
 try:os.killpg(pid,0);return True
 except ProcessLookupError:return False
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}
def run(name,argv,cwd=repo):
 frozen();row={'name':name,'argv':list(map(str,argv)),'owned_process_group':True,'process_tree_reaped':False};q['commands'].append(row);persist()
 with(evidence/(name+'.stdout.log')).open('xb')as out,(evidence/(name+'.stderr.log')).open('xb')as err:
  child=subprocess.Popen(row['argv'],cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True)
  try:row['returncode']=child.wait(timeout=120)
  finally:
   if alive(child.pid):
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:pass
    if alive(child.pid):os.killpg(child.pid,signal.SIGKILL)
   child.wait(timeout=5)
   for _ in range(100):
    if not alive(child.pid):break
    time.sleep(.01)
   row['process_tree_reaped']=not alive(child.pid);persist()
  assert row['returncode']==0 and row['process_tree_reaped']
 print('PASS '+name,flush=True)
persist()
try:
 frozen();sys.path.insert(0,str(tools))
 from pathlib import PureWindowsPath
 from pooled_runtime import BOUNDED_PROFILE,load_profile,compatibility_patch_bytes
 from build_valid_mel_candidate import source_inventory
 from windows_compat import apply_windows_lf_compat
 from platform_bounded_sources import verify_completed_source as verify_original_bounded
 from windows_q8_candidate_sources import load_recipe,install_q8_patch
 from windows_q8_candidate_verify import verify_completed_source
 from windows_q8_candidate_metrics import bounded_audit,PATH
 _,lock=load_profile(BOUNDED_PROFILE);_,recipe=load_recipe()
 original=bounded_audit();out=evidence
 run('clone-native',['git','clone','--no-hardlinks','--no-checkout',cache/'NeMo-Speech.cpp',source])
 run('checkout-native',['git','-C',source,'checkout','--detach',lock['runtime']['revision']])
 run('clone-llama',['git','clone','--no-hardlinks','--no-checkout',cache/'NeMo-Speech.cpp/llama.cpp',source/'llama.cpp'])
 run('checkout-llama',['git','-C',source/'llama.cpp','checkout','--detach',lock['runtime']['llama_revision']])
 for index,item in enumerate(lock['runtime']['compatibility_patches']):
  patch=tools/item['path'];run('check-patch-'+str(index),['git','apply','--check',patch],source)
  run('apply-patch-'+str(index),['git','apply',patch],source);shutil.copyfile(patch,out/patch.name)
 patch=out/'bounded-inference.patch';patch.write_bytes(compatibility_patch_bytes(lock))
 verify_original_bounded(source,None,lock,patch,True)
 install_q8_patch(source,out,'mx4')
 verify_completed_source(source,None,lock,patch,True)
 apply_windows_lf_compat(source,out);build=source/'build/gen2-cpu-diar'
 verify_completed_source(source,build,lock,patch,True)
 materialized=build/'_deps/llama.cpp'
 run('materialize-twenty-patches',['cmake','-DSOURCE_DIR='+str(source/'llama.cpp'),'-DPATCH_DIR='+str(source/'patches'),'-DDEST_DIR='+str(materialized),'-P',source/'cmake/llama_cpp.cmake'])
 expected=original['files'].copy();expected[PATH]=recipe['ggml']['candidate_sgemm'];actual=source_inventory(materialized)
 assert actual==expected and len(actual)==3425
 old_root=PureWindowsPath(original['compiled_tree_path']).parents[3];compiler_inputs={}
 for name,identity in original['all_compiler_sources'].items():
  relative=PureWindowsPath(name).relative_to(old_root).as_posix()
  if relative=='build/gen2-cpu-diar/_deps/llama.cpp/'+PATH:identity=recipe['ggml']['candidate_sgemm']
  assert desc(source/relative)==identity,relative
  compiler_inputs[relative]=identity
 assert len(compiler_inputs)==97
 for name in ['src/asr/diar/diarizer.cpp','src/runtime/ggml/backend.cpp']:
  assert b'[DEBUG-worker-v1]'not in(source/name).read_bytes()
 verify_completed_source(source,build,lock,patch,True);frozen()
 q.update(completed=True,materialized_files=actual,changed_ggml_files=[name for name in actual if actual[name]!=original['files'][name]],
  native_source_hashes=recipe['source_sha256_after'],compiler_inputs=compiler_inputs,diagnostic_worker_telemetry=False,
  recipe=desc(tools/'windows-q8-candidate.recipe.json'))
except BaseException as error:q['error']=repr(error);raise
finally:persist()
print(json.dumps({'completed':q['completed'],'commands':len(q['commands']),'files':len(q.get('materialized_files',{})),'compiler_inputs':len(q.get('compiler_inputs',{})),'changed':q.get('changed_ggml_files'),'compiler_or_model':False}))
