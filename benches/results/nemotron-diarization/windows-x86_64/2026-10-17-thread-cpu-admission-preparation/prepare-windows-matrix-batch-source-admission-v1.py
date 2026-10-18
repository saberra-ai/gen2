"""Frozen source-only composition/materialization; no compiler, SDK or model invocation."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import shutil
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'windows-matrix-batch-profile-source-admission-v1';root.mkdir();evidence=root/'evidence';evidence.mkdir();source=root/'source'
def desc(path):
 raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p_path=tools/'windows-matrix-batch-profile.protocol.json';p_raw=p_path.read_bytes();p=json.loads(p_raw)
fixed={str(tools/name):value for name,value in p['tools'].items()}|{str(repo/name):value for name,value in p['repository_files'].items()}
fixed[str(Path(__file__))]=desc(Path(__file__));fixed[str(p_path)]=desc(p_path)
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
 from pooled_runtime import BOUNDED_PROFILE,load_profile,compatibility_patch_bytes
 from build_valid_mel_candidate import source_inventory
 from windows_compat import apply_windows_lf_compat
 from windows_worker_metrics import DIARIZER,BACKEND,transform
 from windows_matrix_batch_profile_sources import install_profile,CPU_SOURCE
 from windows_matrix_batch_profile_verify import verify_completed_source
 from windows_matrix_batch_profile_metrics import canonical_audit
 from ggml_matrix_profile_source import PATH
 _,lock=load_profile(BOUNDED_PROFILE)
 run('clone-native',['git','clone','--no-hardlinks','--no-checkout',cache/'NeMo-Speech.cpp',source])
 run('checkout-native',['git','-C',source,'checkout','--detach',lock['runtime']['revision']])
 run('clone-llama',['git','clone','--no-hardlinks','--no-checkout',cache/'NeMo-Speech.cpp/llama.cpp',source/'llama.cpp'])
 run('checkout-llama',['git','-C',source/'llama.cpp','checkout','--detach',lock['runtime']['llama_revision']])
 for index,item in enumerate(lock['runtime']['compatibility_patches']):
  patch=tools/item['path'];run('check-patch-'+str(index),['git','apply','--check',patch],source)
  run('apply-patch-'+str(index),['git','apply',patch],source);shutil.copyfile(patch,evidence/patch.name)
 patch=evidence/'bounded-inference.patch';patch.write_bytes(compatibility_patch_bytes(lock))
 for name in (DIARIZER,BACKEND):
  path=source/name;path.write_bytes(transform(name,path.read_bytes(),4))
 install_profile(source,evidence)
 verify_completed_source(source,None,lock,patch,True,p['variants']['4']['source_sha256'])
 apply_windows_lf_compat(source,evidence);build=source/'build/gen2-cpu-diar'
 verify_completed_source(source,build,lock,patch,True,p['variants']['4']['source_sha256'])
 materialized=build/'_deps/llama.cpp'
 run('materialize-twenty-patches',['cmake','-DSOURCE_DIR='+str(source/'llama.cpp'),'-DPATCH_DIR='+str(source/'patches'),'-DDEST_DIR='+str(materialized),'-P',source/'cmake/llama_cpp.cmake'])
 expected=canonical_audit()['files'].copy();expected[PATH]=CPU_SOURCE;actual=source_inventory(materialized)
 assert actual==expected and len(actual)==3425
 verify_completed_source(source,build,lock,patch,True,p['variants']['4']['source_sha256']);frozen()
 q.update(completed=True,materialized_files=actual,changed_ggml_files=[PATH],profiled_cpu_source=desc(materialized/PATH),source_hashes=p['variants']['4']['source_sha256'])
except BaseException as error:q['error']=repr(error);raise
finally:persist()
print(json.dumps({'completed':q['completed'],'commands':len(q['commands']),'materialized_files':len(q.get('materialized_files',{})),'compiler_or_model':False}))
