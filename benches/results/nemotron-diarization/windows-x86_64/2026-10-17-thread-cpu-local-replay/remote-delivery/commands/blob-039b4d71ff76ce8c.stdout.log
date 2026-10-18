"""Frozen source-only composition/materialization; no compiler, SDK or model invocation."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import shutil
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'windows-thread-cpu-profile-source-admission-v1';root.mkdir();evidence=root/'evidence';evidence.mkdir();source=root/'source'
def desc(path):
 raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p_path=tools/'windows-thread-cpu-profile.protocol.json';p_raw=p_path.read_bytes();p=json.loads(p_raw)
assert {'bytes':len(p_raw),'sha256':hashlib.sha256(p_raw).hexdigest()}=={'bytes':670257,'sha256':'e9809ffe025bbe127c27daa5f8dfa07916cab7fe24a40d074651103b8c3cfc66'}
fixed={str(tools/name):value for name,value in p['tools'].items()}|{str(repo/name):value for name,value in p['repository_files'].items()}
fixed[str(Path(__file__))]=desc(Path(__file__));fixed[str(p_path)]=desc(p_path)
fixed[str(tools/'remote_q8_apack_replay.py')]={'bytes': 22591, 'sha256': '6b7f4c6d3a9a672ad0a89c790fa0d51608adad4c3667395c2da1329c68e48b08'}
q={'completed':False,'scope':__doc__,'inputs':fixed,'commands':[],'compiled_sdk_or_model':False}
def persist():(root/'qualification.json').write_text(json.dumps(q,indent=2)+'\n')
def frozen():assert all(desc(Path(name))==value for name,value in fixed.items())
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1','PYTHONDONTWRITEBYTECODE':'1'}
def run(name,argv,cwd=repo):
 frozen()
 row={'name':name,'command':list(map(str,argv)),'completed':False}
 q['commands'].append(row);persist()
 try:
  row.update(bounded_process(row['command'],evidence/(name+'.stdout.log'),evidence/(name+'.stderr.log'),cwd=cwd,env=env,timeout=180,limit=8*1024**2))
 finally:persist()
 assert row['returncode']==0 and row['leader_reaped']and row['group_gone']and row['completed']
 frozen();print('PASS '+name,flush=True)
persist()
try:
 frozen();sys.path.insert(0,str(tools))
 from remote_q8_apack_replay import bounded_process
 from pooled_runtime import BOUNDED_PROFILE,load_profile,compatibility_patch_bytes
 from build_valid_mel_candidate import source_inventory
 from windows_compat import apply_windows_lf_compat
 from windows_worker_metrics import DIARIZER,BACKEND,transform
 from windows_thread_cpu_profile_sources import install_profile,CPU_SOURCE
 from windows_thread_cpu_profile_verify import verify_completed_source
 from windows_thread_cpu_profile_metrics import canonical_audit
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
