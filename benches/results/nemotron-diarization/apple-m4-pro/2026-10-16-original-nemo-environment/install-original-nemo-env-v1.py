import hashlib,json,os,pathlib,shutil,subprocess,sys,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from qualification_storage import StorageGuard,atomic_json
from qualify_valid_mel_pnola import execute_guarded
out=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0/original-nemo-env-v1');envpath=out/'venv';assert not envpath.exists();commands=[]
def record(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
inputs={str(p):record(p)for p in [out/'requirements-wheel.lock',out/'wheels/nemo_toolkit-3.1.0-py3-none-any.whl',pathlib.Path(__file__)]}
result={'phase':'model-free environment construction','model_loaded':False,'model_executed':False,'inputs':inputs,'commands':commands}
def persist():atomic_json(out/'installation.json',result)
guard=StorageGuard(out,3*1024**3,1024**3,128*1024**2)
try:
 for name,argv in [('venv',['/opt/homebrew/bin/uv','venv','--python','/Users/victor/.cache/gen2-nemotron3-n0/python-reference/bin/python',str(envpath)]),('sync',['/opt/homebrew/bin/uv','pip','sync','--python',str(envpath/'bin/python'),'--link-mode','clone','--require-hashes',str(out/'requirements-wheel.lock')]),('freeze',['/opt/homebrew/bin/uv','pip','freeze','--python',str(envpath/'bin/python')]),('check',['/opt/homebrew/bin/uv','pip','check','--python',str(envpath/'bin/python')])]:
  guard.before_model();item={'name':name,'argv':argv};commands.append(item);start=time.monotonic()
  with (out/(name+'.stdout.log')).open('xb')as so,(out/(name+'.stderr.log')).open('xb')as se:
   execute_guarded(argv,cwd=out,environment=os.environ.copy(),stdout=so,stderr=se,guard=guard,record=item,persist=persist,timeout=600)
  item['elapsed_seconds']=time.monotonic()-start;persist();assert item['returncode']==0,name
 result['passed']=True
except BaseException as e:
 result.update(passed=False,error=repr(e));raise
finally:
 guard.release();result['storage']=guard.record;persist()
print(json.dumps({'passed':result['passed'],'commands':len(commands),'storage':guard.record}))
