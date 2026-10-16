import json,os,pathlib,sys,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from qualification_storage import StorageGuard,atomic_json
from qualify_valid_mel_pnola import execute_guarded
out=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0/original-nemo-env-v1');home=out/'import-home';temp=out/'import-tmp';home.mkdir();temp.mkdir()
policy=out/'import.sb';policy.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (subpath "/Users/victor/.cache/gen2-nemotron3-n0/research/native-f32-control") (subpath "/Users/victor/.cache/huggingface") (regex #".*\\\\.(nemo|gguf|safetensors|onnx|npy|f32)$"))\n(deny file-write* (subpath '+json.dumps(str(out/'venv'))+') (subpath '+json.dumps(str(out/'Speech-218d0b443441b7d8aa1c832f47df310f186cf0a4'))+'))\n')
env={'HOME':str(home),'TMPDIR':str(temp)+'/', 'PATH':'/usr/bin:/bin','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','WANDB_MODE':'disabled','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false'}
argv=['/usr/bin/sandbox-exec','-f',str(policy),str(out/'venv/bin/python'),str(out/'import-smoke.py')];result={'argv':argv,'environment':env,'model_execution_authorized':False};guard=StorageGuard(out,3*1024**3,1024**3,128*1024**2)
def persist():atomic_json(out/'import-command.json',result)
start=time.monotonic()
try:
 with (out/'import.stdout.log').open('xb')as so,(out/'import.stderr.log').open('xb')as se:
  execute_guarded(argv,cwd=out,environment=env,stdout=so,stderr=se,guard=guard,record=result,persist=persist,timeout=180)
 result['passed']=result['returncode']==0
finally:
 guard.release();result['elapsed_seconds']=time.monotonic()-start;result['storage']=guard.record;persist()
print(json.dumps(result))
