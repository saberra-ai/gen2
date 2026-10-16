"""Run one committed two-stream continuation, preserving atomic status records."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
protocol=tools/'valid-mel-metal-long-v2.protocol.json';p=json.loads(protocol.read_text());out=Path(p['output_path'])
def rec(path):
 data=path.read_bytes();return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
assert not out.exists()and not list(out.parent.glob(out.name+'.*')), 'preserve previous guarded attempt'
assert rec(Path(__file__))==p['external_launcher']
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-metal-long-v2.protocol.json'],cwd=repo)==protocol.read_bytes()
for records,root,prefix in [(p['tools'],tools,'tools/nemotron-diarization/'),(p['rust_sources'],repo,'')]:
 for name,expected in records.items():
  data=subprocess.check_output(['git','show',source+':'+prefix+name],cwd=repo)
  assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==rec(root/name)==expected
for name,expected in p['inputs'].items():assert rec(Path(name))==expected
sys.path.insert(0,str(tools))
from qualification_storage import atomic_json
profile=Path(str(out)+'.preflights.sb');profile.write_text('(version 1)\n(allow default)\n(deny network*)\n')
env={'PATH':'/usr/bin:/bin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}
steps=[(case['name'],['-m','unittest','discover','-s',str(tools),'-p',case['pattern']])for case in p['preflights']]+[('qualification',[str(tools/'qualify_valid_mel_metal_long_v2.py')])]
for name,args in steps:
 prefix=[str(cache/'evidence-tools-env/bin/python')]if name=='qualification'else['/usr/bin/sandbox-exec','-f',str(profile),str(cache/'evidence-tools-env/bin/python')]
 argv=[*prefix,*args];record={'source_commit':source,'argv':argv,'environment':env};command=Path(str(out)+'.'+name+'.command.json')
 atomic_json(command,record);started=time.monotonic()
 try:
  with Path(str(out)+'.'+name+'.stdout.log').open('xb')as stdout,Path(str(out)+'.'+name+'.stderr.log').open('xb')as stderr:
   result=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
  record['returncode']=result.returncode
 except BaseException as error:
  record['startup_or_wait_error']=str(error)
  raise
 finally:
  record['elapsed_seconds']=time.monotonic()-started
  atomic_json(command,record)
 print(json.dumps({'step':name,'returncode':result.returncode,'elapsed_seconds':record['elapsed_seconds']}),flush=True)
 result.check_returncode()
