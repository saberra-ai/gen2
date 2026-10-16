"""Execute committed fixed package protocol once after29model-free preflights."""
import hashlib,json,pathlib,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';protocol=tools/'valid-mel-macos-package-v2.protocol.json';p=json.loads(protocol.read_text());out=pathlib.Path(p['output_path'])
def rec(path):return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert not out.exists()and not list(out.parent.glob(out.name+'.*')),'previous attempt must be preserved'
assert rec(pathlib.Path(__file__))==p['external_launcher']
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-macos-package-v2.protocol.json'],cwd=repo)==protocol.read_bytes()
for records,root,prefix in [(p['tools'],tools,'tools/nemotron-diarization/'),(p['rust_sources'],repo,'')]:
 for name,expected in records.items():
  blob=subprocess.check_output(['git','show',source+':'+prefix+name],cwd=repo);assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==rec(root/name)==expected
for name,expected in p['inputs'].items():assert rec(pathlib.Path(name))==expected
# Preflights remain network-denied. The controller does file/array admission
# without an outer sandbox; every native child receives the strict profile once.
profile=pathlib.Path(str(out)+'.outer.sb');profile.write_text('(version 1)\n(allow default)\n(deny network*)\n')
env={'PATH':'/usr/bin:/bin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}
steps=[(x['name'],['-m','unittest','discover','-s',str(tools),'-p',x['pattern']])for x in p['preflights']]+[('qualification',[str(tools/'qualify_valid_mel_macos_package_v2.py')])]
for name,args in steps:
 prefix=[str(cache/'evidence-tools-env/bin/python')]if name=='qualification'else['/usr/bin/sandbox-exec','-f',str(profile),str(cache/'evidence-tools-env/bin/python')];argv=[*prefix,*args];record={'source_commit':source,'argv':argv,'environment':env};command=pathlib.Path(str(out)+'.'+name+'.command.json');command.write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic()
 with pathlib.Path(str(out)+'.'+name+'.stdout.log').open('xb')as stdout,pathlib.Path(str(out)+'.'+name+'.stderr.log').open('xb')as stderr:r=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
 record.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-start);command.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'step':name,'returncode':r.returncode,'elapsed_seconds':record['elapsed_seconds']}),flush=True);r.check_returncode()
