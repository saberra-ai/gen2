"""Launch one committed and reviewed CPU candidate finish protocol, preserving all failures."""
import hashlib,json,os,pathlib,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';out=cache/'valid-mel-finish-v1';protocol_path=tools/'valid-mel-finish.protocol.json';p=json.loads(protocol_path.read_text())
def rec(path):return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert not out.exists()and not list(out.parent.glob(out.name+'.*')),'refusing prior attempt prefix overwrite'
assert rec(pathlib.Path(__file__))==p['external_launcher']
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-finish.protocol.json'],cwd=repo)==protocol_path.read_bytes()
for name,expected in p['tools'].items():
 blob=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)
 assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==rec(tools/name)==expected
profile=pathlib.Path(str(out)+'.sb');profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(x)+')'for x in p['denied_paths'])+')\n')
argv=['/usr/bin/sandbox-exec','-f',str(profile),str(cache/'evidence-tools-env/bin/python'),str(tools/'diagnose_valid_mel_finish.py')];env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';meta={'source_commit':source,'argv':argv,'environment_overrides':{'PYTHONDONTWRITEBYTECODE':'1'}};command=pathlib.Path(str(out)+'.command.json');command.write_text(json.dumps(meta,indent=2)+'\n');start=time.monotonic()
with pathlib.Path(str(out)+'.stdout.log').open('xb')as stdout,pathlib.Path(str(out)+'.stderr.log').open('xb')as stderr:
 r=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
meta.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-start);command.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta));r.check_returncode()
