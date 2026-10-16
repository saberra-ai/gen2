"""Run committed fixed taggedCPU neural admission once, after15 model-free preflights."""
import hashlib,json,pathlib,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';protocol_path=tools/'tagged-valid-mel-qualification.protocol.json';p=json.loads(protocol_path.read_text());out=pathlib.Path(p['output_path'])
def rec(path):return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert not out.exists()and not list(out.parent.glob(out.name+'.*')),'refusing prior qualification attempt overwrite'
assert rec(pathlib.Path(__file__))==p['external_launcher']
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/tagged-valid-mel-qualification.protocol.json'],cwd=repo)==protocol_path.read_bytes()
for name,expected in p['tools'].items():
 blob=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)
 assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==rec(tools/name)==expected
for name,expected in p['rust_sources'].items():
 blob=subprocess.check_output(['git','show',source+':'+name],cwd=repo)
 assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==rec(repo/name)==expected
profile=pathlib.Path(str(out)+'.sb');profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(x)+')'for x in p['denied_paths'])+')\n')
env={'PATH':'/usr/bin:/bin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}
steps=[('comparison-preflight',['-m','unittest','discover','-s',str(tools),'-p','test_stream_comparison_valid_mel.py']),('profile-preflight',['-m','unittest','discover','-s',str(tools),'-p','test_runtime_profiles.py']),('qualification',[str(tools/'qualify_tagged_valid_mel.py')])]
for name,args in steps:
 argv=['/usr/bin/sandbox-exec','-f',str(profile),str(cache/'evidence-tools-env/bin/python'),*args];meta={'source_commit':source,'argv':argv,'environment':env};command=pathlib.Path(str(out)+'.'+name+'.command.json');command.write_text(json.dumps(meta,indent=2)+'\n');started=time.monotonic()
 with pathlib.Path(str(out)+'.'+name+'.stdout.log').open('xb')as stdout,pathlib.Path(str(out)+'.'+name+'.stderr.log').open('xb')as stderr:r=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
 meta.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-started);command.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'step':name,'returncode':r.returncode,'elapsed_seconds':meta['elapsed_seconds']}),flush=True);r.check_returncode()
