"""Execute committed tagged construction admission once; zero neural work."""
import hashlib,json,pathlib,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';protocol_path=tools/'valid-mel-tagged-build.protocol.json';p=json.loads(protocol_path.read_text());out=pathlib.Path(p['evidence_path'])
def rec(path):return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for target in [out,pathlib.Path(p['candidate_source_path']),pathlib.Path(p['candidate_prefix_path'])]:assert not target.exists(),'refusing prior construction prefix overwrite'
assert not list(out.parent.glob(out.name+'.*')),'refusing prior outer evidence overwrite'
assert rec(pathlib.Path(__file__))==p['external_launcher']
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-tagged-build.protocol.json'],cwd=repo)==protocol_path.read_bytes()
for name,expected in p['tools'].items():
 blob=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)
 assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==rec(tools/name)==expected
for name,expected in p['receipts'].items():assert rec(repo/name)==expected
out.mkdir();(out/'protocol.json').write_bytes(protocol_path.read_bytes());profile=out/'offline.sb';profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(x)+')'for x in p['denied_paths'])+')\n')
env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1','GIT_LFS_SKIP_SMUDGE':'1'}
commands=[('prepare',[str(cache/'evidence-tools-env/bin/python'),str(tools/'prepare_valid_mel_source.py'),'--upstream',p['original_source_path'],'--destination',p['candidate_source_path'],'--output',str(out/'preparation')]),('build',[str(cache/'evidence-tools-env/bin/python'),str(tools/'build_valid_mel_sdk.py'),'--source',p['candidate_source_path'],'--prefix',p['candidate_prefix_path'],'--output',str(out/'build'),'--jobs','6','--sandbox-profile',str(profile)])]
for name,argv in commands:
 actual=['/usr/bin/sandbox-exec','-f',str(profile),*argv] if name=='prepare' else argv;record={'source_commit':source,'argv':actual,'environment':env};command=out/(name+'.command.json');command.write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic()
 with (out/(name+'.stdout.log')).open('xb')as stdout,(out/(name+'.stderr.log')).open('xb')as stderr:r=subprocess.run(actual,cwd=repo,env=env,stdout=stdout,stderr=stderr)
 record.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-start);command.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'phase':name,'returncode':r.returncode}),flush=True);r.check_returncode()
print(json.dumps({'tagged_construction_completed':True,'neural_calls':0}))
