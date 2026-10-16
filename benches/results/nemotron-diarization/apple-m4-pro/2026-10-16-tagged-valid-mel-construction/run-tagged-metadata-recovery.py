"""Run only the committed metadata recovery protocol; preserve every prior file."""
import hashlib,json,pathlib,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';protocol_path=tools/'valid-mel-tagged-metadata.protocol.json';p=json.loads(protocol_path.read_text());out=pathlib.Path(p['output_path'])
def rec(path):return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert not out.exists()and not list(out.parent.glob(out.name+'.*')),'refusing prior metadata recovery overwrite'
assert rec(pathlib.Path(__file__))==p['external_launcher']
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-tagged-metadata.protocol.json'],cwd=repo)==protocol_path.read_bytes()
for name,expected in p['tools'].items():
 blob=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)
 assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==rec(tools/name)==expected
argv=[str(cache/'evidence-tools-env/bin/python'),str(tools/'recover_valid_mel_build_metadata.py')];env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'};meta={'source_commit':source,'argv':argv,'environment':env};command=pathlib.Path(str(out)+'.command.json');command.write_text(json.dumps(meta,indent=2)+'\n');started=time.monotonic()
with pathlib.Path(str(out)+'.stdout.log').open('xb')as stdout,pathlib.Path(str(out)+'.stderr.log').open('xb')as stderr:r=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
meta.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-started);command.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta));r.check_returncode()
