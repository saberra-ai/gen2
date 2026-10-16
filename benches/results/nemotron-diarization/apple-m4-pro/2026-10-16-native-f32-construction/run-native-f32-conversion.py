"""Execute the frozen local conversion, preserving all command and failure evidence."""
import hashlib,importlib.metadata,json,os,platform,subprocess,tarfile,time
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');native=cache/'NeMo-Speech.cpp'
protocol_path=repo/'tools/nemotron-diarization/native-f32-conversion.protocol.json';p=json.loads(protocol_path.read_text())
record=lambda path:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert 'Python '+platform.python_version()==p['python_version']
for name,version in p['packages'].items():assert importlib.metadata.version(name)==version
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=native,text=True).strip()==p['native_revision']
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=native,text=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=native/'llama.cpp',text=True).strip()==p['gguf_revision']
for group,root in [('converter_files',native),('gguf_files',native/'llama.cpp/gguf-py')]:
 for name,expected in p[group].items():assert record(root/name)==expected
checkpoint=cache/'research/native-f32-control'/p['checkpoint']['name']
assert record(checkpoint)=={key:p['checkpoint'][key]for key in ['bytes','sha256']}
with tarfile.open(checkpoint)as bundle:
 rows=[]
 for member in bundle:
  assert member.isfile()
  data=bundle.extractfile(member).read();rows.append({'name':member.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 assert rows==p['checkpoint_members']
output=Path(p['argv'][-1]).parent;output.mkdir(exist_ok=False)
(output/'protocol.json').write_bytes(protocol_path.read_bytes())
profile=output/'offline.sb';profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (literal "'+str(cache/'assets/model.safetensors')+'"))\n')
argv=['/usr/bin/sandbox-exec','-f',str(profile),*p['argv']]
command={'argv':argv,'environment':p['environment'],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'native_source_commit':p['native_revision'],'protocol_sha256':record(protocol_path)['sha256'],'sandbox_sha256':record(profile)['sha256']}
(output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
start=time.monotonic()
with (output/'converter.stdout.log').open('x')as stdout,(output/'converter.stderr.log').open('x')as stderr:
 result=subprocess.run(argv,cwd=repo,env=p['environment'],stdout=stdout,stderr=stderr)
command.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-start)
(output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
if result.returncode:raise SystemExit(result.returncode)
control=Path(p['argv'][-1]);(output/'construction.json').write_text(json.dumps({'artifact':record(control),'name':control.name,'checkpoint_deserialization_performed':True,'neural_model_inference_performed':False,'release_qualified':False,'artifact_admitted_for_inference':False,'source_tensor_accounting_completed':False},indent=2)+'\n')
print(json.dumps({'returncode':0,'artifact':record(control),'inference_admitted':False,'elapsed_seconds':command['elapsed_seconds']}))
