"""Launch only after the separate Stage A2 protocol has been committed/reviewed."""
import hashlib,json,os,pathlib,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');output=cache/'tail-framing-v1';tools=repo/'tools/nemotron-diarization';python=cache/'python-reference/bin/python';protocol=json.loads((tools/'tail-framing.protocol.json').read_text())
assert not output.exists() and not list(output.parent.glob(output.name + '.*')), 'refusing to overwrite any prior run-prefix evidence'
assert {'bytes': pathlib.Path(__file__).stat().st_size, 'sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()} == protocol['external_launcher']
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/tail-framing.protocol.json'],cwd=repo)==(tools/'tail-framing.protocol.json').read_bytes()
for name,expected in protocol['tools'].items():
 p=tools/name;actual={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};assert actual==expected
profile=pathlib.Path(str(output)+'.sb');profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(p)+')'for p in protocol['denied_paths'])+')\n')
env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONPATH':str(tools)})
preflight=['/usr/bin/sandbox-exec','-f',str(profile),str(python),'-m','unittest','discover','-s',str(tools),'-p','test_tail_framing.py']
def run(argv,label):
 start=time.monotonic();p=subprocess.run(argv,cwd=repo,env=env,capture_output=True);elapsed=time.monotonic()-start
 pathlib.Path(str(output)+'.'+label+'.stdout.log').write_bytes(p.stdout);pathlib.Path(str(output)+'.'+label+'.stderr.log').write_bytes(p.stderr)
 pathlib.Path(str(output)+'.'+label+'.command.json').write_text(json.dumps({'source_commit':source,'argv':argv,'environment_overrides':{k:env[k]for k in ['PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH']},'returncode':p.returncode,'elapsed_seconds':elapsed},indent=2)+'\n');print(label,p.returncode,elapsed,flush=True);p.check_returncode()
run(preflight,'preflight')
argv=['/usr/bin/sandbox-exec','-f',str(profile),str(python),str(tools/'diagnose_tail_framing.py'),'--assets',str(cache/'assets'),'--corpus',str(cache/'quality-corpus-v1'),'--evidence',str(cache/'quality qualification 125f907'),'--frontend-evidence',str(cache/'frontend-diagnostic-ec88564'),'--native-source',str(cache/'NeMo-Speech.cpp'),'--sdk',protocol['linked_sdk_path'],'--probe',str(cache/'frontend-tail-probe-v1/frontend-tail-probe'),'--probe-build',str(cache/'frontend-tail-probe-v1/build.json'),'--output',str(output)]
run(argv,'execution')
