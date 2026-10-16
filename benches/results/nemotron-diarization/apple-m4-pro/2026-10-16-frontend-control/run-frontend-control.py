import hashlib,json,os,subprocess,time
from pathlib import Path
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');base=cache/'frontend-diagnostic-ec88564'
profile=cache/'frontend-diagnostic-ec88564.sb'
profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (literal "'+str(cache/'assets/model.safetensors')+'") (literal "'+str(cache/'research/native-f32-control/Nemotron-3-Diarization.nemo')+'"))\n')
env={'PATH':'/usr/bin:/bin','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
argv=['/usr/bin/sandbox-exec','-f',str(profile),str(cache/'python-reference/bin/python'),str(repo/'tools/nemotron-diarization/diagnose_frontend.py'),'--assets',str(cache/'assets'),'--corpus',str(cache/'quality-corpus-v1'),'--native-source',str(cache/'NeMo-Speech.cpp'),'--sdk',str(cache/'sdk-cpu'),'--probe',str(cache/'frontend-control-build/frontend-probe'),'--output',str(base)]
record={'argv':argv,'environment':env,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'sandbox_sha256':hashlib.sha256(profile.read_bytes()).hexdigest()}
start=time.monotonic()
with Path(str(base)+'.stdout.log').open('x')as stdout,Path(str(base)+'.stderr.log').open('x')as stderr:
 result=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
record.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-start)
Path(str(base)+'.command.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'returncode':result.returncode,'elapsed_seconds':record['elapsed_seconds']}))
raise SystemExit(result.returncode)
