"""Run the registered three-case native CPU diagnostic and retain outer status."""
import json,subprocess,time
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');output=cache/'native-f32-execution-v1'
argv=[str(cache/'python-reference/bin/python'),str(repo/'tools/nemotron-diarization/diagnose_native_f32.py'),'--evidence',str(cache/'quality qualification 125f907'),'--corpus',str(cache/'quality-corpus-v1'),'--construction',str(repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-native-f32-construction'),'--model',str(cache/'native-f32-conversion-v1/Nemotron-3-Diarization.f32-control.gguf'),'--helper',str(cache/'quality qualification 125f907/native-cpu'),'--sdk',str(cache/'release CPU qualification d01ef80/nemo'),'--output',str(output)]
env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
command={'argv':argv,'environment':env,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()};started=time.monotonic()
with Path(str(output)+'.stdout.log').open('x')as stdout,Path(str(output)+'.stderr.log').open('x')as stderr:
 completed=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
command.update(returncode=completed.returncode,elapsed_seconds=time.monotonic()-started)
Path(str(output)+'.command.json').write_text(json.dumps(command,indent=2)+'\n')
print(json.dumps({'returncode':completed.returncode,'elapsed_seconds':command['elapsed_seconds']}))
raise SystemExit(completed.returncode)
