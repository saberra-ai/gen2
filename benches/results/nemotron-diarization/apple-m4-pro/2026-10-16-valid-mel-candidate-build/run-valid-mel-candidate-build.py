"""Run only the committed build-only candidate protocol, preserving each attempt."""
import hashlib,json,os,pathlib,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';output=cache/'valid-mel-candidate-build-v1';protocol=json.loads((tools/'valid-mel-candidate-build.protocol.json').read_text())
assert not output.exists()and not list(output.parent.glob(output.name+'.*')),'refusing prior build evidence overwrite'
record=lambda p:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert record(pathlib.Path(__file__))==protocol['external_launcher']
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-candidate-build.protocol.json'],cwd=repo)==(tools/'valid-mel-candidate-build.protocol.json').read_bytes()
for name,expected in protocol['tools'].items():assert record(tools/name)==expected
argv=[str(cache/'evidence-tools-env/bin/python'),str(tools/'build_valid_mel_candidate.py'),'--source',protocol['candidate_source_path'],'--prefix',protocol['candidate_prefix_path'],'--preparation',str(cache/'valid-mel-source-v1/prepared.json'),'--output',str(output)]
start=time.monotonic();p=subprocess.run(argv,cwd=repo,capture_output=True);elapsed=time.monotonic()-start
pathlib.Path(str(output)+'.stdout.log').write_bytes(p.stdout);pathlib.Path(str(output)+'.stderr.log').write_bytes(p.stderr);pathlib.Path(str(output)+'.command.json').write_text(json.dumps({'source_commit':source,'argv':argv,'returncode':p.returncode,'elapsed_seconds':elapsed},indent=2)+'\n');print(p.stdout.decode(),flush=True);print(p.stderr.decode(),flush=True);p.check_returncode()
