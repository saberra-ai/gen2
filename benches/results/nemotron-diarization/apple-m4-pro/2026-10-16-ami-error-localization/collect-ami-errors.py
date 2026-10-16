"""Collect and independently replay the preregistered saved-output diagnostic."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
base=cache/'ami-error-localization-5113a30'
output=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-ami-error-localization'
tools=repo/'tools/nemotron-diarization'
python=cache/'python-reference/bin/python'
read=lambda p:json.loads(p.read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source='5113a30161b3592479b460b129a6d21f78ffdabe'
protocol=read(base/'protocol.json');result=read(base/'result.json')
assert digest(base/'protocol.json')==digest(tools/'ami-error-localization.protocol.json')==result['protocol_sha256']
assert result['source_commit']==source
assert all(result[name] is False for name in ['release_qualified','original_gate_status_changed','inference_performed','reference_changed'])
for name,expected in protocol['tools'].items():
 assert digest(tools/name)==expected
 assert hashlib.sha256(subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)).hexdigest()==expected
argv=[str(python),str(tools/'diagnose_ami_errors.py'),'--evidence',str(cache/'quality qualification 125f907'),'--reference-diagnostic',str(repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-ami-reference-diagnostic')]
with tempfile.TemporaryDirectory() as temporary:
 replay=Path(temporary)/'replay'
 completed=subprocess.run([*argv,'--output',str(replay)],cwd=repo,text=True,capture_output=True,check=True)
 assert read(replay/'result.json')==result and (replay/'protocol.json').read_bytes()==(base/'protocol.json').read_bytes()
 output.mkdir(exist_ok=False)
 shutil.copytree(base,output,dirs_exist_ok=True)
 (output/'receiver-replay.stdout.log').write_text(completed.stdout)
 (output/'receiver-replay.stderr.log').write_text(completed.stderr)
(output/'tools').mkdir()
for name in protocol['tools']:
 shutil.copy2(tools/name,output/'tools'/name)
shutil.copy2(__file__,output/'collect-ami-errors.py')
rows=result['results'];assert len(rows)==12 and len(result['independent_scorer_calibrations'])==4
max_error=0.0
for row in rows:
 for axis in row['partitions'].values():
  for key in ['SCORED_TIME','SCORED_SPEAKER','MISSED_SPEAKER','FALARM_SPEAKER','SPEAKER_ERROR']:
   delta=abs(sum(bin[key]for bin in axis.values())-row['total'][key]);max_error=max(max_error,delta)
   assert delta<1e-8
assert {row['backend']for row in rows}=={'cpu','metal'}
assert {row['profile']for row in rows}==set(protocol['profiles'])
collection={'schema_version':1,'source_commit':source,'release_qualified':False,'original_gate_status_changed':False,
 'inference_performed':False,'reference_changed':False,'results':12,'official_synthetic_controls':4,
 'all_previous_official_results_reproduced':True,'all_partitions_exhaustive_and_additive':True,
 'maximum_partition_component_rounding_seconds':max_error,'independent_receiver_replay_exact':True,
 'execution_argv':[*argv,'--output',str(base)],'receiver_replay_returncode':completed.returncode,
 'limits':protocol['limitations']}
(output/'qualification.json').write_text(json.dumps(collection,indent=2)+'\n')
files={path.relative_to(output).as_posix():{'bytes':path.stat().st_size,'sha256':digest(path)}for path in output.rglob('*')if path.is_file()}
(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'evidence_files':len(files),'results':12,'max_rounding_seconds':max_error}))
