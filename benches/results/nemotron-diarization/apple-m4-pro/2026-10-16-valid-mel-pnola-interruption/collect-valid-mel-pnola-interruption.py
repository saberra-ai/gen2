"""Collect interrupted Pnola attempt and preparation history; no neural calls."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
source=cache/'valid-mel-pnola-runtime-v1';r=json.loads((source/'qualification.json').read_text());p=json.loads((source/'protocol.json').read_text());commands=json.loads((source/'commands.json').read_text());out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-pnola-interruption';assert not out.exists()
assert not r['adapter_gates_passed']and not r['cases']and r['storage']['released']and r['source_commit']=='959f5a83ccf5e5788a400497e26c5ebe1f91bbcb'
assert len(commands)==2 and commands[0]['name']=='list'and commands[0]['returncode']==0 and commands[1]['name']=='cpu-recorded-test'and commands[1]['returncode']==-9 and commands[1]['forced_process_group_termination']
assert descriptor(source/'protocol.json')==r['protocol']
try:
 import os
 os.kill(commands[1]['pid'],0)
except ProcessLookupError:pass
else:raise ValueError('interrupted helper is still present')
out.mkdir(parents=True);shutil.copytree(source,out/'raw')
for name in ['valid-mel-pnola-helper-build-v1','valid-mel-pnola-helper-build-v2','valid-mel-pnola-inputs-v1','valid-mel-pnola-inputs-v2']:
 for path in sorted((cache/name).iterdir()):
  if path.is_file()and path.suffix in {'.json','.jsonl','.log','.sb'}:
   dest=out/'preparation'/name/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
for name in ['build-valid-mel-pnola-helpers.py','build-valid-mel-pnola-helpers-v2.py','prepare-valid-mel-pnola-inputs.py','prepare-valid-mel-pnola-inputs-v2.py','cleanup-stale-pnola-incremental-v1.py','pnola-incremental-cleanup-v1.json']:
 shutil.copy2(cache/name,out/name)
for name,expected in p['tools'].items():
 data=subprocess.check_output(['git','show',r['source_commit']+':tools/nemotron-diarization/'+name],cwd=repo)
 assert{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==expected
 dest=out/'execution-tools'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
shutil.copy2(Path(__file__),out/Path(__file__).name)
record={'schema_version':1,'adapter_gates_passed':False,'release_qualified':False,'source_commit':r['source_commit'],'protocol':r['protocol'],'interruption_reason':'Implementer found capture-result validation assumed full Diarization geometry/frame-count fields, while actual live snapshot is compact. Explicit SIGINT stopped the parent during first CPU recorded test, before any model case completed.','controller_exit_status':130,'completed_model_cases':0,'model_free_listing_passed':True,'interrupted_command':commands[1]['name'],'interrupted_child_status':-9,'child_process_group_killed_and_reaped':True,'reserve_released':True,'new_native_or_neural_calls_during_collection':0,'preparation_history':['First helper compilation failed because model-denying sandbox blocked Tauri resource staging; zero helper/model execution.','Second helper test/CLI compilation succeeded offline with model reads denied and explicit TAURI_CONFIG bundle.resources/frameworks empty; this is not signed bundle construction.','First clone-only staging incorrectly expected80.04s source WAV; actual60s source is repeated/truncated to80.04s by existing live verifier. Zero native/model/helper calls.','Second clone-only staging succeeds; exact245regularfiles/twoaliases and both profile-bound v2packages are preserved in the preparation record.'],'limits':['No Pnola adapter case passes from this interrupted attempt. No real-model parity, capture/live, GUI, microphone, quality, performance or release claim.','Original failed/interrupted records are retained unchanged; the separate compact-schema continuation requires a fresh frozen protocol.'],'files':{str(path.relative_to(out)):descriptor(path)for path in sorted(out.rglob('*'))if path.is_file()and '__pycache__'not in path.parts}}
(out/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'target':str(out),'members':len(record['files'])+1,'completed_model_cases':0}))
