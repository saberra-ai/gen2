"""Collect actual scorer-bytecode inventory rejection; no native/model execution."""
from pathlib import Path
import json,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from sdk_archive import inventory
root=cache/'ami-prefix-diagnostic-v1';p=json.loads((root/'protocol.json').read_text());q=json.loads((root/'qualification.json').read_text());source='c4e2071e9b937261b5d30f01c10ca7729ea2ba5d'
assert q['source_commit']==source and q['diagnostic_completed']is False and q['commands']==q['neural_commands_started']==[] and q['error']=="ValueError('prefix input inventory changed')"
assert descriptor(root/'protocol.json')==q['protocol']and(root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/ami-prefix-diagnostic.protocol.json'],cwd=repo)
pins={'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()};pins.update(p['repository_files']);verify_git_sources(repo,source,pins)
actual=inventory(Path(p['work_path']));assert all(actual.get(n)==v for n,v in p['work_inventory'].items())
extra={n:v for n,v in actual.items()if n not in p['work_inventory']};assert set(extra)=={'scoring/__pycache__','scoring/__pycache__/md_eval.cpython-312.pyc'}
assert q['storage']['released']and len(json.loads((root/'scorer-calibrations.json').read_text()))==4
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-ami-prefix-import-failure';assert not out.exists();out.mkdir()
for path in root.iterdir():
 if path.is_file()and path.suffix in {'.json','.log','.sb'}:shutil.copy2(path,out/path.name)
for path in [Path(__file__),cache/'ami-prefix-diagnostic-v1.controller.log',cache/'ami-prefix-readonly-import-red.log',cache/'ami-prefix-readonly-import-green.log']:shutil.copy2(path,out/path.name)
r={'schema_version':1,'diagnostic_completed':False,'release_qualified':False,'source_commit':source,'source_pins_verified':len(pins),'native_or_model_commands_started':0,'all_original_151_input_entries_unchanged':True,'unexpected_import_generated_entries':extra,'official_synthetic_controls_completed':4,'original_quality_gate_remains_failed':True,'limits':['The parent official-scorer import generated __pycache__ inside the protected staging tree. The first frozen() command guard rejects the added entries before any sandbox command, version query or model load.','Original protocol, output and changed staging remain preserved. Corrected execution requires fresh immutable staging and a separately frozen protocol; no inference retry occurred during collection.']}
(out/'collection-qualification.json').write_text(json.dumps(r,indent=2)+'\n');files={x.relative_to(out).as_posix():descriptor(x)for x in out.rglob('*')if x.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,len(files)+1,'members')
