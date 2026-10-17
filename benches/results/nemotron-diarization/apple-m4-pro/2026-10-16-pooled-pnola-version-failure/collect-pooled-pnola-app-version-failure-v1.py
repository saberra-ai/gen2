"""Collect the actual strict alias-image rejection; never execute app/native/models."""
from pathlib import Path
import json,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from sdk_archive import inventory
from package_macos import verify
from macos_loaded_libraries import audit_loaded_libraries
root=cache/'pooled-pnola-complete-app-runtime-v1';q=json.loads((root/'qualification.json').read_text());p=json.loads((root/'protocol.json').read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert q['source_commit']==source and not q['automatic_packaged_resolution_gates_passed']and q['neural_commands_started']==[]
assert q['error'].startswith("ValueError('unexpected non-system dyld image: ")and q['error'].endswith("/nemo/lib/libnemo_speech_asr_c.dylib')")
assert [(x['name'],x['returncode'])for x in q['commands']]==[('runtime-controls',0),('premodel-controls',0),('version-only',0)]
assert descriptor(root/'protocol.json')==q['protocol'] and(root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-pooled-pnola-app.protocol.json'],cwd=repo)
pins={'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()};pins.update(p['repository_files']);verify_git_sources(repo,source,pins)
work=Path(p['work_path']);assert inventory(work)==p['work_inventory']and inventory(Path(p['app_path']))==p['app_inventory'];manifest=verify(work/'nemo')
assert not(work/'nemo/lib/libnemo_speech_asr_c.dylib').is_symlink()
assert descriptor(work/'nemo/lib/libnemo_speech_asr_c.dylib')==descriptor(work/'nemo/lib'/manifest['entry_library'])
try:audit_loaded_libraries((root/'version-only.stderr.log').read_text(),work/'nemo',manifest,work/'bin/load-probe')
except ValueError as error:assert repr(error)==q['error']
else:raise AssertionError('strict auditor failed to reproduce original alias rejection')
assert q['storage']['released']
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-pnola-version-failure';assert not out.exists();out.mkdir()
for path in root.iterdir():
 if path.is_file()and path.suffix in {'.json','.log','.sb'}:shutil.copy2(path,out/path.name)
shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(cache/'pooled-pnola-complete-app-runtime-v1.controller.log',out/'controller.log')
r={'schema_version':1,'automatic_packaged_resolution_gates_passed':False,'release_qualified':False,'source_commit':source,'source_pins_verified':len(pins),'native_version_queries_started':1,'app_or_neural_commands_started':0,'strict_alias_rejection_reproduced_without_native_calls':True,'work_and_app_inventories_unchanged':True,'limits':['Both controls and version-only process return0, then strict image audit rejects the materialized public entry alias. No app process or model inference ran.','Tauri resource staging permits a byte-identical regular-file alias. Native reference version probe must load the manifest canonical entry directly, without widening the image audit. Original failed protocol and output remain unchanged.']}
(out/'collection-qualification.json').write_text(json.dumps(r,indent=2)+'\n');files={x.relative_to(out).as_posix():descriptor(x)for x in out.rglob('*')if x.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,len(files)+1,'members')
