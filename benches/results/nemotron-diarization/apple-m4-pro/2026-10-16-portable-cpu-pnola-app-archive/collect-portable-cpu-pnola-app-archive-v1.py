"""Retain local complete-app archive metadata, not opaque app/model payloads."""
import ast,hashlib,json,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory
source=cache/'portable-cpu-pnola-app-delivery-v1';p=json.loads((source/'protocol.json').read_text());q=json.loads((source/'qualification.json').read_text());m=json.loads((source/'portable-cpu-pnola-app-files.json').read_text());v=json.loads((source/'archive-byte-verification.json').read_text())
assert q['archive_construction_passed'] is True and q['models_or_app_executed'] is False and q['release_qualified'] is False
assert descriptor(cache/'archive-portable-cpu-pnola-app-v1.py')==p['launcher']
assert descriptor(cache/'verify-portable-cpu-pnola-app-archive-v1.py')==v['verifier']
assert descriptor(source/'protocol.json')==v['archive_construction_protocol']
for name,value in p['tools'].items():assert descriptor(tools/name)==value
# Additive post-execution binding; preserve the executed four-tool protocol.
prior_root=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-portable-cpu-pnola-app-construction'
prior_files=json.loads((prior_root/'collection.json').read_text())['files']
assert descriptor(prior_root/'protocol.json')==prior_files['protocol.json']
constructor=json.loads((prior_root/'protocol.json').read_text())
assert descriptor(cache/'portable-cpu-pnola-app-build-v1/protocol.json')==prior_files['protocol.json']
queue=['build_valid_mel_candidate.py','qualification_storage.py','qualify_valid_mel_pnola.py','sdk_archive.py'];closure=set()
while queue:
 name=queue.pop()
 if name in closure:continue
 closure.add(name)
 for node in ast.walk(ast.parse((tools/name).read_text())):
  modules=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
  for module in modules:
   path=tools/((module or '').split('.')[0]+'.py')
   if path.is_file():queue.append(path.name)
closure_pins={name:descriptor(tools/name) for name in sorted(closure)}
for name,value in closure_pins.items():
 assert value==constructor['tools'][name]
 raw=subprocess.check_output(['git','show','7db170d0c8f17b3aea670769577d2d3cd2deaf5b:tools/nemotron-diarization/'+name],cwd=repo)
 assert value=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
addendum={'schema_version':1,'all_local_transitive_python_imports':closure_pins,'verification_timing':'After archive execution, against the already retained prior constructor protocol and historical7db source; not a claim that all9imports were individually pinned by the executed4tool archive protocol.','original_archive_protocol_preserved':True,'archive_bytes_unchanged':True,'initial_byte_verifier_correction':'The first verifier assumed root-relative tar names and rejected the fixed sdk/ root before payload checks. Only the verifier mapping was corrected to canonical sdk/ members; original archive and manifest unchanged. The full final stream check then passed.','no_archive_rebuild_or_model_app_execution':True}
build=cache/'portable-cpu-pnola-app-build-v1/build.json';assert descriptor(build)==p['constructor_receipt']
b=json.loads(build.read_text());assert b['app_construction_passed'] is True
assert inventory(Path(p['source_app']))==m['files']==p['inventory']==b['inventory']
assert len(m['files'])==q['inventory_entries']==v['entries']==246 and v['regular_files']==225
assert v['all_archive_entries_match_original_inventory'] is True and v['models_or_app_executed'] is False and v['extracted'] is False
assert v['release_qualified'] is False
archive=source/m['archive'];h=hashlib.sha256()
with archive.open('rb') as stream:
 while chunk:=stream.read(1024**2):h.update(chunk)
actual={'bytes':archive.stat().st_size,'sha256':h.hexdigest()}
assert actual==q['archive_descriptor']==v['archive']=={'bytes':573473763,'sha256':'2b74246cb6deb65380edf23c3fbcc14134423d5a547a32f9b6e7b19f868bfe93'}
assert {k:value for k,value in m.items() if k!='files'}==q['archive']
c=q['command'];assert c['returncode']==0 and not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes')
assert c['argv'][:3]==['/usr/bin/sandbox-exec','-f',str(source/'archive.sb')]
assert c['argv'][4:]==['-c',p['archive_code'],p['source_app'],str(source)]
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum'] and q['storage']['reserve_reported_allocated_bytes']>=p['storage']['reserve_bytes']
assert p['neural_admission_floor_unchanged_bytes']==5*1024**3
r={'schema_version':1,'app_source_commit':b['source_commit'],'local_archive_construction_and_byte_verification_passed':True,'archive':actual,'entries':246,'regular_files':225,'models_or_app_executed':False,'delivered_to_fresh_receiver':False,'release_qualified':False,'storage':q['storage'],'scope':__doc__,'limits':['Archive is in owned local cache; no cloud asset, public release, fresh-machine app execution, GUI, microphone, quality, performance or signing/notarization gate is established.','Exact full app contains model bytes for later offline delivery; only metadata/hashes/scripts/logs are retained in Git.','Independent standard-library tar iteration checks all canonical sdk/ member names, uniqueness, modes, types, alias targets and regular-file lengths/digests without extraction.','Model-free archive construction uses3GiB start/1GiB running/256MiB reserve;5GiB neural admission remains unchanged.']}
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-portable-cpu-pnola-app-archive';assert not out.exists();out.mkdir()
for path in source.iterdir():
 if path.is_file() and path.suffix!='.gz':shutil.copy2(path,out/path.name)
for name in ['archive-portable-cpu-pnola-app-v1.py','verify-portable-cpu-pnola-app-archive-v1.py','collect-portable-cpu-pnola-app-archive-v1.py']:
 shutil.copy2(cache/name,out/name)
(out/'source-binding-addendum.json').write_text(json.dumps(addendum,indent=2)+'\n')
(out/'collection-qualification.json').write_text(json.dumps(r,indent=2)+'\n');files={f.name:descriptor(f) for f in out.iterdir() if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'members':len(files)+1,'passed':True,'output':str(out)}))
