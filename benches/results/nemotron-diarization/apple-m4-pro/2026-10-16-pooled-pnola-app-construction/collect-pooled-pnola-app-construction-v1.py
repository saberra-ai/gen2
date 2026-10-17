"""Collect static complete pooled app construction; never execute app/native/models."""
from pathlib import Path
import json,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from sdk_archive import inventory
from verify_pooled_pnola_app import verify_app
from package_macos import verify
root=cache/'pooled-pnola-complete-app-build-v1';q=json.loads((root/'build.json').read_text());p=json.loads((root/'protocol.json').read_text());source='f21c5616d51d6f20600ccee51489ca8178ec13f3'
assert q['source_commit']==p['source_commit']==source and q['app_construction_passed']is True and q['models_loaded_for_inference']is False and q['native_version_queries_or_neural_calls']==0 and 'error'not in q
assert [(x['name'],x['returncode'])for x in q['commands']]==[('cpu-build',0),('cpu-clone',0),('cpu-signature',0)]
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes']>=1024**3
pins=dict(p['gen2_files']);pins.update({'tools/nemotron-diarization/'+n:v for n,v in p['extra_tools'].items()});verify_git_sources(repo,source,pins)
verify_git_sources(repo/'vendor/llama-cpp-sys-2/llama.cpp',p['vendor_revision'],p['vendor_source_files'])
for n,v in p['pnola_source_files'].items():assert descriptor(Path('/Users/victor/workspace/pnola')/n)==v
for n,v in p['asset_inputs'].items():assert inventory(Path(n))==v
for n,v in p['asset_files'].items():assert descriptor(Path(n))==v
app=root/'cpu/Pnola.app';assert inventory(app)==q['apps']['cpu']['files'];replay=verify_app(app);assert replay==json.loads((root/'cpu-static-verification.json').read_text())
assert replay['app_structure_verified']and replay['runtime_profile']=='gen2-valid-mel-pool-v1'and replay['backend']=='cpu'and replay['release_qualified']is False
assert len(replay['frameworks'])==94 and replay['logical_runtime_library_count']==92
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-pnola-app-construction';assert not out.exists();out.mkdir()
for path in root.iterdir():
 if path.is_file()and path.suffix in {'.json','.log','.sb'}:shutil.copy2(path,out/path.name)
for path in [cache/'build-pooled-pnola-complete-app-v1.py',cache/'pooled-pnola-complete-app-build-v1.controller.log',Path(__file__)]:shutil.copy2(path,out/path.name)
record={'schema_version':1,'complete_pooled_cpu_app_construction_passed':True,'release_qualified':False,'source_commit':source,'parent_git_source_pins_verified':len(pins),'vendor_source_pins_verified':len(p['vendor_source_files']),'pnola_source_pins_verified':len(p['pnola_source_files']),'app_inventory_entries_rehashed':len(q['apps']['cpu']['files']),'static_verification_replayed':True,'framework_files':94,'logical_nemo_libraries':92,'model_or_native_version_queries_during_construction':0,'app_or_neural_calls_during_collection':0,'limits':['Complete same-host ad-hoc pooled CPU app construction and static payload/signature/dependency/license checks only. App and model were not executed.','Private app source, binary and weights are not published. Metadata inventories and construction logs are retained; actual app is local. No GUI, microphone, fresh-machine, public Developer ID/notarization, accuracy or release claim.']}
(out/'collection-qualification.json').write_text(json.dumps(record,indent=2)+'\n');files={x.relative_to(out).as_posix():descriptor(x)for x in out.rglob('*')if x.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,len(files)+1,'members')
