from pathlib import Path
import hashlib,json,shutil,subprocess
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'bounded-native-admission-v2'
read=lambda path:json.loads(Path(path).read_text())
def desc(path):
 raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
q,p=read(root/'qualification.json'),read(root/'protocol.json');source='0e5080daf82a5070c99ff418943812b5cb6b3311'
assert q['source_commit']==source and q['admit_only'] is True and q['deliveries_admitted'] is True
assert q['native_parity_passed'] is False and q['realtime_gate_passed'] is False and q['release_qualified'] is False
assert q['commands']==q['neural_commands_started']==[] and q['comparisons']==q['loaded_images']=={} and 'error' not in q
assert not (root/'sdk').exists() and not (root/'helper-source').exists() and list((root/'work').iterdir())==[]
assert all(key not in q for key in ['acquisitions','helper_acquisitions','used_compiler_inputs','helper_archive'])
assert (root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-bounded-native-v2.protocol.json'],cwd=repo)
for key in ['producer','native_reference']:
 files={str(path.relative_to(root/key)):desc(path) for path in (root/key).rglob('*') if path.is_file()}
 assert files==p[key]['files']
 api=q['deliveries'][key]
 assert api['id']==p[key]['id'] and api['name']==p[key]['name'] and api['workflow_run']['id']==p[key]['run_id']
 assert api['workflow_run']['head_sha']==p[key]['source_commit'] and api['digest']=='sha256:'+p[key]['zip']['sha256']
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes']==p['storage']['reserve_bytes']
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-native-admission-v2';assert not out.exists();out.mkdir()
for path in [root/'protocol.json',root/'qualification.json',cache/'bounded-native-admission-v2.controller.log',Path(__file__),
             cache/'bounded-native-schema-red.log',cache/'bounded-native-schema-green.log',cache/'platform-ci/model-free-262-bounded-native-v2-local.log',
             *list((cache/'bounded-native-inputs-v2').iterdir())]:shutil.copy2(path,out/path.name)
summary={'schema_version':1,'source_commit':source,'both_full_deliveries_admitted':True,'members_verified':{'producer':200,'native_reference':33},
 'sdk_restored':False,'helpers_compiled_or_executed':0,'model_assets_acquired':0,'model_calls':0,'commands_started':0,
 'complete_declared_identity_contract_verified':True,'storage_reserve_released':True,'original_failure_relabelled':False,
 'release_qualified':False,'limits':['Local pure artifact admission; no SDK restoration, native library/helper execution or inference. Corrected M1 numerical/performance test and remaining public/lifecycle/quality/app/release gates stay pending.']}
(out/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n');files={str(path.relative_to(out)):desc(path) for path in out.iterdir() if path.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(out,len(files)+1,'members')
