"""Retain saved first-audio bytes; independently recheck, never run inference."""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'pooled-native-local-v2'
from build_valid_mel_candidate import descriptor
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from qualify_pooled_native import compare_history
from sdk_archive import inventory
q=json.loads((root/'qualification.json').read_text());p=json.loads((root/'protocol.json').read_text())
source='618d7969b112d1e923e15db19fe81eb2c0b892f4'
assert q['source_commit']==source and q['native_parity_passed'] and q['release_qualified'] is False
assert q['neural_commands_started']==['native-12s','native-60s']
assert [r['name'] for r in q['commands']]==['build-native-stream','build-failed-load-probe','model-free-controls','failed-loads','runtime-controls','native-12s','native-60s']
assert all(r['returncode']==0 for r in q['commands'])
assert descriptor(root/'protocol.json')==q['protocol']
assert (root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-pooled-native.protocol.json'],cwd=repo)
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
 for name,expected in p[group].items():
  raw=subprocess.check_output(['git','show',source+':'+prefix+name],cwd=repo)
  assert expected=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},name
for stem,destination in [('pooled-cpu-sdk',root/'sdk'),('pooled-cpu-runtime',root/'work/nemo')]:
 m=json.loads((root/'construction'/(stem+'-files.json')).read_text());assert inventory(destination)==m['files']
helpers=json.loads((root/'pooled-native-helpers-files.json').read_text());assert inventory(root/'work/bin')==helpers['files']
assert descriptor(root/helpers['archive'])=={'bytes':helpers['archive_bytes'],'sha256':helpers['archive_sha256']}
for name,entry in p['helper_inputs'].items():assert descriptor(root/'helper-source'/name)=={'bytes':entry['size'],'sha256':entry['sha256']}
assert descriptor(root/'helper-source/native-stream.cpp')==p['derived_helper']
manifest=verify(root/'work/nemo')
for name,executable in [('failed-loads','failed-load-probe'),('native-12s','native-stream'),('native-60s','native-stream')]:
 assert audit_loaded_libraries((root/(name+'.stderr.log')).read_text(),root/'work/nemo',manifest,root/'work/bin'/executable)==q['loaded_images'][name]
 assert q['loaded_images'][name]['private_library_count']==92
for case in p['cases']:
 actual=root/'results'/(case['name']+'.json');reference=root/case['reference']
 replay=compare_history(json.loads(actual.read_text()),Path(str(actual)+'.f32').read_bytes(),json.loads(reference.read_text()),Path(str(reference)+'.f32').read_bytes())
 assert all(q['comparisons'][case['name']][k]==v for k,v in replay.items()) and replay['bit_exact']
f=q['failed_load_control'];assert f['failed_model_construction_attempts']==101 and f['valid_models_loaded']==f['frontend_or_neural_calls']==0
assert all(len(f[k])==101 for k in ['immediate_thread_counts','threads_after_each_failure','requested_settling_wait_milliseconds'])
assert set(f['threads_after_each_failure'])=={1} and set(f['requested_settling_wait_milliseconds'])=={10}
failed=cache/'pooled-native-local-v1';bad=json.loads((failed/'qualification.json').read_text())
assert not bad['native_parity_passed'] and bad['neural_commands_started']==[] and bad['commands']==[] and bad['error']=="KeyError('id')"
assert q['storage']['released'] and bad['storage']['released']
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-native';assert not out.exists();out.mkdir()
def retain(src,relative):
 dst=out/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert descriptor(src)==descriptor(dst)
for path in root.iterdir():
 if path.is_file() and path.suffix in {'.json','.log','.sb'}:retain(path,path.name)
for directory in ['results','helper-source']:
 for path in (root/directory).rglob('*'):
  if path.is_file():retain(path,path.relative_to(root))
retain(failed/'qualification.json','initial-input-admission-failure/qualification.json')
retain(cache/'pooled-native-local-v1.controller.log','initial-input-admission-failure/controller.log')
retain(cache/'portable-macos-runtime-ci-v2/download-identity.json','initial-input-admission-failure/wrapped-input-receipt.json')
for name in ['pooled-native-sandbox-red.log','pooled-native-sandbox-green.log']:
 retain(cache/name,'sandbox-regression/'+name)
retain(cache/'platform-ci/model-free-216-pooled-native-local.log','model-free-216.log')
retain(Path(__file__),Path(__file__).name)
record={'schema_version':1,'source_commit':source,'native_parity_passed':True,'all42historical_source_and_workflow_pins_verified':True,'native_processes':2,'frames_compared':[1200,6000],'probabilities_compared':57600,'all_probabilities_bit_exact':True,'raw_turns_compared':32,'all_raw_turns_exact':True,'failed_model_attempts':101,'three_strict_private_library_audits':92,'helper_payload_archive_rehashed':True,'sdk_package_and_helper_inventories_unchanged':True,'initial_metadata_input_failure_retained_with_zero_native_or_neural_commands':True,'release_qualified':False,'limits':['Actual M4 CPU execution of immutable M1-built candidate. Exactly two fresh C ABI audio processes; no neural retry.','First local invocation supplied a wrapped historical download receipt instead of raw API metadata and failed closed during artifact admission before any command. Fresh invocation uses the correct API response, with unchanged source/protocol.','Warm RTF0.0446453/0.128474 and external time/RSS are descriptive. Historical reference hosts differ; this is no matched speedup or supported live-performance claim.','Rust/public lifecycle, long/concurrent/mixed, accuracy/attribution, platform/app/signing/notary/release gates remain separate. Opaque SDK/runtime/helpers remain outside Git with exact archive/member descriptors.']}
(out/'collection-qualification.json').write_text(json.dumps(record,indent=2)+'\n')
files={path.relative_to(out).as_posix():descriptor(path) for path in out.rglob('*') if path.is_file()}
(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
assert all(descriptor(out/name)==expected for name,expected in files.items())
print(out,'members',len(files)+1)
