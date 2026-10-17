"""Collect delivered M1 C ABI parity without executing binaries or inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,zipfile
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools));cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'pooled-native-ci-v1';e=root/'evidence'
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory
from package_macos import verify
from macos_loaded_libraries import audit_loaded_libraries
from qualify_pooled_native import compare_history
p=json.loads((e/'protocol.json').read_text());q=json.loads((e/'qualification.json').read_text());api=json.loads((root/'download-identity.json').read_text());delivery=json.loads((root/'delivery.json').read_text());ci=json.loads((root/'run.json').read_text())
source='618d7969b112d1e923e15db19fe81eb2c0b892f4'
assert ci['status']=='completed' and ci['conclusion']=='success' and ci['headSha']==source and ci['url'].endswith('/38986310483')
assert api['id']==12083481818 and api['workflow_run']['head_sha']==source and api['workflow_run']['id']==38986310483
assert descriptor(root/'artifact.zip')==delivery['zip']=={'bytes':381613,'sha256':'bd6b8e7ece63f687e974c7b2b047bc0e87f5690d8958af568cc0d5576042dcf3'}
assert api['digest']=='sha256:'+delivery['zip']['sha256'] and api['size_in_bytes']==381613
with zipfile.ZipFile(root/'artifact.zip') as z:
 assert len(z.infolist())==len({m.filename for m in z.infolist()})==len(delivery['files'])==33
 for m in z.infolist():assert (e/m.filename).read_bytes()==z.read(m) and descriptor(e/m.filename)==delivery['files'][m.filename]
assert q['source_commit']==source and q['native_parity_passed'] and q['release_qualified'] is False
assert q['neural_commands_started']==['native-12s','native-60s'] and len(q['commands'])==7 and all(r['returncode']==0 for r in q['commands'])
assert descriptor(e/'protocol.json')==q['protocol']
assert (e/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-pooled-native.protocol.json'],cwd=repo)
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
 for name,expected in p[group].items():
  b=subprocess.check_output(['git','show',source+':'+prefix+name],cwd=repo);assert expected=={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
package=cache/'pooled-macos-construction-ci-v1/restored-pooled-cpu-runtime';manifest=verify(package);helpers=root/'restored-helpers';m=json.loads((e/'pooled-native-helpers-files.json').read_text());assert inventory(helpers)==m['files']
for name,exe in [('failed-loads','failed-load-probe'),('native-12s','native-stream'),('native-60s','native-stream')]:
 original=q['loaded_images'][name];remote_package=Path(next(iter(original['private_libraries']))).parents[1];local=helpers/exe
 trace=(e/(name+'.stderr.log')).read_text().replace(original['executable'],str(local)).replace(str(remote_package),str(package))
 replay=audit_loaded_libraries(trace,package,manifest,local)
 assert json.loads(json.dumps(replay).replace(str(local),original['executable']).replace(str(package),str(remote_package)))==original and replay['private_library_count']==92
for case,ref in [('native-12s',cache/'portable-cpu-profile-ci-v1/evidence/native.json'),('native-60s',cache/'portable-macos-runtime-ci-v2/evidence/results/native.json')]:
 target=e/'results'/(case+'.json');replay=compare_history(json.loads(target.read_text()),Path(str(target)+'.f32').read_bytes(),json.loads(ref.read_text()),Path(str(ref)+'.f32').read_bytes());assert replay['bit_exact']
 assert all(q['comparisons'][case][k]==v for k,v in replay.items())
f=q['failed_load_control'];assert f['failed_model_construction_attempts']==101 and f['valid_models_loaded']==f['frontend_or_neural_calls']==0 and set(f['threads_after_each_failure'])=={1} and set(f['requested_settling_wait_milliseconds'])=={10}
assert json.loads((e/'host.json').read_text())['cpu']=='Apple M1 (Virtual)' and q['storage']['released']
out=repo/'benches/results/nemotron-diarization/macos-arm64/2026-10-16-pooled-native';assert not out.exists();out.mkdir()
for path in e.rglob('*'):
 if path.is_file() and path.suffix!='.gz':
  destination=out/path.relative_to(e);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination);assert descriptor(path)==descriptor(destination)
for name in ['delivery.json','download-identity.json','run.json']:shutil.copy2(root/name,out/name)
shutil.copy2(__file__,out/Path(__file__).name)
(out/'collection-qualification.json').write_text(json.dumps({'schema_version':1,'source_commit':source,'ci_url':ci['url'],'native_parity_passed':True,'delivered_members_verified':33,'historical_tool_and_workflow_pins_verified':42,'probabilities_compared':57600,'all_probabilities_bit_exact':True,'raw_turns_compared':32,'all_raw_turns_exact':True,'three_strict_private_library_audits':92,'helper_payloads_independently_rehashed':True,'failed_model_attempts':101,'warm_rtfs':[q['comparisons'][n]['warm_rtf'] for n in ['native-12s','native-60s']],'release_qualified':False,'limits':['Actual M1 virtual CPU execution of the immutable M1-built owned-pool candidate. Exactly two fresh C ABI audio processes; no neural retry.','Descriptive shared-VM warm RTF1.86610/2.01168 still exceeds1. This is no uncontended live performance or matched speedup trial.','No public Rust/lifecycle/long/mixed/quality/attribution/desktop/app/signing/notary/release qualification. Opaque helper binary archive remains outsideGit with exact member/ZIP/archive descriptors.']},indent=2)+'\n')
files={path.relative_to(out).as_posix():descriptor(path) for path in out.rglob('*') if path.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,'members',len(files)+1)
