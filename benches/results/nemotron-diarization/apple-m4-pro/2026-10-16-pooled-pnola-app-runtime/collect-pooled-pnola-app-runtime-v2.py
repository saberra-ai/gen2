"""Replay saved complete pooled app/native results; never execute app/helpers/models."""
from pathlib import Path
import array,json,math,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from sdk_archive import inventory
from package_macos import verify
from pooled_pnola_metrics import native_reference,recorded_matches,live_matches
from macos_loaded_libraries import audit_loaded_libraries
from verify_valid_mel_pnola_app import audit_app_images
root=cache/'pooled-pnola-complete-app-runtime-v2';read=lambda p:json.loads(p.read_text());p=read(root/'protocol.json');q=read(root/'qualification.json');source='d3adaf379778bbb54dafbc7abe48fc9661bc2acf'
assert q['source_commit']==source and q['automatic_packaged_resolution_gates_passed']is True and'error'not in q and q['release_qualified']is False
assert q['microphone_used']is False and q['artifact_environment_overrides_used']is False
assert descriptor(root/'protocol.json')==q['protocol']and(root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-pooled-pnola-app-v2.protocol.json'],cwd=repo)
pins={'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()};pins.update(p['repository_files']);verify_git_sources(repo,source,pins)
work=Path(p['work_path']);app=Path(p['app_path']);assert inventory(work)==p['work_inventory']and inventory(app)==p['app_inventory']
for n,v in p['preparation_records'].items():assert descriptor(Path(n))==v
cases=['native-60s','native-80s','cpu-recorded','cpu-live','cpu-rejects-metal'];assert q['neural_commands_started']==cases and[c['name']for c in q['cases']]==cases
assert[c['name']for c in q['commands']]==['runtime-controls','premodel-controls','version-only',*cases]and all(c['returncode']==0 for c in q['commands'])
package=work/'nemo';manifest=verify(package);assert manifest==verify(app/'Contents/Resources/nemo')
verified=read(app.parents[1]/'cpu-static-verification.json');assert verified['app']==str(app)
images={};images['version-only']=audit_loaded_libraries((root/'version-only.stderr.log').read_text(),package,manifest,work/'bin/load-probe')
natives={}
for n,wav,duration in [('native-60s','ami.wav',60.),('native-80s','live-80s.wav',80.04)]:
 native=read(root/'results'/(n+'.json'));native_reference(native,duration);assert native['input_pcm_f32le_sha256']==p['pcm_sha256'][wav]
 raw=array.array('f');raw.frombytes((root/'results'/(n+'.json.f32')).read_bytes());assert len(raw)==round(duration*100)*8 and all(math.isfinite(v)and 0<=v<=1 for v in raw)
 images[n]=audit_loaded_libraries((root/(n+'.stderr.log')).read_text(),package,manifest,work/'bin/native-stream');natives[n]=native
comparisons={}
for n in cases[2:]:
 actual=read(root/'results'/(n+'.json'))
 comparisons[n]=(live_matches(actual,natives['native-80s'])if n=='cpu-live'else recorded_matches(actual,natives['native-60s'],p['bark_text'],p['wav_sha256'],unavailable=n=='cpu-rejects-metal'))
 assert comparisons[n]==next(c['comparison']for c in q['cases']if c['name']==n)
 images[n]=audit_app_images((root/(n+'.stderr.log')).read_text(),app,verified);assert images[n]['process_local_add_image_history_observed']
for n,v in images.items():assert v==q['loaded_images'][n]
assert set(images)==set(q['loaded_images'])and q['storage']['released']and q['storage']['minimum_observed_free_bytes']>=1024**3
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-pnola-app-runtime';assert not out.exists();out.mkdir()
for path in root.rglob('*'):
 if path.is_file()and path.suffix in {'.json','.log','.sb','.f32','.txt'}and not path.is_relative_to(root/'home')and not path.is_relative_to(root/'tmp'):
  dest=out/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
shutil.copy2(__file__,out/Path(__file__).name)
r={'schema_version':1,'automatic_packaged_resolution_gates_passed':True,'release_qualified':False,'controller_source_commit':source,'app_construction_source_commit':read(app.parents[1]/'build.json')['source_commit'],'source_pins_verified':len(pins),'independent_native_reference_frames':{n:v['native_frame_count']for n,v in natives.items()},'all_protected_app_and_work_entries_rehashed':True,'actual_commands':8,'fresh_neural_processes':5,'comparisons_replayed':comparisons,'three_full_private_native_image_audits':True,'three_hardened_app_callback_image_audits':True,'new_native_or_app_or_neural_calls_during_collection':0,'limits':['Actual same-host ad-hoc complete CPU app automatic packaged recording/live/recovery/export and explicit Metal rejection only; no artifact environment path overrides.','Native references expose full frame/probability coverage. App live timeline is compact and its probability rows/native frame count are not observed; turn comparison against the exact independent80.04s CABI reference stays1e-6.','No physical microphone, GUI, fresh-machine, held-out accuracy, human word gold, uncontended performance, Developer ID/notarization or release claim. Private app binary/source/weights remain local.']}
(out/'collection-qualification.json').write_text(json.dumps(r,indent=2)+'\n');files={x.relative_to(out).as_posix():descriptor(x)for x in out.rglob('*')if x.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,len(files)+1,'members',comparisons)
