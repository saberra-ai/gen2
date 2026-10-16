"""Retain failedsandbox admission with zero native target/model execution."""
import hashlib,json,pathlib,shutil,subprocess
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');src=cache/'valid-mel-macos-package-qualification-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-macos-sandbox-failure';source='0aa0b6926af40175c3a03f855cf0e52c43bc2747'
source=subprocess.check_output(['git','rev-parse','0aa0b69'],cwd=repo,text=True).strip()
def rec(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
p=json.loads((src/'protocol.json').read_text());assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-macos-package.protocol.json'],cwd=repo)==(src/'protocol.json').read_bytes()
commands=json.loads((src/'commands.json').read_text());assert len(commands)==1 and commands[0]['name']=='version-only'and commands[0]['returncode']==71
assert(src/'version-only.stderr.log').read_text()=='sandbox-exec: sandbox_apply: Operation not permitted\n'and(src/'version-only.stdout.log').stat().st_size==0
assert {str(f.relative_to(src))for f in src.rglob('*')if f.is_file()}=={'commands.json','protocol.json','runtime.sb','version-only.stdout.log','version-only.stderr.log'}
for phase in p['preflights']:
 n=phase['name'];j=json.loads(pathlib.Path(str(src)+'.'+n+'.command.json').read_text());log=pathlib.Path(str(src)+'.'+n+'.stderr.log').read_text();assert j['returncode']==0 and 'Ran '+str(phase['tests'])+' tests'in log and '\nOK\n'in log
assert json.loads(pathlib.Path(str(src)+'.qualification.command.json').read_text())['returncode']==1
assert not out.exists();out.mkdir()
for f in src.iterdir():
 if f.is_file():shutil.copy2(f,out/f.name)
for f in src.parent.glob(src.name+'.*'):
 if f.is_file():shutil.copy2(f,out/('outer'+f.name.removeprefix(src.name)))
for f in(cache/'valid-mel-macos-sandbox-diagnosis-v1').iterdir():
 if f.is_file():d=out/'diagnosis'/f.name;d.parent.mkdir(exist_ok=True);shutil.copy2(f,d)
for n in ['qualify_valid_mel_macos_package.py']:
 (out/n).write_bytes(subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+n],cwd=repo))
shutil.copy2(cache/'run-valid-mel-macos-package.py',out/'run-valid-mel-macos-package.py');shutil.copy2(__file__,out/pathlib.Path(__file__).name)
q={'source_commit':source,'protocol_sha256':rec(src/'protocol.json')['sha256'],'preflight_tests_passed':29,'outer_qualification_returncode':1,'first_native_command_returncode':71,'failure':'sandbox_apply: Operation not permitted, before target exec','native_target_started':False,'native_runtime_loaded':False,'models_instantiated':0,'neural_calls':0,'criteria_changed':False,'release_qualified':False,'model_free_diagnosis':{'exact_nested_profile_python_controller_true_returncode':71,'same_inner_profile_single_true_returncode':0},'limits':['Failed admission preserved unchanged. No version probe, missing-library, speech or lifecycle target ran.','Separate source-pinned controller without its outer sandbox may retain the exact strict sandbox on every nativechild and model-free preflight. It requires a new protocol/output attempt before execution; no numerical or release gate changes.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={str(f.relative_to(out)):rec(f)for f in sorted(out.rglob('*'))if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'members':len(files),'neural_calls':0,'native_target_started':False}))
