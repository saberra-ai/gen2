"""Preserve failed long-session attempt and replay only its completed30m pair."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
src=cache/'valid-mel-metal-long-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-long-failure'
def read(p):return json.loads(p.read_text())
def rec(p):
 d=p.read_bytes();return {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
p=read(src/'protocol.json');commit='7b9dba16692bdfa432f0c8e300858a679f230830'
assert rec(src/'protocol.json')['sha256']=='1e6a56ce36abe76bd00df034b2f6ad63e156a2aee63105880dca6b52852d4f67'
assert subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/valid-mel-metal-long.protocol.json'],cwd=repo)==(src/'protocol.json').read_bytes()
for name,expected in p['tools'].items():assert rec(tools/name)==expected
for name,expected in p['inputs'].items():assert rec(Path(name))==expected
commands=read(src/'commands.json')
assert [c['name']for c in commands]==['30m-native','30m-public','60m-native']
assert [c.get('returncode')for c in commands]==[0,0,None]
assert not (src/'result.json').exists()and not(src/'60m-native.json').exists()and not(src/'60m-native.json.f32').exists()
assert not list(src.glob('60m-public*'))
assert 'ios_base::clear: unspecified iostream_category error'in(src/'60m-native.stderr.log').read_text()
outer=read(Path(str(src)+'.qualification.command.json'));assert outer['source_commit']==commit and 'returncode'not in outer
for case in p['preflights']:
 prefix=str(src)+'.'+case['name'];assert read(Path(prefix+'.command.json'))['returncode']==0
 text=Path(prefix+'.stderr.log').read_text();assert 'Ran '+str(case['tests'])+' tests'in text and '\nOK\n'in text
sys.path.insert(0,str(tools))
from valid_mel_long_metrics import audit_case
from macos_loaded_libraries import audit_loaded_libraries
public=read(src/'30m-public.json');native=read(src/'30m-native.json')
comparison=audit_case(public,native,(src/'30m-native.json.f32').read_bytes(),(src/'30m-public.f32').read_bytes(),(src/'30m-native.stderr.log').read_text(),(src/'30m-public.stderr.log').read_text(),p['cases'][0],backend='metal',device='Apple M4 Pro')
assert comparison==read(src/'30m-comparison.json')
work=Path(p['work_path']);manifest=read(work/'nemo/package.json')
loaded={c['name']:audit_loaded_libraries((src/(c['name']+'.stderr.log')).read_text(),work/'nemo',manifest,c['argv'][7])for c in commands}
assert all(v['private_library_count']==93 for v in loaded.values())
assert not out.exists();shutil.copytree(src,out)
for f in src.parent.glob(src.name+'.*'):
 if f.is_file():shutil.copy2(f,out/('outer'+f.name.removeprefix(src.name)))
for name in ['run-valid-mel-metal-long.py','freeze-valid-mel-metal-long.py','collect-valid-mel-metal-long-failure.py','incremental-cleanup-metal-long.json']:shutil.copy2(cache/name,out/name)
for name in p['tools']:
 target=out/'execution-tools'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/'+name],cwd=repo))
q={'schema_version':1,'source_commit':commit,'protocol_sha256':rec(src/'protocol.json')['sha256'],'fixed_metal_30m_1h_compaction_parity_passed':False,'completed_30m_pair_passed':True,'completed_30m_comparison':comparison,'one_hour_qualified':False,'one_hour_native_command_started':True,'one_hour_native_terminal_returncode_recorded':False,'one_hour_native_terminal_arrays_available':False,'one_hour_public_command_started':False,'native_runtime_process_traces_observed':3,'private_images_per_trace':93,'loaded_image_audits':loaded,'model_free_preflight_tests_passed':31,'failure_observations':{'launcher_terminal_unified_exec_session':83314,'launcher_terminal_exit_code':1,'reported_launcher_error':'OSError: [Errno 28] No space left on device while writing valid-mel-metal-long-v1.qualification.command.json','observation_source':'Original unified exec terminal tool response; not a reconstructed child return code','native_stderr_error':'ios_base::clear: unspecified iostream_category error','diagnostic_limit':'Contemporaneous ENOSPC and native output persistence error support a storage interruption; missing terminal data prevents one-hour correctness or successful NN-completion claims. Child/controller return codes were not persisted and are not inferred.','disk_recovery':'Removed15oldGen2debugincrementalcompiler-cache directories,4665536791bytes, with no active cargo/rustc. All native/helper executables, source inputs, weights, audio and evidence preserved.'},'original_attempt_records_unchanged':True,'original_quality_gate_changed':False,'release_qualified':False,'limits':p['limits']+['The attempt failed. Only the completed30m native/public pair is accepted; no1h parity, completion or all-Metal-long-pass claim. A separate preregistered bounded continuation is required for additional NN work. Disk cleanup overlapped this attempt; timings are ordinary background observations, not uncontended benchmarks.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={str(f.relative_to(out)):rec(f)for f in sorted(out.rglob('*'))if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'members':len(files),'30m_passed':True,'1h_qualified':False,'attempt_failed':True,'private_image_records':279}))
