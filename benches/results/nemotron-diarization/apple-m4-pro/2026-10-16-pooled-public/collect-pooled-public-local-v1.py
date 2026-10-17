"""Collect saved public Rust qualification; never execute native/model helpers."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools));cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'pooled-public-local-v1'
from build_valid_mel_candidate import descriptor
from compare_stream_pooled import compare
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from sdk_archive import inventory,restore_archive
source='f72176c51f39b595b8ac9ea243a46a9390c24f13'
q=json.loads((root/'qualification.json').read_text());p=json.loads((root/'protocol.json').read_text())
assert q['source_commit']==source and q['pooled_public_runtime_passed'] and q['release_qualified'] is False
assert q['model_execution_admitted'] and q['lifecycle_tests_passed']==p['lifecycle_tests'] and len(p['lifecycle_tests'])==7
expected=['compile-controls','rustc-version','load-helper-build','lifecycle-build','public-helper-build','runtime-controls','lifecycle-controls','lifecycle-reference-readable','version-only','missing-library','lifecycle-list','cpu-rejects-metal','public']+['lifecycle-'+str(i) for i in range(7)]
assert [c['name'] for c in q['commands']]==expected
assert all(c['returncode']==c['expected_returncode']==(1 if c['name'] in ['missing-library','cpu-rejects-metal'] else 0) for c in q['commands'])
assert q['inference_commands_started']==['cpu-rejects-metal','public','lifecycle-0','lifecycle-1','lifecycle-2','lifecycle-3','lifecycle-4','lifecycle-6']
assert descriptor(root/'protocol.json')==q['protocol']
assert (root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-pooled-public.protocol.json'],cwd=repo)
pins={'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()};pins.update(p['repository_files']);verify_git_sources(repo,source,pins)
for stem,destination in [('pooled-cpu-sdk',root/'sdk'),('pooled-cpu-runtime',root/'work/nemo')]:
 m=json.loads((root/'producer'/(stem+'-files.json')).read_text());assert inventory(destination)==m['files']
helpers=json.loads((root/'pooled-public-helpers-files.json').read_text());assert inventory(root/'work/bin')==helpers['files']
assert descriptor(root/helpers['archive'])=={'bytes':helpers['archive_bytes'],'sha256':helpers['archive_sha256']}
restore_archive(root/helpers['archive'],helpers,root/'independent-helper-restore')
assert inventory(root/'independent-helper-restore')==helpers['files']
manifest=verify(root/'work/nemo');full=0;without=0
for name,audit in q['loaded_images'].items():
 exe=root/'work/bin'/('load-probe' if name=='version-only' else 'public-stream' if name in ['cpu-rejects-metal','public'] else 'native-lifecycle')
 entire=name not in ['lifecycle-list','lifecycle-5']
 assert audit_loaded_libraries((root/(name+'.stderr.log')).read_text(),root/'work/nemo',manifest,exe,entire)==audit
 assert audit['private_library_count']==(92 if entire else 0)
 full+=entire;without+=not entire
assert full==9 and without==2
native_path=root/'native-reference/results/native-60s.json';native=json.loads(native_path.read_text());public=json.loads((root/'results/public.json').read_text())
replay=compare(public,native,Path(str(native_path)+'.f32').read_bytes(),(root/'results/public.f32').read_bytes());assert replay==q['comparison'] and replay['retained_rows']==6000
assert public['input_pcm_f32le_sha256']==native['input_pcm_f32le_sha256']==p['pcm_sha256'] and public['input_repeats']==native['input_repeats']==1
assert (root/'reference/ami.json.f32').read_bytes()==Path(str(native_path)+'.f32').read_bytes()
assert json.loads((root/'reference/ami.json').read_text())=={'segments':[{'start':t['start_seconds'],'end':t['end_seconds'],'speaker':t['speaker']} for t in native['turns']]}
for i,name in enumerate(p['lifecycle_tests']):
 text=(root/('lifecycle-'+str(i)+'.stdout.log')).read_text();assert re.search(r'^test '+re.escape(name)+r' \.\.\. ok$',text,re.MULTILINE) and '1 passed; 0 failed; 0 ignored;' in text
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes']>=1024**3
# Preserve raw logs and the public/reference output, excluding all weights,
# SDK/runtime/archive payloads and Cargo targets from Git.
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-public';assert not out.exists();out.mkdir()
def retain(path,relative):
 destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination);assert descriptor(path)==descriptor(destination)
for path in root.iterdir():
 if path.is_file() and path.suffix in {'.json','.log','.sb'}:retain(path,path.name)
for name in ['helper-source','results','reference']:
 for path in (root/name).rglob('*'):
  if path.is_file():retain(path,path.relative_to(root))
for suffix in ['', '.f32']:retain(Path(str(native_path)+suffix),'native-reference/native-60s.json'+suffix)
retain(cache/'platform-ci/model-free-224-pooled-public-local.log','model-free-224.log')
retain(cache/'pooled-public-focused-preflight.log','focused-preflight.log')
retain(Path(__file__),Path(__file__).name)
record={'schema_version':1,'source_commit':source,'pooled_public_runtime_passed':True,'historical_source_pins_verified':len(pins),'public_frames_compared':6000,'public_probabilities_compared':48000,'public_and_native_probabilities_bit_exact':replay['probabilities_bit_exact'],'turns_compared':replay['turns'],'maximum_turn_error_seconds':replay['maximum_turn_error_seconds'],'seven_real_lifecycle_tests_passed':p['lifecycle_tests'],'actual_model_reloads_in_existing_lifecycle_test':100,'strict_private_library_audits':{'full92':full,'no_private_runtime':without},'helper_archive_payloads_independently_restored_and_rehashed':True,'sdk_package_helper_inventories_unchanged':True,'release_qualified':False,'limits':['Actual M4 CPU execution of the current Rust consumer with the immutable M1-built candidate. Native60s reference is reused from a separately successful M1 C ABI delivery; no redundant native neural run.','Exactly20commands with expected failure statuses retained for missing-private-library and CPU-Metal rejection. Eight inference-admitted commands include the CPU-Metal rejection; no exact neural-forward count is claimed.','All7unchanged lifecycle tests run once, including100reloads, queue/cancellation/drop, short-tail/idle, unevenPCM/reset/admission and offline cache. No inference retry.','Timing/RSS are descriptive and not matched speedup or uncontended live qualification. M1sharedVM native RTFacross1remainsfailed.','No new long-session/Metal/other-platform/mixed/quality/word-attribution/app/GUI/physicalmicrophone/freshcompleteapp/signing/notary/release gate. No default profile promotion. Opaque helper binaries remain outsideGit with exact archive/member descriptors.']}
(out/'collection-qualification.json').write_text(json.dumps(record,indent=2)+'\n');files={path.relative_to(out).as_posix():descriptor(path) for path in out.rglob('*') if path.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');assert all(descriptor(out/n)==v for n,v in files.items());print(out,'members',len(files)+1,'comparison',replay)
