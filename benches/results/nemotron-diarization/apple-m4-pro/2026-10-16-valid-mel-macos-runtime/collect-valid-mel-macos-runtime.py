"""Replay saved packaged CPU output and dyld traces; never execute native code."""
import hashlib,json,pathlib,re,shutil,subprocess,sys,wave
import numpy as np
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');src=cache/'valid-mel-macos-package-qualification-v2';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-macos-runtime';tools=repo/'tools/nemotron-diarization'
def rec(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
p=json.loads((src/'protocol.json').read_text());r=json.loads((src/'result.json').read_text());source=r['source_commit'];assert source==subprocess.check_output(['git','rev-parse','52d9346'],cwd=repo,text=True).strip()
assert rec(src/'protocol.json')['sha256']==r['protocol_sha256']=='c599e4130ebcd2d5238646b28de8497df31157f6d32a9d5e1521f83f09070183'
assert subprocess.check_output(['git','show',source+':tools/nemotron-diarization/valid-mel-macos-package-v2.protocol.json'],cwd=repo)==(src/'protocol.json').read_bytes()
for records,root,prefix in [(p['tools'],tools,'tools/nemotron-diarization/'),(p['rust_sources'],repo,'')]:
 for name,expected in records.items():
  blob=subprocess.check_output(['git','show',source+':'+prefix+name],cwd=repo);assert rec(root/name)=={'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==expected
for name,expected in p['inputs'].items():assert rec(pathlib.Path(name))==expected
for name,expected in p['tree_files'].items():
 root=pathlib.Path(name);actual={str(q.relative_to(root)):{**rec(q),'symlink_target':str(q.readlink())if q.is_symlink()else None}for q in root.rglob('*')if q.is_file()};assert actual==expected
commands=json.loads((src/'commands.json').read_text());expected_names=['version-only','missing-library','native','public']+['lifecycle-'+str(i)for i in range(7)];assert [c['name']for c in commands]==expected_names
for c in commands:assert c['returncode']==(1 if c['name']=='missing-library'else 0)and not c.get('timed_out')and c['argv'][:4]==['/usr/bin/sandbox-exec','-f',str(src/'runtime.sb'),'/usr/bin/env']and c['argv'][4]=='DYLD_PRINT_LIBRARIES=1'
for phase in p['preflights']:
 name=phase['name'];c=json.loads(pathlib.Path(str(src)+'.'+name+'.command.json').read_text());log=pathlib.Path(str(src)+'.'+name+'.stderr.log').read_text();assert c['returncode']==0 and 'Ran '+str(phase['tests'])+' tests'in log and '\nOK\n'in log
outer=json.loads(pathlib.Path(str(src)+'.qualification.command.json').read_text());assert outer['returncode']==0 and outer['argv'][0]==str(cache/'evidence-tools-env/bin/python')
sys.path.insert(0,str(tools));from qualify_valid_mel_macos_package import compare_complete_60s;from macos_loaded_libraries import audit_loaded_libraries;from package_macos import verify_files,verify_package_provenance
work=pathlib.Path(p['work_path']);manifest=json.loads((work/'nemo/package.json').read_text());verify_files(work/'nemo',manifest['files']);verify_package_provenance(work/'nemo',manifest)
native=json.loads((src/'native.json').read_text());public=json.loads((src/'public.json').read_text());raw=(src/'native.json.f32').read_bytes();actual=(src/'public.f32').read_bytes();comparison=compare_complete_60s(public,native,raw,actual);assert comparison==r['comparison']==json.loads((src/'comparison.json').read_text())and raw==actual
with wave.open(str(work/'assets/ami_en2002d_2132.wav'),'rb')as wav:pcm=np.frombuffer(wav.readframes(960000),dtype='<i2').astype('<f4')/32768
assert hashlib.sha256(pcm.tobytes()).hexdigest()==p['pcm_sha256']==native['input_pcm_f32le_sha256']==public['input_pcm_f32le_sha256']
assert(src/'version-only.stdout.log').read_text().strip()==p['runtime_version']and not(src/'missing-library.stdout.log').stat().st_size
assert 'Library not loaded:'in(src/'missing-library.stderr.log').read_text()and pathlib.Path(p['missing_library_path']).name in(src/'missing-library.stderr.log').read_text()
loaded={}
for c in commands:
 name=c['name']
 if name=='missing-library':continue
 require=name!='lifecycle-5';loaded[name]=audit_loaded_libraries((src/(name+'.stderr.log')).read_text(),work/'nemo',manifest,c['argv'][5],require)
assert loaded==r['loaded_image_audits']==json.loads((src/'loaded-libraries.json').read_text())and len(loaded)==10 and sum(v['private_library_count']for v in loaded.values())==828
for i,test in enumerate(p['lifecycle_tests']):
 text=(src/('lifecycle-'+str(i)+'.stdout.log')).read_text();assert re.search(r'^test '+re.escape(test)+r' \.\.\. ok$',text,re.MULTILINE)and '1 passed; 0 failed; 0 ignored;'in text
assert r['lifecycle_tests_passed']==p['lifecycle_tests']and r['fixed_packaged_cpu_parity_and_lifecycle_passed']is True and r['release_qualified']is False
assert json.loads((src/'reference/ami.json').read_text())=={'segments':[{'start':s['start_seconds'],'end':s['end_seconds'],'speaker':s['speaker']}for s in native['turns']]}and(src/'reference/ami.json.f32').read_bytes()==raw
assert not out.exists();out.mkdir()
for q in src.rglob('*'):
 if q.is_file()and not q.is_relative_to(src/'tmp'):
  dest=out/q.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,dest)
for q in src.parent.glob(src.name+'.*'):
 if q.is_file():shutil.copy2(q,out/('outer'+q.name.removeprefix(src.name)))
for directory in ['valid-mel-macos-runtime-inputs-v1','valid-mel-macos-load-probe-v1','valid-mel-macos-loader-preflight-v1']:
 root=cache/directory;dest=out/'preparation'/directory;dest.mkdir(parents=True)
 for q in root.iterdir():
  if q.is_file()and q.name!='load-probe':shutil.copy2(q,dest/q.name)
shutil.copy2(cache/'run-valid-mel-macos-package-v2.py',out/'run-valid-mel-macos-package-v2.py');shutil.copy2(__file__,out/pathlib.Path(__file__).name)
q={'source_commit':source,'scope':'Same-host relocated macOS arm64 CPU model/loader/public-native/lifecycle qualification only. Not supported desktop release.','protocol_sha256':r['protocol_sha256'],'fixed_packaged_cpu_parity_and_lifecycle_passed':True,'private_libraries_per_native_process':92,'full_runtime_processes_audited':9,'audited_private_image_records':828,'warm_provisioning_native_images':0,'missing_private_library_failed_before_model_creation':True,'all6000rows_native_public_byte_identical':True,'turns':comparison['turns'],'maximum_probability_error':comparison['maximum_probability_error'],'maximum_turn_error_seconds':comparison['maximum_turn_error_seconds'],'actual_normalized_pcm_sha256':p['pcm_sha256'],'lifecycle_tests_passed':p['lifecycle_tests'],'actual100reload_test_passed':True,'model_free_preflight_tests_passed':29,'qualification_elapsed_seconds':outer['elapsed_seconds'],'neural_inference_performed':True,'internal_neural_forward_counts_instrumented':False,'prior_zeroNN_sandbox_failure_retained':True,'original_quality_gate_changed':False,'release_qualified':False,'limits':p['limits'],'temporary_files_after_tests':{str(q.relative_to(src/'tmp')):rec(q)for q in(src/'tmp').rglob('*')if q.is_file()}}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={str(q.relative_to(out)):rec(q)for q in sorted(out.rglob('*'))if q.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'members':len(files),'rows':6000,'turns':comparison['turns'],'lifecycle_tests':7,'audited_private_image_records':828,'neural_inference_performed':True}))
