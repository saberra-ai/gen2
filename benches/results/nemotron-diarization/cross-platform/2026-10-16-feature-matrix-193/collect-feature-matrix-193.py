"""Retain exact current-pushed-branch feature matrix, without replacing old evidence."""
import hashlib,json,pathlib,shutil
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci');out=repo/'benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-193';ci=json.loads((cache/'feature-38967939643.json').read_text())
def record(p):return{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert ci['headSha']=='e1c219a56782eabf7cf612935550871c62561c5d'and ci['databaseId']==38967939643 and ci['status']=='completed'and ci['conclusion']=='success'and len(ci['jobs'])==13
assert all(j['status']=='completed'and j['conclusion']=='success'for j in ci['jobs'])
platforms=['ubuntu-latest','windows-latest','macos-latest'];features=['','backend-diar-nemo','asr-dynamic,backend-diar-nemo','backend-llamacpp,asr-dynamic,backend-diar-nemo']
assert {j['name']for j in ci['jobs']}=={'evidence-tools'}|{f'features ({os}, {flags})'if flags else f'features ({os})'for os in platforms for flags in features}
for j in ci['jobs']:
 if j['name']=='evidence-tools':
  names=['Run python -m unittest discover -s tools/nemotron-diarization','Run python -m unittest discover -s integrations/pnola','Run node --test integrations/pnola/overlay/tests/desktop-assets.test.mjs']
 else:names=['Run cargo fmt','Run cargo clippy','Run cargo test']
 for name in names:
  matches=[s for s in j['steps']if s['name'].startswith(name)];assert len(matches)==1 and matches[0]['conclusion']=='success'
log=(cache/'feature-38967939643.log').read_text();assert 'Ran 193 tests'in log
assert not out.exists();out.mkdir()
shutil.copy2(cache/'feature-38967939643.json',out/'ci-success.json');shutil.copy2(__file__,out/pathlib.Path(__file__).name)
q={'schema_version':1,'source_commit':ci['headSha'],'ci_url':ci['url'],'all_thirteen_feature_jobs_passed':True,'model_free_evidence_tests_passed':193,'platforms':platforms,'feature_sets':features,'all_twelve_feature_jobs_ran_fmt_clippy_tests_successfully':True,'evidence_and_pnola_python_and_desktop_asset_test_steps_passed':True,'combined_log_retained_outside_git':record(cache/'feature-38967939643.log'),'release_qualified':False,'limits':['CI source is e1c219a. This proves its12OS/feature combinations and193model-free evidence tests, including both observed CTest output formats, portable Mac controls and existing current-platform/original-NeMo evidence tools. It does not execute the app/model/corpus gates.','No model/nativeSDK is required by this workflow. It does not replace real-model, desktop, distribution, held-out-quality or uncontended-performance gates. Earlier network-failure and102-test success receipts remain unchanged.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={p.relative_to(out).as_posix():record(p)for p in out.rglob('*')if p.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'files':len(files),'all_jobs':13,'evidence_tests':193}))
