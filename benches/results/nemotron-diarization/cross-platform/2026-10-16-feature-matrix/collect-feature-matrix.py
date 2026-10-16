"""Retain original network-only checkout failure beside full subsequent success."""
import hashlib,json,shutil
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci');output=repo/'benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix'
read=lambda p:json.loads(p.read_text());record=lambda p:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
ci=read(cache/'feature-38903243271.json');old=read(cache/'feature-38901164547-attempt1.json')
assert ci['conclusion']=='success'and ci['headSha']=='ec88564ec7f7b62e92b73156481630413b9928a9'and len(ci['jobs'])==13 and all(j['conclusion']=='success'for j in ci['jobs'])
assert sum(j['conclusion']=='success'for j in old['jobs'])==12 and sum(j['conclusion']=='failure'for j in old['jobs'])==1
failure,=[j for j in old['jobs']if j['conclusion']=='failure'];assert failure['name']=='features (macos-latest, backend-llamacpp,asr-dynamic,backend-diar-nemo)'
log=(cache/'feature-38901164547-macos-mixed.log').read_text();assert 'Failed to connect to github.com port 443'in log
assert all(step['conclusion']=='skipped'for step in failure['steps']if step['name'].startswith('Run cargo'))
full=(cache/'feature-38903243271.log').read_text();assert 'Ran 102 tests'in full
output.mkdir(parents=True,exist_ok=False)
for name,source in [('ci-success.json','feature-38903243271.json'),('ci-original-fetch-failure.json','feature-38901164547-attempt1.json'),('original-macos-fetch-failure.log','feature-38901164547-macos-mixed.log')]:shutil.copy2(cache/source,output/name)
qualification={'schema_version':1,'source_commit':ci['headSha'],'ci_url':ci['url'],'all_thirteen_feature_jobs_passed':True,'model_free_evidence_tests_passed':102,'platforms':['ubuntu-latest','windows-latest','macos-latest'],'feature_sets':['','backend-diar-nemo','asr-dynamic,backend-diar-nemo','backend-llamacpp,asr-dynamic,backend-diar-nemo'],'original_failed_run':old['url'],'original_failed_source':old['headSha'],'original_run_twelve_jobs_passed':True,'original_macos_checkout_failed_before_compilation_due_to_github_connectivity':True,'failed_job_rerun_attempt':{'exit_code':1,'reason':'Resource not accessible by personal access token'},'successful_control':'Fresh authorized working-branch source push triggered the full matrix; the formerly blocked macOS mixed job and every other job passed.','combined_success_log_retained_outside_git':record(cache/'feature-38903243271.log'),'release_qualified':False,'limits':['Feature compatibility and model-free tests only. Does not replace actual model/native platform receipts, held-out-quality, desktop, distribution or performance gates.','CI source is ec88564. Subsequent committed changes are protocol/artifact evidence and documentation; no completion or main push is inferred from this matrix.']}
(output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
shutil.copy2(__file__,output/'collect-feature-matrix.py')
files={p.relative_to(output).as_posix():record(p)for p in output.rglob('*')if p.is_file()};(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'evidence_files':len(files),'all_feature_jobs_passed':13,'model_free_tests':102}))
