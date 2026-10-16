from pathlib import Path
import hashlib,json,shutil,sys,subprocess
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'quality qualification 125f907'
dest=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-frozen-corpus'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
import qualify_corpus as q
q.validate_protocol(repo/'tools/nemotron-diarization/corpus.protocol.json',cache/'quality-corpus-v1')
manifest=q.validate_corpus(cache/'quality-corpus-v1')
s=json.loads((root/'summary.json').read_text())
assert len(s['results'])==16 and not s['all_slice_gates_passed']
assert {(x['slice'],x['backend']) for x in s['results']}=={(x['id'],b) for x in manifest['slices'] for b in ['cpu','metal']}
for x in s['results']:
 for p,h in x['sha256'].items():assert sha(Path(p))==h,p
 assert x['wrapper_parity']['maximum_probability_error']==0
 assert x['wrapper_parity']['maximum_turn_error_seconds']==0
 assert x['gates']['q8_degradation_at_most_0_02']
 assert x['all_slice_gates_passed']==all(x['gates'].values())
 assert json.loads((root/(x['slice']+'-'+x['backend']+'-score.json')).read_text())==x
for item in manifest['slices']:
 q.validate_oracle_environment(json.loads((root/(item['id']+'-fp32.json')).read_text()))
dest.mkdir()
raw={}; artifacts={}
for p in sorted(root.iterdir()):
 if not p.is_file():continue
 if p.suffix in ('.json','.log','.sb','.py'):
  shutil.copy2(p,dest/p.name);artifacts[p.name]=sha(p)
 else:raw[p.name]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
shutil.copy2(__file__,dest/'collect.py');artifacts['collect.py']=sha(dest/'collect.py')
ci=subprocess.check_output(['gh','run','view','38860761229','--repo','saberra-ai/gen2','--json','headSha,status,conclusion,jobs'])
ci_record=json.loads(ci)
assert ci_record['conclusion']=='success' and len(ci_record['jobs'])==13 and all(j['conclusion']=='success' for j in ci_record['jobs'])
(dest/'feature-ci.json').write_bytes(ci);artifacts['feature-ci.json']=sha(dest/'feature-ci.json')
receipt={'schema_version':1,'release_qualified':False,'scope':s['scope'],'source_commit':s['source_commit'],
 'frozen_protocol_sha256':s['protocol_sha256'],'corpus_manifest_sha256':sha(cache/'quality-corpus-v1/manifest.json'),
 'audio_slices':8,'distinct_recordings':7,'cpu_metal_comparisons':16,'all_wrapper_comparisons_exact':True,
 'all_q8_degradation_bounds_passed':True,'all_slice_quality_gates_passed':False,
 'passed_slice_backend_pairs':sum(x['all_slice_gates_passed'] for x in s['results']),
 'failed_slice_backend_pairs':[{'slice':x['slice'],'backend':x['backend'],'gates':x['gates']} for x in s['results'] if not x['all_slice_gates_passed']],
 'per_slice_metrics':[{'slice':x['slice'],'backend':x['backend'],'fp32_der':x['fp32']['der'],'q8_shared_der':x['q8_shared_threshold']['der'],'q8_production_der':x['q8_production_default']['der'],'q8_minus_fp32':x['q8_minus_fp32_absolute_der'],'all_gates_passed':x['all_slice_gates_passed']} for x in s['results']],
 'execution':'Sequential fresh native CPU/Metal and original FP32 processes with network denied; no model-output selection, threshold/profile tuning or reference replacement. Fixed 9/4 low-latency geometry. All failed outcomes retained.',
 'timing_scope':'Quality and wrapper parity only. Editing, artifact collection, CI inspection and OS background activity occurred; no clean throughput or capture/UI-delay claim.',
 'reference_caveat':'AMI transcriber segment protocol differs from NVIDIA published forced-alignment RTTMs. Frozen failures remain authoritative under this protocol; no direct reproduction-discrepancy claim. See docs/research/nemotron-quality-gaps.md.',
 'limitations':['Vox and ICSI are reported in checkpoint training inventory; local tuning-heldout does not imply checkpoint-unseen.', 'Two AMI channels share one meeting and both fail 20% gates.', 'ICSI native production default is close to the 20% boundary; no generalization margin claimed.', 'Full release, broader Bark attribution and platform/deployment gates remain open.'],
 'feature_ci':{'run_id':38860761229,'head_sha':ci_record['headSha'],'all_13_jobs_passed':True},'artifacts':artifacts,'raw_artifacts':raw}
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'comparisons':16,'passed':receipt['passed_slice_backend_pairs'],'failed':receipt['failed_slice_backend_pairs'],'dest':str(dest)},indent=2))
