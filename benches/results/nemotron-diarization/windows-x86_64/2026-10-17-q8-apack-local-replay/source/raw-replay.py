"""Recheck returned raw evidence using unchanged frozen admission AST/metrics."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path,PureWindowsPath
import re
import sys

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
pre_path=cache/'q8-apack-returned-replay-pre-execution-v2.json'
pre=json.loads(pre_path.read_bytes())
# Bootstrap source admission uses only stdlib before the project helper import.
def bootstrap_descriptor(path):
    h=hashlib.sha256();size=0
    with path.open('rb')as f:
        while block:=f.read(1024**2):size+=len(block);h.update(block)
    return {'bytes':size,'sha256':h.hexdigest()}
for name,wanted in pre['files'].items():assert bootstrap_descriptor(Path(name))==wanted,name
assert str(tools/'remote_q8_apack_replay.py')in pre['files']
sys.path.insert(0,str(tools))
from remote_q8_apack_replay import descriptor,verify_collection,RESULTS,PARENT,SOURCE,RUN,JOB,ARCHIVE
local=cache/'q8-apack-local-bundle-admission-v1'
admission=json.loads((local/'admission.json').read_bytes())
assert admission['remote_source_commit']=='cc655ab95f2bb800696f1f5c9a27d69321627b27'
assert admission['remote_run_id']==39080172371 and admission['remote_job_id']==117535343984
assert admission['remote_result_commit']=='ba5cce755ab7c4673892ccfc9b0bf4992e6fc26f'
assert admission['archive']=={'bytes':17392754,'sha256':'0a31a38d6e136bdf48a66b7b106745ab1c175e601e6f44b2af438cbe207bcd16'}
materialized=local/'materialized'
collections={name:verify_collection(materialized/name)for name in RESULTS}
assert collections==admission['collections']
delivered=materialized/RESULTS[0];static=materialized/RESULTS[1]
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
p=read(delivered/'protocol.json')
assert descriptor(delivered/'protocol.json')=={'bytes':885307,'sha256':'329e0afea3c8e4f05d6158e58e801519a0dfb3d76fa0e1b27afa9d00eb7e5c2f'}
assert (delivered/'protocol.json').read_bytes()==(tools/'windows-q8-apack.protocol.json').read_bytes()
for group,base in [('tools',tools),('repository_files',repo)]:
    for name,wanted in p[group].items():assert descriptor(base/name)==wanted,name
for name,wanted in p['tools'].items():assert descriptor(delivered/'sources'/name)==wanted,name

# Imports occur only after complete frozen local source admission. No returned
# Python source, executable or object is executed.
from windows_q8_apack_metrics import ORDER,SHAPES,GUARD_CODES,admit_guard,admit_half_table,validate_output,timing_evidence
from q8_apack_metrics import admit_packing

q=read(delivered/'qualification.json')
assert q['source_commit']==SOURCE and q['diagnostic_completed']is True and len(q['cases'])==12
assert all(q[name]is False for name in ['release_qualified','package_qualified','public_or_lifecycle_qualified','sdk_or_model_qualified'])
members=read(delivered/'receiver/delivery-artifact-members.json')
delivery=read(delivered/'receiver/delivery-qualification.json')
assert delivery['archive']==ARCHIVE and delivery['source_commit']==SOURCE and delivery['run_id']==RUN and delivery['job_id']==JOB
assert descriptor(delivered/'receiver/delivery-artifact-members.json')==delivery['artifact_members']
for row in q['commands']:
    for suffix in ('.stdout.log','.log'):
        name='evidence/'+row['name']+suffix
        assert descriptor(delivered/name)==members[name],name

def execute(nodes,env,label):
    tree=ast.Module(body=nodes,type_ignores=[])
    identity=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    exec(compile(tree,label,'exec'),env)
    return identity

supervision_source=repo/PARENT/'prepared/replay-windows-q8-apack-supervision-v1.py'
supervision_body=ast.parse(supervision_source.read_bytes()).body
start=next(i for i,n in enumerate(supervision_body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='summaries'for t in n.targets))
stop=next(i for i,n in enumerate(supervision_body[start:],start)if isinstance(n,ast.Assert)and isinstance(n.test,ast.Compare)
    and ast.unparse(n.test.left)=='descriptor(Path(__file__))')
scope={**globals()}
supervision_ast=execute(supervision_body[start:stop],scope,'unchanged-original-supervision-AST')
assert len(scope['summaries'])==95 and all(row['normal_exit_and_cleanup_admitted']for row in scope['summaries'])
assert scope['summaries']==read(delivered/'receiver/supervision-replay-v1.json')['commands']

collector_source=repo/PARENT/'prepared/collect-windows-q8-apack-v1.py'
collector_body=ast.parse(collector_source.read_bytes()).body
helpers=[n for n in collector_body if isinstance(n,ast.FunctionDef)and n.name in ('admit_kernel_command','admit_packing_case')]
assert len(helpers)==2
block,=[n for n in collector_body if isinstance(n,ast.If)and ast.unparse(n.test)=="q['diagnostic_completed'] is True"
    and any(isinstance(child,ast.For)and ast.unparse(child.iter)=="enumerate(q['cases'])"for child in n.body)]
start=next(i for i,n in enumerate(block.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='reference'for t in n.targets))
assert isinstance(block.body[-1],ast.Expr)and ast.unparse(block.body[-1].value.func)=='result.update'
scope={**globals(),'evidence':delivered/'evidence',
    'ci_evidence':PureWindowsPath(q['build']['kernel_translation_units']['control']['source']).parent}
numerical_ast=execute(helpers+block.body[start:-1],scope,'unchanged-original-vector-half-packing-timing-AST')
timing=scope['timing']
assert timing==read(delivered/'receiver/collection-replay-v1.json')['timing_comparison']
assert timing==read(delivered/'collection-qualification.json')['timing_comparison']

# Reuse the original function selector on both actual saved disassemblies.
inspector=repo/PARENT/'prepared/inspect-q8-row-loops-v1.py'
inspect_node,=[n for n in ast.parse(inspector.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='inspect_loop']
scope={**globals()};static_ast=execute([inspect_node],scope,'unchanged-original-loop-selector-AST')
functions={};negative_controls=0
for role,method in [('control','gemmMx4<2>'),('apack','gemmPackA<2>')]:
    for name,path in [('kernel-'+role+'.obj',static/'objects'/('kernel-'+role+'.obj')),
            ('disasm-'+role+'.stdout.log',static/'disassembly'/('disasm-'+role+'.stdout.log'))]:
        assert descriptor(path)==members['evidence/'+name]
    text=(static/'disassembly'/('disasm-'+role+'.stdout.log')).read_text()
    functions[role]=scope['inspect_loop'](text,method,8)
    for wrong_method,wrong_fmas in [(method,4),(method+'missing',8)]:
        try:scope['inspect_loop'](text,wrong_method,wrong_fmas)
        except ValueError:negative_controls+=1
        else:raise AssertionError('wrong function/FMA selector accepted')
assert functions==read(static/'inspection.json')['functions']and negative_controls==4
feature_run=read(static/'feature-ci/run.json');feature_jobs=read(static/'feature-ci/jobs.json')
assert feature_run['id']==39077372958 and feature_run['head_sha']==SOURCE and feature_run['status']=='completed'and feature_run['conclusion']=='success'
assert feature_jobs['total_count']==len(feature_jobs['jobs'])==13
assert all(row['run_id']==feature_run['id']and row['head_sha']==SOURCE and row['status']=='completed'and row['conclusion']=='success'for row in feature_jobs['jobs'])

primary=set(timing['primary_shapes'])
four=[row for process in timing['processes'][-4:]for row in process['kernel_comparison']if row['shape']in primary]
observers=[row for row in timing['observer_comparison']if row['shape']in primary]
rows=[row for case in q['cases']for row in case['observed']['rows']]
intervals=[interval for row in rows for sample in row['worker_intervals_ns']for interval in sample]
result={'schema_version':1,'source_commit':SOURCE,'run_id':RUN,'job_id':JOB,'native_archive':ARCHIVE,
    'local_bundle_admission':descriptor(local/'admission.json'),'input_collections':collections,
    'current_frozen_tools_verified':len(p['tools']),'current_frozen_repository_references_verified':len(p['repository_files']),
    'retained_frozen_tools_verified':len(p['tools']),'commands_verified':95,
    'command_returncodes':dict(Counter(str(row['returncode'])for row in q['commands'])),
    'all_saved_native_command_cleanup_admitted':True,'full_vectors_verified':1056,'full_half_tables_verified':12,
    'full_packing_pairs_verified':12,'support_controls_verified':36,'raw_dispatch_timings':sum(len(row['timed_ns'])for row in rows),
    'worker_intervals':len(intervals),'zero_duration_worker_intervals':sum(v['end_ns']==v['start_ns']for v in intervals),
    'four_worker_primary':{'comparisons':len(four),'speed_passes':sum(r['both_candidates_at_least_5_percent_faster_than_both_controls']for r in four),
        'control_drift_passes':sum(r['control_drift_at_most_10_percent']for r in four),
        'candidate_repeat_drift_passes':sum(r['candidate_repeat_drift_at_most_10_percent']for r in four)},
    'primary_observer':{'comparisons':len(observers),'drift_passes':sum(r['control_drift_at_most_10_percent']for r in observers),
        'overhead_passes':sum(r['both_observers_at_most_5_percent_over_both_controls']for r in observers),
        'both_passes':sum(r['bounded_interpretation_supported']for r in observers)},
    'timing_comparison':timing,'static_functions':functions,'static_negative_controls':negative_controls,'producer_feature_jobs_verified':13,
    'unchanged_source_ast':{'supervision':supervision_ast,'numerical':numerical_ast,'static':static_ast},
    'new_compiler_kernel_sdk_model_calls':0,'production_or_release_qualified':False,
    'limits':'Local replay of complete retained raw vectors/half/packing/guards, saved95WindowsJob records, timing/observer and selected objects/disassemblies using unchanged original AST/metrics. Original full nativeZIP/canonical/pristine source delivery was independently admitted remotely and is not re-downloaded or rehashed locally. No causal CPU-time/traffic, full-model speed or release inference.'}
for name,wanted in pre['files'].items():assert bootstrap_descriptor(Path(name))==wanted,name
for group,base in [('tools',tools),('repository_files',repo)]:
    for name,wanted in p[group].items():assert descriptor(base/name)==wanted,name
for name,wanted in p['tools'].items():assert descriptor(delivered/'sources'/name)==wanted,name
assert collections=={name:verify_collection(materialized/name)for name in RESULTS}
with(cache/'q8-apack-returned-evidence-replay-v2.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in ('timing_comparison','static_functions')}))
