"""Prepare source-bound saved-data receivers; do not consume artifacts or run native code."""
import ast
import hashlib
import json
from pathlib import Path

REPO=Path('/Users/victor/workspace/gen2-nemotron3-n0')
CACHE=Path('/Users/victor/.cache/gen2-nemotron3-n0')
SOURCE='15c905d8815190a54f25452e7e51378c24f85914'
BASE=REPO/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-matrix-batch-profile'
TARGET=CACHE/'thread-cpu-receivers-prepared-v3'

def desc(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def once(raw,old,new):
    if raw.count(old)!=1:raise ValueError('exact receiver seam required: '+old[:100])
    return raw.replace(old,new)

manifest=json.loads((BASE/'collection.json').read_bytes())['files']
baseline_raw=(BASE/'qualification.json').read_bytes()
if desc(baseline_raw)!=manifest['qualification.json']:raise ValueError('retained command baseline changed')
baseline_names=[row['name']for row in json.loads(baseline_raw)['commands']]
if len(baseline_names)!=59 or len(set(baseline_names))!=59:raise ValueError('complete original command sequence required')
names={
 'admit-windows-matrix-batch-profile-delivery-v1.py':'admit-windows-thread-cpu-profile-delivery-v1.py',
 'replay-windows-matrix-batch-profile-supervision-v1.py':'replay-windows-thread-cpu-profile-supervision-v1.py',
 'collector.py':'collect-windows-thread-cpu-profile-v1.py',
}
prepared={};derivation={}
for oldname,newname in names.items():
    original=(BASE/oldname).read_bytes()
    if desc(original)!=manifest[oldname]:raise ValueError('retained receiver origin changed')
    raw=original.decode()
    for old,new in [
        ('96e5e6bc95eea8e096603323d89ef6fff1b8daa7',SOURCE),
        ('39051809615','39083120058'),('117451767146','117544022272'),
        ('windows-matrix-batch-profile','windows-thread-cpu-profile'),
        ('windows_matrix_batch_profile_','windows_thread_cpu_profile_'),
        ('04e3b9a4dc3376ed577b7e3bc5df9b314b7d8c7e8350b5115394d51fe70aa225',
         'e9809ffe025bbe127c27daa5f8dfa07916cab7fe24a40d074651103b8c3cfc66'),
        ('564099','670257'),("len(p['tools']) == 122","len(p['tools']) == 135"),
        ("len(p['repository_files']) == 2445","len(p['repository_files']) == 2880"),
        ("'tools_verified': 122","'tools_verified': 135"),
        ("'repository_files_verified': 2445","'repository_files_verified': 2880"),
        ('2026-10-17-matrix-batch-profile','2026-10-17-thread-cpu-profile'),
    ]:
        raw=raw.replace(old,new)
    if oldname.startswith('admit-'):
        raw=once(raw,"             'Actual owned-background timeout and assignment controls',",
            "             'Thread clock API adapter and complete counter traces',\n             'Actual owned-background timeout and assignment controls',")
        raw=once(raw,"log.count('Ran 10 tests') == 2", "log.count('Ran 10 tests') == 2 and log.count('Ran 8 tests') == 1")
        raw=once(raw,"'actual_batch_profile_control_tests':10,", "'actual_batch_profile_control_tests':10, 'actual_thread_clock_tests':8,")
    if oldname=='collector.py':
        raw=once(raw,"assert q['commands'] == read(delivered / 'evidence/commands.json')",
            "assert q['commands'] == read(delivered / 'evidence/commands.json')\nassert [row['name']for row in q['commands']]=="+repr(baseline_names))
        raw=once(raw,'from ggml_matrix_profile import MAX_LOG_BYTES, summarize',
            'from ggml_thread_cpu_profile import MAX_LOG_BYTES, summarize')
        first=raw.index("assert len(control['inputs']) == 4\n")
        last=raw.index('control_reference={};control_files={}\n',first)
        raw=raw[:first]+'''assert len(control['inputs']) == 7
control_build,=[row for row in q['commands']if row['name']=='matrix-control-build']
control_evidence=PureWindowsPath(control_build['cwd'])
control_source=PureWindowsPath(next(name for name in control['inputs']if PureWindowsPath(name).name=='ggml_matrix_profile_control.cpp'))
assert control_source.parts[-3:]==('tools','nemotron-diarization','ggml_matrix_profile_control.cpp')
assert control_source==PureWindowsPath(next(row for row in q['commands']if row['name']=='receiver-host')['cwd'])/'tools/nemotron-diarization/ggml_matrix_profile_control.cpp'
library_names={'gen2_nemo_windows_ggml.lib','gen2_nemo_windows_ggml-base.lib','gen2_nemo_windows_ggml-cpu.lib'}
input_groups={}
assert len({PureWindowsPath(name)for name in control['inputs']})==7
for name,value in control['inputs'].items():
    path=PureWindowsPath(name)
    assert path.is_absolute()
    input_groups.setdefault(path.name,[]).append((path,value))
assert set(input_groups)==library_names|{'ggml_matrix_profile_control.cpp'}
assert len(input_groups['ggml_matrix_profile_control.cpp'])==1
assert descriptor(tools/'ggml_matrix_profile_control.cpp')==input_groups['ggml_matrix_profile_control.cpp'][0][1]
original_libraries={}
for basename in library_names:
    original_and_retained=input_groups[basename]
    assert len(original_and_retained)==2
    assert original_and_retained[0][1]==original_and_retained[1][1]
    retained=[(path,value)for path,value in original_and_retained if path.parent.name=='matrix-control-inputs']
    assert len(retained)==1
    path,value=retained[0]
    assert descriptor(evidence/'matrix-control-inputs'/basename)==value
    assert path==control_evidence/'matrix-control-inputs'/basename
    original,=[path for path,_ in original_and_retained if path.parent.name!='matrix-control-inputs']
    assert original==materialized.parents[1]/'ggml/src'/basename
    original_libraries[basename]=str(original)
assert control_evidence==PureWindowsPath(next(row for row in q['commands']if row['name']=='matrix-control-1-0')['command'][0]).parent
assert control_build['command']==[
    build['probe_compiler']['path'],'/nologo','/std:c++17','/O2','/EHsc','/TP',str(control_source),
    '/I'+str(materialized/'ggml/include'),'/link',original_libraries['gen2_nemo_windows_ggml.lib'],
    '/OUT:'+str(control_evidence/'matrix-profile-control.exe'),
    original_libraries['gen2_nemo_windows_ggml-base.lib'],original_libraries['gen2_nemo_windows_ggml-cpu.lib']]
''' + raw[last:]
        raw=once(raw,'cases, signatures, operations, matrix_groups = [], [], [], []',
            'cases, signatures, operations, matrix_groups, thread_cpu_totals = [], [], [], [], []')
        raw=once(raw,"        by_worker = {}\n",'''        by_worker = {}
        cpu_by_worker={}
        for graph in profile['graphs']:
            for worker in graph['workers']:
                key=str(worker['thread'])
                slot=cpu_by_worker.setdefault(key,{name:0 for name in ('cycles','kernel_100ns','user_100ns','final_wait_cycles')})
                for name in slot:slot[name]+=worker['thread_cpu'][name]
        thread_cpu_totals.append({'index':index,'per_worker':cpu_by_worker})
''')
        raw=once(raw,"    phase = 'case-' + str(index)\n",'''    phase = 'case-' + str(index)
    model_command,=[row for row in q['commands']if row['name']==phase]
    model_root=control_evidence.parent
    assert model_command['command']==[
        str(model_root/'sdk/bin/gen2_native_stream_probe_valid_mel.exe'),
        str(model_root/'assets/Nemotron-3-Diarization.q8_0.gguf'),
        str(model_root/'assets/ami_en2002d_2132.wav'),'1','-1',str(control_evidence/(phase+'.json'))]
    assert PureWindowsPath(model_command['cwd'])==PureWindowsPath(next(row for row in q['commands']if row['name']=='receiver-host')['cwd'])
''')
        raw=once(raw,"'fused': 0, 'kernel_us': 0, 'barrier_us': 0}",
            "'fused': 0, 'kernel_us': 0, 'barrier_us': 0, 'kernel_cycles':0, 'barrier_cycles':0}")
        raw=once(raw,"('roots', 'fused', 'kernel_us', 'barrier_us')", "('roots', 'fused', 'kernel_us', 'barrier_us', 'kernel_cycles', 'barrier_cycles')")
        raw=once(raw,"'first':[0,0,0,0],'second':[0,0,0,0],'conversion':[0,0],'inner_wait_us':0,'chunks':[0,0]}",
            "'first':[0,0,0,0],'second':[0,0,0,0],'conversion':[0,0],'inner_wait_us':0,'chunks':[0,0],\n                        **{name:0 for name in ('first_cycles','second_cycles','conversion_cycles','inner_wait_cycles','chunk_cycles','kernel_cycles')}}")
        raw=once(raw,"('kernel_us','inner_wait_us')", "('kernel_us','inner_wait_us','first_cycles','second_cycles','conversion_cycles','inner_wait_cycles','chunk_cycles','kernel_cycles')")
        raw=once(raw,"assert sum(value['kernel_us']for value in groups.values())==mul_mat['kernel_us']",
            "assert sum(value['kernel_us']for value in groups.values())==mul_mat['kernel_us']\n            assert sum(value['kernel_cycles']for value in groups.values())==mul_mat['kernel_cycles']")
        raw=once(raw,"if name in ('protocol.json', 'qualification.json') or (name.startswith('evidence/')",
            "if name in ('protocol.json', 'qualification.json') or name.startswith('evidence/matrix-control-inputs/') or (name.startswith('evidence/')")
        raw=once(raw,"'full_sequence_bit_exact': True, 'timing_comparison':timing,",
            "'full_sequence_bit_exact': True, 'timing_comparison':timing,'per_worker_thread_cpu_totals':thread_cpu_totals,'cpu_time_units':'100ns','cycle_units':'raw-thread-cycles',")
        raw=once(raw,"'Physical compiler aliases and model-free build-tree import-library hashes are actual CI observations; saved replay does not stat absent Windows inputs.',",
            "'Physical compiler aliases remain saved Windows observations. All three first-bound graph import libraries are retained and verified against both original and retained CI input identities.',\n        'Raw thread cycles and CPU100ns durations remain separate from wall microseconds; no frequency conversion, exact waits, pure arithmetic or causal scheduling inference. Preserve zeros and all original observer bounds.',")
    ast.parse(raw)
    prepared[newname]=raw.encode()
    derivation[newname]={'origin':str(BASE/oldname),'original':desc(original),'prepared':desc(prepared[newname])}
if TARGET.exists():raise ValueError('fresh preparation required')
TARGET.mkdir()
for name,raw in prepared.items():
    with(TARGET/name).open('xb')as f:f.write(raw)
report={'source_commit':SOURCE,'run_id':39083120058,'job_id':117544022272,'preparer':desc(Path(__file__).read_bytes()),
    'complete_command_sequence_origin':{'path':str(BASE/'qualification.json'),'descriptor':desc(baseline_raw),'commands':len(baseline_names)},
    'native_protocol':desc((REPO/'tools/nemotron-diarization/windows-thread-cpu-profile.protocol.json').read_bytes()),
    'prepared':derivation,'artifact_consumed':False,'receivers_invoked':False,'new_native_calls':0,
    'remaining':'Review all adaptations; add actual-source preparation, frozen remote API/archive identity, bounded process and result-bundle transport before invoking. Original8GiB full-native admission floor remains unchanged.'}
with(TARGET/'preparation.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'prepared':list(prepared),'receivers_invoked':False,'directory':str(TARGET)}))
