"""Prepare pinned saved-data receivers for the distinct two-row experiment."""
import hashlib
import json
from pathlib import Path

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
parent = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-worker-timing'
manifest = json.loads((parent / 'collection.json').read_text())['files']
source, run, job = '259966ceca113f50d7d172fa63269b0bcb17b37a', 39066832336, 117496146656


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def once(raw, old, new):
    assert raw.count(old) == 1, old
    return raw.replace(old, new)


def common(raw):
    for old, new in [('windows-q8-worker-timing', 'windows-q8-pair'),
            ('7fadd1497f58f2d287dc44e90c1337283074ad5e', source),
            ('39056905799', str(run)), ('117466626836', str(job))]:
        assert old in raw
        raw = raw.replace(old, new)
    return raw


origins, outputs = {}, {}
for kind, name in [('admit', 'admit-windows-q8-worker-timing-delivery-v1.py'),
        ('supervision', 'replay-windows-q8-worker-timing-supervision-v1.py'),
        ('collect', 'collect-windows-q8-worker-timing-v1.py'),
        ('publish', 'publish-windows-q8-worker-timing-v1.py')]:
    raw = (cache / name).read_bytes()
    assert identity(raw) == manifest['receiver/' + name]
    assert raw == (parent / 'receiver' / name).read_bytes()
    origins[name] = identity(raw)
    text = common(raw.decode())
    if kind == 'admit':
        text = once(text, "'Preserve successful or failed worker timing evidence'",
            "'Two-row source, full-vector and unchanged decision controls',\n             'Preserve successful or failed two-row experiment evidence'")
        text = once(text, "'bytes': 785549, 'sha256': '0eb0d37780ead508c3622cc57c4b7f0ae8ba32da3ebe317e0d46417707a26d97'",
            "'bytes': 862413, 'sha256': '865d607acc7de3d9306a7874ca657ddfec1ab894c9f50558e642cff7cbcb9959'")
        text = once(text, "len(p['tools']) == 120 and len(p['repository_files']) == 3420",
            "len(p['tools']) == 131 and len(p['repository_files']) == 3745")
        text = once(text, "'tools_verified': 120, 'repository_files_verified': 3420",
            "'tools_verified': 131, 'repository_files_verified': 3745")
        text = once(text, "log.count('Ran 6 tests') == 2", "log.count('Ran 6 tests') == 3")
        text = once(text, "'actual_worker_timing_control_tests':6,", "'actual_worker_timing_control_tests':6, 'actual_pair_control_tests':6,")
        text = once(text, "['control','mx4','mx4','control']", "['control','pair','pair','control']")
    elif kind == 'supervision':
        text = once(text, "'disasm-mx4'", "'disasm-pair'")
        text = once(text, "and type(status) is int and status == value.get('observed_worker_exit_code') == value.get('reaped_worker_exit_code')",
            "and type(status) is int and type(value.get('observed_worker_exit_code')) is int\n             and type(value.get('reaped_worker_exit_code')) is int\n             and status == value.get('observed_worker_exit_code') == value.get('reaped_worker_exit_code')")
    elif kind == 'collect':
        text = once(text, 'from windows_q8_worker_timing_metrics import ', 'from windows_q8_pair_metrics import ')
        text = once(text, 'from q8_mx4_source import PATH, ORIGINAL, transform',
            'from q8_pair_source import PATH, ORIGINAL, transform, control_source')
        text = text.replace('q8_worker_timing_benchmark', 'q8_pair_benchmark').replace('sgemm-mx4.cpp', 'sgemm-pair.cpp')
        text = text.replace("['control','mx4','mx4','control']", "['control','pair','pair','control']")
        text = once(text, "assert len(audit['patch_series']) == 19", "control_matches = ((evidence/'sgemm-control.cpp').read_bytes() == control_source((tools/'fixtures/q8-tile-sgemm.cpp').read_bytes())\n                   and desc(evidence/'sgemm-control.cpp') == p['control_source'])\nassert len(audit['patch_series']) == 19")
        text = once(text, "'original_sgemm_matches': original_matches,", "'original_sgemm_matches': original_matches, 'production_control_sgemm_matches': control_matches,")
        text = once(text, 'assert canonical_complete and archive_matches and original_matches and candidate_matches and all(patches_matching)',
            'assert canonical_complete and archive_matches and original_matches and control_matches and candidate_matches and all(patches_matching)')
        text = once(text, "len(build['project_inputs'])==len(project_inputs)==4", "len(build['project_inputs'])==len(project_inputs)==5")
        text = once(text, "'sgemm-pair.cpp':p['candidate_source'],", "'sgemm-pair.cpp':p['candidate_source'],'sgemm-control.cpp':p['control_source'],")
        text = once(text, "assert set(units)=={'control','mx4'}", "assert set(units)=={'control','pair'}")
        text = once(text, "canonical=PureWindowsPath(units['control']['source']).parents[4]\n    candidate=PureWindowsPath(units['mx4']['source'])",
            "control=PureWindowsPath(units['control']['source'])\n    ci_evidence=control.parent\n    canonical=ci_evidence.parent/'canonical'\n    candidate=PureWindowsPath(units['pair']['source'])\n    assert control.name=='sgemm-control.cpp' and candidate==ci_evidence/'sgemm-pair.cpp'")
        text = once(text, "elif path==candidate: expected=p['candidate_source']", "elif path==candidate: expected=p['candidate_source']\n        elif path==control: expected=p['control_source']")
        text = once(text, "if variant=='mx4':assert row['command'].count('GEN2_Q8_MX4_EXPECTED=1')==1",
            "assert row['command'].count(('GEN2_Q8_MX4_EXPECTED' if variant=='control' else 'GEN2_Q8_PAIR_EXPECTED')+'=1')==1")
        # Bind actual whole native benchmark argv and cwd, not just role fields.
        marker = 'identity=desc(Path(__file__))'
        function = '''def admit_kernel_command(row, evidence, repo, arguments):
    expected = [str(evidence/'q8_tile_benchmark.exe'), *arguments]
    assert row['command'] == expected and PureWindowsPath(row['cwd']) == repo

'''
        text = once(text, marker, function + marker)
        text = once(text, "assert controls==case['support_controls']",
            "assert controls==case['support_controls']\n        ci_repo=PureWindowsPath(q['commands'][0]['cwd'])\n        for mode in GUARD_CODES:\n            call,=[r for r in q['commands'] if r['name']==phase+'-guard-'+mode]\n            admit_kernel_command(call,ci_evidence,ci_repo,['--support-control',mode])")
        text = once(text, "assert row['command'][1:3]==[variant,str(workers)]",
            "admit_kernel_command(row,ci_evidence,ci_repo,[variant,str(workers),str(ci_evidence/phase)])")
    elif kind == 'publish':
        text = text.replace('2026-10-17-q8-worker-timing', '2026-10-17-q8-pair')
        text = once(text, "files['receiver/'+Path(__file__).name]=Path(__file__)",
            "files['receiver/'+Path(__file__).name]=Path(__file__)\nfor name in ['fetch-windows-q8-pair-artifact-v1.py','windows-q8-pair-receivers-pre-execution-v1.json','test-q8-pair-receiver-controls-v1.py','q8-pair-receiver-controls-local-v1.json']:\n    files['receiver/'+name]=cache/name\nfor name in ['artifact-download-v2.json','artifact-download-v2.stderr.log']:\n    files['receiver/'+name]=root/name")
    compile(text, name, 'exec')
    outputs[name.replace('windows-q8-worker-timing', 'windows-q8-pair')] = text.encode()
raw = (cache / 'fetch-windows-q8-candidate-artifact-v2.py').read_bytes()
assert identity(raw) == {'bytes': 5703, 'sha256': '36091c3b0db2c6f4b9cf28ba20a1916f0b9b1042b100177e2cd5858a17dbb053'}
origins['fetch-windows-q8-candidate-artifact-v2.py'] = identity(raw)
text = raw.decode().replace('windows-q8-candidate-ci-v2', 'windows-q8-pair-ci-v1').replace('39063644421', str(run)).replace('117486629205', str(job)).replace('fee2177adcd7b75b488c67bea5a593ea17df65de', source).replace('nemotron-windows-q8-candidate', 'nemotron-windows-q8-pair')
compile(text, 'fetch-windows-q8-pair-artifact-v1.py', 'exec')
outputs['fetch-windows-q8-pair-artifact-v1.py'] = text.encode()
for name, raw in outputs.items():
    with (cache / name).open('xb') as stream:
        stream.write(raw)
metadata = {'parents': origins, 'outputs': {name: identity(raw) for name, raw in outputs.items()},
    'source_commit': source, 'run_id': run, 'job_id': job, 'new_compiler_sdk_model_calls': 0}
with (cache / 'windows-q8-pair-receivers-pre-execution-v1.json').open('x') as stream:
    stream.write(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata))
