"""Recheck model-denied preparation without executing a helper or model."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'cpu-threads-prepare-v1'
tools = repo / 'tools/nemotron-diarization'
source = '53d98e1a82c700ab55a5faa03403c1260a5f1804'
read = lambda path:json.loads(Path(path).read_text())
def desc(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p = read(root / 'protocol.json')
for name,value in p['tools'].items(): assert desc(tools / name) == value,name
for name,value in p['repository_files'].items(): assert desc(repo / name) == value,name
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-cpu-threads.protocol.json'],cwd=repo)
sys.path.insert(0,str(tools))
from cpu_thread_metrics import validate_failure_control,validate_self_test,used_dependencies
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from sdk_archive import inventory,restore_archive
verify_git_sources(repo,source,{**{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()},**p['repository_files']})
q = read(root / 'qualification.json')
assert q['source_commit'] == source and q['prepare_only'] is True and q['preparation_completed'] is True
assert q['comparison_completed'] is False and q['release_qualified'] is False and q['neural_commands_started'] == []
assert 'acquisitions' not in q and 'error' not in q and list((root / 'work/assets').iterdir()) == []
assert [row['name'] for row in q['commands']] == p['preparation_order'] and len(q['commands']) == 28
for row in q['commands']:
    assert row['returncode'] == row['expected_returncode'] == (1 if row['name'] == 'lifetime-failure-control' else 0)
    assert row['inference_admitted'] is False and row['argv'][:2] == ['/usr/bin/sandbox-exec','-f']
    assert not row.get('forced_process_group_termination') and not row.get('disk_pressure_abort_free_bytes')
protected = read(root / 'protected-inputs.json')
for path,expected in protected.items(): assert inventory(Path(path)) == expected,path
for key in ['producer','native_reference']:
    d = q['deliveries'][key]
    assert d['id'] == p[key]['id'] and d['workflow_run']['head_sha'] == p[key]['source_commit']
    assert d['digest'] == 'sha256:'+p[key]['zip']['sha256']
package = root / 'work/nemo'
manifest = verify(package)
assert manifest == read(root / 'package.json') and manifest['runtime_reported_version'] == p['runtime_version']
helper = root / 'work/bin/native-thread-probe'
assert desc(helper) == q['binary']
for name,validator,key in [('lifetime-failure-control',validate_failure_control,'failure_control'),('lifetime-self-test',validate_self_test,'self_test')]:
    value = validator(read(root / (name+'.stdout.log')))
    assert value == q[key]
    audited = audit_loaded_libraries((root / (name+'.stderr.log')).read_text(),package,manifest,helper)
    assert audited == q['loaded_images'][name] and audited['private_library_count'] == 92
assert 'injected lifetime evidence control' in (root / 'lifetime-failure-control.stderr.log').read_text()
used = set().union(*(used_dependencies((root / (name+'.d')).read_text()) for name in ['probe','backend','logging']))
assert used == set(q['used_compiler_inputs'])
for path,expected in q['used_compiler_inputs'].items(): assert desc(path) == expected,path
for base,key in [(root / 'helper-native','helper_native_expected')]:
    for name,expected in p[key].items(): assert desc(base / name) == expected,name
archive = read(root / 'thread-helpers-files.json')
assert {k:v for k,v in archive.items() if k != 'files'} == q['helper_archive']
replay = cache / 'cpu-threads-prepare-v1-replayed-helpers'
assert not replay.exists()
restore_archive(root / archive['archive'],archive,replay)
assert inventory(replay) == inventory(root / 'work/bin')
assert q['storage']['released'] is True and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-cpu-threads-preparation'
assert not out.exists()
out.mkdir()
files = [path for path in root.iterdir() if path.is_file() and path.suffix in {'.json','.log','.sb','.d'}]
before = {path.name:desc(path) for path in files}
for path in files: shutil.copy2(path,out / path.name)
for path in [Path(__file__),cache / 'cpu-threads-prepare-v1.controller.log',cache / 'platform-ci/model-free-256-cpu-threads-review.log']:
    shutil.copy2(path,out / path.name)
for path in (cache / 'cpu-threads-inputs-v1').iterdir(): shutil.copy2(path,out / path.name)
assert before == {path.name:desc(path) for path in files}
summary = {'schema_version':1,'source_commit':source,'preparation_completed':True,'model_acquisitions':0,'model_calls':0,
    'model_denied_commands':28,'backend_lifetimes_per_control':22,'strict_private_library_counts':[92,92],
    'compiler_dependencies_verified':len(used),'partial_failure_evidence_verified':True,
    'success_settled_counts':[[row['settled'] for row in case['observations']] for case in q['self_test']['cases']],
    'opaque_helper_archive':q['helper_archive'],'archive_replayed_and_inventory_verified':True,
    'protected_inputs_independently_rechecked':list(protected),'host':read(root / 'host.json'),
    'new_collection_helper_or_model_calls':0,'release_qualified':False,'limits':p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
entries = {str(path.relative_to(out)):desc(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
print(out,len(entries)+1,'members')
