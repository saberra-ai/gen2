"""Replay delivered ABBA evidence with no helper/model execution."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'cpu-threads-ci-v1'
saved = root / 'evidence'
tools = repo / 'tools/nemotron-diarization'
source = '2995f0e0b25be69c6b9d4c5f05dc53b3cf22fa26'
read = lambda path:json.loads(Path(path).read_text())
def desc(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p = read(saved / 'protocol.json')
for name,expected in p['tools'].items(): assert desc(tools / name) == expected,name
for name,expected in p['repository_files'].items(): assert desc(repo / name) == expected,name
assert (saved / 'protocol.json').read_bytes() == subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-cpu-threads.protocol.json'],cwd=repo)
before = {str(path.relative_to(saved)):desc(path) for path in saved.rglob('*') if path.is_file()}
ci,api = read(root / 'run.json'),read(root / 'artifacts.json')['artifacts']
assert ci['databaseId'] == 39003266888 and ci['headSha'] == source and ci['status'] == 'completed' and ci['conclusion'] == 'success'
assert len(api) == 1
item = api[0]
assert item['workflow_run']['id'] == ci['databaseId'] and item['workflow_run']['head_sha'] == source
assert item['name'] == 'nemotron-macos-cpu-threads' and item['size_in_bytes'] == desc(root / 'artifact.zip')['bytes']
assert item['digest'] == 'sha256:'+desc(root / 'artifact.zip')['sha256']
with zipfile.ZipFile(root / 'artifact.zip') as archive:
    assert len(archive.infolist()) == len(before) == 86 and {m.filename for m in archive.infolist()} == set(before)
    for name in before: assert archive.read(name) == (saved / name).read_bytes()
assert not any(path.is_symlink() for path in saved.rglob('*'))
sys.path.insert(0,str(tools))
from cpu_thread_metrics import ORDER,compare_thread_trial,used_dependencies,validate_failure_control,validate_self_test
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify
from sdk_archive import inventory,restore_archive
verify_git_sources(repo,source,{**{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()},**p['repository_files']})
q = read(saved / 'qualification.json')
assert q['source_commit'] == source and q['prepare_only'] is False and q['preparation_completed'] is True and q['comparison_completed'] is True
assert q['release_qualified'] is False and 'error' not in q
assert q['neural_commands_started'] == p['neural_order'] and p['order'] == ORDER == [4,3,3,4]
assert [row['name'] for row in q['commands']] == p['preparation_order']+p['neural_order'] and len(q['commands']) == 32
for row in q['commands']:
    assert row['returncode'] == row['expected_returncode'] == (1 if row['name'] == 'lifetime-failure-control' else 0)
    assert row['inference_admitted'] == (row['name'] in p['neural_order'])
    assert row['argv'][:2] == ['/usr/bin/sandbox-exec','-f'] and 'DYLD_PRINT_LIBRARIES' not in row['environment']
    assert not row.get('forced_process_group_termination') and not row.get('disk_pressure_abort_free_bytes')
remote = Path(q['commands'][0]['cwd'])
package = cache / 'pooled-macos-construction-ci-v1/restored-pooled-cpu-runtime'
manifest = verify(package)
assert manifest == read(saved / 'package.json') and manifest['runtime_reported_version'] == p['runtime_version']
archive = read(saved / 'thread-helpers-files.json')
assert {k:v for k,v in archive.items() if k != 'files'} == q['helper_archive']
helpers = root / 'replayed-helpers'
assert not helpers.exists()
restore_archive(saved / archive['archive'],archive,helpers)
helper = helpers / 'native-thread-probe'
assert desc(helper) == q['binary']
local = cache / 'cpu-threads-prepare-v1'
mappings = {str(remote / name):local / name for name in ['helper-native','ggml-headers','sdk']}
mappings[str(remote / 'work/nemo')] = package
mappings[str(remote / 'work/bin')] = helpers
for key,folder in [('producer','pooled-macos-construction-ci-v1'),('native_reference','pooled-native-ci-v1')]:
    prior = cache / folder
    assert desc(prior / 'artifact.zip') == p[key]['zip']
    assert {str(path.relative_to(prior / 'evidence')):desc(path) for path in (prior / 'evidence').rglob('*') if path.is_file()} == p[key]['files']
    delivery = q['deliveries'][key]
    assert delivery['id'] == p[key]['id'] and delivery['workflow_run']['head_sha'] == p[key]['source_commit']
    assert delivery['digest'] == 'sha256:'+p[key]['zip']['sha256']
    mappings[str(remote / key)] = prior / 'evidence'
protected = read(saved / 'protected-inputs.json')
assert set(protected) == set(mappings)|{str(remote / 'work/assets')}
for path,local_path in mappings.items(): assert inventory(local_path) == protected[path],path
for name,expected in p['helper_native_expected'].items(): assert desc(local / 'helper-native' / name) == expected,name
used = set().union(*(used_dependencies((saved / (name+'.d')).read_text()) for name in ['probe','backend','logging']))
assert used == set(q['used_compiler_inputs'])
for path,expected in q['used_compiler_inputs'].items():
    matches = [(prefix,target) for prefix,target in mappings.items() if Path(path).is_relative_to(prefix)]
    if matches:
        assert len(matches) == 1
        prefix,target = matches[0]
        actual = target / Path(path).relative_to(prefix)
    else:
        assert path.endswith('/tools/nemotron-diarization/native_thread_probe.cpp')
        actual = tools / 'native_thread_probe.cpp'
    assert desc(actual) == expected,path
for name,validator,key in [('lifetime-failure-control',validate_failure_control,'failure_control'),('lifetime-self-test',validate_self_test,'self_test')]:
    assert validator(read(saved / (name+'.stdout.log'))) == q[key]
assert 'injected lifetime evidence control' in (saved / 'lifetime-failure-control.stderr.log').read_text()
for name in ['lifetime-failure-control','lifetime-self-test']+p['neural_order']:
    remote_helper = remote / 'work/bin/native-thread-probe'
    def remap(match):
        path = match[2]
        if path.startswith(('/usr/lib/','/System/Library/')): return match[0]
        if path == str(remote_helper): return match[1]+str(helper)
        relative = Path(path).relative_to(remote / 'work/nemo/lib')
        assert relative.as_posix() in set(manifest['libraries'])|{'libnemo_speech_asr_c.dylib'}
        return match[1]+str(package / 'lib' / relative)
    trace = (saved / (name+'.stderr.log')).read_text()
    audited = audit_loaded_libraries(re.sub(r'(dyld\[\d+\]: <[A-Fa-f0-9-]+> )(/[^\r\n]+)',remap,trace),package,manifest,helper)
    normalized = json.loads(json.dumps(audited).replace(str(helper),str(remote_helper)).replace(str(package),str(remote / 'work/nemo')))
    assert normalized == q['loaded_images'][name] and audited['private_library_count'] == 92
assert q['acquisitions'] == [{'asset':entry['name'],'bytes':entry['size'],'sha256':entry['sha256']} for entry in p['assets']]
assets = protected[str(remote / 'work/assets')]
assert set(assets) == {entry['name'] for entry in p['assets']}|{'input.f32'}
for entry in p['assets']: assert {key:assets[entry['name']][key] for key in ['bytes','sha256']} == {'bytes':entry['size'],'sha256':entry['sha256']}
assert assets['input.f32']['bytes'] == 768000 and assets['input.f32']['sha256'] == p['pcm_f32le_sha256']
reference = cache / 'pooled-native-ci-v1/evidence/results/native-12s.json'
golden = Path(str(reference)+'.f32').read_bytes()
trials = []
for name,threads in zip(p['neural_order'],ORDER):
    value = read(saved / 'results' / (name+'.json'))
    raw = (saved / 'results' / (name+'.json.f32')).read_bytes()
    result = compare_thread_trial(value,raw,read(reference),golden,threads)
    assert result == q['comparisons'][name]
    trace = (saved / (name+'.stderr.log')).read_text()
    usage = re.search(r'^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$',trace,re.M)
    wall,user,system = map(float,usage.groups())
    rss = int(re.search(r'(\d+)\s+maximum resident set size',trace)[1])
    trials.append({'name':name,**result,'time_l':{'real_seconds':wall,'user_seconds':user,'system_seconds':system,'cpu_seconds_per_wall_second':(user+system)/wall,'peak_rss_bytes':rss}})
assert {path.name:desc(path) for path in (saved / 'results').iterdir()} == q['result_files']
host = read(saved / 'host.json')
assert host['cpu'] == p['expected_receiver_cpu'] and host['cpu_counts'] == p['expected_cpu_counts']
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
assert before == {str(path.relative_to(saved)):desc(path) for path in saved.rglob('*') if path.is_file()}
out = repo / 'benches/results/nemotron-diarization/macos-arm64/2026-10-17-cpu-threads'
assert not out.exists()
out.mkdir()
for name in before:
    if name.endswith('.f32') or name.endswith('.tar.gz'): continue
    target = out / name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(saved / name,target)
for path in [root / 'run.json',root / 'artifacts.json',Path(__file__),cache / 'extract-cpu-threads-v1.py']:
    shutil.copy2(path,out / path.name)
summary = {'schema_version':1,'source_commit':source,'ci_url':ci['url'],'artifact':item,'opaque_zip':desc(root / 'artifact.zip'),
    'delivered_members_verified':86,'strict_private_image_audits_replayed':[92]*6,'compiler_dependencies_verified':len(used),
    'all_saved_payloads_bound_before_and_after':before,'new_collection_helper_or_model_calls':0,
    'asset_bytes_are_recorded_descriptors_not_delivered_payloads':True,'trials':trials,'host':host,'release_qualified':False,'limits':p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
entries = {str(path.relative_to(out)):desc(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
print(out,len(entries)+1,'members')
print(json.dumps(trials,indent=2))
