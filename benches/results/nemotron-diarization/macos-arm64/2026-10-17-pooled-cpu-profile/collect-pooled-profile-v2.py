"""Verify delivered pooled sampling evidence without another model/profiler call."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = cache / 'pooled-cpu-profile-ci-v1'
saved = root / 'evidence'
source = '761c9ca1613d09fa68eb4c0fad957131a775edd2'
read = lambda path:json.loads(Path(path).read_text())
def desc(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p = read(saved / 'protocol.json')
for name,expected in p['tools'].items():
    assert desc(tools / name) == expected,name
for name,expected in p['repository_files'].items():
    assert desc(repo / name) == expected,name
assert (saved / 'protocol.json').read_bytes() == subprocess.check_output([
    'git','show',source+':tools/nemotron-diarization/macos-pooled-cpu-profile.protocol.json'],cwd=repo)
before = {str(path.relative_to(saved)):desc(path) for path in saved.rglob('*') if path.is_file()}
ci = read(root / 'run.json')
api = read(root / 'artifacts.json')['artifacts']
assert ci['databaseId'] == 39001644761 and ci['headSha'] == source
assert ci['status'] == 'completed' and ci['conclusion'] == 'success' and len(api) == 1
a = api[0]
assert a['id'] == 12089168327 and a['name'] == 'nemotron-macos-pooled-profile'
assert a['workflow_run']['id'] == ci['databaseId'] and a['workflow_run']['head_sha'] == source
assert a['size_in_bytes'] == desc(root / 'artifact.zip')['bytes'] == 200631
assert a['digest'] == 'sha256:'+desc(root / 'artifact.zip')['sha256'] == 'sha256:fc2022f82804c6eba1355e646d57105b8b7d4d26fff428177afa377e53c93fa7'
with zipfile.ZipFile(root / 'artifact.zip') as archive:
    members = archive.infolist()
    assert len(members) == len(before) == 25 and {m.filename for m in members} == set(before)
    for member in members:
        path = Path(member.filename)
        assert not path.is_absolute() and '..' not in path.parts and not member.is_dir()
        assert stat.S_IFMT(member.external_attr>>16) in {0,stat.S_IFREG}
        assert archive.read(member) == (saved / path).read_bytes()
assert not any(path.is_symlink() for path in saved.rglob('*'))
sys.path.insert(0,str(tools))
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files,verify_package_provenance
from profile_macos_pooled_cpu import compare_sampled_history
from profile_macos_portable_cpu import CONTROL_SOURCE
from sdk_archive import inventory,restore_archive

verify_git_sources(repo,source,{**{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()},**p['repository_files']})
r = read(saved / 'qualification.json')
assert r['source_commit'] == source and r['profile_completed'] is True and r['model_execution_admitted'] is True
assert r['release_qualified'] is False and r['neural_commands_started'] == ['native']
assert [row['name'] for row in r['commands']] == p['command_order']
for row in r['commands']:
    assert row['returncode'] == 0 and not row.get('forced_process_group_termination') and not row.get('disk_pressure_abort_free_bytes')
    assert row['argv'][:2] == ['/usr/bin/sandbox-exec','-f'] and 'DYLD_PRINT_LIBRARIES' not in row['environment']
remote = Path(r['commands'][0]['cwd'])
assert all(row['cwd'] == str(remote) for row in r['commands'])
for name,folder in [('producer','pooled-macos-construction-ci-v1'),('native_reference','pooled-native-ci-v1')]:
    prior = cache / folder
    assert desc(prior / 'artifact.zip') == p[name]['zip']
    assert {str(path.relative_to(prior / 'evidence')):desc(path) for path in (prior / 'evidence').rglob('*') if path.is_file()} == p[name]['files']
    delivery = r['deliveries'][name]
    assert delivery['id'] == p[name]['id'] and delivery['workflow_run']['head_sha'] == p[name]['source_commit']
    assert delivery['digest'] == 'sha256:'+p[name]['zip']['sha256']
package = cache / 'pooled-macos-construction-ci-v1/restored-pooled-cpu-runtime'
manifest = read(saved / 'package.json')
assert manifest == read(package / 'package.json')
verify_files(package,manifest['files'])
verify_package_provenance(package,manifest)
helper_manifest = read(saved / 'profile-helpers-files.json')
assert {k:v for k,v in helper_manifest.items() if k != 'files'} == r['helper_archive']
helpers = root / 'replayed-helpers'
assert not helpers.exists()
restore_archive(saved / helper_manifest['archive'],helper_manifest,helpers)
assert inventory(helpers) == p['helpers']
protected = read(saved / 'protected-before.json')
assert protected[str(remote / 'work/bin')] == p['helpers']
assert protected[str(remote / 'work/nemo')] == inventory(package)
for name,folder in [('producer','pooled-macos-construction-ci-v1'),('native_reference','pooled-native-ci-v1')]:
    assert protected[str(remote / name)] == inventory(cache / folder / 'evidence')
assert (saved / 'control.c').read_text() == CONTROL_SOURCE
assert desc(saved / 'work/control') == r['control_executable']
assert r['acquisitions'] == [{'asset':entry['name'],'bytes':entry['size'],'sha256':entry['sha256']} for entry in p['assets']]
for entry in p['assets']:
    assert {key:protected[str(remote / 'work/assets')][entry['name']][key] for key in ['bytes','sha256']} == {'bytes':entry['size'],'sha256':entry['sha256']}
for name in ['sampling-control','native']:
    sample = r['sampling'][name]
    report = (saved / (name+'.sample.txt')).read_text()
    trace = (saved / (name+'.stderr.log')).read_text()
    assert sample['returncode'] == 0 and 'error' not in sample and desc(saved / (name+'.sample.txt')) == sample['report']
    remote_executable = remote / ('work/control' if name == 'sampling-control' else 'work/bin/native-stream')
    executable = saved / 'work/control' if name == 'sampling-control' else helpers / 'native-stream'
    assert sample['executable'] == str(remote_executable)
    assert sample['argv'] == ['/usr/bin/sample',str(sample['pid']),str(1 if name == 'sampling-control' else 5),'10','-mayDie','-file',str(remote / (name+'.sample.txt'))]
    assert re.search(r'^Process:\s+.*\['+str(sample['pid'])+r'\]',report,re.M)
    assert set(re.findall(r'^dyld\[(\d+)\]: <[A-Fa-f0-9-]+> /',trace,re.M)) == {str(sample['pid'])}
    assert re.search(r'^dyld\['+str(sample['pid'])+r'\]: <'+re.escape(sample['target_dyld_uuid'])+r'> '+re.escape(str(remote_executable))+r'$',trace,re.M)
    assert sample['target_binary_image'] in report and '<'+sample['target_dyld_uuid']+'>' in sample['target_binary_image']
    def remap(match):
        path = match[2]
        if path.startswith(('/usr/lib/','/System/Library/')):
            return match[0]
        if path == str(remote_executable):
            return match[1]+str(executable)
        relative = Path(path).relative_to(remote / 'work/nemo/lib')
        assert relative.as_posix() in set(manifest['libraries'])|{'libnemo_speech_asr_c.dylib'}
        return match[1]+str(package / 'lib' / relative)
    audited = audit_loaded_libraries(re.sub(r'(dyld\[\d+\]: <[A-Fa-f0-9-]+> )(/[^\r\n]+)',remap,trace),
        package,manifest,executable,require_runtime=name == 'native')
    normalized = json.loads(json.dumps(audited).replace(str(executable),str(remote_executable)).replace(str(package),str(remote / 'work/nemo')))
    assert normalized == r['loaded_images'][name] and audited['private_library_count'] == (92 if name == 'native' else 0)
results = saved / 'results'
assert {path.name:desc(path) for path in results.iterdir()} == r['results']
value = read(results / 'native.json')
reference = cache / 'pooled-native-ci-v1/evidence/results/native-12s.json'
assert compare_sampled_history(value,(results / 'native.json.f32').read_bytes(),read(reference),Path(str(reference)+'.f32').read_bytes()) == r['comparison']
assert r['comparison']['values_compared'] == 9600 and r['comparison']['bit_exact'] is True
assert value['input_pcm_f32le_sha256'] == p['pcm_f32le_sha256'] and r['native_rtf'] == value['elapsed_micros']/1e6/12.
trace = (saved / 'native.stderr.log').read_text()
usage = re.search(r'^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$',trace,re.M)
wall,user,system = map(float,usage.groups())
rss = int(re.search(r'(\d+)\s+maximum resident set size',trace)[1])
host = read(saved / 'host.json')
assert host['cpu'] == p['expected_receiver_cpu']
assert r['storage']['released'] and r['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert r['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
assert before == {str(path.relative_to(saved)):desc(path) for path in saved.rglob('*') if path.is_file()}
out = repo / 'benches/results/nemotron-diarization/macos-arm64/2026-10-17-pooled-cpu-profile'
assert not out.exists()
out.mkdir()
for name in before:
    if name in {'work/control','profile-helpers.tar.gz','results/native.json.f32'}:
        continue
    target = out / name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(saved / name,target)
for path in [root / 'run.json',root / 'artifacts.json',Path(__file__),cache / 'extract-pooled-profile-v1.py',
             cache / 'platform-ci/model-free-254-pooled-profile-local.log']:
    shutil.copy2(path,out / path.name)
summary = {'schema_version':1,'profile_completed':True,'source_commit':source,'ci_url':ci['url'],
    'artifact':a,'opaque_zip':desc(root / 'artifact.zip'),'delivered_members_verified':25,
    'local_tools_checked_before_import':27,'native_helper_and_control_payloads_independently_rehashed':True,
    'strict_private_image_audits_replayed':[0,92],'all1200rows_and5turns_bit_exact':True,
    'all_delivered_result_bytes_bound_before_and_after':before,'new_collection_model_calls':0,
    'acquired_model_audio_are_recorded_descriptors_not_delivered_payloads':True,
    'native_rtf':r['native_rtf'],'time_l':{'real_seconds':wall,'user_seconds':user,'system_seconds':system,
        'cpu_seconds_per_wall_second':(user+system)/wall,'peak_rss_bytes':rss},
    'host':host,'release_qualified':False,'limits':p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
files = {str(path.relative_to(out)):desc(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(out,len(files)+1,'members')
print(json.dumps(summary['time_l'],indent=2))
