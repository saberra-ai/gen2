"""Replay saved profiling evidence; execute no delivered binaries or models."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'windows-session-profile-ci-v1'
source = 'c183c6cf3501c8b45992a53638ddbbd4b943e449'
run_id = 39027810779


def descriptor(path):
    value, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            size += len(block)
            value.update(block)
    return {'bytes': size, 'sha256': value.hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}


def git_bytes(name):
    return subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo, env=env)


collector_identity = descriptor(Path(__file__))
metadata = {name: descriptor(root / name) for name in ['producer-run.json', 'artifacts.json']}
run = read(root / 'producer-run.json')
assert run['id'] == run_id and run['head_sha'] == source and run['conclusion'] == 'success'
assert run['repository']['full_name'] == 'saberra-ai/gen2'
artifact, = read(root / 'artifacts.json')['artifacts']
assert artifact['name'] == 'nemotron-windows-session-profile' and artifact['expired'] is False
assert artifact['workflow_run']['id'] == run_id and artifact['workflow_run']['head_sha'] == source
archive_identity = {'bytes': artifact['size_in_bytes'], 'sha256': artifact['digest'].removeprefix('sha256:')}
assert descriptor(root / 'artifact.zip') == archive_identity
protocol_bytes = git_bytes('tools/nemotron-diarization/windows-session-profile.protocol.json')
p = json.loads(protocol_bytes)
# Admit the complete evaluator closure from committed bytes before local imports.
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = git_bytes(prefix + name)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
sys.path.insert(0, str(tools))
from pooled_runtime import BOUNDED_PROFILE, load_profile
from profile_windows_sessions import RECEIPT
from windows_archive import MAX_ENTRIES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, safe_relative
from windows_session_timing import ORDER, extract_delivery, file_inventory, summarize
from windows_worker_metrics import compare_native, telemetry

members, folded, total = {}, set(), 0
with zipfile.ZipFile(root / 'artifact.zip') as bundle:
    entries = bundle.infolist()
    assert 0 < len(entries) <= MAX_ENTRIES
    for item in entries:
        safe_relative(item.filename)
        assert item.filename.casefold() not in folded
        folded.add(item.filename.casefold())
        assert not item.is_dir() and not item.flag_bits & 1
        assert stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
        assert item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        assert 0 <= item.file_size <= MAX_FILE_BYTES
        total += item.file_size
        assert total <= MAX_TOTAL_BYTES
        value, size = hashlib.sha256(), 0
        with bundle.open(item) as stream:
            while block := stream.read(1024**2):
                size += len(block)
                assert size <= item.file_size
                value.update(block)
        assert size == item.file_size
        members[item.filename] = {'bytes': size, 'sha256': value.hexdigest()}
work = root / 'collection-replay-v1'
work.mkdir()
delivered = work / 'delivered'
extract_delivery(root / 'artifact.zip', archive_identity, members, delivered)
assert (delivered / 'protocol.json').read_bytes() == protocol_bytes
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source and q['protocol'] == descriptor(delivered / 'protocol.json')
assert q['profile_completed'] is True and q['order'] == ORDER == p['order']
assert all(q[key] is False for key in ['release_qualified', 'package_qualified', 'public_or_lifecycle_qualified'])
assert q['source_delivery'] == p['source_delivery'] and q['limits'] == p['limits']
assert q['host'] == read(delivered / 'receiver-host.stdout.log')
assert q['host']['avx2'] is True and q['host']['fma'] is True
producer = read(repo / RECEIPT / 'qualification.json')
sdk_files = {name.replace('\\', '/'): value for name, value in producer['builds']['4']['installed_files'].items()}
assert q['sdk_files'] == sdk_files
retained_source = cache / 'windows-worker-diagnostic-ci-v2'
assert descriptor(retained_source / 'artifact.zip') == p['source_delivery']['archive']
assert file_inventory(retained_source / 'unpacked/sdk-4') == sdk_files
assert q['native_probe'] == sdk_files['bin/gen2_native_stream_probe_valid_mel.exe']
assert q['private_pe']['pe_report_sha256'] == descriptor(delivered / 'windows-private-pe.json')['sha256']
assert read(delivered / 'windows-private-pe.json') == read(repo / RECEIPT / 'evidence-4/windows-private-pe.json')
_, lock = load_profile(BOUNDED_PROFILE)
assert q['assets'] == {item['name']: {'bytes': item['size'], 'sha256': item['sha256']}
                       for item in lock['artifacts'] if item['role'] != 'reference'}
source_run = read(delivered / 'source-run.stdout.log')
source_artifact = read(delivered / 'source-artifact.stdout.log')
assert source_run['id'] == p['source_delivery']['run_id'] and source_run['conclusion'] == 'success'
assert source_run['head_sha'] == p['source_delivery']['head_sha']
assert source_run['repository']['full_name'] == 'saberra-ai/gen2'
assert source_artifact['id'] == p['source_delivery']['artifact_id']
assert source_artifact['workflow_run']['id'] == source_run['id']
assert source_artifact['workflow_run']['head_sha'] == source_run['head_sha']
assert source_artifact['expired'] is False
assert source_artifact['size_in_bytes'] == p['source_delivery']['archive']['bytes']
assert source_artifact['digest'] == 'sha256:' + p['source_delivery']['archive']['sha256']
commands = q['commands']
case_names = ['case-' + str(index) + '-' + mode for index, mode in enumerate(ORDER)]
expected_commands = ['github-cli-version', 'source-run', 'source-artifact', 'download-sdk-evidence', 'receiver-host'] + case_names
assert [row['name'] for row in commands] == expected_commands
assert all(type(row['returncode']) is int and row['returncode'] == 0 and 'error' not in row for row in commands)
assert len(q['cases']) == 4
reference_path = repo / RECEIPT / 'evidence-4/case-0-cap-4.json'
reference, reference_raw = read(reference_path), Path(str(reference_path) + '.f32').read_bytes()
cases = []
for index, mode in enumerate(ORDER):
    case, phase = q['cases'][index], case_names[index]
    assert case['index'] == index and case['mode'] == mode and case['command_started'] is True
    assert case['timing_environment'] == ('1' if mode == 'timing' else None)
    target = delivered / (phase + '.json')
    result, raw = read(target), Path(str(target) + '.f32').read_bytes()
    assert result == case['result'] and descriptor(target) == case['result_file']
    assert result['push_count'] == 58 and type(result['push_count']) is int
    assert descriptor(Path(str(target) + '.f32')) == case['probabilities']
    log = (delivered / (phase + '.stderr.log')).read_text()
    assert telemetry(log, 4) == case['workers'] and case['workers']['effective_threads'] == 4
    assert compare_native(reference, result, reference_raw, raw, 4, 4) == case['numerical_comparison']
    summary = summarize(log, mode, result['elapsed_micros'])
    assert summary == case['session_profile']
    cases.append({'index': index, 'mode': mode, 'elapsed_micros': result['elapsed_micros'],
                  'probabilities': case['probabilities'], 'workers': case['workers'],
                  'numerical_comparison': case['numerical_comparison'], 'session_profile': summary})
assert file_inventory(delivered) == members and descriptor(root / 'artifact.zip') == archive_identity
assert descriptor(Path(__file__)) == collector_identity
assert all(descriptor(root / name) == expected for name, expected in metadata.items())
receipt = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-session-profile'
assert not receipt.exists()
receipt.mkdir()
shutil.copytree(delivered, receipt / 'evidence')
assert file_inventory(receipt / 'evidence') == members
for name in metadata:
    shutil.copyfile(root / name, receipt / name)
    assert descriptor(receipt / name) == metadata[name]
shutil.copyfile(Path(__file__), receipt / 'collector.py')
assert descriptor(receipt / 'collector.py') == collector_identity
(receipt / 'artifact-members.json').write_text(json.dumps(members, indent=2) + '\n')
qualification = {'schema_version': 1, 'collection_completed': True, 'source_commit': source,
    'run_id': run_id, 'artifact_id': artifact['id'], 'archive': archive_identity,
    'protocol': descriptor(delivered / 'protocol.json'), 'tools_verified': len(p['tools']),
    'repository_files_verified': len(p['repository_files']), 'artifact_members_verified': len(members),
    'sdk_files': len(sdk_files), 'successful_commands': len(commands), 'cases': cases,
    'sdk_payload_scope': 'New CI producer rehashes its downloaded/copy SDK; collection compares its manifest and PE report with the prior receipt and rehashes the retained prior ZIP/66-file SDK. New profiling artifact excludes SDK payloads.',
    'release_qualified': False, 'package_qualified': False, 'public_or_lifecycle_qualified': False,
    'new_native_or_model_calls_during_collection': 0, 'limits': p['limits']}
(receipt / 'collection-qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = file_inventory(receipt)
(receipt / 'collection.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
print(json.dumps({'receipt': str(receipt), 'files': len(files) + 1, 'cases': [
    {key: case[key] for key in ['index', 'mode', 'elapsed_micros']} for case in cases]}))
