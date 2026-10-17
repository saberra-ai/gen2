"""Collect the completed first-build Job failure using saved bytes only."""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import stat
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-compiler-diagnostic-ci-v1')
source = '8f9ad71cfe9045e17135c4be1bd3633fc1a1a37c'
run_id, job_id, artifact_id = 39029556195, 117387093060, 12100021994


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
metadata = {name: descriptor(root / name) for name in ['producer-run.json', 'artifacts.json', 'job.json', 'job.log']}
run, job = read(root / 'producer-run.json'), read(root / 'job.json')
assert run['id'] == run_id and run['head_sha'] == source and run['conclusion'] == 'failure'
assert run['repository']['full_name'] == 'saberra-ai/gen2'
assert job['id'] == job_id and job['run_id'] == run_id and job['head_sha'] == source and job['conclusion'] == 'failure'
for name, status in [('Model-free compiler selection and provenance controls', 'success'),
        ('Actual Windows process-tree cleanup controls without compiler or model', 'success'),
        ('Build both admitted SDKs and run fixed compiler sequence', 'failure'),
        ('Preserve successful or failed compiler evidence', 'success')]:
    step, = [row for row in job['steps'] if row['name'] == name]
    assert step['conclusion'] == status
artifact, = read(root / 'artifacts.json')['artifacts']
assert artifact['id'] == artifact_id and artifact['expired'] is False
assert artifact['name'] == 'nemotron-windows-compiler-diagnostic'
assert artifact['workflow_run']['id'] == run_id and artifact['workflow_run']['head_sha'] == source
archive_identity = {'bytes': 63248, 'sha256': 'caf4c9142794d28540ee600bf922ca714274691b84bf9ba80b2a34053f0945c2'}
assert descriptor(root / 'artifact.zip') == archive_identity
assert artifact['size_in_bytes'] == archive_identity['bytes'] and artifact['digest'] == 'sha256:' + archive_identity['sha256']
protocol_bytes = git_bytes('tools/nemotron-diarization/windows-compiler-diagnostic.protocol.json')
p = json.loads(protocol_bytes)
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = git_bytes(prefix + name)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
# Bind every ZIP payload before importing admitted evaluators or extracting.
members, folded, total = {}, set(), 0
with zipfile.ZipFile(root / 'artifact.zip') as bundle:
    entries = bundle.infolist()
    assert len(entries) == 40
    for item in entries:
        path = PureWindowsPath(item.filename)
        assert not path.is_absolute() and not path.drive and '..' not in path.parts
        assert item.filename.casefold() not in folded
        folded.add(item.filename.casefold())
        assert not item.is_dir() and not item.flag_bits & 1
        assert stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
        assert item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        assert 0 <= item.file_size <= 256 * 1024**2
        total += item.file_size
        assert total <= 512 * 1024**2
        value, size = hashlib.sha256(), 0
        with bundle.open(item) as stream:
            while block := stream.read(1024**2):
                size += len(block)
                assert size <= item.file_size
                value.update(block)
        assert size == item.file_size
        members[item.filename] = {'bytes': size, 'sha256': value.hexdigest()}
sys.path.insert(0, str(tools))
from pooled_runtime import BOUNDED_PROFILE, compatibility_patch_bytes, load_profile
from windows_session_timing import extract_delivery, file_inventory

work = root / 'collection-replay-v1'
work.mkdir()
delivered = work / 'delivered'
extract_delivery(root / 'artifact.zip', archive_identity, members, delivered)
assert (delivered / 'protocol.json').read_bytes() == protocol_bytes
q = read(delivered / 'qualification.json')
assert q['source_commit'] == source and q['protocol'] == descriptor(delivered / 'protocol.json')
assert q['diagnostic_completed'] is False and q['order'] == p['order'] and q['limits'] == p['limits']
assert all(q[key] is False for key in ['release_qualified', 'package_qualified', 'public_or_lifecycle_qualified'])
assert q['cases'] == [] and set(q['builds']) == set(q['commands']) == {'msvc'} and 'assets' not in q
assert q['builds']['msvc']['construction_completed'] is False
assert q['error'] == "RuntimeError('owned command left live descendant processes')"
commands = q['commands']['msvc']
expected_commands = ['clone-native', 'checkout-native', 'clone-llama', 'checkout-llama',
    'inference-patch-check-0', 'inference-patch-0', 'inference-patch-check-1', 'inference-patch-1',
    'inference-patch-check-2', 'inference-patch-2', 'cmake-version', 'git-line-ending-config', 'sdk-build']
assert [row['name'] for row in commands] == expected_commands
assert commands == read(delivered / 'evidence-msvc/commands.json')
assert all(row['process_tree_reaped'] is True for row in commands)
assert all(type(row['returncode']) is int and row['returncode'] == 0 and 'error' not in row for row in commands[:-1])
assert commands[-1]['error'] == q['error'] and 'returncode' not in commands[-1]
assert all(row['process_supervision'] == 'Windows kill-on-close Job with gated worker' for row in commands)
assert not any(name.startswith(('sdk-', 'evidence-clang-cl/')) or '/case-' in name
               or Path(name).suffix.lower() in {'.f32', '.dll', '.exe', '.obj', '.lib'} for name in members)
_, lock = load_profile(BOUNDED_PROFILE)
evidence = delivered / 'evidence-msvc'
assert (evidence / 'bounded-inference.patch').read_bytes() == compatibility_patch_bytes(lock)
for patch in lock['runtime']['compatibility_patches']:
    assert descriptor(evidence / Path(patch['path']).name) == {key: patch[key] for key in ['bytes', 'sha256']}
prior = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4'
assert (evidence / 'diagnostic-inference.diff').read_bytes() == (prior / 'diagnostic-inference.diff').read_bytes()
assert (evidence / 'sdk-build.stdout.log').read_text().splitlines()[-1].startswith('==> done. Binaries in ')
log = (root / 'job.log').read_text(encoding='utf-8-sig')
assert 'Ran 8 tests' in log and 'Ran 5 tests' in log and 'owned command left live descendant processes' in log
assert file_inventory(delivered) == members
assert descriptor(root / 'artifact.zip') == archive_identity and descriptor(Path(__file__)) == collector_identity
assert all(descriptor(root / name) == value for name, value in metadata.items())
receipt = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-job-failure'
assert not receipt.exists()
receipt.mkdir()
shutil.copytree(delivered, receipt / 'evidence')
assert file_inventory(receipt / 'evidence') == members
for name, expected in metadata.items():
    shutil.copyfile(root / name, receipt / name)
    assert descriptor(receipt / name) == expected
shutil.copyfile(Path(__file__), receipt / 'collector.py')
assert descriptor(receipt / 'collector.py') == collector_identity
for name, expected in p['tools'].items():
    path = receipt / 'sources' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(git_bytes('tools/nemotron-diarization/' + name))
    assert descriptor(path) == expected
(receipt / 'artifact-members.json').write_text(json.dumps(members, indent=2) + '\n')
result = {'schema_version': 1, 'collection_completed': True, 'source_commit': source, 'run_id': run_id,
    'job_id': job_id, 'artifact_id': artifact_id, 'archive': archive_identity,
    'protocol': descriptor(delivered / 'protocol.json'), 'tools_verified': len(p['tools']),
    'repository_files_verified': len(p['repository_files']), 'artifact_members_verified': len(members),
    'supervised_commands': len(commands), 'successful_returncodes_recorded': 12, 'verified_reaped_commands': 13,
    'failure': q['error'], 'actual_windows_job_control_tests': 5, 'actual_compiler_control_tests': 8,
    'native_inference_cases_started': 0, 'clang_build_started': False, 'installed_sdk_delivered': False,
    'release_qualified': False, 'package_qualified': False, 'public_or_lifecycle_qualified': False,
    'new_native_or_compiler_or_model_calls_during_collection': 0,
    'limits': ['Build output contains completion marker, but failed supervisor did not retain the wrapped SDK command exit code.',
        'Top-level worker exited with descendants still active; original helper terminated/reaped them before failure persistence.',
        'First SDK installation/admission and all Clang/model work were never reached. No compiler speed comparison is established.',
        'Remaining process identities were not recorded; do not attribute them to a named tool from this evidence.']}
(receipt / 'collection-qualification.json').write_text(json.dumps(result, indent=2) + '\n')
files = file_inventory(receipt)
(receipt / 'collection.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
print(json.dumps({'receipt': str(receipt), 'files': len(files) + 1, 'qualification': result}))
