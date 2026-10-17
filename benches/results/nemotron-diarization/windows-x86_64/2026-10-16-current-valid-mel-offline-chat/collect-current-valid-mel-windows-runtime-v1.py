"""Verify saved Windows tagged runtime evidence, without native or model execution."""
import os
os.environ.update(GIT_NO_LAZY_FETCH="1",GIT_NO_REPLACE_OBJECTS="1")
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import descriptor
from runtime_profiles import validate_provenance
from windows_archive import byte_inventory, restore_archive
from frozen_git_sources import verify_git_sources

base = cache / 'current-valid-mel-windows-runtime-ci-v1/evidence'
collector = cache / 'current-valid-mel-windows-runtime-collection-tools-v1'
out = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-current-valid-mel-offline-chat'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
r = read(base / 'qualification.json'); p = read(base / 'protocol.json'); f = read(base / 'frozen-inputs.json')
run=read(base.parent/'run.json');run['id']=run['databaseId'];run['head_sha']=run['headSha'];job,=run['jobs'];job['id']=job['databaseId'];artifact=read(base.parent/'download-identity.json')
assert not out.exists()
assert run['id']==artifact['workflow_run']['id']==38990890737 and artifact['id']==12085543613 and not artifact['expired']
assert run['conclusion']==job['conclusion']=='success'
assert run['head_sha']==artifact['workflow_run']['head_sha']==r['source_commit']==f['source_commit']=='4083013d56c59f67ad766ade1c331c3dd274eb67'
archive=base.parent/'artifact.zip'
assert artifact['digest']=='sha256:'+descriptor(archive)['sha256'] and artifact['size_in_bytes']==archive.stat().st_size
with zipfile.ZipFile(archive)as z:
    assert {i.filename for i in z.infolist()if not i.is_dir()}=={x.relative_to(base).as_posix()for x in base.rglob('*')if x.is_file()}
    for item in z.infolist():
        if not item.is_dir():assert (base/item.filename).read_bytes()==z.read(item)
assert (base / 'protocol.json').read_bytes() == subprocess.check_output(['git','show',run['head_sha']+':tools/nemotron-diarization/current-valid-mel-windows-runtime.protocol.json'],cwd=repo)
assert descriptor(base / 'protocol.json') == r['protocol']
for key in ['tools','repository_files']:assert p[key]==f[key]
pins={'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()};pins.update(p['repository_files']);verify_git_sources(repo,run['head_sha'],pins)
for n,v in p['gitlinks'].items():verify_git_sources(repo/n,v,p['vendor_sources'])
for n,v in p['tools'].items():assert descriptor(tools/n)==v
assert p['gitlinks'] == f['gitlinks']
for name, expected in p['gitlinks'].items():
    assert subprocess.check_output(['git', 'ls-tree', run['head_sha'], name], cwd=repo, text=True).split()[2] == expected
assert descriptor(base / 'native-receipt.json') == f['native_receipt'] == p['native_receipt']
sdk = cache / 'valid-mel-platform-ci-v1/windows/nemo-valid-mel-sdk'
assert byte_inventory(sdk) == p['sdk_inventory'] == f['sdk_inventory']
producer = read(base / 'producer-run.json')
assert producer['id'] == p['producer']['run_id'] == 38940008466
assert producer['head_sha'] == p['producer']['source_commit'] == '8fbef3e8a46a034b8e0ffa8f096409db683a923e'
assert producer['conclusion'] == 'success' and producer['repository']['full_name'] == p['producer']['repository'] == 'saberra-ai/gen2'
for key in ['package_construction_passed', 'setup_and_fresh_image_state_passed', 'offline_container_gates_passed', 'chat_diarization_gates_passed']:
    assert r[key] is True
assert r['release_qualified'] is False and r['redistribution_qualified'] is False
assert read(base / 'commands.json') == r['commands'] and len(r['commands']) == 4
assert [c['name'] for c in r['commands']] == ['construction', 'preflight', 'central-setup', 'offline-chat']
assert all(c['returncode'] == 0 for c in r['commands'])
for key, directory in [('construction_receipt', 'construction'), ('setup_receipt', 'central-setup'), ('runtime_receipt', 'runtime')]:
    assert descriptor(base / directory / 'receipt.json') == r[key]
for directory in ['construction', 'runtime']:
    validate_provenance(read(base / directory / 'receipt.json'))
restored = cache / 'current-valid-mel-windows-restored-runtime-v1'
archive_manifest = read(base / 'construction/development-windows-runtime-files.json')
if not restored.exists():
    restore_archive(base / 'construction' / archive_manifest['archive'], archive_manifest, restored)
assert byte_inventory(restored) == archive_manifest['files']
subprocess.run([sys.executable, str(collector / 'collect-windows-central-chat.py'), str(base / 'runtime'),
                str(base / 'construction'), str(out), str(base / 'central-setup')], check=True)
for name in ['qualification.json', 'protocol.json', 'frozen-inputs.json', 'commands.json', 'native-receipt.json', 'producer-run.json',
             'construction.stdout.log', 'construction.stderr.log', 'preflight.stdout.log', 'preflight.stderr.log',
             'central-setup.stdout.log', 'central-setup.stderr.log', 'offline-chat.stdout.log', 'offline-chat.stderr.log']:
    (out / 'wrapper').mkdir(exist_ok=True)
    shutil.copy2(base / name, out / 'wrapper' / name)
for name in ['run.json','download-identity.json','delivery.json']:shutil.copy2(base.parent/name,out/name)
shutil.copy2(collector/'collect-windows-central-chat.py',out/'collect-windows-central-chat.py')
for name, expected in p['tools'].items():
    dest = out / 'execution-tools' / name; dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tools / name, dest)
shutil.copy2(__file__, out / Path(__file__).name)
record = {'schema_version': 1, 'offline_container_gates_passed': True, 'chat_diarization_gates_passed': True,
          'release_qualified': False, 'redistribution_qualified': False, 'producer_source_commit': r['source_commit'],
          'collection_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
          'run_id': run['id'], 'job_id': job['id'], 'artifact_id': artifact['id'], 'artifact_zip': descriptor(archive),
          'artifact_zip_verified_against_github_digest': True, 'protocol': r['protocol'],
          'source_pins': {'tools': len(p['tools']), 'repository_files': len(p['repository_files']), 'gitlinks': p['gitlinks'], 'vendor_files':len(p['vendor_sources'])},
          'original_sdk_inventory_rehashed_entries': len(p['sdk_inventory']), 'new_native_or_neural_calls_during_collection': 0,
          'scope': 'Saved Windows Server 2022 AMD64 v2 five-private-DLL package, actual publisher VC setup, offline and CPU chat evidence. Receiver uses pure metadata/PE/array auditors only.',
          'limits': ['Actual Windows signatures, installation, System32 hashes, container controls and model execution are recorded CI observations.',
                     'Opaque runtime/helper archives were restored and byte-verified on this receiver; their binary payloads and large models are excluded from Git evidence.',
                     'No Bark/Laya, physical desktop, generic Windows baseline, heldout quality, uncontended live performance, redistribution or release qualification.']}
record['files'] = {str(path.relative_to(out)): descriptor(path) for path in sorted(out.rglob('*')) if path.is_file()}
(out / 'receiver-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'target': str(out), 'members': len(record['files']) + 1, 'protocol': r['protocol']}))
