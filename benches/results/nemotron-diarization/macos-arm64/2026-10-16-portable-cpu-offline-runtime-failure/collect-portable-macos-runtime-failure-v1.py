"""Retain the pre-model launcher-trace failure; inspect saved bytes only."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import zipfile
import stat

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import descriptor
from compare_stream_valid_mel import compare
from macos_loaded_libraries import audit_loaded_libraries
from sdk_archive import inventory, restore_archive
from package_macos import verify_files, verify_package_provenance
from qualify_macos_portable_runtime import policy

commit = '63aabe75e875d390450b456e5d4a5683258f8c83'
ci = json.loads((cache / 'platform-ci/portable-runtime-38970855035.final.json').read_text())
assert ci['databaseId'] == 38970855035 and ci['status'] == 'completed' and ci['headSha'] == commit
source = cache / 'portable-macos-runtime-ci-v1/evidence'
archive_path = cache / 'portable-macos-runtime-ci-v1/artifact.zip'
artifacts = json.loads((cache / 'platform-ci/portable-runtime-38970855035.artifacts.json').read_text())['artifacts']
assert len(artifacts) == 1
artifact = artifacts[0]
assert artifact['id'] == 12076527982 and artifact['name'] == 'nemotron-macos-portable-runtime'
assert artifact['workflow_run']['id'] == ci['databaseId'] and artifact['workflow_run']['head_sha'] == commit
assert artifact['size_in_bytes'] == archive_path.stat().st_size
assert artifact['digest'] == 'sha256:' + descriptor(archive_path)['sha256']
with zipfile.ZipFile(archive_path) as archive:
    members = archive.infolist()
    assert len(members) == len({m.filename for m in members})
    expected_files = set()
    for member in members:
        path = Path(member.filename)
        assert not path.is_absolute() and '..' not in path.parts and not stat.S_ISLNK(member.external_attr >> 16)
        if not member.is_dir():
            expected_files.add(member.filename)
            assert (source / path).read_bytes() == archive.read(member)
    assert {str(f.relative_to(source)) for f in source.rglob('*') if f.is_file()} == expected_files
    assert not any(f.is_symlink() for f in source.rglob('*'))
p = json.loads((source / 'protocol.json').read_text())
r = json.loads((source / 'qualification.json').read_text())
assert r['source_commit'] == commit and descriptor(source / 'protocol.json') == r['protocol']
assert (source / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', commit + ':tools/nemotron-diarization/macos-portable-runtime.protocol.json'], cwd=repo)
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = subprocess.check_output(['git', 'show', commit + ':' + prefix + name], cwd=repo)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected == descriptor(repo / (prefix + name))
assert ci['conclusion'] == 'failure' and r['fixed_receiver_runtime_passed'] is False
assert r['model_execution_admitted'] is False and r['inference_commands_started'] == []
assert r['error'] == "ValueError('unexpected non-system dyld image: /usr/bin/sandbox-exec')"
assert [x['name'] for x in r['commands']] == ['compile-controls', 'native-helper-build', 'load-helper-build',
    'lifecycle-build', 'public-helper-build', 'runtime-controls', 'version-only']
assert all(x['returncode'] == 0 for x in r['commands']) and r['loaded_images'] == {}
assert (source / 'version-only.stdout.log').read_text().strip() == p['runtime_provenance']['runtime_reported_version']
manifest = json.loads((source / 'package.json').read_text())
command = r['commands'][-1]
executable = command['argv'][3]
remote_root = Path(command['cwd'])
remote_package = remote_root / 'work/nemo/lib'
observed = set()
for image in re.findall(r'dyld\[\d+\]: <[A-Fa-f0-9-]+> (/[^\r\n]+)', (source / 'version-only.stderr.log').read_text()):
    if image.startswith(('/usr/lib/', '/System/Library/')):
        continue
    if image in {'/usr/bin/sandbox-exec', executable}:
        observed.add(image)
        continue
    relative = Path(image).relative_to(remote_package).as_posix()
    if relative == 'libnemo_speech_asr_c.dylib':
        relative = manifest['entry_library']
    assert relative in manifest['libraries']
    observed.add(relative)
assert observed == set(manifest['libraries']) | {'/usr/bin/sandbox-exec', executable}
helper_manifest = json.loads((source / 'receiver-helpers-files.json').read_text())
assert {k: v for k, v in helper_manifest.items() if k != 'files'} == r['helper_archive']
helpers = cache / 'portable-macos-runtime-ci-v1/restored-helpers'
restore_archive(source / helper_manifest['archive'], helper_manifest, helpers)
build = json.loads((source / 'build-inputs.json').read_text())
assert build['source_pins'] == p['repository_files'] and build['helper_inputs'] == p['helper_inputs']
assert build['compiled_helpers'] == helper_manifest['files'] == inventory(helpers)
assert build['model_or_frontend_execution_during_build'] is False
record = {'schema_version': 1, 'source_commit': commit, 'ci_url': ci['url'], 'artifact': artifact,
          'opaque_download': descriptor(archive_path), 'all_delivered_files_match_opaque_zip': True,
          'fixed_receiver_runtime_passed': False, 'model_execution_admitted': False,
          'inference_commands_started': [], 'release_qualified': False,
          'saved_version_query_returned_expected_tag': True,
          'all_seven_native_or_control_commands_returned_zero': True,
          'saved_trace_expected_private_components': len(manifest['libraries']),
          'saved_trace_has_sandbox_launcher_and_no_other_foreign_images': True,
          'failure_is_strict_image_audit_rejection_of_launcher': True,
          'helper_archive_restored_byte_exact_without_execution': r['helper_archive'],
          'host': json.loads((source / 'host.json').read_text()), 'storage': r['storage'],
          'scope': 'Specific failed M1 receiver before model admission. Builds/controls/version return0; inherited DYLD tracing includes sandbox-exec, which strict audit rejects. Missing-library and all inference/lifecycle cases never started.',
          'limits': p['limits'] + ['Byte-only collection restores compiled helpers and diagnoses the saved image history without loading/executing any helper, library, version/state function, model or app.']}
passed = False

out = repo / 'benches/results/nemotron-diarization/macos-arm64' / (
    '2026-10-16-portable-cpu-offline-runtime' if passed else '2026-10-16-portable-cpu-offline-runtime-failure')
assert not out.exists()
out.mkdir(parents=True)
for path in source.rglob('*'):
    if path.is_file() and path.suffix != '.gz':
        dest = out / 'raw' / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
for name in ['portable-runtime-38970855035.final.json', 'portable-runtime-38970855035.artifacts.json']:
    shutil.copy2(cache / 'platform-ci' / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
(out / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'passed': passed, 'members': len(files) + 1, 'output': str(out)}))
