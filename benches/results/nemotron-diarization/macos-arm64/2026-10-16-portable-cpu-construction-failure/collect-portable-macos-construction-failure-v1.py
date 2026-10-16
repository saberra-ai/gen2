"""Retain the specific CTest-summary construction failure; no native execution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import stat

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import descriptor

ci = json.loads((cache / 'platform-ci/portable-38966947865.final.json').read_text())
source = cache / 'portable-macos-construction-ci-v1/evidence'
artifact_records = json.loads((cache / 'platform-ci/portable-38966947865.artifacts.json').read_text())['artifacts']
assert len(artifact_records) == 1
artifact = artifact_records[0]
assert artifact['id'] == 12075017664 and artifact['name'] == 'nemotron-macos-portable-construction'
assert artifact['workflow_run']['id'] == 38966947865 and artifact['workflow_run']['head_sha'] == ci['headSha']
archive_path = cache / 'portable-macos-construction-ci-v1/artifact.zip'
assert archive_path.stat().st_size == artifact['size_in_bytes']
assert 'sha256:' + descriptor(archive_path)['sha256'] == artifact['digest']
with zipfile.ZipFile(archive_path) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names))
    expected_files = set()
    for member in archive.infolist():
        name = Path(member.filename)
        assert not name.is_absolute() and '..' not in name.parts
        assert not stat.S_ISLNK(member.external_attr >> 16)
        if not member.is_dir():
            expected_files.add(member.filename)
            assert (source / name).read_bytes() == archive.read(member)
    assert {str(path.relative_to(source)) for path in source.rglob('*') if path.is_file()} == expected_files
    assert not any(path.is_symlink() for path in source.rglob('*'))
q = json.loads((source / 'qualification.json').read_text())
p = json.loads((source / 'protocol.json').read_text())
commit = '5e5c4d0a389eabc57dc80bc421e804858a088131'
assert ci['databaseId'] == 38966947865 and ci['status'] == 'completed'
assert ci['headSha'] == q['source_commit'] == commit
assert descriptor(source / 'protocol.json') == q['protocol']
assert (source / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', commit + ':tools/nemotron-diarization/macos-portable-construction.protocol.json'], cwd=repo)
assert q['model_calls'] == 0 and q['model_or_frontend_execution_admitted'] is False and q['release_qualified'] is False
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = subprocess.check_output(['git', 'show', commit + ':' + prefix + name], cwd=repo)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
assert ci['conclusion'] == 'failure' and q['construction_and_package_passed'] is False
assert [(x['name'], x['returncode']) for x in q['commands']] == [('sandbox-controls', 0), ('prepare', 0), ('build', 1)]
build_commands = json.loads((source / 'build/commands.json').read_text())
assert len(build_commands) == 24 and all(x['returncode'] == 0 for x in build_commands)
assert [x['phase'] for x in build_commands] == ['configure', 'llama-archive', *['llama-patch-' + str(i) for i in range(19)], 'build', 'install', 'model-free-state-test']
state_output = (source / 'build/model-free-state-test.stdout.log').read_text()
assert '1/1 Test #7: diar_state' in state_output and 'Passed' in state_output and '100% tests passed out of 1' in state_output
assert 'required model-free state test did not execute and pass' in (source / 'build.stderr.log').read_text()
assert not (source / 'build/build.json').exists() and not (source / 'source-library-mapping.json').exists()
result = {'schema_version': 1, 'source_commit': commit, 'ci_url': ci['url'],
          'portable_construction_passed': False, 'model_calls': 0, 'release_qualified': False,
          'real_model_or_receiver_inference_qualified': False, 'protocol': q['protocol'],
          'artifact': artifact, 'extracted_evidence_byte_matches_opaque_zip': True,
          'scope': 'The specific failed M1 construction attempt, retained by byte-only collection. Native compilation/install and the state command returned0; the producer validator rejected changed CTest summary wording. Packaging, terminal architecture/source audits and version query never ran.',
          'actual_host': json.loads((source / 'host.json').read_text()),
          'completed_native_build_commands': 24, 'saved_state_test_passed': True,
          'failure_stage': 'producer state-test output validation',
          'storage': q['storage'], 'build_commands': build_commands}
success = False

out = repo / 'benches/results/nemotron-diarization/macos-arm64' / (
    '2026-10-16-portable-cpu-construction' if success else '2026-10-16-portable-cpu-construction-failure')
assert not out.exists()
out.mkdir(parents=True)
for path in source.rglob('*'):
    if not path.is_file() or path.is_symlink() or path.is_relative_to(source / 'source-libraries') or path.suffix in {'.tar', '.gz'}:
        continue
    destination = out / 'raw' / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
for name in ['portable-38966947865.final.json', 'portable-38966947865.artifacts.json']:
    shutil.copy2(cache / 'platform-ci' / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
result['opaque_download'] = descriptor(cache / 'portable-macos-construction-ci-v1/artifact.zip')
result['limits'] = ['No real-model/hardware baseline, lifecycle, GUI, microphone, performance, quality or public release qualification.',
                    'Opaque CI ZIP and pristine GGML source tar remain in cache/CI. This failure emitted no SDK/runtime filesystem archive or original-library payload mapping; no receiver artifact is admitted.']
(out / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'success': success, 'members': len(files) + 1, 'output': str(out)}))
