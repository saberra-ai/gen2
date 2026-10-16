"""Collect a completed construction attempt; read payload bytes, never load them."""
import hashlib
import json
from pathlib import Path
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
from macos_cpu_architecture import audit_portable_commands
from package_macos import verify_files, verify_package_provenance
from sdk_archive import inventory, restore_archive
from ctest_output import require_single_test_pass
from platform_valid_mel_sources import verify_completed_source
from runtime_profiles import load_profile
from build_valid_mel_candidate import source_inventory

ci = json.loads((cache / 'platform-ci/portable-38967939615.final.json').read_text())
source = cache / 'portable-macos-construction-ci-v2/evidence'
artifact_records = json.loads((cache / 'platform-ci/portable-38967939615.artifacts.json').read_text())['artifacts']
assert len(artifact_records) == 1
artifact = artifact_records[0]
assert artifact['id'] == 12074849757 and artifact['name'] == 'nemotron-macos-portable-construction'
assert artifact['workflow_run']['id'] == 38967939615 and artifact['workflow_run']['head_sha'] == ci['headSha']
archive_path = cache / 'portable-macos-construction-ci-v2/artifact.zip'
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
commit = 'e1c219a56782eabf7cf612935550871c62561c5d'
assert ci['databaseId'] == 38967939615 and ci['status'] == 'completed'
assert ci['headSha'] == q['source_commit'] == commit
assert descriptor(source / 'protocol.json') == q['protocol']
assert (source / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', commit + ':tools/nemotron-diarization/macos-portable-construction-v2.protocol.json'], cwd=repo)
assert q['model_calls'] == 0 and q['model_or_frontend_execution_admitted'] is False and q['release_qualified'] is False
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = subprocess.check_output(['git', 'show', commit + ':' + prefix + name], cwd=repo)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
        assert descriptor(repo / (prefix + name)) == expected
success = ci['conclusion'] == 'success' and q['construction_and_package_passed'] is True
assert success
result = {'schema_version': 1, 'source_commit': commit, 'ci_url': ci['url'],
          'portable_construction_passed': success, 'model_calls': 0, 'release_qualified': False,
          'real_model_or_receiver_inference_qualified': False, 'protocol': q['protocol'],
          'artifact': artifact, 'extracted_evidence_byte_matches_opaque_zip': True,
          'scope': 'Exact fresh-host construction attempt and byte-only receiver collection; no model, frontend, native library, state/version query or app is executed during collection.'}

def sections(path):
    raw = path.read_bytes()
    magic, cpu, subcpu, kind, count, command_bytes, flags, reserved = struct.unpack_from('<8I', raw)
    assert magic == 0xfeedfacf and cpu == 0x100000c and kind == 6
    end = 32 + command_bytes
    assert end <= len(raw)
    cursor = 32
    result = {}
    for _ in range(count):
        command, size = struct.unpack_from('<II', raw, cursor)
        assert size >= 8 and cursor + size <= end
        if command == 0x19:
            assert size >= 72
            segment = struct.unpack_from('<II16sQQQQiiII', raw, cursor)
            nsections = segment[-2]
            assert size == 72 + nsections * 80
            for index in range(nsections):
                row = struct.unpack_from('<16s16sQQIIIIIIII', raw, cursor + 72 + index * 80)
                key = (row[1].rstrip(b'\0').decode(), row[0].rstrip(b'\0').decode())
                assert key not in result
                section_size, offset, section_flags = row[3], row[4], row[8]
                zerofill = section_flags & 0xff in {1, 0xc, 0x12}
                assert zerofill or (offset >= end and offset + section_size <= len(raw))
                result[key] = (row[2:], None if zerofill else hashlib.sha256(raw[offset:offset + section_size]).hexdigest())
        cursor += size
    assert cursor == end and result
    return result


if success:
    assert all(c['returncode'] == 0 for c in q['commands'])
    assert [c['name'] for c in q['commands']] == ['sandbox-controls', 'prepare', 'build', 'package']
    build = json.loads((source / 'build/build.json').read_text())
    assert build['model_free_state_test_completed'] and build['version_function_only_executed']
    assert not build['model_instantiated_or_inference_performed']
    assert all(c['returncode'] == 0 for c in build['commands'])
    require_single_test_pass((source / 'build/model-free-state-test.stdout.log').read_text(), 'diar_state')
    assert q['cpu_architecture'] == build['cpu_architecture'] == audit_portable_commands(
        json.loads((source / 'build/compile_commands.json').read_text()))
    assert json.loads((source / 'build/compiler-inputs.json').read_text()) == build['actual_compiler_sources']
    materialized = json.loads((source / 'build/llama-materialized-audit.json').read_text())
    assert materialized['exact_independent_replay_match'] and len(materialized['patch_series']) == 19
    assert descriptor(source / 'build/pristine-llama.tar') == materialized['archive']
    # Compare the full emitted materialized source inventory and every compiler
    # input to the previously qualified pinned source, using bytes and git only.
    local_source = cache / 'NeMo-Speech-valid-mel-tagged-v1'
    local_build = local_source / 'build/cpu-diar'
    local_materialized = local_build / '_deps/llama.cpp'
    lock_path, lock = load_profile()
    patch = tools / lock['runtime']['compatibility_patches'][0]['path']
    verify_completed_source(local_source, local_build, lock, patch, False)
    assert source_inventory(local_materialized) == materialized['files']
    producer_materialized = Path(materialized['compiled_tree_path'])
    producer_root = producer_materialized.parents[3]
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=local_source).decode().split('\0'))
    for filename, expected in build['actual_compiler_sources'].items():
        path = Path(filename)
        if path.is_relative_to(producer_materialized):
            local_file = local_materialized / path.relative_to(producer_materialized)
        else:
            relative = path.relative_to(producer_root).as_posix()
            assert relative in tracked
            local_file = local_source / relative
        assert descriptor(local_file) == expected

    restored = {}
    for stem, key in [('portable-cpu-sdk', 'sdk_archive'), ('portable-cpu-runtime', 'package_archive')]:
        manifest = json.loads((source / (stem + '-files.json')).read_text())
        assert {k: v for k, v in manifest.items() if k != 'files'} == q[key]
        destination = cache / 'portable-macos-construction-ci-v2' / ('restored-' + stem)
        restore_archive(source / manifest['archive'], manifest, destination)
        assert inventory(destination) == manifest['files']
        restored[key] = destination
    sdk, package = restored['sdk_archive'], restored['package_archive']
    manifest = json.loads((package / 'package.json').read_text())
    verify_files(package, manifest['files'])
    verify_package_provenance(package, manifest)
    assert (package / 'provenance/build.json').read_bytes() == (source / 'build/build.json').read_bytes()
    mapping = json.loads((source / 'source-library-mapping.json').read_text())
    assert set(mapping) == set(manifest['libraries'])
    for name, record in manifest['libraries'].items():
        original = source / 'source-libraries' / name
        assert descriptor(original) == {k: mapping[name][k] for k in ['bytes', 'sha256']}
        assert mapping[name]['source'] == record['source'] and descriptor(original)['sha256'] == record['source_sha256']
        assert sections(original) == sections(package / 'lib' / name)
        if record['sdk_library'] is not None:
            assert descriptor(original) == descriptor(sdk / 'lib' / record['sdk_library'])
    result.update(cpu_architecture=q['cpu_architecture'], libraries=len(mapping),
                  sdk_archive=q['sdk_archive'], package_archive=q['package_archive'],
                  all_archives_restored_with_exact_inventory=True,
                  all_original_and_private_library_sections_match=True,
                  independent_ggml_patch_replay_count=19,
                  host=json.loads((source / 'host.json').read_text()))

out = repo / 'benches/results/nemotron-diarization/macos-arm64' / (
    '2026-10-16-portable-cpu-construction-v2')
assert not out.exists()
out.mkdir(parents=True)
for path in source.rglob('*'):
    if not path.is_file() or path.is_symlink() or path.is_relative_to(source / 'source-libraries') or path.suffix in {'.tar', '.gz'}:
        continue
    destination = out / 'raw' / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
for path in package.rglob('*'):
    if path.is_file() and not path.is_symlink() and not path.is_relative_to(package / 'lib'):
        destination = out / 'package-metadata' / path.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
for name in ['portable-38967939615.final.json', 'portable-38967939615.artifacts.json']:
    shutil.copy2(cache / 'platform-ci' / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
result['opaque_download'] = descriptor(cache / 'portable-macos-construction-ci-v2/artifact.zip')
result['limits'] = ['No real-model/hardware baseline, lifecycle, GUI, microphone, performance, quality or public release qualification.',
                    'SDK/runtime/original-library/pristine-source archive bytes remain in cache/CI; Git retains exact descriptors and saved build/source evidence.']
(out / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'success': success, 'members': len(files) + 1, 'output': str(out)}))
