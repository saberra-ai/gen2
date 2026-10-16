"""Retain a completed receiver attempt; replay saved bytes, never native code."""
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

commit = '7db170d0c8f17b3aea670769577d2d3cd2deaf5b'
ci = json.loads((cache / 'platform-ci/portable-runtime-38972811283.final.json').read_text())
assert ci['databaseId'] == 38972811283 and ci['status'] == 'completed' and ci['headSha'] == commit
source = cache / 'portable-macos-runtime-ci-v2/evidence'
archive_path = cache / 'portable-macos-runtime-ci-v2/artifact.zip'
artifacts = json.loads((cache / 'platform-ci/portable-runtime-38972811283.artifacts.json').read_text())['artifacts']
assert len(artifacts) == 1
artifact = artifacts[0]
assert artifact['id'] == 12080325210 and artifact['name'] == 'nemotron-macos-portable-runtime'
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
    ['git', 'show', commit + ':tools/nemotron-diarization/macos-portable-runtime-v2.protocol.json'], cwd=repo)
for group, prefix in [('tools', 'tools/nemotron-diarization/'), ('repository_files', '')]:
    for name, expected in p[group].items():
        raw = subprocess.check_output(['git', 'show', commit + ':' + prefix + name], cwd=repo)
        assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected == descriptor(repo / (prefix + name))
passed = ci['conclusion'] == 'success' and r.get('fixed_receiver_runtime_passed') is True
record = {'schema_version': 1, 'source_commit': commit, 'ci_url': ci['url'], 'artifact': artifact,
          'opaque_download': descriptor(archive_path), 'all_delivered_files_match_opaque_zip': True,
          'fixed_receiver_runtime_passed': passed, 'producer_reported_runtime_passed': r.get('fixed_receiver_runtime_passed'),
          'ci_conclusion': ci['conclusion'], 'model_execution_admitted': r['model_execution_admitted'],
          'inference_commands_started': r['inference_commands_started'], 'underlying_forward_count_instrumented': False,
          'release_qualified': False, 'scope': 'Fresh M1 virtual offline receiver attempt plus byte-only collection, not an app, accuracy or release qualification.',
          'limits': p['limits'] + ['Collection executes only Python byte/hash/trace comparisons and git reads. No model/native library/version/state/app is executed.']}
assert passed, 'success collector requires actual terminal success; preserve failures separately'
if passed:
    assert len(ci['jobs']) == 1 and ci['jobs'][0]['name'] == 'receiver' and ci['jobs'][0]['conclusion'] == 'success'
    api = r['producer_artifact']
    assert api['id'] == p['producer']['id'] and api['workflow_run']['id'] == p['producer']['run_id']
    assert api['workflow_run']['head_sha'] == p['producer']['source_commit']
    assert api['digest'] == 'sha256:' + p['producer']['zip']['sha256']
    assert r['model_execution_admitted'] and r['lifecycle_tests_passed'] == p['lifecycle_tests'] and r['release_qualified'] is False
    names = ['compile-controls', 'native-helper-build', 'load-helper-build', 'lifecycle-build', 'public-helper-build',
             'runtime-controls', 'version-only', 'missing-library', 'lifecycle-list', 'cpu-rejects-metal',
             'native', 'public', *['lifecycle-' + str(i) for i in range(7)]]
    assert [c['name'] for c in r['commands']] == names
    for command in r['commands']:
        assert command['returncode'] == (1 if command['name'] in {'missing-library', 'cpu-rejects-metal'} else 0)
        assert command['argv'][0:2] == ['/usr/bin/sandbox-exec', '-f']
        assert not command.get('forced_process_group_termination') and not command.get('disk_pressure_abort_free_bytes')
    producer = cache / 'portable-macos-construction-ci-v2'
    assert descriptor(producer / 'artifact.zip') == p['producer']['zip']
    package = producer / 'restored-portable-cpu-runtime'
    original_manifest = json.loads((producer / 'evidence/portable-cpu-runtime-files.json').read_text())
    assert inventory(package) == original_manifest['files']
    manifest = json.loads((package / 'package.json').read_text())
    assert manifest == json.loads((source / 'package.json').read_text())
    verify_files(package, manifest['files'])
    verify_package_provenance(package, manifest)
    helper_manifest = json.loads((source / 'receiver-helpers-files.json').read_text())
    assert {k: v for k, v in helper_manifest.items() if k != 'files'} == r['helper_archive']
    helpers = cache / 'portable-macos-runtime-ci-v2/restored-helpers'
    restore_archive(source / helper_manifest['archive'], helper_manifest, helpers)
    build = json.loads((source / 'build-inputs.json').read_text())
    assert build['source_pins'] == p['repository_files'] and build['helper_inputs'] == p['helper_inputs']
    assert build['model_or_frontend_execution_during_build'] is False
    acquisitions = json.loads((source / 'acquisitions.json').read_text())
    lock = json.loads((tools / 'runtime-valid-mel.lock.json').read_text())
    assets = {a['name']: {'bytes': a['size'], 'sha256': a['sha256']} for a in lock['artifacts'] if a['name'] in p['asset_names']}
    assets.update({'helper/' + name: {'bytes': item['size'], 'sha256': item['sha256']} for name, item in p['helper_inputs'].items()})
    assert acquisitions == assets
    assert build['compiled_helpers'] == helper_manifest['files'] == inventory(helpers)
    static = json.loads((source / 'helper-load-commands.json').read_text())
    assert set(static) == {'native-stream', 'load-probe', 'native-lifecycle', 'public-stream'}
    for name, item in static.items():
        assert descriptor(helpers / name) == item['binary'] and item['architectures'] == 'arm64'
        header = struct.unpack_from('<4I', (helpers / name).read_bytes())
        assert header[0] == 0xfeedfacf and header[1] == 0x100000c and header[3] == 2
    remote_root = Path(r['commands'][0]['cwd'])
    assert all(c['cwd'] == str(remote_root) for c in r['commands'])
    compile_env = r['commands'][0]['environment']
    assert compile_env['RUSTFLAGS'] == '-C target-cpu=generic -C link-arg=-mmacosx-version-min=26.0'
    assert compile_env['CFLAGS'] == compile_env['CXXFLAGS'] == '-march=armv8.2-a+dotprod+fp16'
    assert compile_env == build['compiler_environment']
    cpp = next(c for c in r['commands'] if c['name'] == 'native-helper-build')
    cpp_source = next(a for a in cpp['argv'] if a.endswith('/tools/nemotron-diarization/native_stream_probe_valid_mel.cpp'))
    remote_repo = Path(cpp_source).parents[2]
    assert (source / 'compile.sb').read_text() == policy(denied_reads=[remote_root / 'work/assets'],
        readonly=[remote_root / 'sdk', remote_root / 'work/nemo', remote_root / 'producer', remote_repo, '/opt/homebrew'])
    assert (source / 'runtime.sb').read_text() == policy(denied_reads=[remote_root / 'sdk', remote_root / 'producer',
        remote_repo, remote_root / 'cargo-target', remote_root / 'helper-source', '/opt/homebrew',
        Path(compile_env['CARGO_HOME']), Path(compile_env['RUSTUP_HOME']), remote_root.parent / 'portable-delivery'],
        readonly=[remote_root / 'work', remote_root / 'sdk', remote_root / 'producer', remote_repo])
    for c in r['commands']:
        compile_phase = c['name'] in names[:5]
        assert 'DYLD_PRINT_LIBRARIES' not in c['environment']
        assert c['argv'][2] == str(remote_root / ('compile.sb' if compile_phase else 'runtime.sb'))
        if not compile_phase:
            assert 'DYLD_PRINT_LIBRARIES' not in c['environment']
            assert c['argv'][3:5] == ['/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1']
    assert r['inference_commands_started'] == ['cpu-rejects-metal', 'native', 'public'] + [
        'lifecycle-' + str(i) for i in range(7) if i != 5]
    remote_package = remote_root / 'work/nemo'
    remote_binary = remote_root / 'work/bin'
    image_pattern = r'(dyld\[\d+\]: <[A-Fa-f0-9-]+> )(/[^\r\n]+)'
    audits = {}
    for name, expected in r['loaded_images'].items():
        command = next(c for c in r['commands'] if c['name'] == name)
        executable = command['argv'][5]
        assert expected['executable'] == executable and Path(executable).parent == remote_binary
        assert expected['executable_sha256'] == descriptor(helpers / Path(executable).name)['sha256']
        def remap(match):
            path = match[2]
            if path.startswith(('/usr/lib/', '/System/Library/')):
                return match[0]
            if path == executable:
                return match[1] + str(helpers / Path(path).name)
            relative = Path(path).relative_to(remote_package / 'lib')
            assert relative.as_posix() in set(manifest['libraries']) | {'libnemo_speech_asr_c.dylib'}
            return match[1] + str(package / 'lib' / relative)
        text = (source / (name + '.stderr.log')).read_text()
        mapped = re.sub(image_pattern, remap, text)
        actual = audit_loaded_libraries(mapped, package, manifest, helpers / Path(executable).name,
                                        name not in {'lifecycle-list', 'lifecycle-5'})
        # Reverse only these exact local roots to compare the original producer record.
        normalized = json.loads(json.dumps(actual).replace(str(package), str(remote_package))
                                .replace(str(helpers), str(remote_binary)))
        assert normalized == expected
        audits[name] = actual['private_library_count']
    assert set(audits) == {'version-only', 'lifecycle-list', 'cpu-rejects-metal', 'native', 'public'} | {'lifecycle-' + str(i) for i in range(7)}
    public = json.loads((source / 'results/public.json').read_text())
    native = json.loads((source / 'results/native.json').read_text())
    raw, actual_raw = (source / 'results/native.json.f32').read_bytes(), (source / 'results/public.f32').read_bytes()
    assert len(raw) == len(actual_raw) == 192000 and raw == actual_raw
    assert public['retained_probabilities'] == {'first_frame': 0, 'rows': 6000} and native['retained_rows'] == 6000
    assert native['input_pcm_f32le_sha256'] == public['input_pcm_f32le_sha256'] == p['pcm_sha256']
    comparison = compare(public, native, raw, actual_raw)
    assert comparison == r['comparison']
    for index, name in enumerate(p['lifecycle_tests']):
        text = (source / ('lifecycle-' + str(index) + '.stdout.log')).read_text()
        assert re.search(r'^test ' + re.escape(name) + r' \.\.\. ok$', text, re.MULTILINE)
        assert '1 passed; 0 failed; 0 ignored;' in text
    assert (source / 'version-only.stdout.log').read_text().strip() == p['runtime_provenance']['runtime_reported_version']
    missing = (source / 'missing-library.stderr.log').read_text()
    assert not (source / 'missing-library.stdout.log').read_bytes() and 'Library not loaded:' in missing and p['missing_library_name'] in missing
    assert 'requested Metal device is unavailable in the native SDK' in (source / 'cpu-rejects-metal.stderr.log').read_text()
    assert json.loads((source / 'reference/ami.json').read_text()) == {'segments': [
        {'start': t['start_seconds'], 'end': t['end_seconds'], 'speaker': t['speaker']} for t in native['turns']]}
    assert (source / 'reference/ami.json.f32').read_bytes() == raw
    assert json.loads((source / 'host.json').read_text())['cpu'] == p['expected_receiver_cpu']
    assert r['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
    assert r['storage']['released'] and r['storage']['reserve_reported_allocated_bytes'] >= 256 * 1024**2
    record.update(comparison=comparison, lifecycle_tests_passed=p['lifecycle_tests'],
                  private_image_records=sum(audits.values()), audited_processes=len(audits),
                  host=json.loads((source / 'host.json').read_text()), storage=r['storage'],
                  helper_archive=r['helper_archive'])
else:
    record['producer_error'] = r.get('error')
    record['storage'] = r.get('storage')
    record['partial_commands'] = r['commands']

out = repo / 'benches/results/nemotron-diarization/macos-arm64' / (
    '2026-10-16-portable-cpu-offline-runtime-v2' if passed else '2026-10-16-portable-cpu-offline-runtime-v2-failure')
assert not out.exists()
out.mkdir(parents=True)
for path in source.rglob('*'):
    if path.is_file() and path.suffix != '.gz':
        dest = out / 'raw' / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
for name in ['portable-runtime-38972811283.final.json', 'portable-runtime-38972811283.artifacts.json']:
    shutil.copy2(cache / 'platform-ci' / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
(out / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'passed': passed, 'members': len(files) + 1, 'output': str(out)}))
