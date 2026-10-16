"""Receiver audit of opaque Windows runtime/helpers and recorded offline calls."""
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from compare_stream import compare
from package_windows import CRT, verify
from qualify_linux_package import require_lifecycle
from windows_archive import byte_inventory, digest, restore_archive
from windows_pe import inspect_pe
from windows_runtime_audit import loaded_modules

base, package, output = map(Path, sys.argv[1:])
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
ci, receipt, frozen = read(Path(str(base) + '-ci.json')), read(base / 'receipt.json'), read(base / 'frozen-inputs.json')
source = '855a83c2f56ec0267cd0b915a63eb55654b9dfec'
assert ci['conclusion'] == 'success' and ci['headSha'] == receipt['source_commit'] == frozen['source_commit'] == source
assert receipt['offline_container_gates_passed'] is True and receipt['release_qualified'] is False and receipt['redistribution_qualified'] is False
assert digest(base / 'frozen-inputs.json') == receipt['frozen_inputs_sha256']
for name, expected in frozen['tools'].items():
    data = subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/' + name], cwd=repo)
    assert hashlib.sha256(data).hexdigest() == expected == digest(repo / 'tools/nemotron-diarization' / name), name
construction, archive = read(package / 'receipt.json'), read(package / 'development-windows-runtime-files.json')
assert construction['package_construction_passed'] is True and construction['native_receipt_sha256'] == frozen['native_receipt_sha256']
assert construction['package_manifest_sha256'] == frozen['package_manifest_sha256'] == receipt['package_manifest_sha256']
for key in ['archive', 'archive_bytes', 'archive_sha256']:
    assert archive[key] == construction['archive'][key] == frozen['archive'][key]
native = read(base / 'native/results/native.json')
public = read(base / 'public/results/public.json')
raw = (base / 'native/results/native.json.f32').read_bytes()
comparison = compare(public, native, raw, (base / 'public/results/public.f32').read_bytes())
assert comparison == receipt['native_public_comparison']
assert public['input_pcm_f32le_sha256'] == native['input_pcm_f32le_sha256'] and native['input_repeats'] == public['input_repeats'] == 1
assert json.loads((base / 'public/results/child.stdout.log').read_text()) == public
require_lifecycle((base / 'lifecycle-list/results/child.stdout.log').read_text())
require_lifecycle((base / 'lifecycle/results/child.stdout.log').read_text(), completed=True)
helper_manifest = read(base / 'development-windows-test-helpers-files.json')
assert {key: value for key, value in helper_manifest.items() if key != 'files'} == frozen['test_helper_archive']
audits, commands_reconstructed = {}, {}
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    restored = restore_archive(package / archive['archive'], archive, root / 'receiver relocated runtime ü')
    manifest = verify(restored)
    assert manifest == read(package / 'runtime/package.json')
    assert byte_inventory(restored) == byte_inventory(package / 'runtime') == byte_inventory(package / 'relocated runtime ü') == archive['files']
    assert digest(restored / 'package.json') == frozen['package_manifest_sha256']
    for name, row in frozen['files'].items():
        if name == 'relocated runtime ü':
            assert row == archive['files']['.']
        elif name.startswith('relocated runtime ü/'):
            assert row == archive['files'][name.removeprefix('relocated runtime ü/')]
    helpers = restore_archive(base / helper_manifest['archive'], helper_manifest, root / 'receiver helpers ü')
    assert {path.name for path in helpers.iterdir()} == CRT | set(frozen['helpers'])
    for name, expected in frozen['helpers'].items():
        pe = inspect_pe(helpers / name)
        assert pe == read(base / (name + '-pe.json'))
        assert {key: pe[key] for key in ['bytes', 'sha256']} == expected
    for name in CRT:
        assert (helpers / name).read_bytes() == (restored / 'lib' / name).read_bytes()
    for name, entry in helper_manifest['files'].items():
        assert frozen['files']['bin' if name == '.' else 'bin/' + name] == entry
    lock = read(repo / 'tools/nemotron-diarization/runtime.lock.json')
    for row in lock['artifacts']:
        if row['role'] != 'reference':
            assert frozen['files']['assets/' + row['name']] == {'type': 'file', 'bytes': row['size'], 'sha256': row['sha256']}
    lifecycle = read(base / 'lifecycle-inputs.json')
    expected_reference = {'segments': [{'start': turn['start_seconds'], 'end': turn['end_seconds'], 'speaker': turn['speaker']} for turn in native['turns']]}
    reference_bytes = (json.dumps(expected_reference) + '\r\n').encode()
    assert lifecycle == {**frozen['files'], 'reference': {'type': 'directory'},
                        'reference/ami.json': {'type': 'file', 'bytes': len(reference_bytes), 'sha256': hashlib.sha256(reference_bytes).hexdigest()},
                        'reference/ami.json.f32': {'type': 'file', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
    library, assets = 'C:\\inputs\\relocated runtime ü\\lib\\' + manifest['entry_library'], 'C:\\inputs\\assets'
    wav = assets + '\\ami_en2002d_2132.wav'
    calls = [('native', 'native-probe.exe', [assets + '\\Nemotron-3-Diarization.q8_0.gguf', wav, '1', '-1', 'C:\\results\\native.json'], True, False),
             ('public', 'public-stream.exe', [assets, library, wav, '1', 'C:\\results\\public.f32'], True, False),
             ('lifecycle-list', 'native-lifecycle.exe', ['--ignored', '--list'], False, False),
             ('lifecycle', 'native-lifecycle.exe', ['--ignored', '--test-threads=1'], True, False),
             ('missing-private-library', 'public-stream.exe', [assets, library, wav, '1', 'C:\\results\\negative.f32'], False, True)]
    for name, executable, argv, capture, negative in calls:
        control, child = read(base / name / 'results/controls.json'), read(base / name / 'results/child.json')
        assert control['input_read'] and control['input_write_denied'] and control['input_write_win32_error'] == 5
        assert control['development_tools_absent_from_path'] and control['visual_studio_directory_absent']
        assert control['process_architecture'] == 'AMD64' and not control['operational_non_loopback_interfaces']
        assert child['completed'] is True and not child.get('error') and ((child['returncode'] == 0) != negative)
        observed, = read(base / (name + '-inspect.stdout.log'))
        assert observed['Image'] == receipt['image_id'] == frozen['image']['Id']
        assert observed['HostConfig']['Isolation'] == 'process' and observed['HostConfig']['NetworkMode'] == 'none'
        assert observed['State']['ExitCode'] == 0 and observed['State']['Running'] is False
        mounts = {row['Destination'].lower(): row for row in observed['Mounts']}
        assert set(mounts) == {'c:\\inputs', 'c:\\results'} and mounts['c:\\inputs']['RW'] is False and mounts['c:\\results']['RW'] is True
        expected_mount = PureWindowsPath('D:/a/_temp/windows-runtime-offline') / name
        assert PureWindowsPath(mounts['c:\\inputs']['Source']) == expected_mount / 'inputs ü'
        assert PureWindowsPath(mounts['c:\\results']['Source']) == expected_mount / 'results'
        environment = {'SystemRoot': 'C:\\Windows', 'WINDIR': 'C:\\Windows', 'PATH': 'C:\\Windows\\System32;C:\\Windows',
                       'TEMP': 'C:\\results\\tmp', 'TMP': 'C:\\results\\tmp', 'USERPROFILE': 'C:\\results', 'LOCALAPPDATA': 'C:\\results\\local',
                       'GEN2_NEMOTRON_ASSETS': assets, 'GEN2_NEMO_LIBRARY': library, 'GEN2_NEMOTRON_REFERENCE': 'C:\\inputs\\reference', 'GEN2_NEMO_EXECUTION': 'cpu'}
        if executable == 'native-probe.exe':
            environment['PATH'] = 'C:\\inputs\\relocated runtime ü\\lib;' + environment['PATH']
        command = {'executable': 'C:\\inputs\\bin\\' + executable, 'argv': argv, 'command_line': subprocess.list2cmdline(argv),
                   'capture_modules': capture, 'expected_failure': negative, 'environment': environment}
        command_bytes = (json.dumps(command, indent=2) + '\n').replace('\n', '\r\n').encode()
        before = dict(frozen['files'] if name in {'native', 'public'} else lifecycle)
        if negative:
            before.pop('relocated runtime ü/lib/gen2_nemo_windows_ggml-base.dll')
        before['command.json'] = {'type': 'file', 'bytes': len(command_bytes), 'sha256': hashlib.sha256(command_bytes).hexdigest()}
        assert read(base / name / 'frozen-inputs.json') == before
        assert control['input_sha256'] == before['control.txt']['sha256']
        commands_reconstructed[name] = command_bytes
        if capture:
            audited = loaded_modules(child, manifest['libraries'], executable, frozen['helpers'][executable])
            assert audited == read(base / name / 'loaded-modules.json')
            audits[name] = audited
    negative = base / 'missing-private-library/results'
    assert 'os error 126' in (negative / 'child.stderr.log').read_text() and not (negative / 'negative.f32').exists()
    assert receipt['negative_control_passed'] is True
command_records = read(base / 'commands.json')
assert all(row['returncode'] == 0 for row in command_records)
for name, *_ in calls:
    assert [row['name'] for row in command_records].count(name + '-remove') == 1
output.mkdir(parents=True, exist_ok=False)
for path in base.iterdir():
    if path.is_file() and path.suffix != '.zip':
        shutil.copy2(path, output / path.name)
    elif path.is_dir():
        shutil.copytree(path, output / path.name)
(output / 'package').mkdir()
for name in ['receipt.json', 'development-windows-runtime-files.json']:
    shutil.copy2(package / name, output / 'package' / name)
shutil.copy2(package / 'runtime/package.json', output / 'package/package.json')
shutil.copytree(package / 'runtime/metadata', output / 'package/metadata')
shutil.copytree(package / 'runtime/notices', output / 'package/notices')
(output / 'reconstructed-commands').mkdir()
for name, data in commands_reconstructed.items():
    (output / 'reconstructed-commands' / (name + '.json')).write_bytes(data)
shutil.copy2(Path(str(base) + '-ci.json'), output / 'ci.json')
shutil.copy2(Path(str(base) + '-ci.log'), output / 'github-job.log')
shutil.copy2(__file__, output / 'collect-windows-offline.py')
qualification = {'schema_version': 1, 'source_commit': source, 'run_url': ci['url'], 'ci_job_id': ci['jobs'][0]['databaseId'],
                 'offline_container_gates_passed': True, 'release_qualified': False, 'redistribution_qualified': False,
                 'native_source_commit': receipt['native_source_commit'], 'image_id': receipt['image_id'],
                 'receiver_harness_note': 'Initial command-spec reconstruction used LF for indented JSON and stopped before collection. Corrected to the producer Windows CRLF bytes on every line; all five exact frozen hashes then matched without rerunning inference.',
                 'native_public_comparison_recomputed': comparison, 'actual_pcm_sha256': native['input_pcm_f32le_sha256'],
                 'all_seven_lifecycle_tests_reparsed': True, 'all_five_containers_controls_reparsed': True,
                 'all_five_command_specs_reconstructed_exact_frozen_bytes': True,
                 'runtime_archive_entries_reverified': len(archive['files']), 'helper_archive_entries_reverified': len(helper_manifest['files']),
                 'private_dlls_observed_per_helper': {name: len(row['private_native']) for name, row in audits.items()},
                 'crt_paths_observed_per_helper': {name: [row['path'] for row in data['crt']] for name, data in audits.items()},
                 'negative_missing_private_library_error126_no_output': True,
                 'runtime_archive': construction['archive'], 'test_helper_archive': frozen['test_helper_archive'],
                 'limits': receipt['limits'] + ['macOS receiver rehashed opaque archives/helpers/DLLs and reparsed recorded results/controls; actual Windows calls, immutability and system-module disk hashes remain CI observations.',
                                              'Observed module closure applies to recorded helpers/calls, not every potential dynamic load or process namespace.']}
(output / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
         for path in output.rglob('*') if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'evidence_files': len(files), 'native_public': comparison, 'private_modules': qualification['private_dlls_observed_per_helper']}))
