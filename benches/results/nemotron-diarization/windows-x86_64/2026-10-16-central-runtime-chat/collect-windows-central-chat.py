"""Receiver audit of five private Windows DLLs, verified central prerequisite, and nine recorded offline calls."""
import hashlib
import io
import tarfile
import json
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
base, package, output, setup_base = map(Path, sys.argv[1:])
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
ci = read(Path(str(base) + '-ci.json'))
source = ci['headSha']
workflow_bytes = subprocess.check_output(['git', 'show', source + ':.github/workflows/diarization-windows-central.yml'], cwd=repo)
assert b'--output "${{ runner.temp }}/windows-central-runtime-offline"' in workflow_bytes
# Re-run the exact producer's audited Python tools, preserving older receipts
# when later additive notice/packaging changes exist in the working checkout.
source_temp = tempfile.TemporaryDirectory()
source_root = Path(source_temp.name)
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(
        ['git', 'archive', source, 'tools/nemotron-diarization', 'examples/nemotron_chat_coexistence.rs'], cwd=repo))) as bundle:
    bundle.extractall(source_root, filter='data')
sys.path.insert(0, str(source_root / 'tools/nemotron-diarization'))
from compare_stream import compare
from package_windows import CRT, verify
from qualify_linux_package import require_lifecycle
from windows_archive import byte_inventory, digest, restore_archive
from windows_pe import inspect_pe
from windows_runtime_audit import loaded_modules
from linux_chat_audit import audit_results
from windows_central_state import audit_state

base, package, output, setup_base = map(Path, sys.argv[1:])
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
ci, receipt, frozen = read(Path(str(base) + '-ci.json')), read(base / 'receipt.json'), read(base / 'frozen-inputs.json')
chat_frozen = read(base / 'chat-frozen-inputs.json')
setup = read(setup_base / 'receipt.json')
setup_frozen = read(setup_base / 'frozen-inputs.json')
central = frozen['central_setup']['central_crt']
assert setup['source_commit'] == setup_frozen['source_commit'] == source
assert setup['setup_and_fresh_image_state_passed'] is True
assert digest(setup_base / 'receipt.json') == frozen['central_setup']['receipt_sha256']
assert digest(setup_base / 'frozen-inputs.json') == setup['frozen_inputs_sha256'] == frozen['central_setup']['frozen_inputs_sha256']
assert setup['provisioned_image_id'] == frozen['image']['Id']
assert setup['base_image_id'] == frozen['central_setup']['base_image']['Id']
minimum = read(repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-static/minimum-x64-cab-inspection.json')
assert central == {row['runtime_name']: {key: row[key] for key in ['bytes','sha256']} for row in minimum['selected_dlls']}
central_state = read(setup_base / 'fresh-state/state.json')
audit_state(central_state['after'], central, read(source_root / 'tools/nemotron-diarization/windows-vc-runtime.lock.json')['version'])
assert central_state['mode'] == 'Observe' and central_state['installer_executed'] is False
assert receipt['missing_central_runtime_control_passed'] is True
assert chat_frozen['source_commit'] == source
assert receipt['chat_diarization_gates_passed'] is True
assert digest(base / 'chat-frozen-inputs.json') == receipt['chat_frozen_inputs_sha256']
assert ci['conclusion'] == 'success' and ci['headSha'] == receipt['source_commit'] == frozen['source_commit'] == source
assert receipt['offline_container_gates_passed'] is True and receipt['release_qualified'] is False and receipt['redistribution_qualified'] is False
assert digest(base / 'frozen-inputs.json') == receipt['frozen_inputs_sha256']
for name, expected in {**frozen['tools'], **chat_frozen['tools']}.items():
    data = subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/' + name], cwd=repo)
    assert hashlib.sha256(data).hexdigest() == expected == digest(source_root / 'tools/nemotron-diarization' / name), name
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
    assert manifest['format'] == 'gen2-nemo-windows-cpu-central-v1' and len(manifest['libraries']) == 5
    assert {path.name for path in helpers.iterdir()} == set(frozen['helpers'])
    for name, expected in frozen['helpers'].items():
        pe = inspect_pe(helpers / name)
        assert pe == read(base / (name + '-pe.json'))
        assert {key: pe[key] for key in ['bytes', 'sha256']} == expected
    for name, entry in helper_manifest['files'].items():
        assert frozen['files']['bin' if name == '.' else 'bin/' + name] == entry
    lock = read(repo / 'tools/nemotron-diarization/runtime.lock.json')
    for row in lock['artifacts']:
        if row['role'] != 'reference':
            assert frozen['files']['assets/' + row['name']] == {'type': 'file', 'bytes': row['size'], 'sha256': row['sha256']}
    chat_helpers_manifest = read(base / 'development-windows-chat-helpers-files.json')
    assert chat_helpers_manifest == chat_frozen['chat_helper_archive']
    chat_helpers = restore_archive(base / chat_helpers_manifest['archive'], chat_helpers_manifest, root / 'receiver chat helpers ü')
    assert {path.name for path in chat_helpers.iterdir()} == set(frozen['helpers']) | {'chat-coexistence.exe'}
    for name, entry in chat_helpers_manifest['files'].items():
        assert chat_frozen['files']['bin' if name == '.' else 'bin/' + name] == entry
        if name in frozen['helpers']:
            assert (chat_helpers / name).read_bytes() == (helpers / name).read_bytes()
    chat_pe = inspect_pe(chat_helpers / 'chat-coexistence.exe')
    assert chat_pe == read(base / 'chat-helper-pe.json')
    openmp_pe = read(base / 'chat-openmp-pe.json')
    static_openmp, = [row for row in minimum['selected_dlls'] if row['runtime_name'] == 'vcomp140.dll']
    assert {key: value for key, value in openmp_pe.items() if key != 'scope'} == static_openmp
    chat_helper = {key: chat_pe[key] for key in ['bytes', 'sha256']}
    openmp = {key: openmp_pe[key] for key in ['bytes', 'sha256']}
    provenance = read(base / 'chat-openmp-provenance.json')
    assert digest(base / 'chat-openmp-provenance.json') == chat_frozen['openmp_provenance_sha256']
    assert {key: provenance[key] for key in openmp} == openmp and provenance['signature_status'] == 'Valid'
    assert 'O=Microsoft Corporation,' in provenance['signer_subject']
    assert PureWindowsPath(provenance['path']) == PureWindowsPath('C:/Windows/System32/vcomp140.dll')
    assert provenance['central_setup_receipt_sha256'] == digest(setup_base / 'receipt.json')
    observed_openmp, = [row for row in central_state['after']['files'] if row['name'] == 'vcomp140.dll']
    assert all(provenance[key] == value for key,value in observed_openmp.items())
    assert openmp == central['vcomp140.dll']
    chat_lock = read(source_root / 'tools/nemotron-diarization/linux-chat.lock.json')
    assert chat_frozen['chat_fixture_files'] == {row['name']: {'bytes': row['size'], 'sha256': row['sha256']} for row in chat_lock['artifacts']}
    assert chat_frozen['fingerprints'] == {'executable': chat_helper['sha256'], 'chat_model': chat_lock['artifacts'][0]['sha256'],
        'diarization_model': frozen['files']['assets/Nemotron-3-Diarization.q8_0.gguf']['sha256'],
        'library': manifest['libraries'][manifest['entry_library']]['sha256'],
        'wav': frozen['files']['assets/ami_en2002d_2132.wav']['sha256']}
    assert chat_frozen['example_sha256'] == digest(source_root / 'examples/nemotron_chat_coexistence.rs')
    assert chat_frozen['chat_llama_revision'] == subprocess.check_output(['git', 'ls-tree', source, 'vendor/llama-cpp-sys-2/llama.cpp'], cwd=repo, text=True).split()[2]
    assert chat_frozen['chat_features'] == ['backend-llamacpp', 'backend-diar-nemo']
    chat_results = root / 'chat-results'
    chat_results.mkdir()
    stages = {}
    for order in ['isolated', 'chat-first', 'diar-first']:
        shutil.copytree(base / order / 'results' / order, chat_results / order)
        stages[order] = {}
        for stage in (['isolated'] if order == 'isolated' else ['both-loaded', 'reloaded']):
            rows = read(chat_results / order / (stage + '.modules.json'))
            stages[order][stage] = loaded_modules({'stable_snapshots': 1, 'modules': rows}, manifest['libraries'],
                'chat-coexistence.exe', chat_helper, private_required=order != 'isolated', openmp=openmp, central_crt=central)
    assert stages == read(base / 'chat-module-stages.json')
    chat_comparisons = audit_results(chat_results, native, raw, chat_frozen['fingerprints'])
    assert chat_comparisons == receipt['chat_comparisons'] == read(base / 'chat-comparison.json')
    lifecycle = read(base / 'lifecycle-inputs.json')
    expected_reference = {'segments': [{'start': turn['start_seconds'], 'end': turn['end_seconds'], 'speaker': turn['speaker']} for turn in native['turns']]}
    reference_bytes = (json.dumps(expected_reference) + '\r\n').encode()
    assert lifecycle == {**chat_frozen['files'], 'reference': {'type': 'directory'},
                        'reference/ami.json': {'type': 'file', 'bytes': len(reference_bytes), 'sha256': hashlib.sha256(reference_bytes).hexdigest()},
                        'reference/ami.json.f32': {'type': 'file', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
    library, assets = 'C:\\inputs\\relocated runtime ü\\lib\\' + manifest['entry_library'], 'C:\\inputs\\assets'
    wav = assets + '\\ami_en2002d_2132.wav'
    calls = [('native', 'native-probe.exe', [assets + '\\Nemotron-3-Diarization.q8_0.gguf', wav, '1', '-1', 'C:\\results\\native.json'], True, False),
             ('public', 'public-stream.exe', [assets, library, wav, '1', 'C:\\results\\public.f32'], True, False),
             ('lifecycle-list', 'native-lifecycle.exe', ['--ignored', '--list'], False, False),
             ('lifecycle', 'native-lifecycle.exe', ['--ignored', '--test-threads=1'], True, False),
             ('missing-private-library', 'public-stream.exe', [assets, library, wav, '1', 'C:\\results\\negative.f32'], False, True)]
    calls.append(('missing-central-runtime', 'public-stream.exe', [assets, library, wav, '1', 'C:\\results\\absent.f32'], False, True))
    for order in ['isolated', 'chat-first', 'diar-first']:
        calls.append((order, 'chat-coexistence.exe', [order, 'C:\\chat\\Qwen3-0.6B-Q4_K_M.gguf', assets, library, wav,
                                                   'C:\\results\\' + order], True, False))
    for name, executable, argv, capture, negative in calls:
        control, child = read(base / name / 'results/controls.json'), read(base / name / 'results/child.json')
        assert control['input_read'] and control['input_write_denied'] and control['input_write_win32_error'] == 5
        assert control['development_tools_absent_from_path'] and control['visual_studio_directory_absent']
        assert control['process_architecture'] == 'AMD64' and not control['operational_non_loopback_interfaces']
        assert child['completed'] is True and not child.get('error') and ((child['returncode'] == 0) != negative)
        observed, = read(base / (name + '-inspect.stdout.log'))
        expected_image = frozen['central_setup']['base_image']['Id'] if name == 'missing-central-runtime' else frozen['image']['Id']
        assert observed['Image'] == expected_image
        assert receipt['image_id'] == frozen['image']['Id']
        assert observed['HostConfig']['Isolation'] == 'process' and observed['HostConfig']['NetworkMode'] == 'none'
        assert observed['State']['ExitCode'] == 0 and observed['State']['Running'] is False
        mounts = {row['Destination'].lower(): row for row in observed['Mounts']}
        is_chat = executable == 'chat-coexistence.exe'
        expected_destinations = {'c:\\inputs', 'c:\\results'} | ({'c:\\chat'} if is_chat else set())
        assert set(mounts) == expected_destinations and mounts['c:\\inputs']['RW'] is False and mounts['c:\\results']['RW'] is True
        if is_chat:
            assert mounts['c:\\chat']['RW'] is False
            assert PureWindowsPath(mounts['c:\\chat']['Source']) == PureWindowsPath('D:/a/_temp/windows-central-runtime-offline/locked chat fixture ü')
        expected_mount = PureWindowsPath('D:/a/_temp/windows-central-runtime-offline') / name
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
        before = dict(chat_frozen['files'] if name in {'native', 'public'} else lifecycle)
        if name == 'missing-private-library':
            before.pop('relocated runtime ü/lib/gen2_nemo_windows_ggml-base.dll')
        before['command.json'] = {'type': 'file', 'bytes': len(command_bytes), 'sha256': hashlib.sha256(command_bytes).hexdigest()}
        assert read(base / name / 'frozen-inputs.json') == before
        assert control['input_sha256'] == before['control.txt']['sha256']
        assert (base / name / 'inputs ü/command.json').read_bytes() == command_bytes
        commands_reconstructed[name] = command_bytes
        if capture:
            audited = loaded_modules(child, manifest['libraries'], executable,
                chat_helper if is_chat else frozen['helpers'][executable],
                central_crt=central, **({'private_required': name != 'isolated', 'openmp': openmp} if is_chat else {}))
            assert audited == read(base / name / 'loaded-modules.json')
            audits[name] = audited
    negative = base / 'missing-private-library/results'
    assert 'os error 126' in (negative / 'child.stderr.log').read_text() and not (negative / 'negative.f32').exists()
    assert receipt['negative_control_passed'] is True
    absent = base / 'missing-central-runtime/results'
    assert read(absent / 'child.json')['returncode'] == -1073741515 and not (absent / 'absent.f32').exists()
    assert all(name in read(base / 'missing-central-runtime/frozen-inputs.json') for name in
               ['relocated runtime ü/lib/' + name for name in manifest['libraries']])
command_records = read(base / 'commands.json')
assert all(row['returncode'] == 0 for row in command_records)
for name, *_ in calls:
    assert [row['name'] for row in command_records].count(name + '-remove') == 1
    created = (base / (name + '-create.stdout.log')).read_text(encoding='utf-8-sig').strip()
    inspected, = read(base / (name + '-inspect.stdout.log'))
    assert len(created) == 64 and all(char in '0123456789abcdef' for char in created) and inspected['Id'] == created
    for suffix in ['start','inspect','remove']:
        command, = [row for row in command_records if row['name'] == name + '-' + suffix]
        assert command['argv'][-1] == created
assert len({(base / (name + '-create.stdout.log')).read_text(encoding='utf-8-sig').strip() for name,*_ in calls}) == 9
with tempfile.TemporaryDirectory() as temporary:
    audited_setup = Path(temporary) / 'setup-receipt'
    subprocess.run([sys.executable, str(Path(__file__).with_name('collect-windows-central-setup-full.py')),
                    str(setup_base), str(base) + '-ci.json', str(audited_setup)], check=True)
    # A sub-receipt explicitly qualifies setup/state only; the parent adds model behavior below.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(audited_setup, output / 'central-setup')
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
shutil.copy2(__file__, output / 'collect-windows-central-chat.py')
shutil.copy2(Path(__file__).with_name('collect-windows-central-setup-full.py'), output / 'collect-windows-central-setup-full.py')
qualification = {'schema_version': 1, 'source_commit': source, 'run_url': ci['url'], 'ci_job_id': ci['jobs'][0]['databaseId'],
                 'offline_container_gates_passed': True, 'release_qualified': False, 'redistribution_qualified': False,
                 'native_source_commit': receipt['native_source_commit'], 'image_id': receipt['image_id'],
                 'chat_diarization_gates_passed': True, 'chat_comparisons_recomputed': chat_comparisons,
                 'chat_model_bytes_checked_by_receiver': False,
                 'openmp_installed_observation_matches_independently_retained_publisher_CAB': openmp,
                 'central_system32_runtime_model_calls_passed': True,
                 'central_setup_receipt_sha256': digest(setup_base / 'receipt.json'),
                 'no_app_local_crt_or_openmp_payloads': True,
                 'chat_helper_archive': {key: row for key, row in chat_helpers_manifest.items() if key != 'files'},
                 'native_public_comparison_recomputed': comparison, 'actual_pcm_sha256': native['input_pcm_f32le_sha256'],
                 'all_seven_lifecycle_tests_reparsed': True, 'all_nine_containers_controls_reparsed': True,
                 'all_nine_original_command_specs_reparsed_exact_frozen_bytes': True,
                 'all_nine_distinct_canonical_container_id_chains_reparsed': True,
                 'runtime_archive_entries_reverified': len(archive['files']), 'helper_archive_entries_reverified': len(helper_manifest['files']),
                 'private_dlls_observed_per_helper': {name: len(row['private_native']) for name, row in audits.items()},
                 'crt_paths_observed_per_helper': {name: [row['path'] for row in data['crt']] for name, data in audits.items()},
                 'negative_missing_private_library_error126_no_output': True,
                 'negative_missing_central_runtime_status_dll_not_found_no_output': True,
                 'runtime_archive': construction['archive'], 'test_helper_archive': frozen['test_helper_archive'],
                 'limits': receipt['limits'] + ['macOS receiver rehashed opaque five-DLL runtime/helper bytes, replayed setup-input/static-publisher binding, and reparsed nine recorded calls, actual stage modules, ten chat outcomes and six native comparisons; large model bytes, actual Windows calls, immutability and system-module disk hashes remain CI observations.',
                                              'Observed module closure applies to recorded helpers/calls, not every potential dynamic load or process namespace.']}
(output / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
         for path in output.rglob('*') if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'evidence_files': len(files), 'native_public': comparison, 'private_modules': qualification['private_dlls_observed_per_helper']}))
