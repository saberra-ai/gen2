"""Admit bounded Windows SDKs with complete producer, parity and RTF evidence."""
import hashlib
import json
from pathlib import Path
import re
import shutil

from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from platform_bounded_metrics import compare_complete_60s, derive_native_probe
from pooled_runtime import BOUNDED_PROFILE, load_profile as load_runtime_profile, profile_provenance


PROTOCOL = 'bounded-windows-native.protocol.json'
RESULT_FILES = {
    'comparison.json': 'comparison.json',
    'native.json': 'native-ami.json', 'native.f32': 'native-ami.json.f32',
    'public.json': 'public.json', 'public.f32': 'public.f32',
    'native-probe.cpp': 'native_stream_probe_bounded.cpp',
}


def load_profile():
    return load_runtime_profile(BOUNDED_PROFILE)


def validate_provenance(record):
    path, lock = load_profile()
    for name, expected in profile_provenance(path, lock).items():
        if record.get(name) != expected:
            raise ValueError('bounded SDK provenance differs: ' + name)
    return lock


def validate_native_receipt(build):
    tools = Path(__file__).resolve().parent
    path, lock = load_profile()
    validate_provenance(build)
    protocol_path = tools / PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    flags = ['development_native_gates_passed', 'functional_lifecycle_gates_passed',
             'realtime_gate_passed', 'model_free_state_test_completed']
    if (any(build.get(name) is not True for name in flags)
            or build.get('schema_version') != 2 or build.get('backend') != 'cpu'
            or build.get('release_qualified') is not False
            or not build.get('platform', '').startswith('Windows-')
            or build.get('machine', '').lower() not in {'amd64', 'x86_64'}
            or build.get('actual_reported_version') != lock['runtime']['reported_version']
            or build.get('runner_sha256') != protocol['tools']['platform_bounded_windows_native.py']['sha256']
            or build.get('protocol_sha256') != descriptor(protocol_path)['sha256']
            or build.get('completed_lifecycle_tests') != protocol['required_lifecycle_tests']
            or not re.fullmatch('[0-9a-f]{40}', build.get('source_commit', ''))):
        raise ValueError('completed bounded Windows SDK qualification required')
    pins = {'tools/nemotron-diarization/' + name: value for name, value in protocol['tools'].items()}
    pins.update(protocol['repository_files'])
    pins['tools/nemotron-diarization/' + PROTOCOL] = descriptor(protocol_path)
    verify_git_sources(tools.parents[1], build['source_commit'], pins)
    required = {'sdk-build', 'sdk-install', 'model-free-state-test', 'native-probe-build',
                'version-only', 'native-inference', 'public-inference', 'native-lifecycle-list',
                'windows-provenance', 'windows-default-loader-control'}
    required.update('native-lifecycle-' + str(i) for i in range(7))
    required.update('inference-patch-' + str(i) for i in range(3))
    required.update('inference-patch-check-' + str(i) for i in range(3))
    commands = build.get('commands', [])
    names = [row['name'] for row in commands]
    if (len(names) != len(set(names)) or not required <= set(names)
            or any(type(row.get('returncode')) is not int or row['returncode'] != 0 for row in commands)
            or not build.get('compiler_audit')):
        raise ValueError('bounded SDK command/source-audit evidence incomplete')
    patch_order = [name for name in names if name.startswith('inference-patch-')]
    if patch_order != [name for i in range(3) for name in ['inference-patch-check-' + str(i), 'inference-patch-' + str(i)]]:
        raise ValueError('bounded SDK patch application order differs')
    original = (tools / 'native_stream_probe_valid_mel.cpp').read_bytes()
    derived = derive_native_probe(original)
    probe = build.get('native_probe_source', {})
    if (probe.get('original') != descriptor(tools / 'native_stream_probe_valid_mel.cpp')
            or probe.get('derived') != {'bytes': len(derived), 'sha256': hashlib.sha256(derived).hexdigest()}):
        raise ValueError('bounded SDK derived native probe differs')
    return path, lock


def verify_results(root, build, protocol, *, packaged):
    """Replay retained 6000-row evidence; no DLL, helper, model or subprocess call."""
    names = {key: key if packaged else value for key, value in RESULT_FILES.items()}
    read = lambda name: json.loads((root / names[name]).read_text(encoding='utf-8-sig'))
    public, native = read('public.json'), read('native.json')
    comparison = compare_complete_60s(public, native,
        (root / names['native.f32']).read_bytes(), (root / names['public.f32']).read_bytes())
    expected_pcm = protocol['cases']['actual_pcm_sha256']
    if (comparison != read('comparison.json') or comparison['realtime_gate_passed'] is not True
            or any(value.get('input_pcm_f32le_sha256') != expected_pcm or value.get('input_repeats') != 1
                   for value in [public, native])):
        raise ValueError('bounded SDK complete parity/PCM/RTF evidence differs')
    derived = root / names['native-probe.cpp']
    tools = Path(__file__).resolve().parent
    if (descriptor(derived) != build['native_probe_source']['derived']
            or derived.read_bytes() != derive_native_probe((tools / 'native_stream_probe_valid_mel.cpp').read_bytes())):
        raise ValueError('bounded SDK retained native probe differs')


def provenance_records(build_path, lock_path, lock):
    tools = Path(__file__).resolve().parent
    records = {'native-receipt.json': build_path,
               'compiler-audit.json': build_path.parent / 'platform-llama-materialized-audit.json',
               'runtime-profile.json': lock_path, 'native-protocol.json': tools / PROTOCOL}
    records.update({Path(item['path']).name: tools / item['path'] for item in lock['runtime']['compatibility_patches']})
    records.update({name: build_path.parent / source for name, source in RESULT_FILES.items()})
    return records


def copy_provenance(build_path, destination):
    build = json.loads(build_path.read_text(encoding='utf-8-sig'))
    lock_path, lock = validate_native_receipt(build)
    records = provenance_records(build_path, lock_path, lock)
    if descriptor(records['compiler-audit.json']) != build['compiler_audit']:
        raise ValueError('bounded SDK compiler audit differs from receipt')
    protocol = json.loads(records['native-protocol.json'].read_text())
    verify_results(build_path.parent, build, protocol, packaged=False)
    destination.mkdir()
    for name, source in records.items():
        shutil.copy2(source, destination / name)


def verify_provenance(package, manifest):
    root = package / 'provenance'
    receipt = root / 'native-receipt.json'
    if descriptor(receipt)['sha256'] != manifest['build_receipt_sha256']:
        raise ValueError('packaged bounded SDK receipt differs')
    build = json.loads(receipt.read_text(encoding='utf-8-sig'))
    lock_path, lock = validate_native_receipt(build)
    validate_provenance(manifest)
    tools = Path(__file__).resolve().parent
    expected = {'runtime-profile.json': descriptor(lock_path), 'compiler-audit.json': build['compiler_audit'],
                'native-protocol.json': descriptor(tools / PROTOCOL)}
    expected.update({Path(item['path']).name: {key: item[key] for key in ['bytes', 'sha256']}
                     for item in lock['runtime']['compatibility_patches']})
    for name, record in expected.items():
        if descriptor(root / name) != record:
            raise ValueError('packaged bounded SDK provenance differs: ' + name)
    if {path.name for path in root.iterdir()} != set(expected) | set(RESULT_FILES) | {'native-receipt.json'}:
        raise ValueError('unexpected bounded SDK provenance files')
    verify_results(root, build, json.loads((root / 'native-protocol.json').read_text()), packaged=True)
    return build
