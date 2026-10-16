"""Admit only a completed tagged platform SDK before package construction."""
import hashlib
import json
from pathlib import Path
import re
import shutil

from build_valid_mel_candidate import descriptor
from runtime_profiles import load_profile, validate_provenance


def validate_native_receipt(build):
    tools = Path(__file__).resolve().parent
    lock_path, lock = load_profile()
    validate_provenance(build)
    protocol_path = tools / 'valid-mel-platform-native.protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if (build.get('schema_version') != 2 or build.get('backend') != 'cpu'
            or build.get('development_native_gates_passed') is not True
            or build.get('actual_reported_version') != lock['runtime']['reported_version']
            or build.get('protocol_sha256') != descriptor(protocol_path)['sha256']
            or build.get('model_free_state_test_completed') is not True
            or build.get('completed_lifecycle_tests') != protocol['required_lifecycle_tests']
            or not re.fullmatch('[0-9a-f]{40}', build.get('source_commit', ''))):
        raise ValueError('completed tagged platform SDK qualification required')
    for key, root in [('tools', tools), ('repository_files', tools.parents[1])]:
        for name, expected in protocol[key].items():
            if descriptor(root / name) != expected:
                raise ValueError('SDK consumer source differs from frozen input: ' + name)
    commands = build.get('commands', [])
    required = {'sdk-build', 'sdk-install', 'model-free-state-test', 'version-only',
                'native-inference', 'public-inference', 'native-lifecycle-list'} | {
                    'native-lifecycle-' + str(i) for i in range(7)}
    names = [c['name'] for c in commands]
    if (len(names) != len(set(names)) or not required <= set(names)
            or any(c.get('returncode') != 0 for c in commands)
            or not build.get('compiler_audit')):
        raise ValueError('platform SDK command/source-audit evidence incomplete')
    return lock_path, lock


def copy_provenance(build_path, destination):
    """Retain exact build receipt, compiler audit, inference patch and protocol."""
    tools = Path(__file__).resolve().parent
    build = json.loads(build_path.read_text())
    lock_path, lock = validate_native_receipt(build)
    audit = build_path.parent / 'platform-llama-materialized-audit.json'
    if descriptor(audit) != build['compiler_audit']:
        raise ValueError('platform SDK compiler audit differs from receipt')
    destination.mkdir()
    records = [('native-receipt.json', build_path), ('compiler-audit.json', audit),
               ('runtime-profile.json', lock_path),
               ('native-protocol.json', tools / 'valid-mel-platform-native.protocol.json'),
               ('inference.patch', tools / lock['runtime']['compatibility_patches'][0]['path'])]
    for name, source in records:
        shutil.copy2(source, destination / name)


def verify_provenance(package, manifest):
    tools = Path(__file__).resolve().parent
    root = package / 'provenance'
    receipt = root / 'native-receipt.json'
    if hashlib.sha256(receipt.read_bytes()).hexdigest() != manifest['build_receipt_sha256']:
        raise ValueError('packaged platform native receipt differs')
    build = json.loads(receipt.read_text())
    lock_path, lock = validate_native_receipt(build)
    validate_provenance(manifest)
    expected = {'runtime-profile.json': descriptor(lock_path),
                'compiler-audit.json': build['compiler_audit'],
                'native-protocol.json': descriptor(tools / 'valid-mel-platform-native.protocol.json'),
                'inference.patch': descriptor(tools / lock['runtime']['compatibility_patches'][0]['path'])}
    for name, record in expected.items():
        if descriptor(root / name) != record:
            raise ValueError('packaged platform provenance differs: ' + name)
    if {p.name for p in root.iterdir()} != set(expected) | {'native-receipt.json'}:
        raise ValueError('unexpected platform provenance files')
    return build
