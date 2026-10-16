"""Verify complete local Pnola app structure and saved actual dyld traces.

Static inspection and signature stripping operate on temporary copies only.
No native library, application, model or version function is executed.
"""
import json
from pathlib import Path
import plistlib
import posixpath
import re
import shutil
import subprocess
import tempfile

from build_valid_mel_candidate import descriptor
from package_macos import inspect_library, system_dependency, verify, version_tuple
from runtime_profiles import load_profile

ENTRY_ALIAS = 'libnemo_speech_asr_c.dylib'
ORT = 'libonnxruntime.dylib'


def require_closed_dependencies(dependencies, expected):
    siblings = {'@loader_path/' + name for name in expected}
    for dependency in dependencies:
        if system_dependency(dependency) and posixpath.normpath(dependency) == dependency:
            continue
        if dependency not in siblings:
            raise ValueError('app framework dependency escapes closure: ' + dependency)


def unsigned_descriptor(path):
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / 'library.dylib'; shutil.copy2(path, copy)
        subprocess.run(['/usr/bin/codesign', '--remove-signature', str(copy)], check=True, capture_output=True)
        return descriptor(copy)


def verify_app(app):
    app = app.resolve(strict=True); contents = app / 'Contents'; resources = contents / 'Resources'
    package = resources / 'nemo'; frameworks = contents / 'Frameworks'
    if any(path.is_symlink() for path in [contents, resources, package, frameworks, contents / 'MacOS', resources / 'nemotron', resources / 'runtime']):
        raise ValueError('app structural directories cannot be symlinks')
    manifest = verify(package)
    expected = set(manifest['libraries']) | {ENTRY_ALIAS, ORT}
    if {path.name for path in frameworks.iterdir()} != expected or any(path.is_symlink() or not path.is_file() for path in frameworks.iterdir()):
        raise ValueError('app Frameworks must contain the exact regular-file closure and public entry copy')
    lock = load_profile()[1]; bundle = json.loads((resources / 'nemotron/bundle.json').read_text())
    if bundle['runtime_profile'] != lock['profile'] or bundle['frame_contract'] != lock['runtime']['frame_contract']:
        raise ValueError('app model-bundle profile differs')
    for item in lock['artifacts']:
        if item['role'] in {'model', 'notice'}:
            if (resources / 'nemotron' / item['name']).is_symlink() or descriptor(resources / 'nemotron' / item['name']) != {'bytes': item['size'], 'sha256': item['sha256']}:
                raise ValueError('app model/card differs')
    license_record = json.loads(Path(__file__).with_name('notices').joinpath('openmdw-1.1/sources.json').read_text())
    if (resources / 'nemotron/LICENSE.OpenMDW-1.1').is_symlink() or descriptor(resources / 'nemotron/LICENSE.OpenMDW-1.1')['sha256'] != license_record['sha256']:
        raise ValueError('app model license differs')
    bark = resources / 'bark'
    if bark.is_symlink() or (bark / 'manifest.json').is_symlink() or descriptor(bark / 'manifest.json')['sha256'] != '8cd94f24f677b74f9e92306039491f23e3e22863205e59a5bdda322026f98f8c':
        raise ValueError('app Bark manifest differs from the qualified Rust release')
    bark_files = json.loads((bark / 'manifest.json').read_text())['files']
    if {path.name for path in bark.iterdir()} != set(bark_files) | {'manifest.json'}:
        raise ValueError('app Bark inventory differs')
    for name, expected_file in bark_files.items():
        path = bark / name
        if Path(name).name != name or path.is_symlink() or descriptor(path) != expected_file:
            raise ValueError('app Bark asset differs: ' + name)
    info = plistlib.loads((contents / 'Info.plist').read_bytes())
    if version_tuple(info['LSMinimumSystemVersion']) != version_tuple(manifest['minimum_macos']):
        raise ValueError('app deployment declaration differs from native package')
    records = {}; unsigned = {}
    for name in sorted(expected):
        path = frameworks / name
        subprocess.run(['/usr/bin/codesign', '--verify', '--strict', str(path)], check=True, capture_output=True)
        original = (resources / 'runtime' / ORT if name == ORT else package / 'lib' / (manifest['entry_library'] if name == ENTRY_ALIAS else name))
        actual = inspect_library(path); prior = inspect_library(original)
        actual_unsigned = unsigned_descriptor(path); original_unsigned = unsigned_descriptor(original)
        if ((actual['rpaths'] and not (name == ORT and actual['rpaths'] == ['@loader_path']))
                or actual != prior or actual_unsigned != original_unsigned):
            raise ValueError('signed app framework executable payload/metadata differs: ' + name)
        require_closed_dependencies(actual['dependencies'], expected)
        records[name] = descriptor(path); unsigned[name] = actual_unsigned
    executable = contents / 'MacOS/pnola-desktop'
    if executable.is_symlink():
        raise ValueError('app executable cannot be a symlink')
    startup = re.findall(r'^\s+(.+) \(compatibility version .+\)$', subprocess.check_output(['/usr/bin/otool', '-L', str(executable)], text=True), re.MULTILINE)
    commands = subprocess.check_output(['/usr/bin/otool', '-l', str(executable)], text=True)
    rpaths = re.findall(r'cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.+) \(offset \d+\)', commands)
    deployment = re.findall(r'cmd LC_BUILD_VERSION\n\s+cmdsize \d+\n\s+platform 1\n\s+minos ([\d.]+)', commands)
    if (not startup or any(not system_dependency(name) for name in startup)
            or any(path not in {'@executable_path/../Frameworks', '@loader_path/../Frameworks'} for path in rpaths)
            or len(deployment) != 1 or version_tuple(deployment[0]) != version_tuple(info['LSMinimumSystemVersion'])):
        raise ValueError('app executable startup/deployment metadata differs')
    if subprocess.check_output(['/usr/bin/lipo', '-archs', str(executable)], text=True).split() != ['arm64']:
        raise ValueError('app executable architecture differs')
    subprocess.run(['/usr/bin/codesign', '--verify', '--deep', '--strict', str(app)], check=True, capture_output=True)
    return {'schema_version': 1, 'app_structure_verified': True, 'release_qualified': False,
            'scope': 'Complete same-host ad-hoc app structure, current profile/model/license and unsigned framework payload mapping only. No inference, GUI, microphone, Developer ID or notarization claim.',
            'app': str(app), 'backend': manifest['backend'], 'runtime_profile': manifest['runtime_profile'],
            'minimum_macos': info['LSMinimumSystemVersion'], 'executable': descriptor(executable),
            'entry_library': manifest['entry_library'], 'startup_dependencies': startup, 'executable_rpaths': rpaths, 'frameworks': records, 'unsigned_frameworks': unsigned,
            'logical_runtime_library_count': len(manifest['libraries']), 'bark_manifest': descriptor(bark / 'manifest.json'), 'bark_files': bark_files, 'package_manifest': descriptor(package / 'package.json')}


def audit_app_images(trace, app, verified):
    app = app.resolve(strict=True); executable = app / 'Contents/MacOS/pnola-desktop'; frameworks = app / 'Contents/Frameworks'
    if str(app) != verified['app'] or descriptor(executable) != verified['executable']:
        raise ValueError('app executable differs from static verification')
    expected = verified['frameworks']; observed = {}; saw_executable = False
    paths = set(re.findall(r'dyld\[\d+\]: <[A-Fa-f0-9-]+> (/[^\r\n]+)', trace))
    callback_paths = set()
    for line in trace.splitlines():
        if line.startswith('PNOLA_LOADED_IMAGE'):
            match = re.fullmatch(r'PNOLA_LOADED_IMAGE_HEX ((?:[0-9a-f]{2})+)', line)
            if not match:
                raise ValueError('invalid or failed process-local app image record')
            try:
                path = bytes.fromhex(match[1]).decode('utf-8')
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError('invalid process-local app image path') from error
            if not path.startswith('/') or '\x00' in path:
                raise ValueError('invalid process-local app image path')
            callback_paths.add(path)
    paths.update(callback_paths)
    if not paths:
        raise ValueError('missing app dyld telemetry')
    for reported in sorted(paths):
        if reported.startswith(('/usr/lib/', '/System/Library/')):
            continue
        path = Path(reported).resolve(strict=True)
        if path == executable:
            saw_executable = True; continue
        if path.parent != frameworks or path.name not in expected or descriptor(path) != expected[path.name]:
            raise ValueError('unexpected or changed app loaded image: ' + reported)
        observed[path.name] = {'reported_path': reported, **descriptor(path)}
    entry_choices = {verified['entry_library'], ENTRY_ALIAS} & observed.keys()
    required = set(expected) - {verified['entry_library'], ENTRY_ALIAS}
    if not saw_executable or len(entry_choices) != 1 or set(observed) - entry_choices != required:
        raise ValueError('actual app component/provider image set differs')
    return {'executable': str(executable), 'images': observed, 'entry_image': next(iter(entry_choices)),
            'logical_runtime_library_count': len(observed) - 1, 'ort_provider_count': 1,
            'process_local_add_image_history_observed': bool(callback_paths)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path); args = parser.parse_args()
    print(json.dumps(verify_app(args.app), indent=2))
