"""Verify offline APK packaging against the already inspected native artifacts."""
import argparse
import hashlib
import json
import struct
import subprocess
import zipfile
from pathlib import Path


def inspect(a):
    badging = subprocess.check_output([str(a.aapt2), 'dump', 'badging', str(a.apk)], text=True)
    permissions = subprocess.check_output([str(a.aapt2), 'dump', 'permissions', str(a.apk)], text=True)
    if ("package: name='example.gen2.laya'" not in badging or
            "minSdkVersion:'24'" not in badging or "targetSdkVersion:'35'" not in badging):
        raise ValueError('Unexpected package identity or SDK floor')
    if 'uses-permission:' in permissions:
        raise ValueError('Offline harness must not request Android permissions')
    sha = lambda data: hashlib.sha256(data).hexdigest()
    libraries = []
    with zipfile.ZipFile(a.apk) as archive, a.apk.open('rb') as stream:
        names = {name for name in archive.namelist() if name.startswith('lib/')}
        expected = {'lib/' + a.abi + '/' + name for name in
                    ('libgen2_laya_mobile.so', 'libgen2_laya_jni.so', 'libonnxruntime.so')}
        if names != expected:
            raise ValueError('APK must contain exactly the inspected ABI libraries')
        for name in sorted(expected):
            info = archive.getinfo(name)
            stream.seek(info.header_offset)
            header = stream.read(30)
            filename_length, extra_length = struct.unpack_from('<HH', header, 26)
            offset = info.header_offset + 30 + filename_length + extra_length
            if info.compress_type != zipfile.ZIP_STORED or offset % 16384:
                raise ValueError(f'{name}: not uncompressed and 16 KiB aligned')
            digest = sha(archive.read(name))
            if digest != sha((a.libraries / Path(name).name).read_bytes()):
                raise ValueError(f'{name}: packaged library differs from inspected input')
            libraries.append(dict(file=name, sha256=digest, bytes=info.file_size, offset=offset))
        fixtures = sorted(p for p in a.fixture.rglob('*') if p.is_file())
        for path in fixtures:
            name = 'assets/smoke/' + path.relative_to(a.fixture).as_posix()
            if archive.read(name) != path.read_bytes():
                raise ValueError(f'{name}: synthetic fixture changed in packaging')
        if not fixtures or 'classes.dex' not in archive.namelist():
            raise ValueError('Missing Java classes or synthetic assets')
    report = dict(status='APK packaging inspected; device execution not verified',
                  package='example.gen2.laya', abi=a.abi, min_sdk=24, target_sdk=35,
                  apk_sha256=sha(a.apk.read_bytes()), apk_bytes=a.apk.stat().st_size,
                  libraries=libraries, fixture_files=len(fixtures), permissions=permissions,
                  badging=badging)
    workspace = Path(__file__).resolve().parents[2]
    project = workspace / 'examples/laya-mobile/android'
    sources = sorted(list((project / 'app/src').rglob('*')) +
                     [project / name for name in ('LayaNative.java', 'build.gradle',
                       'settings.gradle', 'gradle.properties', 'app/build.gradle')])
    inventory = {p.relative_to(project).as_posix(): sha(p.read_bytes())
                 for p in sources if p.is_file()}
    report['host_sources'] = inventory
    report['host_source_sha256'] = sha(json.dumps(inventory, sort_keys=True).encode())
    a.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(report['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('apk', 'aapt2', 'libraries', 'fixture', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--abi', choices=['arm64-v8a', 'x86_64'], default='arm64-v8a')
    inspect(parser.parse_args())
