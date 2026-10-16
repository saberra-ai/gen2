"""Package the qualified Ubuntu 24.04 x86_64 CPU SDK without host library paths.

This explicit build tool requires readelf, patchelf, dpkg-query and authenticated
APT source indexes. Source acquisition happens during packaging, never inference.
A verified package is not by itself offline-container or coexistence evidence.
"""
import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tempfile

from sdk_archive import inventory

PLATFORM_LIBRARIES = {'libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0',
                      'librt.so.1', 'ld-linux-x86-64.so.2'}
EXTERNAL_PACKAGES = {'libsentencepiece0', 'libstdc++6', 'libgcc-s1'}
SHARED_RUNTIMES = {'libstdc++.so.6', 'libgcc_s.so.1'}
PREFIX = 'libgen2_nemo_linux_'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_environment():
    env = dict(os.environ)
    for name in ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'LD_DEBUG']:
        env.pop(name, None)
    env['LC_ALL'] = 'C.UTF-8'
    return env


def capture(command, **kwargs):
    return subprocess.check_output(list(map(str, command)), env=clean_environment(),
                                   text=True, stderr=subprocess.STDOUT, **kwargs).strip()


def inspect_elf(path, require_soname=True):
    with path.open('rb') as stream:
        header = stream.read(64)
    if len(header) != 64 or header[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', header, 18)[0] != 62:
        raise ValueError(f'expected little-endian x86_64 ELF: {path}')
    dynamic = capture(['readelf', '--dynamic', '--wide', path])
    names = lambda kind: re.findall(r'\(' + kind + r'\).*?\[([^\]]*)\]', dynamic)
    identity = names('SONAME')
    if len(identity) > 1 or (require_soname and len(identity) != 1):
        raise ValueError(f'expected one ELF SONAME: {path}')
    version_text = capture(['readelf', '--version-info', '--wide', path])
    needs = version_text.partition('Version needs section')[2]
    versions = sorted(set(re.findall(r'Name: ((?:GLIBC|GLIBCXX|CXXABI|GCC)_[A-Za-z0-9_.]+)', needs)))
    for value in versions:
        if value.startswith('GLIBC_'):
            if value == 'GLIBC_ABI_DT_RELR':
                continue  # Public marker supported since glibc 2.36.
            match = re.fullmatch(r'GLIBC_([0-9]+(?:\.[0-9]+)+)', value)
            version = [int(x) for x in match[1].split('.')] if match else []
            while len(version) > 2 and version[-1] == 0:
                version.pop()
            if not match or tuple(version) > (2, 39):
                raise ValueError(f'ELF exceeds declared glibc 2.39 baseline: {path}: {value}')
    return {'soname': identity[0] if identity else None, 'needed': names('NEEDED'), 'rpath': names('RPATH'),
            'runpath': names('RUNPATH'), 'required_symbol_versions': versions}


def parse_ldd(output):
    resolved = {}
    for line in output.splitlines():
        match = re.fullmatch(r'\s*(\S+) => (/.+?)\s+\(0x[0-9a-fA-F]+\)\s*', line)
        if match:
            resolved[match[1]] = Path(match[2]).resolve(strict=True)
        else:
            loader = re.fullmatch(r'\s*(/.+?)\s+\(0x[0-9a-fA-F]+\)\s*', line)
            if loader:
                target = Path(loader[1]).resolve(strict=True)
                resolved[Path(loader[1]).name] = target
    return resolved


def resolved_dependencies(path, needed):
    output = capture(['ldd', path])
    resolved = parse_ldd(output)
    for name in needed:
        if Path(name).name != name or name not in resolved:
            raise ValueError(f'unresolved or non-SONAME dependency {name} from {path}: {output}')
    return {name: resolved[name] for name in needed}


def package_metadata(name):
    fields = ['binary:Package', 'Version', 'Architecture', 'source:Package', 'source:Version']
    values = capture(['dpkg-query', '-W', '-f=' + '\t'.join('${' + x + '}' for x in fields), name]).split('\t')
    if len(values) != len(fields) or not all(values):
        raise ValueError(f'incomplete Debian package identity: {name}')
    return dict(zip(fields, values))


def apt_provenance():
    if os.environ.get('APT_CONFIG'):
        raise ValueError('custom APT_CONFIG must be independently qualified')
    config = capture(['apt-config', 'dump'])
    values = {name.lower(): value for name, value in re.findall(r'^([^\s]+) "([^"\n]*)";$', config, re.MULTILINE)}
    for name in ['Acquire::AllowInsecureRepositories', 'Acquire::AllowWeakRepositories',
                 'Acquire::AllowDowngradeToInsecureRepositories', 'APT::Get::AllowUnauthenticated']:
        if values.get(name.lower(), 'false').lower() not in {'false', 'no', '0'}:
            raise ValueError('insecure APT authentication override: ' + name)
    base = Path(values.get('dir', '/')) / values.get('dir::etc', 'etc/apt')
    source_list = base / values.get('dir::etc::sourcelist', 'sources.list')
    source_parts = base / values.get('dir::etc::sourceparts', 'sources.list.d')
    paths = [source_list, *sorted(source_parts.glob('*.list')), *sorted(source_parts.glob('*.sources'))]
    files = {}
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text()
        active = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
        overrides = re.findall(r'(?im)(?:trusted|allow-insecure|allow-weak|allow-downgrade-to-insecure)\s*(?:=|:)\s*([^\s\]#]+)', active)
        if any(value.lower() not in {'false', 'no', '0'} for value in overrides):
            raise ValueError('insecure APT source authentication override: ' + str(path))
        files[str(path)] = {'sha256': digest(path), 'content': text}
    if not files:
        raise ValueError('APT source origins are unavailable')
    keys = [base / values.get('dir::etc::trusted', 'trusted.gpg')]
    key_parts = base / values.get('dir::etc::trustedparts', 'trusted.gpg.d')
    keys.extend([*key_parts.glob('*.gpg'), *key_parts.glob('*.asc')])
    for name, record in files.items():
        active = '\n'.join(line for line in record['content'].splitlines() if not line.lstrip().startswith('#'))
        if Path(name).suffix == '.sources':
            fields = [Parser().parsestr(block).get('Signed-By', '')
                      for block in re.split(r'\n[ \t]*\n', active)]
        else:
            fields = re.findall(r'(?i)signed-by\s*=\s*([^\]\s]+)', active)
        for field in fields:
            for token in field.replace(',', ' ').split():
                token = token.strip(chr(34) + chr(39))
                if token.startswith('/'):
                    keys.append(Path(token))
    return {'config': config, 'source_files': files,
            'keyrings': {str(path): digest(path) for path in sorted(set(keys)) if path.is_file()}}


def owner(path):
    rows = capture(['dpkg-query', '-S', path]).splitlines()
    if len(rows) != 1 or ': ' not in rows[0]:
        raise ValueError(f'ambiguous dependency package owner: {path}')
    name, reported = rows[0].rsplit(': ', 1)
    if ',' in name or Path(reported).resolve(strict=True) != path.resolve(strict=True):
        raise ValueError(f'package owner differs from actual dependency: {path}')
    return package_metadata(name)


def qualified_entry(sdk, build):
    for relative, record in build['installed_files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('unsafe native receipt path')
        path = sdk / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(sdk) or digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'SDK file differs from qualified receipt: {relative}')
    entry = (sdk / 'lib/libnemo_speech_asr_c.so').resolve(strict=True)
    if entry.name != 'libnemo_speech_asr_c.so.1':
        raise ValueError('native C ABI entry has the wrong qualified identity')
    relative = entry.relative_to(sdk).as_posix()
    if entry.parent != sdk / 'lib' or relative not in build['installed_files']:
        raise ValueError('native entry escapes qualified SDK libraries')
    return entry


def discover(entry, sdk, build):
    pending, libraries = [entry], {}
    while pending:
        path = pending.pop()
        if path in libraries:
            continue
        info = inspect_elf(path)
        info.update(source_sha256=digest(path), source_bytes=path.stat().st_size)
        if path.is_relative_to(sdk):
            relative = path.relative_to(sdk).as_posix()
            if path.parent != sdk / 'lib' or relative not in build['installed_files']:
                raise ValueError(f'unqualified SDK library: {path}')
            info['component'] = None
        else:
            component = owner(path)
            if component['binary:Package'].split(':')[0] not in EXTERNAL_PACKAGES:
                raise ValueError(f'unexpected native dependency component: {path}: {component}')
            info['component'] = component
        resolved = resolved_dependencies(path, info['needed'])
        info['platform_dependencies'] = {name: {'path': str(target), 'sha256': digest(target)}
                                         for name, target in resolved.items() if name in PLATFORM_LIBRARIES}
        info['edges'] = {name: target for name, target in resolved.items() if name not in PLATFORM_LIBRARIES}
        pending.extend(info['edges'].values())
        libraries[path] = info
    if libraries[entry]['soname'] != 'libnemo_speech_asr_c.so.1':
        raise ValueError('native C ABI entry has the wrong ELF identity')
    for info in libraries.values():
        for expected, target in info['edges'].items():
            if libraries[target]['soname'] != expected:
                raise ValueError(f'dependency identity differs from requested SONAME: {expected}: {target}')
    return libraries


def check_source_bundle(directory, source, version):
    descriptors = list(directory.glob('*.dsc'))
    if len(descriptors) != 1:
        raise ValueError(f'expected one authenticated source descriptor: {directory}')
    text = descriptors[0].read_text()
    start = text.find('Format:')
    if start < 0:
        raise ValueError('source descriptor has no Debian control data')
    control = Parser().parsestr(text[start:].split('-----BEGIN PGP SIGNATURE-----')[0])
    if control['Source'] != source or control['Version'] != version:
        raise ValueError('downloaded source identity differs from installed binary')
    checksums = control['Checksums-Sha256']
    if not checksums:
        raise ValueError('source descriptor has no SHA256 file inventory')
    expected = {descriptors[0].name}
    for line in checksums.splitlines():
        if not line.strip():
            continue
        sha, size, name = line.split()
        if Path(name).name != name or name in expected:
            raise ValueError('unsafe or duplicate source archive name')
        path = directory / name
        if path.is_symlink() or digest(path) != sha or path.stat().st_size != int(size):
            raise ValueError(f'corresponding source checksum differs: {name}')
        expected.add(name)
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError('source acquisition produced unexpected files')
    return {'source_package': source, 'source_version': version,
            'descriptor_sha256': digest(descriptors[0])}


def copy_notices_and_sources(sdk, libraries, staged, lock):
    original = sdk / 'share/licenses/nemo-speech'
    for name in ['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md']:
        if digest(original / name) != lock['runtime']['source_sha256'][name]:
            raise ValueError(f'native notice differs from immutable source: {name}')
    if not (original / 'third_party/ggml/LICENSE').is_file():
        raise ValueError('native ggml license is absent')
    shutil.copytree(original, staged / 'notices/nemo-speech')
    components, sources = {}, {}
    common_licenses = {'GPL-3'}
    for info in libraries.values():
        component = info['component']
        if component is None:
            continue
        name = component['binary:Package'].split(':')[0]
        if name in components:
            if components[name]['package'] != component:
                raise ValueError(f'mixed versions of dependency component: {name}')
            continue
        notice = Path('/usr/share/doc') / name / 'copyright'
        notice_owner = owner(notice.resolve(strict=True))
        source = component['source:Package']
        version = component['source:Version']
        if (notice_owner['source:Package'], notice_owner['source:Version']) != (source, version):
            raise ValueError(f'copyright source differs from shipped binary: {name}')
        target = staged / 'notices' / name
        target.mkdir(parents=True)
        shutil.copy2(notice, target / 'copyright')
        common_licenses.update(name.rstrip('.') for name in re.findall(
            r'/usr/share/common-licenses/([A-Za-z0-9.+_-]+)', notice.read_text()))
        (target / 'package-policy.txt').write_text(capture(['apt-cache', 'policy', component['binary:Package']]) + '\n')
        key = source + '=' + version
        if key not in sources:
            directory = staged / 'sources' / source
            directory.mkdir(parents=True)
            # APT authenticates configured source indexes; the .dsc then binds
            # all downloaded source tarballs to the installed binary version.
            subprocess.run(['apt-get', 'source', '--download-only', '--only-source', key],
                           cwd=directory, env=clean_environment(), check=True)
            sources[key] = check_source_bundle(directory, source, version)
        components[name] = {'package': component, 'notice_package': notice_owner,
                            'source_bundle': sources[key], 'copyright_sha256': digest(notice)}
    common = staged / 'notices/common-licenses'
    common.mkdir()
    for name in sorted(common_licenses):
        shutil.copy2(Path('/usr/share/common-licenses') / name, common / name)
    return components


def validate_library(info, names):
    if info['rpath'] != ['$ORIGIN'] or info['runpath']:
        raise ValueError('package library has non-relative or ambiguous ELF search paths')
    if info['soname'] not in names:
        raise ValueError('package library SONAME differs from package identity')
    if any(name not in PLATFORM_LIBRARIES and name not in names for name in info['needed']):
        raise ValueError('package ELF dependency escapes the declared closure')


def verify(package):
    manifest = json.loads((package / 'package.json').read_text())
    if manifest['format'] != 'gen2-nemo-linux-cpu-v1' or manifest['architecture'] != 'x86_64' or manifest['backend'] != 'cpu' or manifest['platform_baseline'] != 'Ubuntu 24.04 / glibc 2.39 / AVX2 FMA F16C':
        raise ValueError('unsupported Linux package contract')
    files = inventory(package)
    files.pop('package.json')
    if files != manifest['files']:
        raise ValueError('package filesystem differs from its manifest')
    names = set(manifest['libraries'])
    for name in names:
        if Path(name).name != name or (not name.startswith(PREFIX) and name not in SHARED_RUNTIMES):
            raise ValueError('invalid Linux runtime library name')
        info = inspect_elf(package / 'lib' / name)
        validate_library(info, names)
        if info['soname'] != name or info != manifest['libraries'][name]['final_elf']:
            raise ValueError('ELF metadata differs from package manifest')
    entry = package / 'lib/libnemo_speech_asr_c.so'
    if not entry.is_symlink() or entry.readlink().as_posix() != manifest['entry_library'] or manifest['entry_library'] not in names:
        raise ValueError('package entry alias differs from qualified closure')
    return manifest


def package_sdk(sdk, build_path, destination):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('Linux runtime packaging requires x86_64 Linux')
    if destination.exists():
        raise FileExistsError('refusing to replace an existing runtime package')
    sdk = sdk.resolve(strict=True)
    lock_path = Path(__file__).with_name('runtime.lock.json')
    lock = json.loads(lock_path.read_text())
    build = json.loads(build_path.read_text())
    if not build['development_native_gates_passed'] or build['lock_sha256'] != digest(lock_path) or build['runtime_revision'] != lock['runtime']['revision'] or build['llama_revision'] != lock['runtime']['llama_revision']:
        raise ValueError('native SDK receipt does not establish the pinned native gates')
    commands = [x['command'] for x in build['commands'] if x['name'] == 'sdk-configure']
    required = {'-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_AVX512=OFF', '-DGGML_AVX_VNNI=OFF', '-DGGML_FMA=ON', '-DGGML_F16C=ON'}
    if len(commands) != 1 or not required <= set(commands[0]):
        raise ValueError('native SDK receipt lacks the declared x86_64 ISA baseline')
    entry = qualified_entry(sdk, build)
    libraries = discover(entry, sdk, build)
    apt = apt_provenance()
    names = {path: info['soname'] if info['soname'] in SHARED_RUNTIMES else PREFIX + path.name.removeprefix('lib')
             for path, info in libraries.items()}
    if len(set(names.values())) != len(names):
        raise ValueError('colliding Linux library identities')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.nemo-linux-') as temporary:
        staged = Path(temporary) / 'package'
        (staged / 'lib').mkdir(parents=True)
        components = copy_notices_and_sources(sdk, libraries, staged, lock)
        records = {}
        for path, info in libraries.items():
            if digest(path) != info['source_sha256']:
                raise ValueError(f'native dependency changed during packaging: {path}')
            target = staged / 'lib' / names[path]
            shutil.copy2(path, target)
            target.chmod(0o755)
            command = ['patchelf', '--set-soname', names[path], '--force-rpath', '--set-rpath', '$ORIGIN']
            for old, dependency in info['edges'].items():
                if old != names[dependency]:
                    command.extend(['--replace-needed', old, names[dependency]])
            subprocess.run([*command, str(target)], env=clean_environment(), check=True)
            final = inspect_elf(target)
            validate_library(final, set(names.values()))
            records[names[path]] = {'source': str(path), 'source_sha256': info['source_sha256'],
                                    'source_bytes': info['source_bytes'], 'original_soname': info['soname'],
                                    'platform_dependencies': info['platform_dependencies'], 'final_elf': final}
        (staged / 'lib/libnemo_speech_asr_c.so').symlink_to(names[entry])
        manifest = {'format': 'gen2-nemo-linux-cpu-v1', 'release_qualified': False,
                    'architecture': 'x86_64', 'backend': 'cpu',
                    'platform_baseline': 'Ubuntu 24.04 / glibc 2.39 / AVX2 FMA F16C',
                    'runtime_revision': lock['runtime']['revision'], 'lock_sha256': digest(lock_path),
                    'build_receipt_sha256': digest(build_path), 'entry_library': names[entry],
                    'components': components, 'libraries': records,
                    'apt_provenance': apt,
                    'tools': {tool: {'version': capture([tool, '--version']).splitlines()[0],
                                     'path': shutil.which(tool), 'sha256': digest(Path(shutil.which(tool)))}
                              for tool in ['readelf', 'patchelf', 'dpkg-query', 'apt-get']},
                    'os_release': Path('/etc/os-release').read_text(),
                    'limits': ['Package construction only; independent fresh-container and native coexistence gates still required.',
                               'GCC runtime SONAMEs remain shared; existing host process namespaces need coexistence qualification.',
                               'Installed component identity assumes the trusted CI package installation; source provenance is not a reproducible-build proof.',
                               'Host glibc/loader remain platform dependencies; no musl or older-distro claim.'],
                    'files': inventory(staged)}
        (staged / 'package.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify(staged)
        staged.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-prefix', type=Path)
    parser.add_argument('--build-receipt', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify.resolve())
    else:
        if not all([args.sdk_prefix, args.build_receipt, args.output]):
            parser.error('supply --sdk-prefix, --build-receipt, --output, or --verify')
        result = package_sdk(args.sdk_prefix, args.build_receipt, args.output.resolve())
    print(json.dumps({'verified': True, 'release_qualified': False, 'libraries': len(result['libraries']),
                      'backend': result['backend'], 'platform_baseline': result['platform_baseline']}))


if __name__ == '__main__':
    main()
