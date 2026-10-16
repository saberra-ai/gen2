"""Construct a development Windows runtime from a qualified private-name SDK.

This byte-preserving tool can run on a receiver OS. It does not establish
Microsoft redistribution rights, offline execution or desktop compatibility.
"""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import platform
import shutil
import tempfile

from windows_archive import byte_inventory, digest, restore_archive, safe_relative, write_archive
from windows_naming import NAMES, verify_private_outputs
from windows_pe import inspect_pe

CRT = {'concrt140.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
       'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'}
PLATFORM = {'advapi32.dll', 'dbghelp.dll', 'kernel32.dll'} | {
    f'api-ms-win-crt-{name}-l1-1-0.dll' for name in
    ['convert', 'environment', 'filesystem', 'heap', 'locale', 'math', 'runtime', 'stdio', 'string', 'time', 'utility']}
BASELINE = 'Windows Server 2022 / AMD64 / AVX2 FMA F16C'
EVIDENCE_FILES = ['receipt.json', 'windows-provenance.json', 'windows-private-pe.json',
                  'windows-private-names.json', 'windows-compat.json', 'windows-vcpkg-status.log', 'cmake-cache.log']


def summary(info):
    result = {key: value for key, value in info.items() if key != 'exports'}
    result['export_names_json_sha256'] = hashlib.sha256(json.dumps(info['exports'], separators=(',', ':')).encode()).hexdigest()
    return result


def validate_libraries(libraries):
    names = set(NAMES.values()) | CRT
    if set(libraries) != names:
        raise ValueError('Windows runtime must retain exactly five private native and eight qualified CRT DLLs')
    for name, info in libraries.items():
        if (info['machine'] != 'AMD64' or info['format'] != 'PE32+' or not info['is_dll']
                or info['export_name'] != name or info['bound_imports_present']):
            raise ValueError('unsupported Windows DLL identity/binding form: ' + name)
        dependencies = info['imports'] + info['delay_imports'] + info['forwarder_dependencies']
        if any(dep not in names | PLATFORM for dep in dependencies):
            raise ValueError('Windows DLL dependency escapes declared closure: ' + name)
    reached, pending, platform_names = set(), [NAMES['nemo_speech_asr_c.dll']], set()
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        info = libraries[name]
        for dep in info['imports'] + info['delay_imports'] + info['forwarder_dependencies']:
            if dep in names:
                pending.append(dep)
            else:
                platform_names.add(dep)
    return {'static_entry_closure': sorted(reached), 'platform_imports': sorted(platform_names)}


def qualified_sdk(sdk, evidence, lock):
    build = json.loads((evidence / 'receipt.json').read_text(encoding='utf-8-sig'))
    if (build['development_native_gates_passed'] is not True or build['runtime_revision'] != lock['runtime']['revision']
            or build['llama_revision'] != lock['runtime']['llama_revision']
            or build['lock_sha256'] != digest(Path(__file__).with_name('runtime.lock.json'))
            or build['machine'].lower() not in {'amd64', 'x86_64'} or not build['platform'].startswith('Windows-')):
        raise ValueError('qualified pinned Windows native SDK required')
    records = byte_inventory(sdk)
    expected, folded = set(), set()
    for name, record in build['installed_files'].items():
        name = name.replace('\\', '/')
        safe_relative(name)
        if name.casefold() in folded:
            raise ValueError('colliding qualified SDK paths')
        folded.add(name.casefold())
        expected.add(name)
        if records.get(name) != {'type': 'file', **record}:
            raise ValueError('delivered SDK differs from qualified file: ' + name)
    if {name for name, record in records.items() if record['type'] == 'file'} != expected:
        raise ValueError('unexpected SDK files outside qualified receipt')
    cache_path = evidence / 'cmake-cache.log'
    cache = {}
    for line in cache_path.read_text(encoding='utf-8-sig').splitlines():
        if ':' in line and '=' in line and not line.startswith(('#', '//')):
            key, value = line.split('=', 1)
            key = key.split(':', 1)[0]
            if key in cache:
                raise ValueError('duplicate CMake provenance key')
            cache[key] = value
    required = {'CMAKE_GENERATOR': 'Ninja', 'CMAKE_BUILD_TYPE': 'Release', 'BUILD_SHARED_LIBS': 'ON',
                'GGML_NATIVE': 'OFF', 'GGML_AVX': 'ON', 'GGML_AVX2': 'ON', 'GGML_AVX512': 'OFF',
                'GGML_AVX_VNNI': 'OFF', 'GGML_FMA': 'ON', 'GGML_F16C': 'ON', 'GGML_OPENMP': 'OFF',
                'GGML_BACKEND_DL': 'OFF', 'GGML_CUDA': 'OFF', 'GGML_VULKAN': 'OFF',
                'NEMO_SPEECH_BUILD_ASR': 'OFF', 'NEMO_SPEECH_BUILD_TTS': 'OFF'}
    if any(cache.get(key) != value for key, value in required.items()):
        raise ValueError('SDK CMake cache differs from qualified CPU/ISA profile')
    naming = build['windows_private_names']
    if naming['names'] != NAMES or not naming['generic_imports_absent']:
        raise ValueError('private DLL naming qualification required')
    original = evidence / 'windows-private-pe.json'
    if digest(original) != naming['pe_report_sha256']:
        raise ValueError('qualified PE report changed')
    with tempfile.TemporaryDirectory() as temporary:
        verify_private_outputs(sdk, Path(temporary))
        if json.loads((Path(temporary) / 'windows-private-pe.json').read_text(encoding='utf-8-sig')) != json.loads(original.read_text(encoding='utf-8-sig')):
            raise ValueError('actual SDK PE tables differ from qualified report')
    provenance = json.loads((evidence / 'windows-provenance.json').read_text(encoding='utf-8-sig'))
    if provenance['cmake_cache']['sha256'] != digest(cache_path):
        raise ValueError('actual CMake provenance changed')
    redist = provenance['redistributables']
    if len(redist) != len(CRT) or {row['name'] for row in redist} != CRT:
        raise ValueError('exact CRT source provenance missing')
    for row in redist:
        source, installed = row['source'], row['installed']
        actual = records['bin/' + row['name']]
        if (any(actual['sha256'] != item['sha256'] or actual['bytes'] != item['bytes'] for item in [source, installed])
                or not PureWindowsPath(source['path']).is_relative_to(PureWindowsPath(provenance['redist_root']) / 'x64')
                or PureWindowsPath(source['path']).name.lower() != row['name']
                or row['signature']['status'] != 'Valid' or 'O=Microsoft Corporation' not in row['signature']['signer']['subject']):
            raise ValueError('CRT bytes/source/signature observation differs: ' + row['name'])
    status = [row for row in provenance['vcpkg_metadata'] if PureWindowsPath(row['path']).name == 'windows-vcpkg-status.log']
    if (len(status) != 1 or digest(evidence / 'windows-vcpkg-status.log') != status[0]['sha256']
            or (evidence / 'windows-vcpkg-status.log').stat().st_size != status[0]['bytes']):
        raise ValueError('retained static-dependency status differs from qualified provenance')
    return build, provenance, records


def verify(root):
    manifest = json.loads((root / 'package.json').read_text(encoding='utf-8-sig'))
    if (manifest['format'] != 'gen2-nemo-windows-cpu-v1' or manifest['platform_baseline'] != BASELINE
            or manifest['architecture'] != 'AMD64' or manifest['backend'] != 'cpu'
            or manifest['release_qualified'] is not False or manifest['redistribution_qualified'] is not False):
        raise ValueError('unsupported Windows development runtime contract')
    files = byte_inventory(root)
    files.pop('package.json')
    if files != manifest['files']:
        raise ValueError('Windows runtime filesystem differs from manifest')
    if set(manifest['libraries']) != set(NAMES.values()) | CRT:
        raise ValueError('Windows library names escape the declared set')
    metadata = root / 'metadata'
    native = json.loads((metadata / 'receipt.json').read_text(encoding='utf-8-sig'))
    lock_path = Path(__file__).with_name('runtime.lock.json')
    lock = json.loads(lock_path.read_text())
    if (digest(metadata / 'receipt.json') != manifest['native_receipt_sha256']
            or digest(metadata / 'windows-provenance.json') != manifest['msvc_provenance_sha256']
            or native['development_native_gates_passed'] is not True
            or native['source_commit'] != manifest['native_source_commit']
            or native['lock_sha256'] != manifest['lock_sha256'] or digest(lock_path) != manifest['lock_sha256']
            or native['runtime_revision'] != manifest['runtime_revision'] or manifest['runtime_revision'] != lock['runtime']['revision']
            or native['llama_revision'] != manifest['llama_revision'] or manifest['llama_revision'] != lock['runtime']['llama_revision']):
        raise ValueError('Windows package source/provenance pins differ')
    libraries = {name: summary(inspect_pe(root / 'lib' / name)) for name in manifest['libraries']}
    installed = {name.replace('\\', '/'): record for name, record in native['installed_files'].items()}
    for name, info in libraries.items():
        if any(info[field] != installed['bin/' + name][field] for field in ['sha256', 'bytes']):
            raise ValueError('Windows DLL bytes differ from qualified native SDK')
    if libraries != manifest['libraries'] or validate_libraries(libraries) != manifest['dependency_closure']:
        raise ValueError('Windows PE closure differs from manifest')
    if manifest['entry_library'] != NAMES['nemo_speech_asr_c.dll']:
        raise ValueError('Windows C ABI entry differs')
    return manifest


def package_sdk(sdk, evidence, destination):
    sdk, evidence, destination = Path(sdk).resolve(strict=True), Path(evidence).resolve(strict=True), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('refusing to replace a Windows runtime')
    tools = Path(__file__).resolve().parent
    lock = json.loads((tools / 'runtime.lock.json').read_text())
    evidence_before = {name: digest(evidence / name) for name in EVIDENCE_FILES}
    build, provenance, original = qualified_sdk(sdk, evidence, lock)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.package-windows-') as temporary:
        staged = Path(temporary) / 'runtime'
        (staged / 'lib').mkdir(parents=True)
        for name in sorted(set(NAMES.values()) | CRT):
            shutil.copy2(sdk / 'bin' / name, staged / 'lib' / name)
        notices = sdk / 'share/licenses/nemo-speech'
        for name in ['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md']:
            if digest(notices / name) != lock['runtime']['source_sha256'][name]:
                raise ValueError('pinned NeMo notice differs')
        shutil.copytree(notices, staged / 'notices/nemo-speech')
        (staged / 'notices/gen2').mkdir()
        shutil.copy2(tools.parents[1] / 'LICENSE', staged / 'notices/gen2/LICENSE')
        metadata = staged / 'metadata'
        metadata.mkdir()
        for name in EVIDENCE_FILES:
            shutil.copy2(evidence / name, metadata / name)
            if digest(metadata / name) != evidence_before[name]:
                raise ValueError('qualification evidence changed during copying')
        libraries = {name: summary(inspect_pe(staged / 'lib' / name)) for name in sorted(set(NAMES.values()) | CRT)}
        manifest = {'format': 'gen2-nemo-windows-cpu-v1', 'release_qualified': False, 'redistribution_qualified': False,
                    'architecture': 'AMD64', 'backend': 'cpu', 'platform_baseline': BASELINE,
                    'runtime_revision': build['runtime_revision'], 'llama_revision': build['llama_revision'],
                    'native_source_commit': build['source_commit'], 'native_receipt_sha256': digest(evidence / 'receipt.json'),
                    'lock_sha256': digest(tools / 'runtime.lock.json'), 'entry_library': NAMES['nemo_speech_asr_c.dll'],
                    'libraries': libraries, 'dependency_closure': validate_libraries(libraries),
                    'msvc_provenance_sha256': digest(evidence / 'windows-provenance.json'),
                    'packaging_host': {'platform': platform.platform(), 'python': platform.python_version()},
                    'tools': {name: digest(tools / name) for name in ['package_windows.py', 'windows_archive.py', 'windows_pe.py', 'windows_naming.py']},
                    'limits': ['Development byte-preserving construction only; no execution/relocation/coexistence qualification.',
                               'Conservative eight-CRT payload; static reachability is not proof that fewer files are safe.',
                               'Microsoft DLLs remain unmodified/shared names; source/signature provenance is observed on the qualified build host.',
                               'Applicable Microsoft distribution terms/entitlement and complete static-dependency source provenance remain open.',
                               'No Windows desktop, fresh-machine, accuracy or live/uncontended-performance qualification.'],
                    'files': byte_inventory(staged)}
        with (staged / 'package.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(manifest, indent=2) + '\n')
        verify(staged)
        if byte_inventory(sdk) != original:
            raise ValueError('qualified SDK changed during package construction')
        if {name: digest(evidence / name) for name in EVIDENCE_FILES} != evidence_before:
            raise ValueError('qualification evidence changed during construction')
        staged.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-prefix', type=Path, required=True)
    parser.add_argument('--native-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'schema_version': 1, 'package_construction_passed': False, 'release_qualified': False,
               'redistribution_qualified': False, 'runtime_qualified': False}
    try:
        package = args.output / 'runtime'
        manifest = package_sdk(args.sdk_prefix, args.native_evidence, package)
        archive = write_archive(package, args.output)
        restored = restore_archive(args.output / archive['archive'], archive, args.output / 'relocated runtime ü')
        if verify(restored) != manifest:
            raise ValueError('delivered Windows archive verification differs')
        receipt.update(package_construction_passed=True, native_source_commit=manifest['native_source_commit'],
                       native_receipt_sha256=manifest['native_receipt_sha256'], package_manifest_sha256=digest(package / 'package.json'),
                       archive={key: value for key, value in archive.items() if key != 'files'},
                       scope='Development runtime construction and exact ZIP restoration on the named packaging host; no model/native execution.',
                       packaging_host=manifest['packaging_host'])
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
