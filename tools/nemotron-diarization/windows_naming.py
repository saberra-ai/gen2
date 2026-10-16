"""Guarded build-only private DLL naming and static output verification."""
import hashlib
import json

from windows_pe import inspect_pe


TARGETS = ('nemo_speech_asr_c', 'nemo_speech_asr', 'ggml', 'ggml-base', 'ggml-cpu')
NAMES = {name + '.dll': 'gen2_nemo_windows_' + name + '.dll' for name in TARGETS}
SOURCE_HASHES = {
    'CMakeLists.txt': 'e79e24832e52b440808dd59a4b274ad657ed6e7bb42df992dcaf0ee87f9d39fa',
    'src/asr/CMakeLists.txt': '7cf799a2a8f98de94e749bbdf72bab5cabba8dd61921b78ac9be6fa2f8850e48',
    'scripts/windows/build.ps1': '068c58cceea80cc6b021ec09e60f0caea1241ab2963ba3ffe1a5f7e3b547c68f',
    'vcpkg.json': '3ba4dc4a35bd4cd7958eaa5525714f231ab1729515d05600fee94fcb037b1992',
    'cmake/llama_cpp.cmake': '22357d4a5d8a67716daa8b4e374abd63120432bd57540f97d572a525d5354c7b',
    'llama.cpp/ggml/CMakeLists.txt': 'c118116572c8e5bc8d9302d44cc6806ce69a31face60f2cefdb36801fdf72fd3',
    'llama.cpp/ggml/src/CMakeLists.txt': '38ddf053a1adfc14bb84b7722e3c4ec01909694f7e51e32013f36e7a324fa4f0',
    'llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt': 'ed4d3cf43377accac79bd09b4b93c10725e66f189e1ddfcd528c7c29508cf3d0',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_private_names(source, tools, output):
    for relative, expected in SOURCE_HASHES.items():
        if digest(source / relative) != expected:
            raise ValueError('private Windows naming requires pinned source: ' + relative)
    hook = tools / 'windows_private_dlls.cmake'
    record = {'scope': 'Build-only output/import-library names; logical targets and inference source unchanged.',
              'source_sha256': SOURCE_HASHES, 'hook_sha256': digest(hook), 'names': NAMES,
              'inspector_sha256': digest(tools / 'windows_pe.py'), 'controller_sha256': digest(tools / 'windows_naming.py')}
    (output / 'windows-private-names.json').write_text(json.dumps(record, indent=2) + '\n')
    return hook, record


def verify_private_outputs(prefix, output):
    reports = {}
    for path in sorted((prefix / 'bin').iterdir()):
        if path.suffix.lower() not in {'.dll', '.exe'}:
            continue
        if path.is_symlink():
            raise ValueError('unexpected native output alias: ' + path.name)
        report = inspect_pe(path)
        reports[path.name] = report
        dependencies = report['imports'] + report['delay_imports'] + report['forwarder_dependencies']
        if path.name.lower() in NAMES or any(name in NAMES for name in dependencies):
            raise ValueError('generic NeMo DLL identity remains: ' + path.name)
        if report['bound_imports_present']:
            raise ValueError('bound imports require separate qualification: ' + path.name)
    for name in NAMES.values():
        if name not in reports or not reports[name]['is_dll'] or reports[name]['export_name'] != name:
            raise ValueError('private NeMo DLL output/export identity differs: ' + name)
        if not (prefix / 'lib' / name.replace('.dll', '.lib')).is_file():
            raise ValueError('private NeMo import library missing: ' + name)
    (output / 'windows-private-pe.json').write_text(json.dumps(reports, indent=2) + '\n')
    return {'pe_report_sha256': digest(output / 'windows-private-pe.json'),
            'private_dlls_verified': sorted(NAMES.values()), 'generic_imports_absent': True,
            'limits': 'Static ordinary/delay/forwarded imports only; dynamic loads, offline closure and coexistence remain separate.'}
