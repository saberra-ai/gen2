"""Complete failed tagged construction metadata only; never compile, test or run a model."""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

from artifacts import digest
from build_valid_mel_candidate import descriptor, source_inventory
from runtime_profiles import load_profile


def main():
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'valid-mel-tagged-metadata.protocol.json'
    p = json.loads(protocol_path.read_text())
    output = Path(p['output_path'])
    if output.exists():
        raise ValueError('refusing prior metadata recovery evidence overwrite')
    for name, expected in p['tools'].items():
        if descriptor(tools / name) != expected:
            raise ValueError('metadata recovery tool differs: ' + name)
    for path, expected in p['inputs'].items():
        if descriptor(Path(path)) != expected:
            raise ValueError('tagged construction input differs: ' + path)
    original, source, prefix = [Path(p[key]) for key in ['failed_attempt_path', 'native_source_path', 'sdk_path']]
    old_protocol_path = original / 'protocol.json'
    committed = subprocess.check_output(['git', 'show', p['construction_commit'] +
        ':tools/nemotron-diarization/valid-mel-tagged-build.protocol.json'], cwd=tools)
    if committed != old_protocol_path.read_bytes():
        raise ValueError('original construction protocol differs from its committed admission')
    old_protocol = json.loads(committed)
    for name, expected in old_protocol['tools'].items():
        blob = subprocess.check_output(['git', 'show', p['construction_commit'] +
            ':tools/nemotron-diarization/' + name], cwd=tools)
        if {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()} != expected:
            raise ValueError('original admitted construction tool differs: ' + name)
    build = original / 'build'
    commands = json.loads((build / 'commands.json').read_text())
    phases = ['configure', 'llama-archive'] + ['llama-patch-' + str(i) for i in range(19)] + ['build', 'install', 'model-free-state-test', 'version-only']
    if [row['phase'] for row in commands] != phases or any(row['returncode'] != 0 for row in commands):
        raise ValueError('tagged construction phases did not all complete successfully')
    outer = json.loads((original / 'build.command.json').read_text())
    if outer['returncode'] != 1 or outer['source_commit'] != p['construction_commit']:
        raise ValueError('unexpected original outer failure')
    error = (original / 'build.stderr.log').read_text()
    if 'Error: $HOME must be set to run brew.' not in error or "toolchain['packages'] = capture" not in error:
        raise ValueError('failure is not the admitted late Homebrew metadata error')
    lock_path, lock = load_profile()
    runtime = lock['runtime']
    for name, expected in runtime['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('source changed since tagged construction: ' + name)
    patch = tools / runtime['compatibility_patches'][0]['path']
    def capture(argv, cwd=source):
        return subprocess.check_output(argv, cwd=cwd)
    if capture(['git', 'rev-parse', 'HEAD']).decode().strip() != runtime['revision']:
        raise ValueError('native source base differs')
    if capture(['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all']) != patch.read_bytes():
        raise ValueError('native source diff differs')
    if capture(['git', 'diff', '--cached', '--binary', '--no-ext-diff', '--ignore-submodules=all']):
        raise ValueError('native source staging changed')
    if capture(['git', 'rev-parse', 'HEAD'], source / 'llama.cpp').decode().strip() != runtime['llama_revision'] or capture(['git', 'status', '--porcelain', '--untracked-files=no'], source / 'llama.cpp'):
        raise ValueError('pristine llama changed')
    audit = json.loads((build / 'llama-materialized-audit.json').read_text())
    if source_inventory(Path(audit['compiled_tree_path'])) != audit['files']:
        raise ValueError('materialized compiler inputs changed')
    log = (build / 'model-free-state-test.stdout.log').read_text()
    if not re.search(r'1/1 Test #\d+: diar_state\s+.*Passed', log) or '100% tests passed, 0 tests failed out of 1' not in log:
        raise ValueError('actual model-free state test pass is absent')
    version = (build / 'version-only.stdout.log').read_text().strip()
    if version != runtime['reported_version']:
        raise ValueError('saved actual version differs')
    cache = source / 'build/cpu-diar'
    database = json.loads((cache / 'compile_commands.json').read_text())
    if (not any(row['file'] == str(source / 'src/asr/diar/diar_pipeline.cpp') for row in database)
            or not any(row['file'].startswith(audit['compiled_tree_path'] + '/ggml/') for row in database)):
        raise ValueError('actual compiler database does not bind audited pipeline/GGML inputs')
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    profile = output / 'offline.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* ' + ' '.join('(subpath ' + json.dumps(x) + ')' for x in p['denied_paths']) + ')\n')
    records = []
    def run(name, argv, overrides=None):
        actual = ['/usr/bin/sandbox-exec', '-f', str(profile), *argv]
        environment = {'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'LC_ALL': 'C', **(overrides or {})}
        record = {'name': name, 'argv': actual, 'environment': environment}
        records.append(record)
        (output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        started = time.monotonic()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            result = subprocess.run(actual, cwd=source, env=environment, stdout=stdout, stderr=stderr)
        record.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
        (output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        result.check_returncode()
        return (output / (name + '.stdout.log')).read_text().strip()
    toolchain = {name: run(name, [path, '--version']).splitlines()[0]
                 for name, path in [('cmake', '/opt/homebrew/bin/cmake'), ('ninja', '/opt/homebrew/bin/ninja'), ('clang++', '/usr/bin/clang++')]}
    toolchain['sdk'] = run('xcode-sdk', ['/usr/bin/xcrun', '--show-sdk-version'])
    toolchain['packages'] = run('brew-packages', ['/opt/homebrew/bin/brew', 'list', '--versions', 'sentencepiece', 'abseil', 'protobuf'],
                              {'HOME': p['actual_user_home'], 'HOMEBREW_NO_AUTO_UPDATE': '1'})
    libraries, paths = {}, {}
    for path in sorted((prefix / 'lib').iterdir()):
        if not path.is_file():
            continue
        paths[path.name] = {**descriptor(path), 'symlink_target': str(path.readlink()) if path.is_symlink() else None}
        if not path.is_symlink():
            libraries[path.name] = {'sha256': digest(path), 'size': path.stat().st_size,
                'dependencies': capture(['/usr/bin/otool', '-L', str(path)]).decode(),
                'exports': capture(['/usr/bin/nm', '-gU', str(path)]).decode()}
    if paths != p['sdk_library_paths']:
        raise ValueError('SDK path or alias set changed during metadata recovery')
    for name in ['CMakeCache.txt', 'compile_commands.json']:
        (output / name).write_bytes((cache / name).read_bytes())
    archive = cache / 'src/runtime/ggml/libnemo_speech_runtime_ggml.a'
    record = {'runtime_revision': runtime['revision'], 'llama_revision': runtime['llama_revision'],
        'runtime_profile': lock['profile'], 'runtime_reported_version': version, 'lock_sha256': digest(lock_path),
        'compatibility_patches': runtime['compatibility_patches'], 'frame_contract': runtime['frame_contract'],
        'backend': 'cpu', 'preset': 'cpu-diar', 'metal_shaders_embedded': False,
        'platform': platform.platform(), 'machine': platform.machine(),
        'construction_commit': p['construction_commit'], 'construction_outer_returncode': 1,
        'metadata_recovery_only': True, 'commands': commands, 'metadata_commands': records,
        'toolchain': toolchain, 'libraries': libraries, 'installed_library_paths': paths,
        'runtime_archive': {'path': str(archive), **descriptor(archive)},
        'cmake_cache_sha256': digest(cache / 'CMakeCache.txt'), 'model_free_state_test_completed': True,
        'saved_version_function_query_verified': True, 'new_compilation_or_tests_or_version_query': False,
        'model_instantiated_or_inference_performed': False, 'artifact_admitted_for_inference': False,
        'production_qualified': False, 'release_qualified': False}
    for path, expected in p['inputs'].items():
        if descriptor(Path(path)) != expected:
            raise ValueError('tagged construction input changed during metadata recovery: ' + path)
    (output / 'recovered-build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'metadata_recovered': True, 'new_neural_or_compile_or_test_or_version_calls': 0}))


if __name__ == '__main__':
    main()
