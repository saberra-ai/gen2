"""Construct a fresh tagged macOS arm64 CPU SDK; model-free state test only."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from artifacts import digest
from build_valid_mel_candidate import capture, descriptor, source_inventory
from ctest_output import require_single_test_pass
from macos_cpu_architecture import PORTABLE_ARM64, audit_portable_commands, portable_definitions
from platform_valid_mel_sources import verify_completed_source
from runtime_profiles import audit_cpu_compiler_tree, load_profile, verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'prefix', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--cpu-arch', choices=['native', PORTABLE_ARM64], default='native',
                        help='explicit portable candidate, or the unchanged build-host native default')
    parser.add_argument('--sandbox-profile', type=Path, help='optional stricter child profile retained with evidence')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or args.jobs < 1:
        parser.error('this construction profile requires macOS arm64 and positive jobs')
    source, prefix, output = args.source.resolve(), args.prefix.resolve(), args.output.resolve()
    roots = [source, prefix, output]
    if any(left != right and left.is_relative_to(right) for left in roots for right in roots):
        parser.error('source, prefix and evidence paths must be disjoint')
    if len(set(roots)) != 3 or output.exists() or prefix.exists() or (source / 'build').exists():
        parser.error('refusing existing construction evidence, SDK prefix or source build')
    lock_path, lock = load_profile()
    runtime = lock['runtime']
    verify_source(source, lock)
    output.mkdir(parents=True)
    (output / 'runtime-profile.json').write_bytes(lock_path.read_bytes())
    patch = lock_path.parent / runtime['compatibility_patches'][0]['path']
    (output / 'source.patch').write_bytes(patch.read_bytes())
    profile = output / 'offline.sb'
    if args.sandbox_profile:
        profile.write_bytes(args.sandbox_profile.read_bytes())
    else:
        profile.write_text('(version 1)\n(allow default)\n(deny network*)\n')
    environment = {'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                   'LC_ALL': 'C', 'GIT_LFS_SKIP_SMUDGE': '1'}
    commands = []
    def run(name, argv, cwd=source, overrides=None):
        actual = ['/usr/bin/sandbox-exec', '-f', str(profile), *map(str, argv)]
        env = {**environment, **(overrides or {})}
        record = {'phase': name, 'argv': actual, 'cwd': str(cwd), 'environment': env}
        commands.append(record)
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        started = time.monotonic()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            completed = subprocess.run(actual, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        record.update(returncode=completed.returncode, elapsed_seconds=time.monotonic() - started)
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(json.dumps({'phase': name, 'returncode': completed.returncode}), flush=True)
        completed.check_returncode()
    definitions = {**runtime['cmake_definitions'], 'NEMO_SPEECH_BUILD_TESTS': 'ON',
                   'CMAKE_EXPORT_COMPILE_COMMANDS': 'ON', 'CMAKE_INSTALL_PREFIX': str(prefix)}
    if args.cpu_arch != 'native':
        definitions.update(portable_definitions())
    run('configure', ['/bin/bash', 'scripts/configure.sh', 'cpu-diar',
                      *['-D' + key + '=' + value for key, value in definitions.items()]])
    materialized, inventory = audit_cpu_compiler_tree(source, output, run)
    run('build', ['/opt/homebrew/bin/cmake', '--build', '--preset', 'cpu-diar', '--parallel', str(args.jobs)])
    run('install', ['/opt/homebrew/bin/cmake', '--install', 'build/cpu-diar'])
    run('model-free-state-test', ['/opt/homebrew/bin/ctest', '--test-dir', 'build/cpu-diar',
                                  '--output-on-failure', '-R', '^diar_state$'])
    log = (output / 'model-free-state-test.stdout.log').read_text()
    require_single_test_pass(log, 'diar_state')
    if source_inventory(materialized) != inventory:
        raise ValueError('materialized compiler inputs changed during build')
    if capture(['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all'], source) != patch.read_bytes():
        raise ValueError('root source changed during construction')
    if capture(['git', 'status', '--porcelain', '--untracked-files=no'], source / 'llama.cpp'):
        raise ValueError('pristine llama changed during construction')
    build = source / 'build/cpu-diar'
    verify_completed_source(source, build, lock, patch, False)
    compiler_commands = json.loads((build / 'compile_commands.json').read_text())
    architecture = (audit_portable_commands(compiler_commands) if args.cpu_arch != 'native'
                    else {'mode': 'build-host-native', 'hardware_or_model_qualification': False})
    tracked = {(source / name.decode()).resolve() for name in
               capture(['git', 'ls-files', '-z'], source).split(b'\0') if name}
    compiler_inputs = {}
    for item in compiler_commands:
        path = Path(item['file']).resolve()
        if path.is_relative_to(materialized.resolve()):
            if descriptor(path) != inventory.get(path.relative_to(materialized).as_posix()):
                raise ValueError('actual compiler source differs from independent patch replay')
        elif path not in tracked or path.is_relative_to(build) or not path.is_file():
            raise ValueError('unbound actual SDK compiler source: ' + str(path))
        compiler_inputs[str(path)] = descriptor(path)
    (output / 'compiler-inputs.json').write_text(json.dumps(compiler_inputs, indent=2) + '\n')
    (output / 'cpu-architecture.json').write_text(json.dumps(architecture, indent=2) + '\n')
    entry = prefix / 'lib/libnemo_speech_asr_c.dylib'
    query = ('import ctypes,sys; library=ctypes.CDLL(sys.argv[1]); '
             'version=library.nemo_speech_asr_version; version.restype=ctypes.c_char_p; '
             'print(version().decode("utf-8"))')
    run('version-only', [sys.executable, '-c', query, str(entry)])
    version = (output / 'version-only.stdout.log').read_text().strip()
    if version != runtime['reported_version']:
        raise ValueError('actual tagged C ABI version differs')
    libraries, paths = {}, {}
    for path in sorted((prefix / 'lib').iterdir()):
        if not path.is_file():
            continue
        paths[path.name] = {**descriptor(path), 'symlink_target': str(path.readlink()) if path.is_symlink() else None}
        if not path.is_symlink():
            libraries[path.name] = {'sha256': digest(path), 'size': path.stat().st_size,
                'dependencies': capture(['/usr/bin/otool', '-L', str(path)], source).decode(),
                'exports': capture(['/usr/bin/nm', '-gU', str(path)], source).decode()}
    toolchain = {name: capture([path, '--version'], source).decode().splitlines()[0]
                 for name, path in [('cmake', '/opt/homebrew/bin/cmake'), ('ninja', '/opt/homebrew/bin/ninja'),
                                    ('clang++', '/usr/bin/clang++')]}
    toolchain['sdk'] = capture(['/usr/bin/xcrun', '--show-sdk-version'], source).decode().strip()
    run('brew-packages', ['/opt/homebrew/bin/brew', 'list', '--versions', 'sentencepiece', 'abseil', 'protobuf'],
        overrides={'HOME': str(Path.home()), 'HOMEBREW_NO_AUTO_UPDATE': '1'})
    toolchain['packages'] = (output / 'brew-packages.stdout.log').read_text().strip()
    build = source / 'build/cpu-diar'
    for name in ['CMakeCache.txt', 'compile_commands.json']:
        (output / name).write_bytes((build / name).read_bytes())
    archive = build / 'src/runtime/ggml/libnemo_speech_runtime_ggml.a'
    record = {'runtime_revision': runtime['revision'], 'llama_revision': runtime['llama_revision'],
        'runtime_profile': lock['profile'], 'runtime_reported_version': version, 'lock_sha256': digest(lock_path),
        'compatibility_patches': runtime['compatibility_patches'], 'frame_contract': runtime['frame_contract'],
        'platform': platform.platform(), 'machine': platform.machine(), 'backend': 'cpu', 'preset': 'cpu-diar',
        'metal_shaders_embedded': False, 'commands': commands, 'cmake_definitions': definitions,
        'toolchain': toolchain, 'libraries': libraries, 'installed_library_paths': paths,
        'cpu_architecture': architecture, 'actual_compiler_sources': compiler_inputs,
        'runtime_archive': {'path': str(archive), **descriptor(archive)},
        'cmake_cache_sha256': digest(build / 'CMakeCache.txt'), 'model_free_state_test_completed': True,
        'version_function_only_executed': True, 'model_instantiated_or_inference_performed': False,
        'artifact_admitted_for_inference': False, 'production_qualified': False, 'release_qualified': False}
    (output / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'constructed': True, 'profile': lock['profile'], 'version': version, 'neural_calls': 0}))


if __name__ == '__main__':
    main()
