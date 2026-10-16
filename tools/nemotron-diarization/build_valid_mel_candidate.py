"""Build-only CPU candidate for the reviewed V3 valid-Mel compatibility patch."""
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import tarfile
import tempfile
import time

from artifacts import digest


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def capture(argv, cwd):
    return subprocess.check_output(argv, cwd=cwd)


def source_inventory(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in materialized compiler input: ' + str(path))
        if path.is_file():
            files[path.relative_to(root).as_posix()] = descriptor(path)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'prefix', 'preparation', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'valid-mel-candidate-build.protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('this candidate build protocol is macOS arm64 CPU only')
    if args.output.exists() or args.prefix.exists() or (args.source / 'build').exists():
        parser.error('refusing an existing candidate build, prefix or evidence directory')
    if str(args.source.resolve()) != protocol['candidate_source_path'] or str(args.prefix.resolve()) != protocol['candidate_prefix_path']:
        parser.error('candidate source/prefix path differs from frozen build')
    for name, expected in protocol['tools'].items():
        if descriptor(tools / name) != expected:
            raise ValueError('candidate build tool differs: ' + name)
    for name, expected in protocol['source_files'].items():
        if descriptor(args.source / name) != expected:
            raise ValueError('candidate build source differs: ' + name)
    if descriptor(tools / 'runtime.lock.json') != protocol['original_runtime_lock']:
        raise ValueError('historical runtime lock differs')
    patch = tools / 'patches/v3-finish-valid-mel.patch'
    if descriptor(patch) != protocol['patch']:
        raise ValueError('candidate patch payload differs')
    if descriptor(args.preparation) != protocol['source_preparation']:
        raise ValueError('candidate source preparation record differs')
    preparation = json.loads(args.preparation.read_text())
    if (preparation['source_candidate'] != str(args.source)
            or preparation['source_base_revision'] != protocol['base_revision']
            or preparation['llama_revision'] != protocol['llama_revision']
            or preparation['patch'] != protocol['patch']
            or preparation['patched_file_identity'] != descriptor(args.source / 'src/asr/diar/diar_pipeline.cpp')):
        raise ValueError('candidate source preparation binding differs')
    if capture(['git', 'rev-parse', 'HEAD'], args.source).decode().strip() != protocol['base_revision']:
        raise ValueError('candidate upstream base differs')
    delta = capture(['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all'], args.source)
    if delta != patch.read_bytes():
        raise ValueError('candidate has unrelated source changes')
    if capture(['git', 'diff', '--cached', '--binary', '--no-ext-diff', '--ignore-submodules=all'], args.source):
        raise ValueError('candidate has staged source changes')
    llama = args.source / 'llama.cpp'
    if capture(['git', 'rev-parse', 'HEAD'], llama).decode().strip() != protocol['llama_revision']:
        raise ValueError('candidate llama base differs')
    if capture(['git', 'status', '--porcelain', '--untracked-files=no'], llama):
        raise ValueError('candidate llama must be pristine before configure')
    for tree in [args.source, llama]:
        for flags in [[], ['--ignored']]:
            if capture(['git', 'ls-files', '--others', *flags, '--exclude-standard'], tree):
                raise ValueError('untracked/ignored files could enter candidate build: ' + str(tree))
    args.output.mkdir(parents=True)
    (args.output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (args.output / 'candidate-source.patch').write_bytes(delta)
    (args.output / 'source-preparation.json').write_bytes(args.preparation.read_bytes())
    profile = args.output / 'offline.sb'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* ' +
                       ' '.join('(subpath ' + json.dumps(path) + ')' for path in protocol['denied_paths']) + ')\n')
    environment = {'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'LC_ALL': 'C',
                   'GIT_LFS_SKIP_SMUDGE': '1'}
    commands = []
    flags = ['-D' + key + '=' + value for key, value in protocol['cmake_definitions'].items()]
    flags.append('-DCMAKE_INSTALL_PREFIX=' + str(args.prefix))
    preset = 'cpu-diar'
    phases = [('configure', ['/bin/bash', 'scripts/configure.sh', preset, *flags]),
              ('build', ['/opt/homebrew/bin/cmake', '--build', '--preset', preset, '--parallel', '6']),
              ('install', ['/opt/homebrew/bin/cmake', '--install', 'build/' + preset]),
              ('model-free-state-test', ['/opt/homebrew/bin/ctest', '--test-dir', 'build/' + preset,
                                         '--output-on-failure', '-R', '^diar_state$'])]
    def execute(name, argv, cwd=args.source, overrides=None):
        actual = ['/usr/bin/sandbox-exec', '-f', str(profile.resolve()), *argv]
        child_environment = {**environment, **(overrides or {})}
        metadata = {'phase': name, 'argv': actual, 'cwd': str(cwd), 'environment': child_environment}
        (args.output / (name + '.started.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        started = time.monotonic()
        with (args.output / (name + '.stdout.log')).open('wb') as stdout, (args.output / (name + '.stderr.log')).open('wb') as stderr:
            result = subprocess.run(actual, cwd=cwd, env=child_environment, stdout=stdout, stderr=stderr)
        metadata = {**metadata, 'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started}
        (args.output / (name + '.completed.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        commands.append(metadata)
        print(json.dumps({'phase': name, 'returncode': result.returncode}), flush=True)
        result.check_returncode()
    for name, argv in phases:
        execute(name, argv)
        if name == 'configure':
            # CMake builds a patched archive export, not the pristine submodule.
            # Independently replay its declared series before compiling anything.
            archive = args.output / 'pristine-llama.tar'
            execute('llama-archive', ['git', 'archive', '--format=tar', '-o', str(archive.resolve()),
                                     protocol['llama_revision'], '--', '.', ':(exclude)models',
                                     ':(exclude)docs', ':(exclude)media'], cwd=llama)
            series = []
            with tempfile.TemporaryDirectory(prefix='llama-replay-', dir=args.output) as temporary:
                expected_root = Path(temporary)
                with tarfile.open(archive) as source:
                    source.extractall(expected_root, filter='data')
                for line in (args.source / 'patches/series').read_text().splitlines():
                    patch_name = line.split('#', 1)[0].strip()
                    if not patch_name:
                        continue
                    patch_path = args.source / 'patches' / patch_name
                    normalized = args.output / 'current-llama.patch'
                    normalized.write_bytes(patch_path.read_bytes().replace(b'\r\n', b'\n'))
                    execute('llama-patch-' + str(len(series)),
                            ['git', 'apply', '--whitespace=nowarn', str(normalized.resolve())],
                            cwd=expected_root, overrides={'GIT_CEILING_DIRECTORIES': str(expected_root.parent.resolve())})
                    series.append({'name': patch_name, **descriptor(patch_path)})
                normalized.unlink()
                expected_inventory = source_inventory(expected_root)
                materialized = args.source / 'build/cpu-diar/_deps/llama.cpp'
                if source_inventory(materialized) != expected_inventory:
                    raise ValueError('actual materialized compiler tree differs from independent series replay')
            audit = {'base_revision': protocol['llama_revision'], 'archive': descriptor(archive),
                     'compiled_tree_path': str(materialized), 'exact_independent_replay_match': True,
                     'patch_series': series, 'files': expected_inventory}
            (args.output / 'llama-materialized-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    state_log = (args.output / 'model-free-state-test.stdout.log').read_text()
    if (not re.search(r'1/1 Test #\d+: diar_state\s+.*Passed', state_log)
            or '100% tests passed, 0 tests failed out of 1' not in state_log):
        raise ValueError('required model-free diar_state test did not execute and pass')
    if capture(['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all'], args.source) != delta:
        raise ValueError('candidate source changed during build')
    if source_inventory(materialized) != expected_inventory:
        raise ValueError('materialized compiler source changed during build')
    if capture(['git', 'status', '--porcelain', '--untracked-files=no'], llama):
        raise ValueError('pristine llama submodule changed during build')
    libraries = {}
    for path in sorted((args.prefix / 'lib').iterdir()):
        if path.is_file():
            libraries[path.name] = {**descriptor(path), 'symlink_target': str(path.readlink()) if path.is_symlink() else None,
                                    'dependencies': capture(['/usr/bin/otool', '-L', str(path)], args.source).decode()}
    build = args.source / 'build' / preset
    for name in ['CMakeCache.txt', 'compile_commands.json']:
        (args.output / name).write_bytes((build / name).read_bytes())
    archive = build / 'src/runtime/ggml/libnemo_speech_runtime_ggml.a'
    compiler = capture(['/usr/bin/clang++', '--version'], args.source).decode()
    toolchain = {name: capture([path, '--version'], args.source).decode().splitlines()[0]
                 for name, path in [('cmake', '/opt/homebrew/bin/cmake'), ('ninja', '/opt/homebrew/bin/ninja')]}
    record = {'source_commit': capture(['git', 'rev-parse', 'HEAD'], tools).decode().strip(),
              'scope': protocol['scope'], 'base_revision': protocol['base_revision'], 'patch': descriptor(patch),
              'patched_file': descriptor(args.source / 'src/asr/diar/diar_pipeline.cpp'),
              'llama_revision': protocol['llama_revision'], 'candidate_source_path': str(args.source),
              'candidate_prefix_path': str(args.prefix), 'protocol_sha256': digest(protocol_path),
              'cmake_definitions': protocol['cmake_definitions'], 'commands': commands,
              'platform': platform.platform(), 'compiler': compiler, 'toolchain': toolchain,
              'libraries': libraries, 'runtime_archive': {'path': str(archive), **descriptor(archive)},
              'neural_model_instantiation_or_inference': False, 'candidate_finish_executed': False,
              'model_free_diar_state_test_completed': True, 'artifact_admitted_for_inference': False,
              'production_runtime_changed': False, 'release_qualified': False, 'limits': protocol['limitations']}
    files = {p.relative_to(args.output).as_posix(): descriptor(p) for p in args.output.rglob('*') if p.is_file()}
    record['files'] = files
    (args.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'candidate_cpu_build_completed': True, 'library_paths': len(libraries), 'inference_admitted': False}))


if __name__ == '__main__':
    main()
