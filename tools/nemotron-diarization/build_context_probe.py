"""Build a model-free public-C-ABI context probe against an unchanged CPU package."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import time

from macos_child_environment import sandboxed_command
from macos_loaded_libraries import audit_loaded_libraries
from qualification_storage import StorageGuard, atomic_json
from sdk_archive import inventory


def descriptor(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    raw = args.protocol.read_bytes()
    p = json.loads(raw)
    out = args.output.resolve()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('fixed macOS arm64 construction required')
    if out.exists() or args.output.is_symlink() or out.is_relative_to(repo):
        parser.error('fresh output outside repository required')
    package = Path(p['package_path'])

    def frozen():
        if args.protocol.read_bytes() != raw:
            raise ValueError('construction protocol changed')
        for name, expected in p['tools'].items():
            if descriptor(tools / name) != expected:
                raise ValueError('construction source changed: ' + name)
        for entry in p['source_files'].values():
            if descriptor(entry['path']) != entry['identity']:
                raise ValueError('construction header/source changed: ' + entry['path'])
        if (descriptor(p['compiler']) != p['compiler_identity']
                or descriptor(p['control_python']) != p['control_python_identity']
                or inventory(package) != p['package_inventory']):
            raise ValueError('qualified compiler/package changed')

    frozen()
    out.mkdir(parents=True)
    (out / 'protocol.json').write_bytes(raw)
    result = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'constructed': False, 'model_calls': 0, 'commands': [], 'release_qualified': False,
        'limits': p['limits']}
    guard = None
    try:
        guard = StorageGuard(out, 3 * 2**30, 2**30, 256 * 2**20)
        work = out / 'work'
        source = work / 'sources'
        binary = work / 'bin' / 'native-context'
        binary.parent.mkdir(parents=True)
        for name, entry in p['source_files'].items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry['path'], target)
            if descriptor(target) != entry['identity']:
                raise ValueError('staged construction source differs')
        relocated = work / 'nemo'
        shutil.copytree(package, relocated, symlinks=True)
        if inventory(relocated) != p['package_inventory']:
            raise ValueError('relocated construction package differs')

        def staged_sources():
            paths = list(source.rglob('*'))
            if any(path.is_symlink() for path in paths):
                raise ValueError('staged construction source became an alias')
            identities = {path.relative_to(source).as_posix(): descriptor(path) for path in paths if path.is_file()}
            if identities != {name: entry['identity'] for name, entry in p['source_files'].items()}:
                raise ValueError('staged construction sources changed')
            return identities

        # Baseline read-only system access; construction writes only below out.
        # Direct source/SDK/checkout/Homebrew inputs and all model files are denied.
        profile = out / 'model-free.sb'
        protected = [source, relocated]
        denied_reads = [repo, package, *map(Path, p['denied_read_roots'])]
        profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n'
            + '(allow file-write* (subpath ' + json.dumps(str(out)) + ') (subpath "/private/tmp") (literal "/dev/null"))\n'
            + ''.join('(deny file-write* (subpath ' + json.dumps(str(path)) + '))\n' for path in protected)
            + ''.join('(deny file-read* (subpath ' + json.dumps(str(path)) + '))\n' for path in denied_reads)
            + '(deny file-read* (regex #".*\\.(gguf|onnx|nemo|safetensors|pt|ckpt|npy|f32)$"))\n')
        env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(out), 'TMPDIR': str(out),
               'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1'}

        def run(name, arguments, expected_code=0, trace=False):
            guard.before_model()
            child_env = dict(env)
            if trace:
                child_env['DYLD_PRINT_LIBRARIES'] = '1'
            argv, parent = sandboxed_command(arguments, profile, child_env)
            row = {'name': name, 'argv': argv, 'environment': child_env,
                   'expected_returncode': expected_code}
            result['commands'].append(row)
            atomic_json(out / 'construction.json', result)
            started = time.monotonic()
            try:
                with (out / (name + '.stdout')).open('wb') as stdout, (out / (name + '.stderr')).open('wb') as stderr:
                    process = subprocess.Popen(argv, env=parent, cwd=work, stdout=stdout, stderr=stderr,
                                               start_new_session=True)
                    row['returncode'] = guard.wait(process, row, 120)
                row['elapsed_seconds'] = time.monotonic() - started
                if row['returncode'] != expected_code:
                    raise ValueError('construction command failed: ' + name)
            finally:
                atomic_json(out / 'construction.json', result)

        # Exercise actual policy denials; the .gguf marker contains no model.
        marker = out / 'denied-model.gguf'
        marker.write_bytes(b'model-free policy control\n')
        control = ('from pathlib import Path\n'
            + 'assert Path(' + repr(str(source / 'native_context_probe.cpp')) + ').read_bytes()\n'
            + 'for path, mode in ' + repr([(str(marker), 'rb'), (str(source / 'native_context_probe.cpp'), 'wb'),
                                          (str(package / 'package.json'), 'rb')]) + ':\n'
            + ' try:\n  open(path, mode).close()\n except PermissionError:\n  continue\n'
            + ' raise RuntimeError("sandbox unexpectedly allowed " + path)\n'
            + 'print("three model/source controls denied")\n')
        run('sandbox-controls', [p['control_python'], '-B', '-c', control])
        run('compiler-version', [p['compiler'], '--version'])
        common = [p['compiler'], '-arch', 'arm64', '-mmacosx-version-min=26.0', '-isysroot', p['sdk'],
                  '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-x', 'c++', '-I' + str(source),
                  '-MMD', '-c']
        for name, relative in [('probe', 'native_context_probe.cpp'), ('sha256', 'sha256/sha256.c')]:
            staged_sources()
            run('compile-' + name, [*common, str(source / relative), '-MF', str(work / (name + '.d')),
                                   '-o', str(work / (name + '.o'))])
        dependencies = set()
        for name in ['probe', 'sha256']:
            text = (work / (name + '.d')).read_text().replace('\\\n', ' ')
            dependencies.update(shlex.split(text.split(':', 1)[1]))
        if dependencies != {str(source / name) for name in p['source_files']}:
            raise ValueError('non-system dependency closure differs from seven staged inputs')
        result['non_system_dependencies'] = sorted(dependencies)
        run('link', [p['compiler'], '-arch', 'arm64', '-mmacosx-version-min=26.0', '-isysroot', p['sdk'],
                     str(work / 'probe.o'), str(work / 'sha256.o'), '-L' + str(relocated / 'lib'),
                     '-lnemo_speech_asr_c', '-Wl,-rpath,@loader_path/../nemo/lib', '-o', str(binary)])
        run('invalid-profile', [str(binary), 'unused-model', 'unused-audio', '1', '-1', 'invalid',
                                str(out / 'unexpected.json')], expected_code=1, trace=True)
        trace = (out / 'invalid-profile.stderr').read_text()
        if trace.splitlines()[-1] != 'unknown fixed context profile' or (out / 'unexpected.json').exists():
            raise ValueError('fixed-profile rejection control differs')
        manifest = json.loads((relocated / 'package.json').read_text())
        result['invalid_profile_loaded_images'] = audit_loaded_libraries(trace, relocated, manifest, binary)
        result['binary'] = descriptor(binary)
        result['staged_source_identity'] = staged_sources()
        if inventory(relocated) != p['package_inventory']:
            raise ValueError('relocated package changed during compilation/control')
        frozen()
        result['constructed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        if guard is not None:
            guard.release()
            result['storage_guard'] = guard.record
        atomic_json(out / 'construction.json', result)
        files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*')
                 if path.is_file() and not path.is_symlink()}
        atomic_json(out / 'files.json', {'files': files})
    print(json.dumps({'constructed': result['constructed'], 'model_calls': 0, 'binary': result['binary']}, indent=2))


if __name__ == '__main__':
    main()
