"""Build the fixed paced public-API helper offline without model execution."""
import argparse
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import time

from build_context_probe import descriptor
from macos_child_environment import sandboxed_command
from qualification_storage import StorageGuard, atomic_json


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

    def frozen():
        if args.protocol.read_bytes() != raw:
            raise ValueError('paced build protocol changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, identity in p[group].items():
                if descriptor(root / name) != identity:
                    raise ValueError('paced build source changed: ' + name)
        for path, identity in p['compiler_files'].items():
            if descriptor(path) != identity:
                raise ValueError('paced compiler/config changed: ' + path)

    frozen()
    out.mkdir(parents=True)
    (out / 'protocol.json').write_bytes(raw)
    profile = out / 'model-free.sb'
    mel = repo / 'assets/bark/nemo128.f32'
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n'
        + '(deny file-write* (subpath ' + json.dumps(str(repo)) + '))\n'
        + '(deny file-read* (require-all (regex #".*\\.(gguf|nemo|onnx|safetensors|npy|f32)$") '
          '(require-not (literal ' + json.dumps(str(mel)) + '))))\n')
    env = {'PATH': str(Path(p['cargo']).parent) + ':/usr/bin:/bin:/usr/sbin:/sbin',
           'HOME': '/Users/victor', 'CARGO_HOME': '/Users/victor/.cargo', 'RUSTC': p['rustc'],
           'CARGO_TARGET_DIR': p['target_directory'], 'CARGO_BUILD_JOBS': '2', 'LC_ALL': 'C',
           'PYTHONDONTWRITEBYTECODE': '1'}
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
         'constructed': False, 'model_calls': 0, 'commands': [], 'release_qualified': False, 'limits': p['limits']}
    guard = None
    persist = lambda: atomic_json(out / 'construction.json', r)

    def run(name, argv):
        frozen()
        guard.before_model()
        actual, environment = sandboxed_command(argv, profile, env)
        row = {'name': name, 'argv': actual, 'environment': environment, 'cwd': str(repo)}
        r['commands'].append(row)
        persist()
        started = time.monotonic()
        try:
            with (out / (name + '.stdout.log')).open('xb') as stdout, (out / (name + '.stderr.log')).open('xb') as stderr:
                process = subprocess.Popen(actual, cwd=repo, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
                row['returncode'] = guard.wait(process, row, 600)
            row['elapsed_seconds'] = time.monotonic() - started
            if row['returncode'] != 0:
                raise ValueError('paced construction command failed: ' + name)
        finally:
            persist()

    try:
        guard = StorageGuard(out, 3 * 2**30, 2**30, 256 * 2**20)
        marker = out / 'denied.gguf'
        marker.write_bytes(b'not a model')
        control = '''import pathlib,socket,sys
for operation in [lambda:pathlib.Path(sys.argv[1]).read_bytes(),lambda:pathlib.Path(sys.argv[2]).open('ab'),lambda:socket.create_connection(('127.0.0.1',9),timeout=1)]:
 try:operation()
 except PermissionError:continue
 raise RuntimeError('construction sandbox unexpectedly allowed model/source/network access')
assert pathlib.Path(sys.argv[3]).read_bytes()
print('model/source/network denials and exact mel source exception pass')
'''
        run('sandbox-controls', ['/usr/bin/python3', '-B', '-c', control, str(marker), str(repo / 'Cargo.toml'), str(mel)])
        run('rustc-version', [p['rustc'], '-vV'])
        run('rustfmt-check', [p['rustfmt'], '--edition', '2024', '--check', str(repo / 'examples/nemotron_paced_stream.rs')])
        run('cargo-build', [p['cargo'], 'build', '--offline', '--locked', '--release', '--no-default-features',
                            '--features', 'backend-diar-nemo', '--example', 'nemotron_paced_stream', '--message-format=json'])
        run('error-evidence-regression', [p['cargo'], 'test', '--offline', '--locked', '--release', '--no-default-features',
            '--features', 'backend-diar-nemo', '--example', 'nemotron_paced_stream', '--', '--exact',
            'tests::partial_evidence_survives_replay_error'])
        artifacts = [json.loads(line) for line in (out / 'cargo-build.stdout.log').read_text().splitlines() if line.startswith('{')]
        candidates = [row for row in artifacts if row.get('reason') == 'compiler-artifact'
                      and row['target']['name'] == 'nemotron_paced_stream' and row.get('executable')]
        if len(candidates) != 1:
            raise ValueError('exact paced Cargo executable artifact required')
        artifact = candidates[0]
        binary = Path(artifact['executable'])
        dependency = binary.with_suffix('.d')
        dependencies = shlex.split(dependency.read_text().replace('\\\n', ' ').splitlines()[0].split(':', 1)[1])
        used = {}
        for name in dependencies:
            path = Path(name).resolve(strict=True)
            if path.is_relative_to(repo):
                relative = path.relative_to(repo).as_posix()
                if relative not in p['repository_files'] or descriptor(path) != p['repository_files'][relative]:
                    raise ValueError('unbound Gen2 dependency in paced helper: ' + relative)
                used[relative] = descriptor(path)
        if 'examples/nemotron_paced_stream.rs' not in used or not any(name.startswith('src/diarization/') for name in used):
            raise ValueError('paced helper/core dependency records missing')
        (out / 'bin').mkdir()
        shutil.copy2(binary, out / 'bin/paced-public')
        shutil.copy2(dependency, out / 'paced-public.d')
        r.update(cargo_artifact=artifact, binary=descriptor(out / 'bin/paced-public'),
                 used_gen2_source_files=used, constructed=True)
        frozen()
    except BaseException as error:
        r['constructed'] = False
        r['error'] = repr(error)
        raise
    finally:
        if guard is not None:
            guard.release()
            r['storage'] = guard.record
        persist()
        atomic_json(out / 'files.json', {'files': {path.relative_to(out).as_posix(): descriptor(path)
                    for path in out.rglob('*') if path.is_file()}})
    print(json.dumps({'constructed': True, 'model_calls': 0, 'binary': r['binary']}))


if __name__ == '__main__':
    main()
