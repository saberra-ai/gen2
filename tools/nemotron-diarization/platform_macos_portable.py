"""Construct and retain a portable Mac CPU candidate on a fresh CI host.

No model acquisition, loading, frontend or inference is admitted by this phase.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from build_valid_mel_candidate import descriptor
from macos_cpu_architecture import PORTABLE_ARM64
from qualification_storage import StorageGuard, atomic_json
from qualify_valid_mel_pnola import execute_guarded
from runtime_profiles import load_profile, profile_provenance
from sdk_archive import write_archive


SANDBOX_CONTROL = '''import pathlib, socket, sys
def denied(operation):
    try:
        operation()
    except PermissionError:
        return
    raise RuntimeError("sandbox operation unexpectedly permitted")
for name in sys.argv[1:]:
    denied(lambda name=name: pathlib.Path(name).read_bytes())
denied(lambda: socket.create_connection(("127.0.0.1", 9), timeout=1))
'''


def construction_policy(upstream, repo):
    return ('(version 1)\n(allow default)\n(deny network*)\n'
            + '(deny file-read* (regex #".*\\.(nemo|gguf|safetensors|onnx|npy|f32)$"))\n'
            + '(deny file-write* (subpath ' + json.dumps(str(upstream)) + ')'
            + ' (subpath ' + json.dumps(str(repo)) + '))\n')


def construction_roots(paths):
    lexical = [Path(p).absolute() for p in paths]
    if any(p.exists() or p.is_symlink() for p in lexical[1:]):
        raise ValueError('fresh construction destinations required')
    roots = [p.resolve() for p in lexical]
    if len(set(roots)) != len(roots) or any(a != b and a.is_relative_to(b) for a in roots for b in roots):
        raise ValueError('all source/artifact/evidence roots must be disjoint')
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'source', 'sdk', 'package', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).with_name('macos-portable-construction-v2.protocol.json'))
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('portable candidate construction requires an actual arm64 Mac')
    try:
        upstream, source, sdk, package, output = construction_roots(
            [getattr(args, name) for name in ['upstream', 'source', 'sdk', 'package', 'output']])
    except ValueError as error:
        parser.error(str(error))
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    protocol_path = args.protocol.resolve()
    protocol_bytes = protocol_path.read_bytes()
    p = json.loads(protocol_bytes)
    lock_path, lock = load_profile()

    def frozen():
        if protocol_path.read_bytes() != protocol_bytes or p['runtime_provenance'] != profile_provenance(lock_path, lock):
            raise ValueError('portable construction protocol/profile changed')
        for name, expected in p['tools'].items():
            if descriptor(tools / name) != expected:
                raise ValueError('portable construction tool differs: ' + name)
        for name, expected in p['repository_files'].items():
            if descriptor(repo / name) != expected:
                raise ValueError('portable construction repository input differs: ' + name)

    frozen()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    policy = output / 'construction.sb'
    policy.write_text(construction_policy(upstream, repo))
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'GIT_LFS_SKIP_SMUDGE': '1',
                   'HOMEBREW_NO_AUTO_UPDATE': '1', 'HOMEBREW_NO_INSTALL_CLEANUP': '1'}
    commands = []
    result = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              'protocol': descriptor(protocol_path), 'commands': commands, 'model_calls': 0,
              'model_or_frontend_execution_admitted': False, 'release_qualified': False}
    persist = lambda: atomic_json(output / 'qualification.json', result)
    guard = None

    def run(name, argv, *, sandbox=True, timeout=1200):
        if guard is None:
            raise ValueError('construction storage guard missing')
        guard.before_model()
        actual = ['/usr/bin/sandbox-exec', '-f', str(policy), *map(str, argv)] if sandbox else list(map(str, argv))
        item = {'name': name, 'argv': actual, 'cwd': str(repo)}
        commands.append(item)
        persist()
        started = time.monotonic()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(actual, cwd=repo, environment=environment, stdout=stdout, stderr=stderr,
                            guard=guard, record=item, persist=persist, timeout=timeout)
        item['elapsed_seconds'] = time.monotonic() - started
        persist()
        if item['returncode'] != 0:
            raise ValueError('portable construction phase failed: ' + name)

    try:
        guard = StorageGuard(output, 3 * 1024**3, 1024**3, 256 * 1024**2)
        # Positive files and loopback are sufficient to distinguish real sandbox denial.
        probes = [output / ('denied-probe.' + suffix)
                  for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for probe in probes:
            probe.write_bytes(b'not a model')
            if probe.read_bytes() != b'not a model':
                raise ValueError('sandbox positive control file differs')
        run('sandbox-controls', [sys.executable, '-c', SANDBOX_CONTROL, *probes])
        for probe in probes:
            probe.unlink()
        facts = {'platform': platform.platform(), 'machine': platform.machine(),
                 'python': sys.version, 'cpu': subprocess.check_output(
                     ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                 'os': subprocess.check_output(['/usr/bin/sw_vers'], text=True),
                 'xcode': subprocess.check_output(['/usr/bin/xcodebuild', '-version'], text=True)}
        atomic_json(output / 'host.json', facts)
        run('prepare', [sys.executable, tools / 'prepare_valid_mel_source.py', '--upstream', upstream,
                        '--destination', source, '--output', output / 'preparation'])
        # The builder applies the same construction sandbox to each compiler/test child.
        run('build', [sys.executable, tools / 'build_valid_mel_sdk.py', '--source', source,
                      '--prefix', sdk, '--output', output / 'build', '--jobs', '2',
                      '--cpu-arch', PORTABLE_ARM64, '--sandbox-profile', policy], sandbox=False)
        run('package', [sys.executable, tools / 'package_macos.py', '--sdk-prefix', sdk,
                        '--build-receipt', output / 'build/build.json', '--output', package])
        # Keep original SDK/Homebrew bytes for independent relocation-payload checks
        # after the disposable host disappears. No model/other user files are read.
        originals = output / 'source-libraries'
        originals.mkdir()
        mapping = {}
        manifest = json.loads((package / 'package.json').read_text())
        for name, record in manifest['libraries'].items():
            original = Path(record['source'])
            if descriptor(original)['sha256'] != record['source_sha256']:
                raise ValueError('original package source library changed')
            destination = originals / name
            shutil.copy2(original, destination)
            mapping[name] = {'source': str(original), **descriptor(destination)}
        atomic_json(output / 'source-library-mapping.json', mapping)
        frozen()
        result['sdk_archive'] = write_archive(sdk, output, 'portable-cpu-sdk')
        result['package_archive'] = write_archive(package, output, 'portable-cpu-runtime')
        result['construction_and_package_passed'] = True
        result['cpu_architecture'] = json.loads((output / 'build/cpu-architecture.json').read_text())
        result['real_model_or_receiver_inference_qualified'] = False
    except BaseException as error:
        result.update(construction_and_package_passed=False, error=repr(error))
        raise
    finally:
        if guard is not None:
            guard.release()
        result['storage'] = guard.record if guard is not None else {'admission_failed_before_child': True}
        persist()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
